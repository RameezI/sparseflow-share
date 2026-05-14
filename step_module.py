"""Lightning training and evaluation protocol for YOLOv8-style detectors."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import lightning as L
import torch
from torch import Tensor, nn

from detection_utils import (
    DetectionLoss,
    DetectionLossComponents,
    DetectionMetricsEvaluator,
    DetectionPostprocessor,
)
from models.axon_hillock import AxonHillock

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer and scheduler constants matching the YOLOv8 training protocol."""

    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    lr_final_factor: float = 0.01


def load_pretrained_weights(model: nn.Module, pretrained_weights_path: str) -> None:
    """Transfer weights from a YOLOv8 .pt checkpoint into model."""
    LOGGER.info("Loading pretrained weights: %s", pretrained_weights_path)
    ckpt = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
    state_dict = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info(
        "Weights loaded — missing: %d, unexpected: %d", len(missing), len(unexpected)
    )


class StepModule(L.LightningModule):  # pylint: disable=too-many-instance-attributes
    """Shared training and evaluation protocol for YOLOv8-style detection models."""

    def __init__(
        self,
        model: nn.Module,
        base_learning_rate: float = 0.01,
        batch_size: int = 16,
        pretrained_weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.base_learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.pretrained_weights_path = pretrained_weights_path

        self._postprocessor = DetectionPostprocessor()
        self._val_evaluator = DetectionMetricsEvaluator(
            conf_threshold=self._postprocessor.conf_threshold,
            iou_threshold=self._postprocessor.iou_threshold,
        )
        self._test_evaluator = DetectionMetricsEvaluator(
            conf_threshold=self._postprocessor.conf_threshold,
            iou_threshold=self._postprocessor.iou_threshold,
        )
        self._loss_fn: Optional[DetectionLoss] = None
        self._train_epoch_start: Optional[float] = None
        self._val_epoch_start: Optional[float] = None

        if self.pretrained_weights_path:
            load_pretrained_weights(self.model, self.pretrained_weights_path)

    def forward(self, images: Tensor, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return self.model(images)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Do not persist derived loss buffers; they are rebuilt from Detect metadata."""
        state_dict = checkpoint.get("state_dict", {})
        for key in list(state_dict):
            if key.startswith("_loss_fn."):
                del state_dict[key]

    # ------------------------------------------------------------------
    # Loss — lazily built after strides are initialised
    # ------------------------------------------------------------------

    def _get_loss_fn(self) -> DetectionLoss:
        if self._loss_fn is None:
            detect = next(
                (
                    m
                    for m in self.model.modules()
                    if hasattr(m, "stride") and hasattr(m, "nl")
                ),
                None,
            )
            if detect is None:
                raise RuntimeError(
                    "Cannot find Detect layer to read strides for loss init."
                )
            self._loss_fn = DetectionLoss(
                strides=detect.stride,
                num_classes=detect.nc,
                reg_max=detect.reg_max,
            ).to(self.device)
        return self._loss_fn

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        return float(optimizer.param_groups[0]["lr"])

    def _reset_cuda_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _log_cuda_peak_memory(self, name: str) -> None:
        if self.device.type != "cuda":
            return
        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        self.log(name, peak_mb, on_step=False, on_epoch=True)

    def on_train_epoch_start(self) -> None:
        self._train_epoch_start = time.perf_counter()
        self._reset_cuda_peak_memory()

    def _collect_axon_regularization(self) -> Tensor:
        """Sum regularization terms from all AxonHillock modules in the model."""
        total = torch.tensor(0.0, device=self.device)
        for module in self.model.modules():
            if isinstance(module, AxonHillock):
                total = total + module.last_regularization.to(self.device)
        return total

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        del batch_idx, args, kwargs
        images: Tensor = batch["images"]
        preds = self(images)  # training mode → dict with "boxes", "scores", "feats"
        loss_components = self._get_loss_fn().forward_components(preds, batch)
        det_loss = loss_components.loss
        reg_loss = self._collect_axon_regularization()
        loss = det_loss + reg_loss
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        component_logs = {
            "train/loss_box": loss_components.loss_box,
            "train/loss_cls": loss_components.loss_cls,
            "train/loss_dfl": loss_components.loss_dfl,
            "train/loss_reg": reg_loss,
        }
        self.log_dict(
            component_logs,
            on_epoch=True,
            on_step=False,
            batch_size=images.shape[0],
        )
        current_lr = self._current_lr()
        self.log(
            "train/lr_step",
            current_lr,
            on_step=True,
            on_epoch=False,
            batch_size=images.shape[0],
        )
        self.log(
            "train/lr_epoch",
            current_lr,
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        return loss

    def on_train_epoch_end(self) -> None:
        if self._train_epoch_start is not None:
            duration = time.perf_counter() - self._train_epoch_start
            self.log("train/epoch_duration_sec", duration, on_step=False, on_epoch=True)
        self._log_cuda_peak_memory("train/gpu_memory_peak_mb")

    # ------------------------------------------------------------------
    # Validation steps
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_start = time.perf_counter()
        self._reset_cuda_peak_memory()

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del batch_idx, args, kwargs
        images, targets = self._val_evaluator.preprocess(batch, self.device)
        outputs = self(images)
        preds = self._val_evaluator.postprocess(outputs)
        self._val_evaluator.update(preds, targets)
        loss_components = self._loss_components_from_eval_outputs(outputs, batch)
        self.log(
            "val/loss",
            loss_components.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log_dict(
            {
                "val/loss_box": loss_components.loss_box,
                "val/loss_cls": loss_components.loss_cls,
                "val/loss_dfl": loss_components.loss_dfl,
            },
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )

    def on_validation_epoch_end(self) -> None:
        if self._val_epoch_start is not None:
            duration = time.perf_counter() - self._val_epoch_start
            self.log("val/epoch_duration_sec", duration, on_step=False, on_epoch=True)
        stats = self._val_evaluator.compute()
        self._val_evaluator.reset()
        for key, value in stats.items():
            self.log(f"val/{key}", value, prog_bar=(key in {"mAP50-95", "mAP50"}))
        self._log_cuda_peak_memory("val/gpu_memory_peak_mb")

    def _loss_components_from_eval_outputs(
        self,
        outputs: Any,
        batch: Dict[str, Any],
    ) -> DetectionLossComponents:
        """Compute validation loss from the Detect head aux output returned in eval mode."""
        if not (
            isinstance(outputs, tuple)
            and len(outputs) > 1
            and isinstance(outputs[1], dict)
        ):
            raise RuntimeError(
                "Validation loss requires eval outputs with Detect aux tensors."
            )
        return self._get_loss_fn().forward_components(outputs[1], batch)

    # ------------------------------------------------------------------
    # Test steps
    # ------------------------------------------------------------------

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del batch_idx, args, kwargs
        images, targets = self._test_evaluator.preprocess(batch, self.device)
        preds = self._test_evaluator.postprocess(self(images))
        self._test_evaluator.update(preds, targets)

    def on_test_epoch_end(self) -> None:
        stats = self._test_evaluator.compute()
        self._test_evaluator.reset()
        for key, value in stats.items():
            self.log(f"test/{key}", value)

    # ------------------------------------------------------------------
    # Optimiser + LR schedule
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer_config = OptimizerConfig()
        # Scale LR and weight decay linearly with batch size, matching Ultralytics
        # gradient accumulation protocol (nominal batch size nbs=64).
        nbs = 64
        lr_scale = self.batch_size / nbs
        scaled_lr = self.base_learning_rate * lr_scale
        scaled_wd = optimizer_config.weight_decay * lr_scale
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=scaled_lr,
            momentum=optimizer_config.momentum,
            weight_decay=scaled_wd,
            nesterov=True,
        )
        total_epochs = self.trainer.max_epochs if self.trainer else 100
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(optimizer_config.warmup_epochs, 1),
            end_factor=1.0,
            total_iters=optimizer_config.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - optimizer_config.warmup_epochs, 1),
            eta_min=scaled_lr * optimizer_config.lr_final_factor,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[optimizer_config.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
