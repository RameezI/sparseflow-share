"""Command-line entry points for SparseFlow training, evaluation, and prediction."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from data import CocoDatasetConfig, CocoDetectionDataModule
from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec
from step_module import StepModule


class SparseFlowProgressBar(TQDMProgressBar):
    """Project progress bar that keeps logger bookkeeping out of the display."""

    def get_metrics(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> Dict[str, Any]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def _build_model(model_spec: str, num_classes: int):
    return build_model_from_spec(model_spec=model_spec, num_classes=num_classes)


def _build_datamodule(args: argparse.Namespace) -> CocoDetectionDataModule:
    max_train = getattr(args, "max_train_samples", None) or None
    max_eval = getattr(args, "max_eval_samples", None) or None
    return CocoDetectionDataModule(
        config=CocoDatasetConfig(
            max_train_samples=max_train,
            max_eval_samples=max_eval,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def _cuda_runtime_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.empty(1, device="cuda")
        probe += 1
        torch.cuda.synchronize()
    except RuntimeError as exc:
        logging.warning("CUDA is visible but not usable by this PyTorch build: %s", exc)
        return False
    return True


def _build_trainer(
    max_epochs: Optional[int] = None,
    checkpoint_out: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_name: str = "baseline",
    device: str = "auto",
    check_val_every_n_epoch: int = 1,
) -> L.Trainer:
    callbacks = [SparseFlowProgressBar()]
    enable_checkpointing = False
    if checkpoint_out:
        dirpath = str(Path(checkpoint_out).parent)
        filename = Path(checkpoint_out).stem
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor="val/mAP50-95",
            mode="max",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = f"{filename}-last"
        callbacks.append(checkpoint_callback)
        enable_checkpointing = True

    accelerator = _resolve_accelerator(device)
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name) if log_dir else False

    return L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,
        inference_mode=False,
    )


def _resolve_accelerator(device: str) -> str:
    normalized = device.lower()
    if normalized == "auto":
        return "cuda" if _cuda_runtime_is_usable() else "cpu"
    if normalized == "cuda":
        if not _cuda_runtime_is_usable():
            raise RuntimeError("--device cuda was requested, but CUDA is not usable.")
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported --device value: {device!r}")


def _metrics_from_test_output(output: Any) -> Dict[str, float]:
    if isinstance(output, list) and output:
        return dict(output[0])
    return {}


def _metrics_from_validate_output(output: Any) -> Dict[str, float]:
    return _metrics_from_test_output(output)


def _logged_metrics_with_prefix(
    metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, float]:
    result = {}
    for key, value in metrics.items():
        key_name = str(key)
        if not key_name.startswith(prefix):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        result[key_name] = float(value)
    return result


def _checkpoint_paths(trainer: L.Trainer) -> Dict[str, Optional[str]]:
    ckpt_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
    if not ckpt_callbacks:
        return {"best": None, "last": None}
    callback = ckpt_callbacks[0]
    return {
        "best": callback.best_model_path or None,
        "last": callback.last_model_path or None,
    }


def run_train(args: argparse.Namespace) -> None:
    """Train the selected model and report initial/final validation metrics."""
    if args.check_val_every_n_epoch < 1:
        raise ValueError("--check-val-every-n-epoch must be >= 1.")

    datamodule = _build_datamodule(args)
    model = _build_model(args.model, num_classes=args.num_classes)
    protocol = StepModule(
        model=model,
        base_learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        pretrained_weights_path=args.pretrained_weights or None,
    )
    checkpoint_out = args.checkpoint_out or None
    trainer = _build_trainer(
        max_epochs=args.max_epochs,
        checkpoint_out=checkpoint_out,
        log_dir=args.log_dir or None,
        log_name=args.name,
        device=args.device,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    resume_from_checkpoint = args.resume_from_checkpoint or None
    if resume_from_checkpoint and not Path(resume_from_checkpoint).exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {resume_from_checkpoint}"
        )
    initial_validation_metrics = _metrics_from_validate_output(
        trainer.validate(
            protocol,
            datamodule=datamodule,
            ckpt_path=resume_from_checkpoint,
            verbose=True,
        )
    )
    trainer.fit(protocol, datamodule=datamodule, ckpt_path=resume_from_checkpoint)
    final_validation_metrics = _logged_metrics_with_prefix(
        trainer.callback_metrics,
        "val/",
    )
    metrics = _metrics_from_test_output(
        trainer.test(protocol, datamodule=datamodule, verbose=False)
    )

    checkpoint_paths = _checkpoint_paths(trainer)
    if checkpoint_out:
        if not checkpoint_paths["best"]:
            target = Path(checkpoint_out)
            target.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(target))
            checkpoint_paths["best"] = str(target)
        if not checkpoint_paths["last"]:
            last_target = Path(checkpoint_out).with_name(
                f"{Path(checkpoint_out).stem}-last.ckpt"
            )
            last_target.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(last_target))
            checkpoint_paths["last"] = str(last_target)

    print(
        json.dumps(
            {
                "command": "train",
                "model": args.model,
                "checkpoint": checkpoint_paths["best"],
                "checkpoints": checkpoint_paths,
                "resumed_from_checkpoint": resume_from_checkpoint,
                "device": args.device,
                "validation_cadence": f"every {args.check_val_every_n_epoch} epoch(s)",
                "log_dir": trainer.logger.log_dir if trainer.logger else None,
                "initial_validation_metrics": initial_validation_metrics,
                "final_validation_metrics": final_validation_metrics,
                "metrics": metrics,
            },
            indent=2,
        )
    )


def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate selected model weights or a Lightning checkpoint."""
    if not args.pretrained_weights:
        raise ValueError("Evaluation requires --pretrained-weights.")

    datamodule = _build_datamodule(args)
    protocol = _load_protocol_for_inference(
        args.pretrained_weights,
        args.model,
        args.num_classes,
    )
    trainer = _build_trainer(device=args.device)
    metrics = _metrics_from_test_output(
        trainer.test(protocol, datamodule=datamodule, verbose=False)
    )
    print(
        json.dumps(
            {
                "command": "evaluate",
                "model": args.model,
                "pretrained_weights": args.pretrained_weights,
                "device": args.device,
                "metrics": metrics,
            },
            indent=2,
        )
    )


def run_predict(args: argparse.Namespace) -> None:
    """Run prediction batches with selected weights or a Lightning checkpoint."""
    if not args.pretrained_weights:
        raise ValueError("Inference requires --pretrained-weights.")

    datamodule = _build_datamodule(args)
    protocol = _load_protocol_for_inference(
        args.pretrained_weights,
        args.model,
        args.num_classes,
    )
    trainer = _build_trainer(device=args.device)
    pred_batches = trainer.predict(protocol, datamodule=datamodule)
    print(
        json.dumps(
            {
                "command": "predict",
                "model": args.model,
                "pretrained_weights": args.pretrained_weights,
                "device": args.device,
                "num_batches": len(pred_batches) if pred_batches else 0,
            },
            indent=2,
        )
    )


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )


def _add_eval_limit_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-eval-samples", type=int, default=0)


def _load_protocol_for_inference(
    weights_or_checkpoint: str, model_spec: str, num_classes: int
) -> StepModule:
    model = _build_model(model_spec, num_classes=num_classes)
    path = Path(weights_or_checkpoint)
    if path.suffix == ".ckpt":
        return StepModule.load_from_checkpoint(
            str(path),
            model=model,
            strict=False,
        ).eval()
    return StepModule(model=model, pretrained_weights_path=weights_or_checkpoint).eval()


def build_parser() -> argparse.ArgumentParser:
    """Build the SparseFlow argument parser."""
    parser = argparse.ArgumentParser(description="SparseFlow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model and optionally save a checkpoint",
    )
    _add_shared_args(train_parser)
    _add_eval_limit_arg(train_parser)
    train_parser.add_argument("--pretrained-weights", type=str, default="")
    train_parser.add_argument("--max-epochs", type=int, default=100)
    train_parser.add_argument("--max-train-samples", type=int, default=0)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="checkpoints/baseline.ckpt",
    )
    train_parser.add_argument("--resume-from-checkpoint", type=str, default="")
    train_parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    train_parser.add_argument("--log-dir", type=str, default="logs/tensorboard")
    train_parser.add_argument("--name", type=str, default="baseline")
    train_parser.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a pretrained checkpoint"
    )
    _add_shared_args(eval_parser)
    _add_eval_limit_arg(eval_parser)
    eval_parser.add_argument("--pretrained-weights", type=str, required=True)
    eval_parser.set_defaults(func=run_evaluate)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction with pretrained weights",
    )
    _add_shared_args(predict_parser)
    predict_parser.add_argument("--pretrained-weights", type=str, required=True)
    predict_parser.set_defaults(func=run_predict)

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch the selected command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
