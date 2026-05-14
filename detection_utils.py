"""Detection evaluation, postprocessing, and training-loss utilities for YOLOv8-style models.

Mirrors the structure of Ultralytics BaseValidator/v8DetectionLoss but is
self-contained and works with the SparseFlow datamodule batch format:

    batch = {
        "images": Tensor[B, C, H, W],
        "labels": List[Tensor[N]],   # class indices per image
        "bboxes": List[Tensor[N,4]], # normalised cxcywh per image
    }

Model output (eval mode): (pred_tensor, aux)
    pred_tensor: Tensor[B, 4+nc, A]
        - [:, :4, :] → cxcywh in input-image pixel space
        - [:, 4:, :] → class probabilities (sigmoid applied)

Model output (train mode): dict with keys "boxes" [B, 4*reg_max, A],
    "scores" [B, nc, A], "feats" [list of per-level feature maps].
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops
from torch import Tensor, nn
from torchmetrics.detection import MeanAveragePrecision
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import (
    TaskAlignedAssigner,
    dist2bbox,
    make_anchors,
    bbox2dist,
)


def _normalized_targets(
    labels_list: List[Tensor],
    bboxes_list: List[Tensor],
    image_size: int,
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    """
    Convert lists of normalized labels and bboxes into per-image dicts for metric evaluation.
    Each dict contains:
    - "boxes": Tensor[N, 4] in pixel-space cxcywh format
    - "labels": Tensor[N] of class indices
    """
    result: List[Dict[str, Tensor]] = []
    for labels_i, bboxes_i in zip(labels_list, bboxes_list):
        if bboxes_i.shape[0] == 0:
            result.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                }
            )
            continue
        boxes_cxcywh = bboxes_i.to(device) * image_size
        result.append(
            {
                "boxes": boxes_cxcywh,
                "labels": labels_i.to(device).view(-1).long(),
            }
        )
    return result


def prepare_eval_batch(
    batch: Dict[str, Any], device: torch.device
) -> tuple[Tensor, List[Dict[str, Tensor]]]:
    """Prepare images and targets for metric evaluation."""
    images = batch["images"].to(device)
    targets = _normalized_targets(
        labels_list=batch["labels"],
        bboxes_list=batch["bboxes"],
        image_size=images.shape[-1],
        device=device,
    )
    return images, targets


class DetectionPostprocessor:  # pylint: disable=too-few-public-methods
    """Convert raw model outputs into final per-image detections.

    This utility is prediction-oriented: it extracts the decoded prediction
    tensor, applies confidence filtering, runs class-aware NMS, and returns
    final ``boxes`` / ``scores`` / ``labels`` dictionaries per image.
    """

    def __init__(
        self,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @staticmethod
    def _prediction_tensor(outputs: Tuple[Tensor, ...] | Tensor) -> Tensor:
        """Extract the decoded prediction tensor from the model output."""
        return outputs[0] if isinstance(outputs, tuple) else outputs

    @staticmethod
    def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
        cx, cy, bw, bh = boxes.unbind(-1)
        return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)

    def __call__(self, outputs: Tuple[Tensor, ...] | Tensor) -> List[Dict[str, Tensor]]:
        """Apply confidence threshold and NMS to model predictions.

        Args:
            outputs: A decoded prediction tensor, or a tuple whose first item
                is the decoded prediction tensor with shape ``[B, 4+nc, A]``.

        Returns:
            A list of per-image dicts containing pixel-space ``cxcywh`` boxes,
            confidence scores, and class labels.
        """
        pred = self._prediction_tensor(outputs)
        boxes_cxcywh = pred[:, :4, :].permute(0, 2, 1)  # [B, A, 4]
        class_probs = pred[:, 4:, :].permute(0, 2, 1)  # [B, A, nc]
        scores, labels = class_probs.max(dim=-1)  # [B, A]

        results: List[Dict[str, Tensor]] = []
        for batch_index in range(pred.shape[0]):
            keep_mask = scores[batch_index] >= self.conf_threshold
            boxes_b = boxes_cxcywh[batch_index][keep_mask]
            scores_b = scores[batch_index][keep_mask]
            labels_b = labels[batch_index][keep_mask]

            if boxes_b.shape[0] == 0:
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=pred.device),
                        "scores": torch.zeros((0,), device=pred.device),
                        "labels": torch.zeros(
                            (0,), dtype=torch.long, device=pred.device
                        ),
                    }
                )
                continue

            boxes_xyxy_b = self._cxcywh_to_xyxy(boxes_b)
            nms_idx = tv_ops.batched_nms(
                boxes_xyxy_b, scores_b, labels_b, self.iou_threshold
            )
            results.append(
                {
                    "boxes": boxes_b[nms_idx],
                    "scores": scores_b[nms_idx],
                    "labels": labels_b[nms_idx].long(),
                }
            )
        return results


class DetectionMetricsEvaluator:
    """Accumulate detection predictions and compute COCO-style mAP.
    This utility is evaluation-oriented: it handles the full loop of preparing
    batches, postprocessing model outputs, accumulating metric state, and
    computing final metrics at the end of an epoch.

    usage example:
    ```
    evaluator = DetectionMetricsEvaluator()
    for batch in dataloader:
        images, targets = evaluator.preprocess(batch, device)
        outputs = model(images)
        preds = evaluator.postprocess(outputs)
        evaluator.update(preds, targets)
    final_metrics = evaluator.compute()
    print(final_metrics)
    ```
    """

    def __init__(
        self,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
    ) -> None:
        self.postprocessor = DetectionPostprocessor(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        # max_detection_thresholds must include 100 because pycocotools _summarize() hard-codes
        # maxDets=100 when computing AP@0.5:0.95 (stats[0]). Using [1, 10, 300] omits 100 and
        # causes stats[0] to return -1 (COCO sentinel for "no match found") while stats[1]
        # (AP@0.5) uses p.maxDets[-1] and would still return a valid value.
        self._metric = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 100],
        )
        self._metric.warn_on_many_detections = False

    def preprocess(
        self, batch: Dict[str, Any], device: torch.device
    ) -> tuple[Tensor, List[Dict[str, Tensor]]]:
        """Prepare images and ground truth targets for metric updates."""
        return prepare_eval_batch(batch, device)

    def postprocess(self, outputs: Tuple[Tensor, ...]) -> List[Dict[str, Tensor]]:
        """Convert raw model outputs into metric-ready detections."""
        return self.postprocessor(outputs)

    def update(
        self,
        preds: List[Dict[str, Tensor]],
        targets: List[Dict[str, Tensor]],
    ) -> None:
        """Accumulate a batch of predictions and ground-truth into the metric."""
        self._metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Finalizes and return a flat metrics dict."""
        raw = self._metric.compute()
        return {
            "mAP50-95": float(raw.get("map", float("nan"))),
            "mAP50": float(raw.get("map_50", float("nan"))),
        }

    def reset(self) -> None:
        """Reset accumulated state for a fresh evaluation pass."""
        self._metric.reset()

    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        dataloader: Any,
        device: torch.device,
    ) -> Dict[str, float]:
        """Run a full evaluation loop and return aggregate metrics."""
        model.eval()
        self.reset()
        for batch in dataloader:
            images, targets = self.preprocess(batch, device)
            outputs = model(images)
            preds = self.postprocess(outputs)
            self.update(preds, targets)
        return self.compute()


# ---------------------------------------------------------------------------
# Detection Loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionLossComponents:
    """Weighted detection loss components for logging and optimization."""

    loss: Tensor
    loss_box: Tensor
    loss_cls: Tensor
    loss_dfl: Tensor


class _DFLoss(nn.Module):
    """Distribution Focal Loss over a discrete distribution of reg_max bins."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        """Return per-anchor DFL loss (mean over 4 offsets, keepdim)."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1.0 - wl
        loss = (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        )
        return loss.mean(-1, keepdim=True)


class DetectionLoss(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Detection loss: BCE classification + CIoU box + DFL regression.

    Accepts the raw training-mode output from the Detect head:
        preds = {"boxes": Tensor[B, 4*reg_max, A],
                 "scores": Tensor[B, nc, A],
                 "feats": List[Tensor[B, C, H_i, W_i]]}

    And the SparseFlow batch format:
        batch = {"images": Tensor[B, C, H, W],
                 "labels": List[Tensor[N]],    # class indices (0-indexed)
                 "bboxes": List[Tensor[N, 4]]} # normalised cxcywh

    Loss weights match Ultralytics defaults: box=7.5, cls=0.5, dfl=1.5.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        strides: Tensor,
        num_classes: int = 80,
        reg_max: int = 16,
        loss_box: float = 7.5,
        loss_cls: float = 0.5,
        loss_dfl: float = 1.5,
        tal_topk: int = 10,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.loss_box = loss_box
        self.loss_cls = loss_cls
        self.loss_dfl = loss_dfl

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl = _DFLoss(reg_max)
        # proj vector for converting DFL distribution → scalar offset
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32))
        self.register_buffer("strides", strides.float())

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=num_classes,
            alpha=0.5,
            beta=6.0,
            stride=strides.tolist(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_boxes(self, pred_dist: Tensor, anchor_points: Tensor) -> Tensor:
        """DFL decode: [B, A, 4*reg_max] → [B, A, 4] xyxy in anchor units."""
        b, a, c = pred_dist.shape
        pred_dist = (
            pred_dist.view(b, a, 4, c // 4)
            .softmax(3)
            .matmul(self.proj.to(pred_dist.dtype))
        )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _build_targets(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        labels_list: List[Tensor],
        bboxes_list: List[Tensor],
        batch_size: int,
        imgsz: Tensor,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pack per-image ground-truth into padded batch tensors.

        Returns:
            gt_labels  [B, N_max, 1]  long class indices
            gt_bboxes  [B, N_max, 4]  xyxy in pixel space
            mask_gt    [B, N_max, 1]  bool — True for valid rows
        """
        max_boxes = max((b.shape[0] for b in bboxes_list), default=0)
        if max_boxes == 0:
            z = torch.zeros(batch_size, 0, 1, device=device)
            return z.long(), torch.zeros(batch_size, 0, 4, device=device), z.bool()

        gt_labels = torch.zeros(
            batch_size, max_boxes, 1, dtype=torch.long, device=device
        )
        gt_bboxes = torch.zeros(batch_size, max_boxes, 4, device=device)
        mask_gt = torch.zeros(batch_size, max_boxes, 1, dtype=torch.bool, device=device)

        # imgsz is [H, W] — scale factor for normalised cxcywh → pixel xyxy
        scale = imgsz[[1, 0, 1, 0]]  # [W, H, W, H]

        for i, (labels_i, bboxes_i) in enumerate(zip(labels_list, bboxes_list)):
            n = bboxes_i.shape[0]
            if n == 0:
                continue
            # normalised cxcywh → pixel xyxy
            cxcywh = bboxes_i.to(device) * scale
            cx, cy, bw, bh = cxcywh.unbind(-1)
            xyxy = torch.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1
            )
            gt_labels[i, :n, 0] = labels_i.to(device).view(-1).long()
            gt_bboxes[i, :n] = xyxy
            mask_gt[i, :n, 0] = True

        return gt_labels, gt_bboxes, mask_gt

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_components(  # pylint: disable=too-many-locals
        self, preds: Dict[str, Tensor], batch: Dict[str, Any]
    ) -> DetectionLossComponents:
        """Compute weighted detection loss components for one training step.

        Args:
            preds: dict with "boxes" [B, 4*reg_max, A], "scores" [B, nc, A],
                   "feats" list of per-level tensors.
            batch: SparseFlow batch with "images", "labels", "bboxes".

        Returns:
            Weighted box, class, DFL, and total loss tensors. Components are
            scaled the same way as the returned scalar loss.
        """
        device = preds["boxes"].device
        images: Tensor = batch["images"].to(device)
        batch_size = images.shape[0]
        imgsz = torch.tensor(
            images.shape[2:], dtype=torch.float32, device=device
        )  # [H, W]

        # Permute to [B, A, *]
        pred_dist = preds["boxes"].permute(0, 2, 1).contiguous()  # [B, A, 4*reg_max]
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()  # [B, A, nc]

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.strides, 0.5)

        # Decode predicted boxes (in anchor units, then scale to pixel space)
        pred_bboxes = self._decode_boxes(
            pred_dist, anchor_points
        )  # [B, A, 4] xyxy anchor-units

        # Build ground-truth tensors
        labels_list: List[Tensor] = batch["labels"]
        bboxes_list: List[Tensor] = batch["bboxes"]
        gt_labels, gt_bboxes, mask_gt = self._build_targets(
            labels_list, bboxes_list, batch_size, imgsz, device
        )

        # Task-Aligned Assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).to(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp(min=1)

        # Classification loss (BCE with logits)
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / target_scores_sum

        # Box + DFL losses (foreground anchors only)
        loss_iou = pred_scores.new_zeros(1)
        loss_dfl = pred_scores.new_zeros(1)
        if fg_mask.any():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # stride per foreground anchor: stride_tensor [A, 1] → [B, A, 1] → [N_fg, 1]
            stride_fg = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)[fg_mask]

            # CIoU loss (pixel space)
            iou = bbox_iou(
                pred_bboxes[fg_mask] * stride_fg,
                target_bboxes[fg_mask],
                xywh=False,
                CIoU=True,
            )
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

            # DFL loss — targets in anchor units (same space as decoded pred_dist)
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes / stride_tensor, self.reg_max - 1
            )
            loss_dfl = (
                self.dfl(
                    pred_dist[fg_mask].view(-1, self.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            ).sum() / target_scores_sum

        loss_box = self.loss_box * loss_iou * batch_size
        loss_cls_weighted = self.loss_cls * loss_cls * batch_size
        loss_dfl_weighted = self.loss_dfl * loss_dfl * batch_size
        loss = loss_box + loss_cls_weighted + loss_dfl_weighted

        return DetectionLossComponents(
            loss=loss,
            loss_box=loss_box,
            loss_cls=loss_cls_weighted,
            loss_dfl=loss_dfl_weighted,
        )

    def forward(self, preds: Dict[str, Tensor], batch: Dict[str, Any]) -> Tensor:
        """Compute scalar weighted detection loss for one training step."""
        return self.forward_components(preds, batch).loss
