"""Phase 0 diagnostic: confirm SparseFlow reproduces the YOLOv8n baseline.

This script intentionally compares implementation pieces on the same transformed
Hugging Face COCO batches. It answers four questions:

1. Does the local SparseFlow model produce the same decoded tensor as the
   Ultralytics checkpoint model?
2. Does SparseFlow NMS produce the same mAP as Ultralytics NMS on those tensors?
3. Does SparseFlow image letterboxing match Ultralytics letterboxing?
4. Is the active validation split the same as canonical COCO val2017?

If 1-3 pass, the baseline is reproduced for this project protocol. If 4 differs,
published COCO numbers should not be used as a strict acceptance target for this
Hugging Face split.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np
from datasets import load_dataset
from torch import Tensor, nn
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox as UltralyticsLetterBox
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CocoDatasetConfig, CocoDetectionDataModule  # noqa: E402
from data.transforms import LetterBox as SparseFlowLetterBox  # noqa: E402
from detection_utils import DetectionPostprocessor, prepare_eval_batch  # noqa: E402
from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec  # noqa: E402
from step_module import load_pretrained_weights  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SparseFlow and Ultralytics postprocessing on the same transformed "
            "HF COCO val batches to isolate remaining mAP protocol differences."
        )
    )
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--letterbox-audit-samples", type=int, default=20)
    parser.add_argument("--model-parity-atol", type=float, default=1e-6)
    parser.add_argument("--nms-map-tolerance", type=float, default=1e-3)
    parser.add_argument("--letterbox-pixel-tolerance", type=int, default=0)
    return parser


def _prediction_tensor(outputs: Any) -> Tensor:
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def _new_metric() -> MeanAveragePrecision:
    metric = MeanAveragePrecision(
        box_format="cxcywh",
        iou_type="bbox",
        max_detection_thresholds=[1, 10, 100],
    )
    metric.warn_on_many_detections = False
    return metric


def _metric_result(metric: MeanAveragePrecision) -> Dict[str, float]:
    raw = metric.compute()
    return {
        "mAP50-95": float(raw["map"]),
        "mAP50": float(raw["map_50"]),
    }


def _build_datamodule(args: argparse.Namespace) -> CocoDetectionDataModule:
    datamodule = CocoDetectionDataModule(
        config=CocoDatasetConfig(max_eval_samples=args.max_samples),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="test")
    return datamodule


def _audit_dataset_and_letterbox(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_id = CocoDatasetConfig.dataset_id
    val_split = CocoDatasetConfig.val_split
    val_dataset = load_dataset(dataset_id, split=val_split)
    sparseflow_letterbox = SparseFlowLetterBox(new_shape=(640, 640), scaleup=False)
    ultralytics_letterbox = UltralyticsLetterBox(new_shape=(640, 640), scaleup=False)

    max_pixel_diff = 0
    checked = min(args.letterbox_audit_samples, len(val_dataset))
    mismatches = []
    for index in range(checked):
        sample = val_dataset[index]
        img = np.array(sample["image"].convert("RGB"))[:, :, ::-1]
        raw_boxes = sample["objects"].get("bbox", [])
        bboxes = (
            np.array(raw_boxes, dtype=np.float32)
            if raw_boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        sparseflow_img = sparseflow_letterbox(
            {"img": img.copy(), "bboxes": bboxes.copy()}
        )["img"]
        ultralytics_img = ultralytics_letterbox(image=img.copy())
        diff = int(
            np.abs(
                sparseflow_img.astype(np.int16) - ultralytics_img.astype(np.int16)
            ).max()
        )
        max_pixel_diff = max(max_pixel_diff, diff)
        if diff:
            mismatches.append(
                {
                    "index": index,
                    "max_pixel_diff": diff,
                    "input_shape": list(img.shape),
                    "sparseflow_shape": list(sparseflow_img.shape),
                    "ultralytics_shape": list(ultralytics_img.shape),
                }
            )

    return {
        "dataset_id": dataset_id,
        "val_split": val_split,
        "hf_val_images": len(val_dataset),
        "canonical_coco_val2017_images": 5000,
        "letterbox_checked": checked,
        "letterbox_max_pixel_diff": max_pixel_diff,
        "letterbox_mismatches": mismatches[:5],
    }


def _build_local_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model != "yolov8" and not args.model.startswith("yolov8:"):
        raise ValueError(
            "Ultralytics protocol comparison is only defined for dense yolov8 specs."
        )
    model = (
        build_model_from_spec(args.model, num_classes=args.num_classes)
        .to(device)
        .eval()
    )
    load_pretrained_weights(model, args.weights)
    return model


def _build_checkpoint_model(
    args: argparse.Namespace, device: torch.device
) -> nn.Module:
    return YOLO(args.weights).model.to(device).eval()


def _ultralytics_postprocess(
    outputs: Any,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    num_classes: int,
) -> List[Dict[str, Tensor]]:
    pred = _prediction_tensor(outputs)
    detections = non_max_suppression(
        pred.clone(),
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        max_det=max_det,
        nc=num_classes,
        max_time_img=10.0,
    )

    results: List[Dict[str, Tensor]] = []
    for det in detections:
        if det.numel() == 0:
            results.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=pred.dtype, device=pred.device),
                    "scores": torch.zeros((0,), dtype=pred.dtype, device=pred.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=pred.device),
                }
            )
            continue
        results.append(
            {
                "boxes": _xyxy_to_cxcywh(det[:, :4]),
                "scores": det[:, 4],
                "labels": det[:, 5].long(),
            }
        )
    return results


def _count_summary(counts: List[int]) -> Dict[str, float]:
    values = torch.tensor(counts, dtype=torch.float32)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def _top_detection(preds: List[Dict[str, Tensor]]) -> Dict[str, Any]:
    if not preds or preds[0]["scores"].numel() == 0:
        return {}
    score_index = int(preds[0]["scores"].argmax().item())
    return {
        "label": int(preds[0]["labels"][score_index].item()),
        "score": float(preds[0]["scores"][score_index].item()),
        "box_cxcywh": [
            round(float(x), 3) for x in preds[0]["boxes"][score_index].tolist()
        ],
    }


def _abs_delta(left: float, right: float) -> float:
    return abs(left - right)


def _build_analysis(
    args: argparse.Namespace,
    dataset: Dict[str, Any],
    metric_results: Dict[str, Dict[str, float]],
    first_batch: Dict[str, Any],
) -> Dict[str, Any]:
    local_sparseflow = metric_results["local_model_sparseflow_nms"]
    local_ultralytics = metric_results["local_model_ultralytics_nms"]
    checkpoint_ultralytics = metric_results["checkpoint_model_ultralytics_nms"]

    nms_delta = {
        "mAP50-95": _abs_delta(
            local_sparseflow["mAP50-95"], local_ultralytics["mAP50-95"]
        ),
        "mAP50": _abs_delta(local_sparseflow["mAP50"], local_ultralytics["mAP50"]),
    }
    model_metric_delta = {
        "mAP50-95": _abs_delta(
            local_ultralytics["mAP50-95"], checkpoint_ultralytics["mAP50-95"]
        ),
        "mAP50": _abs_delta(
            local_ultralytics["mAP50"], checkpoint_ultralytics["mAP50"]
        ),
    }

    checks = {
        "model_output_parity": (
            first_batch["local_vs_checkpoint_max_abs_diff"] <= args.model_parity_atol
        ),
        "model_metric_parity": (
            model_metric_delta["mAP50-95"] <= args.nms_map_tolerance
            and model_metric_delta["mAP50"] <= args.nms_map_tolerance
        ),
        "nms_metric_parity": (
            nms_delta["mAP50-95"] <= args.nms_map_tolerance
            and nms_delta["mAP50"] <= args.nms_map_tolerance
        ),
        "letterbox_image_parity": (
            dataset["letterbox_max_pixel_diff"] <= args.letterbox_pixel_tolerance
        ),
        "uses_canonical_coco_val2017": (
            dataset["hf_val_images"] == dataset["canonical_coco_val2017_images"]
        ),
    }
    baseline_reproduced = all(
        checks[name]
        for name in (
            "model_output_parity",
            "model_metric_parity",
            "nms_metric_parity",
            "letterbox_image_parity",
        )
    )

    return {
        "checks": checks,
        "metric_deltas": {
            "sparseflow_nms_vs_ultralytics_nms": nms_delta,
            "local_model_vs_checkpoint_model_with_ultralytics_nms": model_metric_delta,
        },
        "phase0_conclusion": {
            "baseline_reproduced_for_project_protocol": baseline_reproduced,
            "m1_status": "closed" if baseline_reproduced else "open",
            "benchmark_reference": (
                "HF detection-datasets/coco val split"
                if baseline_reproduced
                else "unresolved"
            ),
            "published_coco_metric_caveat": (
                "The active HF val split has 4952 images, not canonical COCO "
                "val2017's 5000 images. Published Ultralytics COCO metrics are "
                "therefore useful for directionality, not exact acceptance, unless "
                "the project switches to canonical COCO val2017."
            ),
            "next_phase": (
                "Phase 1 baseline finetuning" if baseline_reproduced else "Phase 0"
            ),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    datamodule = _build_datamodule(args)
    local_model = _build_local_model(args, device)
    checkpoint_model = _build_checkpoint_model(args, device)
    sparseflow_post = DetectionPostprocessor(
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    metrics = {
        "local_model_sparseflow_nms": _new_metric(),
        "local_model_ultralytics_nms": _new_metric(),
        "checkpoint_model_ultralytics_nms": _new_metric(),
    }
    counts: Dict[str, List[int]] = {key: [] for key in metrics}
    first_batch: Dict[str, Any] = {}

    with torch.no_grad():
        for batch_index, batch in enumerate(datamodule.test_dataloader()):
            images, targets = prepare_eval_batch(batch, device)
            local_outputs = local_model(images)
            checkpoint_outputs = checkpoint_model(images)

            sparseflow_preds = sparseflow_post(local_outputs)
            local_ultra_preds = _ultralytics_postprocess(
                local_outputs,
                args.conf_threshold,
                args.iou_threshold,
                args.max_det,
                args.num_classes,
            )
            checkpoint_ultra_preds = _ultralytics_postprocess(
                checkpoint_outputs,
                args.conf_threshold,
                args.iou_threshold,
                args.max_det,
                args.num_classes,
            )

            preds_by_path = {
                "local_model_sparseflow_nms": sparseflow_preds,
                "local_model_ultralytics_nms": local_ultra_preds,
                "checkpoint_model_ultralytics_nms": checkpoint_ultra_preds,
            }
            for name, preds in preds_by_path.items():
                metrics[name].update(preds, targets)
                counts[name].extend(int(pred["scores"].numel()) for pred in preds)

            if batch_index == 0:
                local_pred = _prediction_tensor(local_outputs)
                checkpoint_pred = _prediction_tensor(checkpoint_outputs)
                diff = (local_pred - checkpoint_pred).abs()
                first_batch = {
                    "image_shape": list(images.shape),
                    "image_range": [float(images.min()), float(images.max())],
                    "first_target_count": int(targets[0]["labels"].numel()),
                    "first_target_labels": targets[0]["labels"][:5].tolist(),
                    "first_target_boxes_cxcywh": [
                        [round(float(x), 3) for x in box]
                        for box in targets[0]["boxes"][:5].tolist()
                    ],
                    "output_shape": list(local_pred.shape),
                    "local_vs_checkpoint_max_abs_diff": float(diff.max()),
                    "local_vs_checkpoint_mean_abs_diff": float(diff.mean()),
                    "top_detection": {
                        name: _top_detection(preds)
                        for name, preds in preds_by_path.items()
                    },
                }

    dataset = _audit_dataset_and_letterbox(args)
    metric_results = {name: _metric_result(metric) for name, metric in metrics.items()}
    result = {
        "weights": args.weights,
        "model": args.model,
        "dataset": dataset,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "max_det": args.max_det,
        "metrics": metric_results,
        "detections_per_image": {
            name: _count_summary(values) for name, values in counts.items()
        },
        "first_batch": first_batch,
        "analysis": _build_analysis(args, dataset, metric_results, first_batch),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
