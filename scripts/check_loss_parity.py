"""Check SparseFlow detection loss parity against Ultralytics v8DetectionLoss.

The check feeds the same deterministic model outputs into both loss
implementations, using equivalent SparseFlow and Ultralytics batch formats.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor
from ultralytics.utils.loss import v8DetectionLoss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection_utils import DetectionLoss  # noqa: E402
from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SparseFlow DetectionLoss against Ultralytics v8DetectionLoss."
    )
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--atol", type=float, default=0.0)
    return parser


def _build_sparseflow_batch(
    images: Tensor,
    labels: list[Tensor],
    bboxes: list[Tensor],
) -> dict[str, Any]:
    return {
        "images": images,
        "labels": labels,
        "bboxes": bboxes,
    }


def _build_ultralytics_batch(
    labels: list[Tensor],
    bboxes: list[Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    batch_idx_parts: list[Tensor] = []
    cls_parts: list[Tensor] = []
    bbox_parts: list[Tensor] = []

    for image_idx, (labels_i, bboxes_i) in enumerate(zip(labels, bboxes)):
        labels_i = labels_i.to(device)
        bboxes_i = bboxes_i.to(device)
        if labels_i.numel() == 0:
            continue
        batch_idx_parts.append(
            torch.full(
                (labels_i.numel(),), image_idx, dtype=torch.float32, device=device
            )
        )
        cls_parts.append(labels_i.float())
        bbox_parts.append(bboxes_i.float())

    if not batch_idx_parts:
        return {
            "batch_idx": torch.zeros(0, dtype=torch.float32, device=device),
            "cls": torch.zeros(0, dtype=torch.float32, device=device),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }

    return {
        "batch_idx": torch.cat(batch_idx_parts),
        "cls": torch.cat(cls_parts),
        "bboxes": torch.cat(bbox_parts),
    }


def _assert_close(name: str, actual: Tensor, expected: Tensor, atol: float) -> float:
    diff = (actual - expected).abs().max().item()
    if diff > atol:
        raise AssertionError(
            f"{name} mismatch: max_abs_diff={diff:.9g}, atol={atol:.9g}, "
            f"actual={actual.detach().cpu().tolist()}, expected={expected.detach().cpu().tolist()}"
        )
    return float(diff)


def _run_case(
    name: str,
    model: torch.nn.Module,
    sparseflow_loss: DetectionLoss,
    ultralytics_loss: v8DetectionLoss,
    images: Tensor,
    labels: list[Tensor],
    bboxes: list[Tensor],
    atol: float,
) -> dict[str, Any]:
    device = images.device
    sparseflow_batch = _build_sparseflow_batch(images, labels, bboxes)
    ultralytics_batch = _build_ultralytics_batch(labels, bboxes, device)

    with torch.no_grad():
        preds = model(images)
        sparseflow_total = sparseflow_loss(preds, sparseflow_batch).reshape(())
        ultralytics_scaled_components, ultralytics_components = ultralytics_loss(
            preds, ultralytics_batch
        )

    ultralytics_total = ultralytics_scaled_components.sum()
    expected_scaled_components = ultralytics_components * images.shape[0]

    total_diff = _assert_close(
        f"{name} total loss",
        sparseflow_total,
        ultralytics_total,
        atol,
    )
    scaling_diff = _assert_close(
        f"{name} batch-size-scaled components",
        ultralytics_scaled_components,
        expected_scaled_components,
        atol,
    )

    no_object_checks: dict[str, Any] = {}
    if ultralytics_batch["bboxes"].numel() == 0:
        cls_component = ultralytics_components[1]
        box_component = ultralytics_components[0]
        dfl_component = ultralytics_components[2]
        if not torch.isfinite(cls_component):
            raise AssertionError(
                f"{name} classification loss is not finite: {cls_component}"
            )
        if box_component.item() != 0.0 or dfl_component.item() != 0.0:
            raise AssertionError(
                f"{name} expected zero box/DFL loss, got "
                f"box={box_component.item()}, dfl={dfl_component.item()}"
            )
        no_object_checks = {
            "classification_finite": True,
            "box_loss_zero": True,
            "dfl_loss_zero": True,
        }

    return {
        "case": name,
        "sparseflow_total": float(sparseflow_total.item()),
        "ultralytics_total": float(ultralytics_total.item()),
        "ultralytics_scaled_components": [
            float(v) for v in ultralytics_scaled_components.detach().cpu().tolist()
        ],
        "ultralytics_unscaled_components": [
            float(v) for v in ultralytics_components.detach().cpu().tolist()
        ],
        "max_total_abs_diff": total_diff,
        "max_batch_scaling_abs_diff": scaling_diff,
        **no_object_checks,
    }


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    model = build_model_from_spec(args.model, num_classes=80).to(device).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    detect = model.model[-1]
    sparseflow_loss = DetectionLoss(
        strides=detect.stride,
        num_classes=detect.nc,
        reg_max=detect.reg_max,
    ).to(device)
    ultralytics_loss = v8DetectionLoss(model)

    object_images = torch.rand(2, 3, args.imgsz, args.imgsz, device=device)
    object_labels = [
        torch.tensor([0, 15], dtype=torch.long, device=device),
        torch.tensor([2], dtype=torch.long, device=device),
    ]
    object_bboxes = [
        torch.tensor(
            [[0.50, 0.50, 0.25, 0.25], [0.25, 0.30, 0.20, 0.20]],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor([[0.70, 0.60, 0.15, 0.20]], dtype=torch.float32, device=device),
    ]

    empty_images = torch.rand(2, 3, args.imgsz, args.imgsz, device=device)
    empty_labels = [
        torch.zeros(0, dtype=torch.long, device=device),
        torch.zeros(0, dtype=torch.long, device=device),
    ]
    empty_bboxes = [
        torch.zeros((0, 4), dtype=torch.float32, device=device),
        torch.zeros((0, 4), dtype=torch.float32, device=device),
    ]

    results = [
        _run_case(
            "objects",
            model,
            sparseflow_loss,
            ultralytics_loss,
            object_images,
            object_labels,
            object_bboxes,
            args.atol,
        ),
        _run_case(
            "no_objects",
            model,
            sparseflow_loss,
            ultralytics_loss,
            empty_images,
            empty_labels,
            empty_bboxes,
            args.atol,
        ),
    ]

    print(
        json.dumps(
            {
                "device": str(device),
                "model": args.model,
                "seed": args.seed,
                "imgsz": args.imgsz,
                "atol": args.atol,
                "passed": True,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
