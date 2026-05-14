"""Check that the local YOLOv8 graph matches the checkpoint model output."""

import argparse
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec  # noqa: E402
from step_module import load_pretrained_weights  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local YOLOv8 output against the Ultralytics checkpoint model."
    )
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model != "yolov8" and not args.model.startswith("yolov8:"):
        raise ValueError(
            "Ultralytics pretrained parity is only defined for dense yolov8 specs."
        )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    local_model = build_model_from_spec(args.model, num_classes=80).to(device).eval()
    load_pretrained_weights(local_model, args.weights)

    checkpoint_model = YOLO(args.weights).model.to(device).eval()
    sample = torch.rand(1, 3, args.imgsz, args.imgsz, device=device)

    with torch.no_grad():
        local_out = local_model(sample)
        checkpoint_out = checkpoint_model(sample)

    local_pred = local_out[0] if isinstance(local_out, tuple) else local_out
    checkpoint_pred = (
        checkpoint_out[0] if isinstance(checkpoint_out, tuple) else checkpoint_out
    )
    diff = (local_pred - checkpoint_pred).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    passed = local_pred.shape == checkpoint_pred.shape and max_abs_diff <= args.atol

    print(
        json.dumps(
            {
                "weights": args.weights,
                "model": args.model,
                "input_shape": list(sample.shape),
                "local_shape": list(local_pred.shape),
                "checkpoint_shape": list(checkpoint_pred.shape),
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "atol": args.atol,
                "passed": passed,
            },
            indent=2,
        )
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
