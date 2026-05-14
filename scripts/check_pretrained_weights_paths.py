"""Validate CLI pretrained weight loading for YOLO .pt and Lightning .ckpt paths."""

# pylint: disable=import-error,wrong-import-position

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import _load_protocol_for_inference  # noqa: E402
from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec  # noqa: E402
from step_module import StepModule  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the script argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Check that --pretrained-weights accepts both an Ultralytics YOLO .pt "
            "file and a SparseFlow Lightning .ckpt file."
        )
    )
    parser.add_argument("--pretrained-weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("checks/pretrained-weights-path.ckpt"),
    )
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.0)
    return parser


def _prediction_tensor(outputs: Any) -> torch.Tensor:
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _save_lightning_checkpoint(protocol: StepModule, checkpoint_path: Path) -> None:
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(protocol)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_path))


def _create_checkpoint_from_pt(args: argparse.Namespace) -> None:
    model = build_model_from_spec(args.model, num_classes=args.num_classes)
    protocol = StepModule(
        model=model,
        pretrained_weights_path=str(args.pretrained_weights),
    ).eval()
    _save_lightning_checkpoint(protocol, args.checkpoint_out)


def main() -> None:
    """Run the pretrained weight path equivalence check."""
    args = build_parser().parse_args()
    if not args.pretrained_weights.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found: {args.pretrained_weights}"
        )

    torch.manual_seed(args.seed)
    _create_checkpoint_from_pt(args)

    pt_protocol = _load_protocol_for_inference(
        str(args.pretrained_weights),
        args.model,
        num_classes=args.num_classes,
    )
    ckpt_protocol = _load_protocol_for_inference(
        str(args.checkpoint_out),
        args.model,
        num_classes=args.num_classes,
    )

    sample = torch.rand(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        pt_pred = _prediction_tensor(pt_protocol(sample))
        ckpt_pred = _prediction_tensor(ckpt_protocol(sample))

    diff = (pt_pred - ckpt_pred).abs()
    max_abs_diff = float(diff.max().item())
    result = {
        "command": "check_pretrained_weights_paths",
        "model": args.model,
        "pt_path": str(args.pretrained_weights),
        "ckpt_path": str(args.checkpoint_out),
        "pt_shape": list(pt_pred.shape),
        "ckpt_shape": list(ckpt_pred.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": float(diff.mean().item()),
        "passed": pt_pred.shape == ckpt_pred.shape and max_abs_diff <= args.atol,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
