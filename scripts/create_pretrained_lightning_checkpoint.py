"""Temporary bridge: save pretrained YOLO weights as a Lightning checkpoint.

Training will eventually produce the checkpoint used for export. Until then,
this script creates the same kind of Lightning checkpoint from pretrained
Ultralytics weights so the export path can be developed realistically.
"""

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec  # noqa: E402
from step_module import StepModule  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a temporary Lightning checkpoint initialized from pretrained YOLO weights."
    )
    parser.add_argument("--pretrained-weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("checkpoints/pretrained-yolov8n.ckpt"),
    )
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _prediction_tensor(outputs):
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


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    model = build_model_from_spec(args.model, num_classes=args.num_classes)
    protocol = StepModule(
        model=model,
        base_learning_rate=args.learning_rate,
        pretrained_weights_path=str(args.pretrained_weights),
    ).eval()
    _save_lightning_checkpoint(protocol, args.checkpoint_out)

    reloaded_model = build_model_from_spec(args.model, num_classes=args.num_classes)
    reloaded_protocol = StepModule.load_from_checkpoint(
        str(args.checkpoint_out),
        model=reloaded_model,
    ).eval()

    sample = torch.rand(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        original_pred = _prediction_tensor(protocol(sample))
        reloaded_pred = _prediction_tensor(reloaded_protocol(sample))
    diff = (original_pred - reloaded_pred).abs()
    max_abs_diff = float(diff.max().item())

    result = {
        "command": "create_pretrained_lightning_checkpoint",
        "pretrained_weights": str(args.pretrained_weights),
        "model": args.model,
        "checkpoint": str(args.checkpoint_out),
        "checkpoint_exists": args.checkpoint_out.exists(),
        "verify_shape": list(reloaded_pred.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": float(diff.mean().item()),
        "passed": max_abs_diff == 0.0,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
