"""Estimate whole-network activation sparsity for a model variant."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from data.coco_detection import CocoDatasetConfig, CocoDetectionDataModule
from models.axon_hillock import AxonHillock
from models.model_builder import build_model_from_spec
from step_module import load_pretrained_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate active neurons per image")
    parser.add_argument("--model", type=str, default="yolov8:n")
    parser.add_argument("--pretrained-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-json",
        type=str,
        default="profiling/activity_estimate.json",
    )
    return parser.parse_args()


class ActivityTracker:
    """Accumulates active/total activation counts across forward passes."""

    def __init__(self) -> None:
        self.active: int = 0
        self.total: int = 0
        self.num_images: int = 0

    def make_hook(self):
        def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
            out = output[0] if isinstance(output, tuple) else output
            self.active += int(torch.count_nonzero(out).item())
            self.total += out.numel()

        return hook

    def record_batch(self, batch_size: int) -> None:
        self.num_images += batch_size

    def summary(self) -> dict[str, Any]:
        active_per_image = self.active // max(self.num_images, 1)
        total_per_image = self.total // max(self.num_images, 1)
        activity = active_per_image / max(total_per_image, 1)
        return {
            "active_neurons_per_image": active_per_image,
            "total_neurons_per_image": total_per_image,
            "activity_ratio": round(activity, 4),
            "sparsity_ratio": round(1.0 - activity, 4),
            "num_images": self.num_images,
        }


def main() -> None:
    args = _parse_args()

    model = build_model_from_spec(args.model, num_classes=80)
    load_pretrained_weights(model, args.pretrained_weights)
    model.eval()

    tracker = ActivityTracker()
    hooks = []

    for module in model.modules():
        if isinstance(module, (nn.SiLU, AxonHillock)):
            hooks.append(module.register_forward_hook(tracker.make_hook()))

    config = CocoDatasetConfig(max_eval_samples=args.num_samples)
    dm = CocoDetectionDataModule(config=config, batch_size=args.batch_size)
    dm.setup("test")
    dataloader = dm.test_dataloader()

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["images"]
            model(images)
            tracker.record_batch(images.shape[0])
            if tracker.num_images >= args.num_samples:
                break

    for hook in hooks:
        hook.remove()

    report = tracker.summary()
    report["model"] = args.model
    report["pretrained_weights"] = args.pretrained_weights

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Model: {args.model}")
    print(f"Images: {report['num_images']}")
    print(f"Active neurons/image: {report['active_neurons_per_image']:,}")
    print(f"Total neurons/image:  {report['total_neurons_per_image']:,}")
    print(f"Activity ratio:       {report['activity_ratio']:.4f}")
    print(f"Sparsity ratio:       {report['sparsity_ratio']:.4f}")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
