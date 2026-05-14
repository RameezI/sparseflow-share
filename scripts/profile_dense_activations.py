"""Profile activation density in the dense YOLOv8 baseline.

For each Conv2d and SiLU layer, registers a forward hook that measures the
fraction of non-zero outputs. Runs N images from HF COCO val, accumulates
stats, and writes a JSON report with per-layer and aggregate density.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coco_detection import CocoDetectionDataModule  # noqa: E402
from models.model_builder import build_model_from_spec  # noqa: E402
from step_module import load_pretrained_weights  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile dense activation density")
    parser.add_argument(
        "--model", type=str, default="yolov8:n", help="Model spec (e.g. yolov8:n)"
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained weights path",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50, help="Number of images to profile"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-json",
        type=str,
        default="profiling/dense_activation_profile.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model = build_model_from_spec(args.model, num_classes=80)
    load_pretrained_weights(model, args.pretrained_weights)
    model.eval()

    layer_stats: dict[str, dict[str, float]] = {}
    layer_macs: dict[str, int] = {}

    def _make_density_hook(name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
            out = output[0] if isinstance(output, tuple) else output
            total = out.numel()
            nonzero = int(torch.count_nonzero(out).item())
            if name not in layer_stats:
                layer_stats[name] = {"total": 0, "nonzero": 0}
            layer_stats[name]["total"] += total
            layer_stats[name]["nonzero"] += nonzero

        return hook

    def _make_mac_hook(name: str, module: nn.Conv2d):
        def hook(_mod: nn.Module, _inputs: Any, output: Tensor) -> None:
            out = output[0] if isinstance(output, tuple) else output
            kernel_macs = module.kernel_size[0] * module.kernel_size[1]
            kernel_macs *= module.in_channels // module.groups
            layer_macs[name] = int(out.numel() * kernel_macs)

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(_make_density_hook(name)))
            hooks.append(module.register_forward_hook(_make_mac_hook(name, module)))
        elif isinstance(module, nn.SiLU):
            hooks.append(module.register_forward_hook(_make_density_hook(name)))

    dm = CocoDetectionDataModule(
        batch_size=args.batch_size, max_eval_samples=args.num_samples
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    images_processed = 0
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["images"]
            model(images)
            images_processed += images.shape[0]
            if images_processed >= args.num_samples:
                break

    for hook in hooks:
        hook.remove()

    per_layer = {}
    for name, stats in layer_stats.items():
        density = stats["nonzero"] / max(stats["total"], 1)
        entry = {"density": round(density, 6), "total_activations": stats["total"]}
        if name in layer_macs:
            entry["macs"] = layer_macs[name]
        per_layer[name] = entry

    total_activations = sum(s["total"] for s in layer_stats.values())
    total_nonzero = sum(s["nonzero"] for s in layer_stats.values())
    aggregate_density = total_nonzero / max(total_activations, 1)

    total_macs = sum(layer_macs.values())

    report = {
        "model": args.model,
        "pretrained_weights": args.pretrained_weights,
        "images_processed": images_processed,
        "aggregate_density": round(aggregate_density, 6),
        "total_macs": total_macs,
        "per_layer": per_layer,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {output_path} ({len(per_layer)} layers, {images_processed} images)")
    print(f"Aggregate density: {aggregate_density:.4f}")


if __name__ == "__main__":
    main()
