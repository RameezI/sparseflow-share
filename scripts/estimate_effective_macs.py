"""Estimate effective MACs for a sparse model given activation sparsity.

For each layer with an AxonHillock activation, combines the dense MAC count
with the measured sparsity ratio to estimate effective compute savings on
hypothetical sparse hardware that can skip zero activations.

Formula: effective_MACs = sum(layer_MACs_i * (1 - layer_sparsity_i))
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
from models.axon_hillock import AxonHillock  # noqa: E402
from models.model_builder import build_model_from_spec  # noqa: E402
from models.sparse_layers import SparseConv2d  # noqa: E402
from step_module import load_pretrained_weights  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate effective MACs")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8_sparse_backbone:n",
        help="Model spec",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained weights path",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20, help="Number of images to run"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-json",
        type=str,
        default="profiling/effective_macs.json",
    )
    return parser.parse_args()


def _build_conv_to_axon_map(model: nn.Module) -> dict[str, str]:
    """Map Conv2d module names to their sibling AxonHillock names in SparseConv2d."""
    mapping: dict[str, str] = {}
    for name, module in model.named_modules():
        if isinstance(module, SparseConv2d):
            conv_name = f"{name}.convolution"
            axon_name = f"{name}.axon"
            mapping[conv_name] = axon_name
    return mapping


def main() -> None:
    args = _parse_args()

    model = build_model_from_spec(args.model, num_classes=80)
    load_pretrained_weights(model, args.pretrained_weights)
    model.eval()

    conv_to_axon = _build_conv_to_axon_map(model)
    axon_modules: dict[str, AxonHillock] = {}
    for name, module in model.named_modules():
        if isinstance(module, AxonHillock):
            axon_modules[name] = module

    layer_macs: dict[str, int] = {}

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
            hooks.append(module.register_forward_hook(_make_mac_hook(name, module)))

    dm = CocoDetectionDataModule(
        batch_size=args.batch_size, max_eval_samples=args.num_samples
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    sparsity_accum: dict[str, list[float]] = {name: [] for name in axon_modules}

    images_processed = 0
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["images"]
            model(images)
            for name, module in axon_modules.items():
                sparsity_accum[name].append(module.last_sparsity_ratio)
            images_processed += images.shape[0]
            if images_processed >= args.num_samples:
                break

    for hook in hooks:
        hook.remove()

    avg_sparsity: dict[str, float] = {}
    for name, values in sparsity_accum.items():
        avg_sparsity[name] = sum(values) / max(len(values), 1)

    dense_macs_total = sum(layer_macs.values())

    per_layer_report = []
    effective_macs_total = 0
    sparse_macs_total = 0

    for conv_name, macs in layer_macs.items():
        axon_name = conv_to_axon.get(conv_name)
        if axon_name and axon_name in avg_sparsity:
            sparsity = avg_sparsity[axon_name]
            effective = int(macs * (1.0 - sparsity))
            sparse_macs_total += macs
            effective_macs_total += effective
            per_layer_report.append(
                {
                    "conv": conv_name,
                    "axon": axon_name,
                    "dense_macs": macs,
                    "sparsity": round(sparsity, 4),
                    "effective_macs": effective,
                }
            )
        else:
            effective_macs_total += macs

    mac_reduction = 1.0 - (effective_macs_total / max(dense_macs_total, 1))
    sparse_mac_reduction = (
        1.0
        - (effective_macs_total - (dense_macs_total - sparse_macs_total))
        / max(sparse_macs_total, 1)
        if sparse_macs_total > 0
        else 0.0
    )

    overall_sparsity = (
        sum(avg_sparsity.values()) / max(len(avg_sparsity), 1) if avg_sparsity else 0.0
    )

    report = {
        "model": args.model,
        "images_processed": images_processed,
        "dense_macs_total": dense_macs_total,
        "effective_macs_total": effective_macs_total,
        "mac_reduction_ratio": round(mac_reduction, 4),
        "sparse_layers_mac_reduction": round(sparse_mac_reduction, 4),
        "mean_activation_sparsity": round(overall_sparsity, 4),
        "num_sparse_layers": len(per_layer_report),
        "per_layer": per_layer_report,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {output_path} ({len(per_layer_report)} sparse layers)")
    print(f"Dense MACs: {dense_macs_total:,}")
    print(f"Effective MACs: {effective_macs_total:,}")
    print(f"MAC reduction: {mac_reduction:.2%}")
    print(f"Mean sparsity: {overall_sparsity:.4f}")


if __name__ == "__main__":
    main()
