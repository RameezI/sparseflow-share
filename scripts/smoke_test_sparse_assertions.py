"""Quick assertion-based smoke test for sparse model variants.

Verifies that each progressive sparse config builds, produces the correct
output shape, has no SiLU in SparseConv paths, and tracks sparsity correctly.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.axon_hillock import AxonHillock  # noqa: E402
from models.model_builder import build_model_from_spec  # noqa: E402

VARIANTS = [
    "yolov8_sparse_stem:n",
    "yolov8_sparse_backbone:n",
    "yolov8_sparse_full:n",
]


def test_variant(spec: str) -> None:
    print(f"Testing {spec}...")
    model = build_model_from_spec(spec, num_classes=80)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    assert out[0].shape == (1, 84, 8400), f"Wrong output shape: {out[0].shape}"

    for name, m in model.named_modules():
        if "SparseConv" in type(m).__name__:
            for sub_name, sub_m in m.named_modules():
                assert not isinstance(
                    sub_m, nn.SiLU
                ), f"SiLU found in {name}.{sub_name}"

    axon_count = 0
    for name, m in model.named_modules():
        if isinstance(m, AxonHillock):
            axon_count += 1
            assert m.stateless, f"{name} is not stateless"
            assert (
                0.0 <= m.last_sparsity_ratio <= 1.0
            ), f"{name} sparsity out of range: {m.last_sparsity_ratio}"
            assert (
                m.last_regularization.item() > 0
            ), f"{name} regularization should be > 0"

    assert axon_count > 0, f"No AxonHillock modules found in {spec}"
    print(f"  PASS: {axon_count} AxonHillock layers, shape OK, no SiLU")


def main() -> None:
    for variant in VARIANTS:
        test_variant(variant)
    print("\nAll sparse assertion tests passed.")


if __name__ == "__main__":
    main()
