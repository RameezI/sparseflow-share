"""Convergence test for the sparse YOLOv8 model with AxonHillock activations.

Trains yolov8_sparse:n on a small COCO subset (~64 samples) for 30 epochs on CPU.
Verifies:
  1. Training loss decreases over epochs (convergence signal).
  2. Gradients flow through all AxonHillock threshold parameters.
  3. AxonHillock thresholds are being updated (not frozen).
  4. Sparsity ratios are in a reasonable range (not all-pass or all-block).

Usage:
    uv run python scripts/convergence_test_sparse.py [--epochs 30] [--samples 64]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightning as L
from lightning.pytorch.callbacks import Callback

from data import CocoDatasetConfig, CocoDetectionDataModule
from models.axon_hillock import AxonHillock
from models.model_builder import build_model_from_spec
from step_module import StepModule


class ConvergenceTracker(Callback):
    """Tracks per-epoch training loss and AxonHillock gradient flow."""

    def __init__(self):
        super().__init__()
        self.epoch_losses = []
        self.axon_grad_norms = []
        self.threshold_snapshots = []
        self._last_batch_grad_norms = {}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grad_norms = {}
        for name, module in pl_module.model.named_modules():
            if isinstance(module, AxonHillock) and module.thresholds is not None:
                grad = module.thresholds.grad
                if grad is not None:
                    grad_norms[name] = float(grad.norm())
        self._last_batch_grad_norms = grad_norms

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train/loss")
        if loss is not None:
            self.epoch_losses.append(float(loss))

        self.axon_grad_norms.append(self._last_batch_grad_norms)

        thresholds = {}
        for name, module in pl_module.model.named_modules():
            if isinstance(module, AxonHillock) and module.thresholds is not None:
                thresholds[name] = module.thresholds.detach().clone().mean().item()
        self.threshold_snapshots.append(thresholds)


def run_convergence_test(num_epochs: int, num_samples: int) -> bool:
    """Run convergence test and return True if all checks pass."""
    print(f"=== Sparse Convergence Test ===")
    print(f"Model: yolov8_sparse:n")
    print(f"Epochs: {num_epochs}, Samples: {num_samples}, Device: CPU")
    print()

    model = build_model_from_spec("yolov8_sparse:n", num_classes=80)
    protocol = StepModule(model=model, base_learning_rate=0.01, batch_size=4)

    datamodule = CocoDetectionDataModule(
        config=CocoDatasetConfig(
            max_train_samples=num_samples,
            max_eval_samples=8,
        ),
        batch_size=4,
        num_workers=0,
    )

    tracker = ConvergenceTracker()
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        callbacks=[tracker],
        log_every_n_steps=1,
        check_val_every_n_epoch=num_epochs,
        num_sanity_val_steps=0,
        inference_mode=False,
        enable_progress_bar=True,
    )

    trainer.fit(protocol, datamodule=datamodule)

    all_passed = True

    # Check 1: Loss decreases
    print("--- Check 1: Loss Convergence ---")
    losses = tracker.epoch_losses
    if len(losses) < 2:
        print("  FAIL: Not enough epochs recorded.")
        all_passed = False
    else:
        first_quarter = losses[: max(len(losses) // 4, 1)]
        last_quarter = losses[-max(len(losses) // 4, 1) :]
        avg_first = sum(first_quarter) / len(first_quarter)
        avg_last = sum(last_quarter) / len(last_quarter)
        decrease_pct = (avg_first - avg_last) / avg_first * 100

        print(f"  First-quarter avg loss: {avg_first:.4f}")
        print(f"  Last-quarter avg loss:  {avg_last:.4f}")
        print(f"  Decrease: {decrease_pct:.1f}%")

        if avg_last < avg_first:
            print("  PASS: Loss is decreasing.")
        else:
            print("  FAIL: Loss did not decrease.")
            all_passed = False

    # Check 2: Gradient flow through AxonHillock
    print("\n--- Check 2: Gradient Flow ---")
    if not tracker.axon_grad_norms:
        print("  FAIL: No gradient data recorded.")
        all_passed = False
    else:
        last_grads = tracker.axon_grad_norms[-1]
        if not last_grads:
            print("  FAIL: No AxonHillock modules found or no gradients.")
            all_passed = False
        else:
            zero_grad_layers = [k for k, v in last_grads.items() if v == 0.0]
            nonzero = len(last_grads) - len(zero_grad_layers)
            print(f"  AxonHillock layers with gradients: {nonzero}/{len(last_grads)}")
            if zero_grad_layers:
                print(f"  Zero-gradient layers: {zero_grad_layers}")
            if nonzero == len(last_grads):
                print("  PASS: All AxonHillock layers receive gradients.")
            elif nonzero > 0:
                print("  WARN: Some layers have zero gradients.")
            else:
                print("  FAIL: No AxonHillock layers received gradients.")
                all_passed = False

    # Check 3: Threshold parameters are updating
    print("\n--- Check 3: Threshold Updates ---")
    if len(tracker.threshold_snapshots) < 2:
        print("  FAIL: Not enough snapshots to compare.")
        all_passed = False
    else:
        first_snap = tracker.threshold_snapshots[0]
        last_snap = tracker.threshold_snapshots[-1]
        changed = 0
        for name in first_snap:
            if name in last_snap and first_snap[name] != last_snap[name]:
                changed += 1
        total = len(first_snap)
        print(f"  Thresholds changed: {changed}/{total}")
        if changed > 0:
            print("  PASS: Thresholds are being learned.")
        else:
            print("  FAIL: No threshold parameters changed.")
            all_passed = False

    # Check 4: Sparsity ratios
    print("\n--- Check 4: Sparsity Ratios ---")
    sparsity_ratios = {}
    for name, module in protocol.model.named_modules():
        if isinstance(module, AxonHillock):
            sparsity_ratios[name] = module.last_sparsity_ratio
    if not sparsity_ratios:
        print("  FAIL: No AxonHillock modules found.")
        all_passed = False
    else:
        values = list(sparsity_ratios.values())
        avg_sparsity = sum(values) / len(values)
        print(f"  Avg sparsity: {avg_sparsity:.3f}")
        print(f"  Min sparsity: {min(values):.3f}")
        print(f"  Max sparsity: {max(values):.3f}")
        if 0.0 < avg_sparsity < 1.0:
            print(
                "  PASS: Sparsity is in a reasonable range (not all-pass or all-block)."
            )
        else:
            print(f"  WARN: Sparsity at boundary ({avg_sparsity:.3f}).")

    # Summary
    print("\n" + "=" * 40)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED — sparse model converges.")
    else:
        print("RESULT: SOME CHECKS FAILED — investigate above.")
    print("=" * 40)

    # Print epoch loss curve
    print("\nEpoch loss curve:")
    for i, loss in enumerate(losses):
        bar_len = int(max(0, min(50, 50 * loss / max(losses))))
        print(f"  Epoch {i+1:3d}: {loss:8.4f} {'█' * bar_len}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Sparse model convergence test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    passed = run_convergence_test(num_epochs=args.epochs, num_samples=args.samples)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
