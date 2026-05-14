"""Diagnostic for mAP gap investigation.

Checks:
  1. COCO HF category ID range (is it 0-79 or 1-90?)
  2. Detection counts per image at different thresholds (NMS sensitivity)
  3. Box coordinate ranges (are predictions in expected pixel-space range?)
  4. Quick mAP sensitivity sweep over NMS IoU thresholds
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import torch
import torchvision.ops as tv_ops
from torchmetrics.detection import MeanAveragePrecision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CocoDatasetConfig, CocoDetectionDataModule
from detection_utils import DetectionPostprocessor, prepare_eval_batch
from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec
from step_module import load_pretrained_weights


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose SparseFlow YOLO mAP gaps.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    return parser


def get_model(model_spec, weights):
    model = build_model_from_spec(model_spec, num_classes=80)
    load_pretrained_weights(model, weights)
    model.eval()
    return model


def get_datamodule(max_samples):
    dm = CocoDetectionDataModule(
        config=CocoDatasetConfig(max_eval_samples=max_samples),
        batch_size=8,
        num_workers=0,
    )
    dm.setup(stage="test")
    return dm


# ── 1. Category ID audit ──────────────────────────────────────────────────────
args = build_parser().parse_args()
print("\n╔══ 1. Category ID audit (first 3 batches) ═══╗")
dm = get_datamodule(max_samples=64)
all_cats: set = set()
for i, batch in enumerate(dm.test_dataloader()):
    for lbl in batch["labels"]:
        vals = lbl.long().tolist() if torch.is_tensor(lbl) else lbl
        if isinstance(vals, list):
            stack = [vals]
            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                else:
                    all_cats.add(int(item))
        else:
            all_cats.add(int(vals))
    if i >= 2:
        break

cat_list = sorted(all_cats)
print(f"  range  : {min(cat_list)} … {max(cat_list)}")
print(f"  unique : {len(cat_list)}  ids = {cat_list[:15]} ...")
print(f"  NOTE   : model outputs class 0..79 (arg-maxed sigmoid).")
print(
    f"  → {'ALIGNED (0-79)' if max(cat_list) <= 79 else 'MISALIGNED — max cat > 79!'}"
)

# ── 2. Detection count + coordinate audit ────────────────────────────────────
print("\n╔══ 2. Prediction stats on first batch ════════╗")
model = get_model(args.model, args.weights)
batch = next(iter(get_datamodule(max_samples=16).test_dataloader()))
images, targets = prepare_eval_batch(batch, torch.device("cpu"))

with torch.no_grad():
    out = model(images)

pred = out[0] if isinstance(out, tuple) else out
boxes_raw = pred[:, :4, :].permute(0, 2, 1)  # [B, A, 4] cxcywh pixels
class_probs = pred[:, 4:, :].permute(0, 2, 1)  # [B, A, nc]
scores, labels = class_probs.max(dim=-1)

print(f"  image size (H=W)            : {images.shape[-1]}")
print(f"  anchors per image           : {boxes_raw.shape[1]}")
print(
    f"  boxes range xmin            : {boxes_raw[:,:,0].min():.1f} … {boxes_raw[:,:,0].max():.1f}"
)
print(
    f"  boxes range ymin            : {boxes_raw[:,:,1].min():.1f} … {boxes_raw[:,:,1].max():.1f}"
)
print(
    f"  boxes range width           : {boxes_raw[:,:,2].min():.1f} … {boxes_raw[:,:,2].max():.1f}"
)
print(
    f"  boxes range height          : {boxes_raw[:,:,3].min():.1f} … {boxes_raw[:,:,3].max():.1f}"
)
print(f"  per-image detections (>0.001) : {(scores>0.001).sum(-1).tolist()}")
print(f"  per-image detections (>0.25)  : {(scores>0.25).sum(-1).tolist()}")

gt = targets[0]
print(f"\n  Ground-truth boxes[0] (cxcywh pixels, first 3):")
for b, l in zip(gt["boxes"][:3].tolist(), gt["labels"][:3].tolist()):
    print(f"    cls={int(l):2d}  box={[round(x,1) for x in b]}")


# ── 3. NMS IoU threshold sensitivity (100 images) ────────────────────────────
print("\n╔══ 3. NMS IoU threshold sweep (100 images) ═══╗")
dm100 = get_datamodule(max_samples=100)

for nms_iou in [0.45, 0.50, 0.60, 0.65, 0.70]:
    postprocessor = DetectionPostprocessor(conf_threshold=0.001, iou_threshold=nms_iou)
    metric = MeanAveragePrecision(
        box_format="cxcywh", iou_type="bbox", max_detection_thresholds=[1, 10, 100]
    )
    metric.warn_on_many_detections = False
    with torch.no_grad():
        for batch in dm100.test_dataloader():
            images, targets = prepare_eval_batch(batch, torch.device("cpu"))
            raw_out = model(images)
            preds = postprocessor(raw_out)
            metric.update(preds, targets)
    raw = metric.compute()
    m50 = float(raw["map_50"])
    m5095 = float(raw["map"])
    print(f"  nms_iou={nms_iou:.2f}  mAP50={m50:.4f}  mAP50-95={m5095:.4f}")

print("\nDone.")
