import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from data import CocoDatasetConfig, CocoDetectionDataModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect val samples produced by CocoDetectionDataModule."
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "dataset_samples"
        ),
    )
    return parser


def _to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """CHW uint8 RGB tensor -> PIL Image."""
    return Image.fromarray(img_tensor.permute(1, 2, 0).numpy())


def _draw_boxes(image: Image.Image, bboxes: torch.Tensor, labels: torch.Tensor) -> None:
    """Draw normalized center-xywh boxes onto image in-place."""
    iw, ih = image.size
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(bboxes):
        x, y, bw, bh = box.tolist()
        x1 = (x - bw / 2) * iw
        y1 = (y - bh / 2) * ih
        x2 = (x + bw / 2) * iw
        y2 = (y + bh / 2) * ih
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=2)
        draw.text((x1 + 2, y1 + 2), str(int(labels[i].item())), fill=(255, 80, 80))


def main() -> None:
    args = build_parser().parse_args()
    cfg = CocoDatasetConfig()
    dm = CocoDetectionDataModule(config=cfg, batch_size=1, num_workers=0)
    dm.setup("validate")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset={cfg.dataset_id}  split={cfg.val_split}  imgsz={cfg.imgsz}")
    for idx, batch in enumerate(dm.val_dataloader()):
        if idx >= args.num_samples:
            break
        img_tensor = batch["images"][0]  # (C, H, W) uint8
        bboxes = batch["bboxes"][0]  # (N, 4) normalised center-xywh
        labels = batch["labels"][0]  # (N,)

        print(f"  sample={idx}  objects={len(bboxes)}")
        for j, box in enumerate(bboxes[:3]):
            print(
                f"    obj[{j}] label={int(labels[j].item())} xywh_norm={box.tolist()}"
            )

        rendered = _to_pil(img_tensor)
        _draw_boxes(rendered, bboxes, labels)
        out_path = save_dir / f"val_{idx:03d}.png"
        rendered.save(out_path)
        print(f"  saved={out_path}")


if __name__ == "__main__":
    main()
