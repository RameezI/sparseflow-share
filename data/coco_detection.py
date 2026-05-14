"""Hugging Face COCO datamodule for SparseFlow detection training and eval."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .transforms import DetectionFormat, TrainDetectionFormat


@dataclass
class CocoDatasetConfig:
    """Configuration for the Hugging Face COCO dataset loader."""

    dataset_id: str = "detection-datasets/coco"
    train_split: str = "train"
    val_split: str = "val"
    imgsz: int = 640
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


def _collate_fn(batch: List[Dict]) -> Dict:
    """Stack images into (B, C, H, W); keep per-image labels and bboxes as lists."""
    return {
        "images": torch.stack([item["img"] for item in batch]),
        "labels": [item["labels"] for item in batch],
        "bboxes": [item["bboxes"] for item in batch],
    }


class CocoDetectionDataModule(L.LightningDataModule):
    """COCO HF datamodule — returns batches of letterboxed images and normalised bboxes."""

    _COCO_DATASET_ID = "detection-datasets/coco"

    def __init__(
        self, config: CocoDatasetConfig, batch_size: int = 8, num_workers: int = 0
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_dataset = None
        self._val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.config.dataset_id != self._COCO_DATASET_ID:
            raise ValueError(
                f"This datamodule only supports dataset_id={self._COCO_DATASET_ID!r}, "
                f"got {self.config.dataset_id!r}."
            )

        load_train = stage in (None, "fit")
        load_val = stage in (None, "fit", "validate", "test", "predict")

        if load_train and self._train_dataset is None:
            ds = load_dataset(self.config.dataset_id, split=self.config.train_split)
            if self.config.max_train_samples is not None:
                ds = ds.select(range(min(self.config.max_train_samples, len(ds))))
            ds.set_transform(TrainDetectionFormat(self.config.imgsz))
            self._train_dataset = ds

        if load_val and self._val_dataset is None:
            ds = load_dataset(self.config.dataset_id, split=self.config.val_split)
            if self.config.max_eval_samples is not None:
                ds = ds.select(range(min(self.config.max_eval_samples, len(ds))))
            ds.set_transform(DetectionFormat(self.config.imgsz, scaleup=False))
            self._val_dataset = ds

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before requesting train_dataloader().")
        return self._loader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before requesting val_dataloader().")
        return self._loader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before requesting test_dataloader().")
        return self._loader(self._val_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before requesting predict_dataloader().")
        return self._loader(self._val_dataset, shuffle=False)
