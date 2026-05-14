"""Image and annotation transforms for YOLO-style detection batches."""

# OpenCV exposes many compiled extension members that pylint cannot inspect.
# pylint: disable=no-member,too-few-public-methods

from typing import Dict

import cv2
import numpy as np
import torch


def _xyxy_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """xyxy (pixel) -> center xywh (pixel)."""
    out = np.empty_like(bboxes)
    out[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    out[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
    out[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    out[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return out


class RandomHorizontalFlip:
    """Randomly flip image and bboxes horizontally.

    Expects data_sample with:
        "img":    HWC numpy array (any dtype)
        "bboxes": (N, 4) float32 array in center-xywh pixel space
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data_sample: Dict) -> Dict:
        if np.random.random() >= self.p:
            return data_sample
        img = data_sample["img"]
        w = img.shape[1]
        data_sample["img"] = img[:, ::-1, :].copy()
        bboxes = data_sample["bboxes"]
        if len(bboxes):
            bboxes = bboxes.copy()
            bboxes[:, 0] = w - bboxes[:, 0]  # flip cx
            data_sample["bboxes"] = bboxes
        return data_sample


class HSVJitter:
    """Apply random HSV colour jitter to a BGR numpy image.

    Gain parameters match Ultralytics YOLOv8 defaults.
    Expects data_sample with "img" as HWC BGR uint8 or float32 numpy array.
    """

    def __init__(self, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, data_sample: Dict) -> Dict:
        if self.hgain == 0 and self.sgain == 0 and self.vgain == 0:
            return data_sample
        img = data_sample["img"]
        gains = (
            np.random.uniform(-1, 1, 3) * np.array([self.hgain, self.sgain, self.vgain])
            + 1
        )
        # Convert to uint8 for cv2.cvtColor if needed
        if img.dtype != np.uint8:
            img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            img_u8 = img
        hue, sat, val = cv2.split(cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV))
        lut_h = np.arange(0, 256, dtype=np.int16)
        lut_s = np.arange(0, 256, dtype=np.int16)
        lut_v = np.arange(0, 256, dtype=np.int16)
        lut_h = (lut_h * gains[0] % 180).clip(0, 179).astype(np.uint8)
        lut_s = (lut_s * gains[1]).clip(0, 255).astype(np.uint8)
        lut_v = (lut_v * gains[2]).clip(0, 255).astype(np.uint8)
        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v))
        )
        out = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        if img.dtype != np.uint8:
            out = out.astype(np.float32) / 255.0
        data_sample["img"] = out
        return data_sample


class Compose:
    """Chains a sequence of callables, passing output of each as input to the next."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, data: Dict) -> Dict:
        """Run all transforms in sequence."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Append a transform to the sequence."""
        self.transforms.append(transform)


class LetterBox:
    """Letterbox-resize an image to new_shape, preserving aspect ratio via grey padding."""

    def __init__(self, new_shape=(640, 640), scaleup=False, center=True, stride=32):
        self.new_shape = new_shape
        self.scaleup = scaleup
        self.center = center
        self.stride = stride

    def __call__(self, data_sample: Dict) -> Dict:
        img = data_sample.get("img")
        assert (
            img is not None
        ), "No image provided: pass 'image' or a labels dict with key 'img'."
        shape = img.shape[:2]  # (h, w)
        new_shape = data_sample.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top = int(round(dh - 0.1)) if self.center else 0
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1)) if self.center else 0
        right = int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        if data_sample.get("ratio_pad"):
            data_sample["ratio_pad"] = (data_sample["ratio_pad"], (left, top))

        data_sample["bboxes"] = self._scale_and_pad_bboxes(
            data_sample["bboxes"], r, dw, dh
        )
        data_sample["img"] = img
        return data_sample

    def _scale_and_pad_bboxes(
        self,
        bboxes: np.ndarray,
        r: float,
        padw: float,
        padh: float,
    ) -> np.ndarray:
        bboxes = bboxes.copy()  # xyxy pixel (HF COCO native format)
        bboxes *= r
        bboxes[:, [0, 2]] += padw
        bboxes[:, [1, 3]] += padh
        return _xyxy_to_xywh(bboxes)  # -> center xywh pixel


class Format:
    """
    Format image and bboxes for detection training.

    Args:
        normalize (bool): Normalize bboxes to [0, 1]. Default True.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, data_sample):
        img = data_sample.pop("img")
        h, w = img.shape[:2]
        labels = data_sample.pop("labels")
        bboxes = data_sample.pop("bboxes")  # center xywh pixel (output of LetterBox)
        nl = len(bboxes)
        if nl and self.normalize:
            bboxes /= np.array([w, h, w, h], dtype=np.float32)
        data_sample["img"] = self._to_tensor(img)
        data_sample["labels"] = torch.from_numpy(labels) if nl else torch.zeros(nl)
        data_sample["bboxes"] = torch.from_numpy(bboxes) if nl else torch.zeros((nl, 4))
        return data_sample

    @staticmethod
    def _to_tensor(img):
        """HWC numpy (BGR) -> CHW torch float32 tensor (RGB, contiguous) in [0, 1]."""
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        return torch.from_numpy(img).float() / 255.0


class DetectionFormat:
    """Transform for HF-style batches — letterbox + normalize only (eval/val use).

    Expects each batch item to have:
        "image":   PIL.Image
        "objects": {"bbox": list[[x1,y1,x2,y2] pixels], "category": list[int]}

    Produces (via LetterBox + Format):
        "img":    float32 CHW tensor  (3, imgsz, imgsz)  — RGB [0, 1]
        "labels": float32 tensor (N,)
        "bboxes": float32 tensor (N, 4)  — normalised cxcywh in [0, 1]
    """

    def __init__(self, imgsz: int = 640, scaleup: bool = False):
        self.pipeline = Compose(
            [
                LetterBox(new_shape=(imgsz, imgsz), scaleup=scaleup),
                Format(normalize=True),
            ]
        )

    def __call__(self, batch):
        return self._process(batch, self.pipeline)

    @staticmethod
    def _process(batch, pipeline):
        imgs, labels_list, bboxes = [], [], []
        for pil_img, objs in zip(batch["image"], batch["objects"]):
            img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # HWC BGR
            raw = objs.get("bbox", [])
            b = (
                np.array(raw, dtype=np.float32)
                if raw
                else np.zeros((0, 4), dtype=np.float32)
            )
            cats = np.array(objs.get("category", []), dtype=np.float32).reshape(-1, 1)
            detection_data = {
                "img": img,
                "labels": cats,
                "bboxes": b,  # xyxy pixel (HF COCO native format)
            }
            out = pipeline(detection_data)
            imgs.append(out["img"])
            labels_list.append(out["labels"])
            bboxes.append(out["bboxes"])
        return {"img": imgs, "labels": labels_list, "bboxes": bboxes}


class TrainDetectionFormat(DetectionFormat):
    """Transform for training: letterbox + HSV jitter + horizontal flip + normalize.

    Mosaic augmentation is intentionally omitted: it requires random access to 3
    additional dataset samples per item, which is incompatible with the HF
    set_transform interface (per-sample/per-batch lazy evaluation). Mosaic would
    require refactoring to a map-style dataset with internal random-access support.

    Applied augmentations (Ultralytics-matched defaults):
        - RandomHorizontalFlip  p=0.5
        - HSVJitter             hgain=0.015, sgain=0.7, vgain=0.4
    """

    def __init__(self, imgsz: int = 640):
        super().__init__(imgsz=imgsz, scaleup=True)
        self.pipeline = Compose(
            [
                LetterBox(new_shape=(imgsz, imgsz), scaleup=True),
                HSVJitter(),
                RandomHorizontalFlip(p=0.5),
                Format(normalize=True),
            ]
        )

    def __call__(self, batch):
        return self._process(batch, self.pipeline)
