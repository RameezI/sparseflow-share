"""Export SparseFlow Lightning checkpoints to deployment formats."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import nncf
import openvino as ov
import torch
from torch import Tensor, nn

from data import CocoDatasetConfig, CocoDetectionDataModule
from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec
from step_module import StepModule


class ExportableDetectionModel(nn.Module):
    """Wrap StepModule/model output so exported artifacts return the prediction tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> Tensor:
        outputs = self.model(images)
        return outputs[0] if isinstance(outputs, tuple) else outputs


def _precision_name(args: argparse.Namespace) -> str:
    if args.int8:
        return "int8"
    if args.fp16:
        return "fp16"
    return "fp32"


def _default_output_dir(checkpoint_path: Path, precision: str) -> Path:
    return Path("exports") / f"{checkpoint_path.stem}-openvino-{precision}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a SparseFlow Lightning checkpoint to OpenVINO."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Lightning checkpoint produced by training or by the temporary pretrained bridge script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for staged weights, exported artifact, and metadata.",
    )
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--fp16",
        action="store_true",
        help="For OpenVINO, save weights compressed to FP16.",
    )
    precision.add_argument(
        "--int8",
        action="store_true",
        help="Run NNCF post-training quantization using HF COCO calibration images.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=300,
        help="Number of HF COCO val images to use for INT8 calibration.",
    )
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.int8 and args.calibration_samples <= 0:
        raise ValueError("--calibration-samples must be positive for INT8 export.")


def _load_export_model(
    checkpoint_path: Path,
    model_spec: str,
    num_classes: int,
) -> ExportableDetectionModel:
    model = build_model_from_spec(model_spec=model_spec, num_classes=num_classes)
    protocol = StepModule.load_from_checkpoint(
        str(checkpoint_path),
        model=model,
        strict=False,
    )
    protocol.eval()
    return ExportableDetectionModel(protocol.model).eval()


def _sample_input(args: argparse.Namespace) -> Tensor:
    return torch.zeros(args.batch, 3, args.imgsz, args.imgsz, dtype=torch.float32)


def _export_openvino(
    model: nn.Module,
    sample_input: Tensor,
    output_dir: Path,
    checkpoint_stem: str,
    compress_to_fp16: bool,
    quantize_int8: bool,
    calibration_dataset: nncf.Dataset | None,
    calibration_samples: int,
) -> Path:
    model_dir = output_dir / f"{checkpoint_stem}_openvino_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    xml_path = model_dir / f"{checkpoint_stem}.xml"
    openvino_model = ov.convert_model(model, example_input=sample_input)
    if quantize_int8:
        if calibration_dataset is None:
            raise ValueError("INT8 export requires a calibration dataset.")
        openvino_model = nncf.quantize(
            openvino_model,
            calibration_dataset,
            subset_size=calibration_samples,
        )
    ov.save_model(
        openvino_model,
        xml_path,
        compress_to_fp16=compress_to_fp16 and not quantize_int8,
    )
    return model_dir


def _calibration_dataset(args: argparse.Namespace) -> nncf.Dataset | None:
    if not args.int8:
        return None

    datamodule = CocoDetectionDataModule(
        config=CocoDatasetConfig(
            imgsz=args.imgsz,
            max_eval_samples=args.calibration_samples,
        ),
        batch_size=args.calibration_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="test")

    def transform_fn(batch: Dict[str, Any]):
        return batch["images"].numpy()

    return nncf.Dataset(datamodule.test_dataloader(), transform_fn)


def _output_shape(model: nn.Module, sample_input: Tensor) -> list[int]:
    with torch.no_grad():
        output = model(sample_input)
    return list(output.shape)


def _write_metadata(
    args: argparse.Namespace,
    exported_path: Path,
    metadata_path: Path,
    output_shape: list[int],
) -> Dict[str, Any]:
    metadata = {
        "command": "export",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "exported_path": str(exported_path),
        "exported_exists": exported_path.exists(),
        "format": "openvino",
        "precision": "int8" if args.int8 else "fp16" if args.fp16 else "fp32",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "num_classes": args.num_classes,
        "calibration_samples": args.calibration_samples if args.int8 else None,
        "calibration_batch_size": args.calibration_batch_size if args.int8 else None,
        "output_shape": output_shape,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_export(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    precision = _precision_name(args)
    output_dir = args.output_dir or _default_output_dir(checkpoint_path, precision)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_export_model(checkpoint_path, args.model, args.num_classes)
    sample_input = _sample_input(args)
    shape = _output_shape(model, sample_input)
    calibration_dataset = _calibration_dataset(args)

    exported_path = _export_openvino(
        model,
        sample_input,
        output_dir,
        checkpoint_path.stem,
        compress_to_fp16=args.fp16,
        quantize_int8=args.int8,
        calibration_dataset=calibration_dataset,
        calibration_samples=args.calibration_samples,
    )

    metadata_path = (
        output_dir / f"{checkpoint_path.stem}_openvino_{precision}_export.json"
    )
    return _write_metadata(
        args=args,
        exported_path=exported_path,
        metadata_path=metadata_path,
        output_shape=shape,
    )


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_export(args), indent=2))


if __name__ == "__main__":
    main()
