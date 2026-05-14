"""Benchmark SparseFlow checkpoints and exported OpenVINO models."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import openvino as ov
import torch
from torch import Tensor, nn

from data import CocoDatasetConfig, CocoDetectionDataModule
from detection_utils import DetectionMetricsEvaluator
from export import ExportableDetectionModel
from models.model_builder import DEFAULT_MODEL_SPEC, build_model_from_spec
from step_module import StepModule

InputProvider = Callable[[], Tensor]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch checkpoint and OpenVINO exports."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Lightning checkpoint to benchmark as the PyTorch reference.",
    )
    parser.add_argument(
        "--openvino-model",
        action="append",
        default=[],
        help=(
            "OpenVINO XML to benchmark. Use NAME=PATH to set a label. "
            "If omitted, exports/ is searched for this checkpoint stem."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--openvino-device", type=str, default="CPU")
    parser.add_argument(
        "--source",
        choices=("dummy", "coco"),
        default="dummy",
        help="Use zero tensors or HF COCO validation images for latency inputs.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--evaluate-map",
        action="store_true",
        help="Also compute mAP50 and mAP50-95 on a small HF COCO validation slice.",
    )
    parser.add_argument("--map-samples", type=int, default=100)
    return parser


def _load_pytorch_model(
    checkpoint_path: Path,
    model_spec: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = build_model_from_spec(model_spec=model_spec, num_classes=num_classes)
    protocol = StepModule.load_from_checkpoint(
        str(checkpoint_path),
        model=model,
        strict=False,
    )
    protocol.eval()
    exportable = ExportableDetectionModel(protocol.model).eval().to(device)
    return exportable


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dummy_provider(batch: int, imgsz: int, device: torch.device) -> InputProvider:
    sample = torch.zeros(batch, 3, imgsz, imgsz, dtype=torch.float32, device=device)

    def next_input() -> Tensor:
        return sample

    return next_input


def _coco_provider(
    batch: int,
    imgsz: int,
    device: torch.device,
    samples: int,
    num_workers: int,
) -> InputProvider:
    datamodule = CocoDetectionDataModule(
        config=CocoDatasetConfig(imgsz=imgsz, max_eval_samples=max(samples, batch)),
        batch_size=batch,
        num_workers=num_workers,
    )
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    iterator = iter(dataloader)

    def next_input() -> Tensor:
        nonlocal iterator
        try:
            batch_dict = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch_dict = next(iterator)
        return batch_dict["images"].to(device)

    return next_input


def _build_input_provider(
    args: argparse.Namespace, device: torch.device
) -> InputProvider:
    if args.source == "dummy":
        return _dummy_provider(args.batch, args.imgsz, device)
    sample_count = (args.warmup + args.iterations) * args.batch
    return _coco_provider(
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        samples=sample_count,
        num_workers=args.num_workers,
    )


def _latency_summary(
    latencies_ms: List[float], images_processed: int
) -> Dict[str, float]:
    total_ms = sum(latencies_ms)
    sorted_latencies = sorted(latencies_ms)

    def percentile(value: float) -> float:
        if not sorted_latencies:
            return float("nan")
        index = min(
            len(sorted_latencies) - 1,
            max(0, round((value / 100.0) * (len(sorted_latencies) - 1))),
        )
        return sorted_latencies[index]

    return {
        "iterations": len(latencies_ms),
        "images": images_processed,
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": percentile(90),
        "p95_ms": percentile(95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "throughput_img_s": images_processed / (total_ms / 1000.0),
    }


def _benchmark_pytorch(
    model: nn.Module,
    input_provider: InputProvider,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> Dict[str, Any]:
    output_shape: List[int] = []
    with torch.inference_mode():
        for _ in range(warmup):
            sample = input_provider()
            output = model(sample)
            _sync_device(device)
            output_shape = list(output.shape)

        latencies_ms: List[float] = []
        images_processed = 0
        for _ in range(iterations):
            sample = input_provider()
            _sync_device(device)
            start = time.perf_counter()
            output = model(sample)
            _sync_device(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            images_processed += sample.shape[0]
            output_shape = list(output.shape)

    return {
        "output_shape": output_shape,
        "latency": _latency_summary(latencies_ms, images_processed),
    }


def _benchmark_openvino(
    xml_path: Path,
    input_provider: InputProvider,
    warmup: int,
    iterations: int,
    openvino_device: str,
) -> Dict[str, Any]:
    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, openvino_device)
    output_shape: List[int] = []

    for _ in range(warmup):
        sample = input_provider().detach().cpu().numpy()
        output = compiled([sample])[0]
        output_shape = list(np.asarray(output).shape)

    latencies_ms: List[float] = []
    images_processed = 0
    for _ in range(iterations):
        sample_tensor = input_provider()
        sample = sample_tensor.detach().cpu().numpy()
        start = time.perf_counter()
        output = compiled([sample])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        images_processed += sample_tensor.shape[0]
        output_shape = list(np.asarray(output).shape)

    return {
        "output_shape": output_shape,
        "latency": _latency_summary(latencies_ms, images_processed),
    }


def _count_parameters(model: nn.Module) -> Dict[str, int]:
    parameters = list(model.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }


def _approximate_macs_per_image(
    model: nn.Module,
    imgsz: int,
    device: torch.device,
) -> int:
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, _inputs: tuple[Any, ...], output: Tensor) -> None:
        nonlocal macs
        output_tensor = output[0] if isinstance(output, tuple) else output
        kernel_macs = module.kernel_size[0] * module.kernel_size[1]
        kernel_macs *= module.in_channels // module.groups
        macs += int(output_tensor.numel() * kernel_macs)

    def linear_hook(
        module: nn.Linear, _inputs: tuple[Any, ...], output: Tensor
    ) -> None:
        nonlocal macs
        output_tensor = output[0] if isinstance(output, tuple) else output
        macs += int(output_tensor.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    try:
        with torch.inference_mode():
            sample = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32, device=device)
            model(sample)
    finally:
        for hook in hooks:
            hook.remove()

    return macs


def _file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _directory_size_bytes(path: Path) -> int:
    if path.is_file():
        return _file_size_bytes(path)
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _precision_from_metadata(export_dir: Path) -> str | None:
    for metadata_path in export_dir.glob("*_openvino_*_export.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        precision = metadata.get("precision")
        if isinstance(precision, str):
            return precision
    return None


def _named_openvino_models(
    specs: Iterable[str],
    checkpoint_stem: str,
) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    if specs:
        for spec in specs:
            name, separator, raw_path = spec.partition("=")
            if separator:
                discovered[name] = Path(raw_path).expanduser().resolve()
            else:
                path = Path(name).expanduser().resolve()
                discovered[path.parent.parent.name] = path
        return discovered

    for xml_path in sorted(
        Path("exports").glob(
            f"*/{checkpoint_stem}_openvino_model/{checkpoint_stem}.xml"
        )
    ):
        export_dir = xml_path.parent.parent
        precision = _precision_from_metadata(export_dir) or export_dir.name
        name = precision
        if name in discovered:
            name = f"{precision}:{export_dir.name}"
        discovered[name] = xml_path.resolve()
    return discovered


def _evaluate_map_pytorch(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    datamodule = CocoDetectionDataModule(
        config=CocoDatasetConfig(imgsz=args.imgsz, max_eval_samples=args.map_samples),
        batch_size=args.batch,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="test")
    evaluator = DetectionMetricsEvaluator()
    return evaluator(model, datamodule.test_dataloader(), device)


def _evaluate_map_openvino(
    xml_path: Path,
    args: argparse.Namespace,
) -> Dict[str, float]:
    core = ov.Core()
    compiled = core.compile_model(core.read_model(xml_path), args.openvino_device)
    datamodule = CocoDetectionDataModule(
        config=CocoDatasetConfig(imgsz=args.imgsz, max_eval_samples=args.map_samples),
        batch_size=args.batch,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="test")
    evaluator = DetectionMetricsEvaluator()
    evaluator.reset()
    for batch in datamodule.test_dataloader():
        images, targets = evaluator.preprocess(batch, torch.device("cpu"))
        output = compiled([images.numpy()])[0]
        preds = evaluator.postprocess(torch.from_numpy(np.asarray(output)))
        evaluator.update(preds, targets)
    return evaluator.compute()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if args.evaluate_map and args.map_samples <= 0:
        raise ValueError("--map-samples must be positive when --evaluate-map is set.")

    device = torch.device(args.device)
    pytorch_model = _load_pytorch_model(
        checkpoint_path,
        args.model,
        args.num_classes,
        device,
    )
    input_provider = _build_input_provider(args, device)
    pytorch_result = _benchmark_pytorch(
        pytorch_model,
        input_provider,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    parameters = _count_parameters(pytorch_model)
    pytorch_result["parameters"] = parameters
    pytorch_result["checkpoint_size_bytes"] = _file_size_bytes(checkpoint_path)
    pytorch_result["approx_macs_per_image"] = _approximate_macs_per_image(
        pytorch_model,
        args.imgsz,
        device,
    )
    pytorch_result["approx_gmacs_per_image"] = (
        pytorch_result["approx_macs_per_image"] / 1e9
    )
    if args.evaluate_map:
        pytorch_result["metrics"] = _evaluate_map_pytorch(pytorch_model, args, device)

    openvino_results: Dict[str, Any] = {}
    for name, xml_path in _named_openvino_models(
        args.openvino_model, checkpoint_path.stem
    ).items():
        if not xml_path.exists():
            raise FileNotFoundError(f"OpenVINO XML not found for {name}: {xml_path}")
        ov_input_provider = _build_input_provider(args, torch.device("cpu"))
        result = _benchmark_openvino(
            xml_path,
            ov_input_provider,
            warmup=args.warmup,
            iterations=args.iterations,
            openvino_device=args.openvino_device,
        )
        result["xml_path"] = str(xml_path)
        result["artifact_size_bytes"] = _directory_size_bytes(xml_path.parent)
        if args.evaluate_map:
            result["metrics"] = _evaluate_map_openvino(xml_path, args)
        openvino_results[name] = result

    output_path = args.output_json
    if output_path is None:
        output_path = (
            Path("benchmarks") / f"{checkpoint_path.stem}_{args.source}_benchmark.json"
        )

    report = {
        "command": "benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "model": args.model,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "source": args.source,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "pytorch": pytorch_result,
        "openvino": openvino_results,
        "output_json": str(output_path),
    }
    _write_json(output_path, report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_benchmark(args), indent=2))


if __name__ == "__main__":
    main()
