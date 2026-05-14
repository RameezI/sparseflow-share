"""YAML-driven model builder for dense and sparse YOLO-style detection graphs."""

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import torch
import yaml
from torch import Tensor, nn
from ultralytics.nn.modules import C2f, Concat, Conv, Detect, SPPF

from .sparse_layers import SparseConv2d

_YOLOV8_BN_EPS = 1e-3
_YOLOV8_BN_MOMENTUM = 0.03
DEFAULT_MODEL_SPEC = "yolov8:n"
_MODEL_CONFIG_DIR = Path(__file__).resolve().parent
_MODULES_WITH_SCALED_CHANNELS = {"Conv", "SparseConv", "C2f", "SPPF"}
_MODULES_THAT_TAKE_REPEAT_COUNT = {"C2f"}


def _make_divisible(value: float, divisor: int = 8) -> int:
    return int(math.ceil(value / divisor) * divisor)


class SparseConv(nn.Module):
    """YOLO parser-compatible sparse conv adapter.

    AxonHillock inside SparseConv2d is the sole nonlinearity — no SiLU.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,  # kept for YAML parser compat
    ):
        super().__init__()
        del activation
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = SparseConv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            stateless=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply sparse convolution (AxonHillock is the sole activation)."""
        return self.conv(x)


_MODULE_BY_NAME = {
    "Conv": Conv,
    "SparseConv": SparseConv,
    "C2f": C2f,
    "SPPF": SPPF,
    "Concat": Concat,
    "Detect": Detect,
}


class ModelBuilder(nn.Module):
    """Build a local YOLO-style detect graph from a model YAML."""

    def __init__(
        self,
        cfg_path: str | Path,
        num_classes: int = 80,
        in_channels: int = 3,
        scale: str = "n",
    ):
        super().__init__()
        self.yaml = _load_yaml(cfg_path)
        self.yaml["channels"] = in_channels
        self.yaml["nc"] = num_classes
        self.yaml["scale"] = scale
        _validate_scale(self.yaml, scale, cfg_path)
        self.model, self.save = _parse_model(self.yaml, input_channels=in_channels)
        _apply_yolov8_checkpoint_defaults(self)
        self._initialize_strides(in_channels)

    def _initialize_strides(self, in_channels: int, probe_size: int = 256) -> None:
        """Compute and set Detect layer strides via a dummy training-mode forward pass."""
        detect = next(
            (
                cast(Any, m)
                for m in self.modules()
                if hasattr(m, "stride") and hasattr(m, "nl")
            ),
            None,
        )
        if detect is None:
            return
        was_training = self.training
        self.train()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, probe_size, probe_size)
            out = self.forward(dummy)
        # Training-mode Detect returns a dict with "feats" (list of per-level feature maps).
        if isinstance(out, dict) and "feats" in out:
            detect.stride = torch.tensor(
                [probe_size / feat.shape[-2] for feat in out["feats"]],
                dtype=torch.float32,
            )
        if not was_training:
            self.eval()

    def forward(self, images: Tensor) -> Any:
        """Run a forward pass through the parsed graph."""
        layer_outputs: List[Any] = []
        current_input: Any = images
        for layer in self.model:
            parsed_layer = cast(Any, layer)
            source = parsed_layer.source
            if source != -1:
                current_input = (
                    layer_outputs[source]
                    if isinstance(source, int)
                    else [
                        current_input if index == -1 else layer_outputs[index]
                        for index in source
                    ]
                )
            current_input = parsed_layer(current_input)
            layer_outputs.append(
                current_input if parsed_layer.layer_index in self.save else None
            )
        return current_input


def _load_yaml(cfg_path: str | Path) -> Dict[str, Any]:
    with Path(cfg_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_scale(config: Dict[str, Any], scale: str, cfg_path: str | Path) -> None:
    scales = config.get("scales")
    if not scales or scale in scales:
        return
    available = ", ".join(sorted(scales))
    raise ValueError(
        f"Unsupported scale '{scale}' for {cfg_path}. Available: {available}"
    )


def _model_config_path(stem: str) -> Path:
    if Path(stem).name != stem or Path(stem).suffix:
        raise ValueError(
            f"Model spec must use a YAML stem under {_MODEL_CONFIG_DIR}, got: {stem!r}"
        )
    cfg_path = _MODEL_CONFIG_DIR / f"{stem}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Model YAML not found for spec stem '{stem}': {cfg_path}"
        )
    return cfg_path


def _parse_model_spec(model_spec: str) -> Tuple[Path, str]:
    spec = model_spec.strip()
    if not spec:
        raise ValueError("Model spec cannot be empty.")
    stem, separator, scale = spec.partition(":")
    if not stem:
        raise ValueError(f"Model spec is missing a YAML stem: {model_spec!r}")
    if separator and (not scale or ":" in scale):
        raise ValueError(
            "Model spec must use the form '<yaml_stem>:<scale>', for example 'yolov8:n'."
        )
    cfg_path = _model_config_path(stem)
    return cfg_path, scale or "n"


def build_model_from_spec(
    model_spec: str = DEFAULT_MODEL_SPEC,
    num_classes: int = 80,
    in_channels: int = 3,
) -> ModelBuilder:
    """Build a model from '<yaml_stem>:<scale>', e.g. 'yolov8:n'."""

    cfg_path, scale = _parse_model_spec(model_spec)
    return ModelBuilder(
        cfg_path=cfg_path,
        num_classes=num_classes,
        in_channels=in_channels,
        scale=scale,
    )


def _apply_yolov8_checkpoint_defaults(model: nn.Module) -> None:
    """Match YOLOv8 checkpoint runtime defaults at graph construction time."""

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eps = _YOLOV8_BN_EPS
            module.momentum = _YOLOV8_BN_MOMENTUM
        if isinstance(module, SPPF):
            module.cv1.act = nn.SiLU(inplace=True)


def _resolve_module(module_name: str) -> type[nn.Module]:
    if module_name.startswith("nn."):
        return getattr(nn, module_name.removeprefix("nn."))
    module_class = _MODULE_BY_NAME.get(module_name)
    if module_class is None:
        raise KeyError(f"Unsupported module '{module_name}' in local parser dictionary")
    return module_class


def _channels_from_source(
    source: int | Sequence[int],
    channels_by_layer: Sequence[int],
) -> int:
    if isinstance(source, int):
        return channels_by_layer[source]
    return sum(channels_by_layer[index] for index in source)


def _parse_model(  # pylint: disable=too-many-locals
    config: Dict[str, Any],
    input_channels: int = 3,
) -> Tuple[nn.Sequential, List[int]]:
    """Build a small YOLOv8 graph from the local YAML format."""

    num_classes = config.get("nc")
    box_regression_bins = config.get("reg_max", 16)
    end_to_end = config.get("end2end")
    depth_multiple = config.get("depth_multiple", 1.0)
    width_multiple = config.get("width_multiple", 1.0)
    max_channels = float("inf")

    scales = config.get("scales")
    if scales:
        scale_name = config.get("scale", "n")
        depth_multiple, width_multiple, max_channels = scales[scale_name]

    layers: List[nn.Module] = []
    channels_by_layer: List[int] = [input_channels]
    saved_layer_indices: List[int] = []

    layer_specs = config["backbone"] + config["head"]
    for layer_index, (source, repeat_count, module_name, raw_args) in enumerate(
        layer_specs
    ):
        module_class = _resolve_module(module_name)
        module_args = [num_classes if value == "nc" else value for value in raw_args]

        scaled_repeat_count = (
            max(round(repeat_count * depth_multiple), 1)
            if repeat_count > 1
            else repeat_count
        )

        if module_name in _MODULES_WITH_SCALED_CHANNELS:
            input_channel_count = _channels_from_source(source, channels_by_layer)
            output_channel_count = module_args[0]
            if output_channel_count != num_classes:
                output_channel_count = _make_divisible(
                    min(output_channel_count, max_channels) * width_multiple,
                    8,
                )
            module_args = [input_channel_count, output_channel_count, *module_args[1:]]
            if module_name in _MODULES_THAT_TAKE_REPEAT_COUNT:
                module_args.insert(2, scaled_repeat_count)
                scaled_repeat_count = 1
        elif module_name == "Concat":
            output_channel_count = _channels_from_source(source, channels_by_layer)
        elif module_name == "Detect":
            output_channel_count = _channels_from_source(source, channels_by_layer)
            module_args.extend(
                [
                    box_regression_bins,
                    end_to_end,
                    [
                        _channels_from_source(index, channels_by_layer)
                        for index in source
                    ],
                ]
            )
            module_class.legacy = True
        else:
            output_channel_count = _channels_from_source(source, channels_by_layer)

        if scaled_repeat_count == 1:
            layer = module_class(*module_args)
        else:
            layer = nn.Sequential(
                *(module_class(*module_args) for _ in range(scaled_repeat_count))
            )
        _tag_layer(layer, layer_index, source, module_name)

        saved_layer_indices.extend(
            index % layer_index
            for index in ([source] if isinstance(source, int) else source)
            if index != -1
        )
        layers.append(layer)

        if layer_index == 0:
            channels_by_layer = []
        channels_by_layer.append(output_channel_count)

    return nn.Sequential(*layers), sorted(saved_layer_indices)


def _tag_layer(
    layer: nn.Module,
    layer_index: int,
    source: int | Sequence[int],
    module_name: str,
) -> None:
    layer.layer_index = layer_index
    layer.source = source
    layer.module_name = module_name
    layer.parameter_count = sum(parameter.numel() for parameter in layer.parameters())

    # Keep Ultralytics-style attribute names for checkpoint/debug compatibility.
    layer.i = layer_index
    layer.f = source
    layer.type = module_name
    layer.np = layer.parameter_count
