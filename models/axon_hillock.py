"""Sparse spiking-inspired activation used by SparseFlow sparse layers."""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,attribute-defined-outside-init
class AxonHillock(nn.Module):
    """Sparse spiking-inspired activation with learnable thresholds and state dynamics.

    When ``stateless=True``, only a learnable threshold is used: activations below
    the threshold are gated to zero using a surrogate gradient.  No temporal state
    (membrane potential, leak, reset, decay) is maintained — suitable for per-image
    spatial sparsity in Phase 3.

    When ``stateless=False`` (default), the full stateful dynamics are active:
    membrane potential accumulates across calls with leak, spike-triggered reset,
    and response decay — suitable for temporal sparsity in Phase 4.
    """

    def __init__(
        self,
        stateless: bool = False,
        granularity_threshold: str = "channel",
        granularity_state_leak: str = "layer",
        granularity_repolarization: str = "layer",
        granularity_response_decay: str = "layer",
        regularization_const_l0: float = 1e-2,
        regularization_const_l1: float = 1e-2,
        detach_state: bool = True,
        surrogate_scale: float = 5.0,
    ) -> None:
        super().__init__()
        valid = {"neuron", "channel", "layer"}
        granularities = {granularity_threshold}
        if not stateless:
            granularities |= {
                granularity_state_leak,
                granularity_repolarization,
                granularity_response_decay,
            }
        if not granularities.issubset(valid):
            raise ValueError(f"Unsupported granularity values: {granularities - valid}")

        self.stateless = stateless
        self.granularity_threshold = granularity_threshold
        self.granularity_state_leak = granularity_state_leak
        self.granularity_repolarization = granularity_repolarization
        self.granularity_response_decay = granularity_response_decay
        self.regularization_const_l0 = regularization_const_l0
        self.regularization_const_l1 = regularization_const_l1
        self.detach_state = detach_state
        self.surrogate_scale = surrogate_scale

        self.thresholds: Optional[nn.Parameter] = None
        self.leak_consts: Optional[nn.Parameter] = None
        self.reset_consts: Optional[nn.Parameter] = None
        self.decay_consts: Optional[nn.Parameter] = None

        if not stateless:
            self.register_buffer("membrane_potentials", None, persistent=False)
            self.register_buffer("spikes", None, persistent=False)
            self.register_buffer("responses", None, persistent=False)

        self.last_regularization = torch.tensor(0.0)
        self.last_sparsity_ratio: float = 0.0

    @staticmethod
    def _shape_from_granularity(
        input_shape: Sequence[int],
        granularity: str,
    ) -> Sequence[int]:
        feature_rank = max(len(input_shape) - 1, 1)
        if granularity == "layer":
            return (1,) * (feature_rank + 1)
        if granularity == "channel":
            shape = [1] * (feature_rank + 1)
            if len(input_shape) > 1:
                shape[1] = input_shape[1]
            return tuple(shape)
        return (1, *tuple(input_shape[1:]))

    @staticmethod
    def _surrogate_heaviside(x: Tensor, scale: float) -> Tensor:
        hard = (x > 0).to(x.dtype)
        soft = torch.sigmoid(scale * x)
        return hard + soft - soft.detach()

    def _lazy_init_params(self, x: Tensor) -> None:
        """Create learnable parameters after the first input reveals feature shape."""
        if self.thresholds is not None:
            return
        input_shape = tuple(x.shape)
        device, dtype = x.device, x.dtype

        self.thresholds = nn.Parameter(
            torch.zeros(
                self._shape_from_granularity(input_shape, self.granularity_threshold),
                device=device,
                dtype=dtype,
            )
        )

        if self.stateless:
            return

        self.leak_consts = nn.Parameter(
            torch.zeros(
                self._shape_from_granularity(input_shape, self.granularity_state_leak),
                device=device,
                dtype=dtype,
            )
        )
        self.reset_consts = nn.Parameter(
            torch.zeros(
                self._shape_from_granularity(
                    input_shape,
                    self.granularity_repolarization,
                ),
                device=device,
                dtype=dtype,
            )
        )
        self.decay_consts = nn.Parameter(
            torch.zeros(
                self._shape_from_granularity(
                    input_shape,
                    self.granularity_response_decay,
                ),
                device=device,
                dtype=dtype,
            )
        )

    def _lazy_init_state(self, x: Tensor) -> None:
        """Create state buffers after the first input reveals feature shape."""
        shape = (1, *x.shape[1:])
        needs_init = (
            self.membrane_potentials is None
            or tuple(self.membrane_potentials.shape) != shape
        )
        if needs_init:
            self.membrane_potentials = torch.zeros(
                shape, device=x.device, dtype=x.dtype
            )
            self.spikes = torch.zeros(shape, device=x.device, dtype=x.dtype)
            self.responses = torch.zeros(shape, device=x.device, dtype=x.dtype)

    def reset_state(self) -> None:
        """Drop cached state so the next forward starts from a blank membrane."""
        if self.stateless:
            return
        self.membrane_potentials = None
        self.spikes = None
        self.responses = None

    def _compute_regularization_and_sparsity(self, output: Tensor) -> None:
        """Compute L0/L1 regularization and sparsity ratio for the given output."""
        n_neurons = max(output.numel(), 1)
        nonzero_count = torch.count_nonzero(output)
        avg_norm_l0 = nonzero_count.to(output.dtype) / n_neurons
        avg_norm_l1 = output.abs().sum() / n_neurons
        self.last_regularization = (
            self.regularization_const_l0 * avg_norm_l0
            + self.regularization_const_l1 * avg_norm_l1
        )
        self.last_sparsity_ratio = 1.0 - (nonzero_count.item() / n_neurons)

    def _stateless_forward(self, x: Tensor) -> Tensor:
        """Threshold-only gating: no temporal state."""
        spikes = self._surrogate_heaviside(x - self.thresholds, self.surrogate_scale)
        output = spikes * x
        self._compute_regularization_and_sparsity(output)
        return output

    def _stateful_forward(self, x: Tensor) -> Tensor:
        """Full stateful dynamics with membrane potential and decay."""
        self._lazy_init_state(x)

        membrane_prev = (
            self.membrane_potentials.detach()
            if self.detach_state
            else self.membrane_potentials
        )
        spikes_prev = self.spikes.detach() if self.detach_state else self.spikes
        responses_prev = (
            self.responses.detach() if self.detach_state else self.responses
        )

        membrane = (
            x + self.leak_consts * membrane_prev + self.reset_consts * spikes_prev
        )
        spikes = self._surrogate_heaviside(
            membrane - self.thresholds, self.surrogate_scale
        )
        responses = spikes * membrane + self.decay_consts * responses_prev

        self._compute_regularization_and_sparsity(responses)

        self.membrane_potentials = membrane.detach()
        self.spikes = spikes.detach()
        self.responses = responses.detach()
        return responses

    def forward(self, x: Tensor) -> Tensor:
        """Apply sparse activation dynamics to an input tensor."""
        self._lazy_init_params(x)
        if self.stateless:
            return self._stateless_forward(x)
        return self._stateful_forward(x)
