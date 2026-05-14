"""Sparse layer wrappers that pair standard layers with AxonHillock activations."""

from torch import Tensor, nn

from .axon_hillock import AxonHillock


class SparseConv2d(nn.Module):
    """2D convolution followed by an AxonHillock activation."""

    def __init__(self, *args, stateless: bool = False, **kwargs):
        super().__init__()
        kwargs.setdefault("bias", False)
        self.convolution = nn.Conv2d(*args, **kwargs)
        self.axon = AxonHillock(stateless=stateless)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution and activation."""
        return self.axon(self.convolution(x))


class SparseLinear(nn.Module):
    """Linear layer followed by an AxonHillock activation."""

    def __init__(self, *args, stateless: bool = False, **kwargs):
        super().__init__()
        kwargs.setdefault("bias", False)
        self.linear = nn.Linear(*args, **kwargs)
        self.axon = AxonHillock(stateless=stateless, granularity_threshold="neuron")

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear projection and activation."""
        return self.axon(self.linear(x))
