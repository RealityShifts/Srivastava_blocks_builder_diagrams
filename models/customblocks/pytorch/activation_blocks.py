# =============================================================================
# ACTIVATION BLOCKS  ·  one standalone node per activation function
# =============================================================================
#
# Drop any of these after a step on the canvas to apply an activation.
# The shape flows straight through ("..."), so they wire onto the output of
# any block without changing axes. Run
#
#     python tools/build_manifest.py
#
# and they appear in the palette under "custom · activation_blocks".
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from pytorch_blocks._typecheck import typecheck


class ReLU(nn.Module):
    """ReLU activation: ``max(0, x)``."""

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.relu(x)


class LeakyReLU(nn.Module):
    """Leaky ReLU: ``x`` if ``x > 0`` else ``negative_slope * x``."""

    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.negative_slope = negative_slope

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.leaky_relu(x, self.negative_slope)


class GELU(nn.Module):
    """Gaussian Error Linear Unit."""

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.gelu(x)


class SiLU(nn.Module):
    """SiLU / Swish activation: ``x * sigmoid(x)``."""

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.silu(x)


class Mish(nn.Module):
    """Mish activation: ``x * tanh(softplus(x))``."""

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * torch.tanh(F.softplus(x))


class ELU(nn.Module):
    """Exponential Linear Unit."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.elu(x, self.alpha)


class Tanh(nn.Module):
    """Hyperbolic tangent activation."""

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.tanh(x)


class Sigmoid(nn.Module):
    """Logistic sigmoid activation."""

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.sigmoid(x)


class Softplus(nn.Module):
    """Softplus activation: ``log(1 + exp(x))``."""

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.softplus(x)


class Softmax(nn.Module):
    """Softmax over a chosen axis (default: last)."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.softmax(x, dim=self.dim)
