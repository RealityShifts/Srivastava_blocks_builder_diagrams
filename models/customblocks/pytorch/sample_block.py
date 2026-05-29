# =============================================================================
# SAMPLE CUSTOM BLOCK  ·  a guided template for new users
# =============================================================================
#
# Drop a file like this into  models/customblocks/pytorch/  and run
#
#     python tools/build_manifest.py
#
# ...and your block shows up in the palette under the section
# "custom · sample_block" (the section name is the file name).
#
# A block only needs to satisfy three rules to be introspected correctly:
#
#   1. It is a torch.nn.Module subclass.
#   2. Its forward() has jaxtyping annotations on EVERY tensor input and on
#      the return value, e.g.  Float[Tensor, "B C_in H W"].  The shape symbols
#      are just names — the builder turns them into axes and tries to bind
#      them to constructor params by name.
#   3. Its __init__ params have plain type hints (int / float / bool / str /
#      list / Optional[...]).  Each one becomes an editable field in the
#      inspector on the right of the canvas.
#
# Below is one minimal block (uncomment-and-go) and one slightly richer block
# that shows how constructor params bind to shape axes.
# =============================================================================

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

# The built-in typecheck decorator is importable because models/blocks/ is on
# sys.path while build_manifest.py runs. It enforces the jaxtyping shapes at
# runtime and is required for the builder to read the annotations.
from pytorch_blocks._typecheck import typecheck


class SampleLinearBlock(nn.Module):
    """The simplest useful block: a single Linear layer + ReLU.

    Start here. Copy this class, rename it, and change the body.
    """

    def __init__(self, in_features: int, out_features: int):
        # ^ both params get a number field in the inspector. Their NAMES matter:
        #   the builder matches "in_features" to the axis it should drive.
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    @typecheck
    def forward(self, x: Float[Tensor, "B D_in"]) -> Float[Tensor, "B D_out"]:
        # "B"     -> batch axis, left symbolic so any batch size flows through.
        # "D_in"  -> bound to in_features  (set it in the inspector -> axis fills in).
        # "D_out" -> bound to out_features.
        return self.linear(x).relu()


class SampleMLPBlock(nn.Module):
    """A richer example: a 2-layer MLP with a configurable hidden size and an
    optional residual connection.

    This demonstrates:
      * three editable params of different types (two int, one bool),
      * a hidden dimension that is NOT exposed as an axis (it stays internal),
      * how the input and output axes stay the same when residual is on.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        use_residual: bool = False,  # bool params render as a checkbox
    ):
        super().__init__()
        self.use_residual = use_residual
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

    @typecheck
    def forward(self, x: Float[Tensor, "B D_in"]) -> Float[Tensor, "B D_out"]:
        out = self.fc2(self.act(self.fc1(x)))
        if self.use_residual:
            # Residual only makes sense when in_features == out_features; the
            # builder won't enforce that, so it's on you to wire it sensibly.
            out = out + x
        return out
