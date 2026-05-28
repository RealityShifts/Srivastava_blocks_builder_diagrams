# Custom blocks

Drop your own block definitions here and they get picked up automatically the
next time you run:

```bash
python tools/build_manifest.py
```

## Layout

```
models/customblocks/
  pytorch/    # *.py files with torch.nn.Module classes (or jaxtyped functions)
  flax/       # *.py files with nnx.Module classes (or jaxtyped functions)
```

Each `.py` file is loaded as a standalone module - no `__init__.py` needed. A
file named `my_block.py` shows up in the palette under the section
**`custom · my_block`**, right after the built-in / utility groups.

## Authoring requirements

Custom blocks are introspected exactly like the built-in ones, so they need
the same machinery:

1. The class must subclass `torch.nn.Module` (or `flax.nnx.Module`).
2. The `forward` (PyTorch) or `__call__` (Flax) method must have
   [jaxtyping](https://docs.kidger.site/jaxtyping/) annotations on every tensor
   input and on the return value. Shape symbols are free text (e.g.
   `Float[Tensor, "B C H W"]`) - the builder will infer them as named axes and
   try to bind them to constructor params by name (`in_ch -> C_in`, etc.).
3. Constructor params should have plain Python type hints (`int`, `float`,
   `bool`, `str`, `list`, or `Optional[...]`); they become editable fields in
   the inspector.

The built-in `_typecheck` helper is importable as
`from pytorch_blocks._typecheck import typecheck` because `models/blocks/`
is on `sys.path` during the build.

## Minimal example

```python
# models/customblocks/pytorch/my_block.py
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from pytorch_blocks._typecheck import typecheck


class DoubleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features)
        self.l2 = nn.Linear(out_features, out_features)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B D_in"]
    ) -> Float[Tensor, "B D_out"]:
        return self.l2(self.l1(x).relu())
```

Run `python tools/build_manifest.py` and `DoubleLinear` will appear in the
PyTorch palette under **`custom · my_block`**.
