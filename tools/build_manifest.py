"""Extract block metadata from models/blocks/{pytorch,flax}_blocks into JSON.

User-defined ("custom") blocks living under ``models/customblocks/<framework>/``
are picked up as well: each ``*.py`` file in those directories is loaded as a
standalone module and any ``nn.Module`` / ``nnx.Module`` (or jaxtyped function)
declared there gets merged into the same per-framework manifest.

The generated manifests are consumed by the web/ Rete editor for shape
inference, validation and Python code generation.

Schema per entry::

    {
      "name": "ConvBlock",
      "module": "pytorch_blocks.core_blocks",
      "framework": "pytorch",
      "kind": "module" | "function",
      "ctor": [
        {"name": "in_ch", "type": "int", "default": null, "required": true},
        ...
      ],
      "inputs": [
        {"name": "x", "shape": ["B", "C_in", "H", "W"], "dtype": "float",
         "optional": false, "variadic": false}
      ],
      "outputs": [
        {"name": "out", "shape": ["B", "C_out", "H_out", "W_out"],
         "dtype": "float"}
      ],
      "bindings": {"C_in": "in_ch", "C_out": "out_ch"}
    }

Run from repo root::

    python tools/build_manifest.py
"""

from __future__ import annotations

import collections.abc
import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SEQUENCE_ORIGINS = (list, tuple, collections.abc.Sequence)

# Disable runtime checks during introspection so importing the packages
# doesn't trip jaxtyped/beartype.
os.environ.setdefault("CHECK_TYPES", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKS_ROOT = REPO_ROOT / "models" / "blocks"
CUSTOM_BLOCKS_ROOT = REPO_ROOT / "models" / "customblocks"
sys.path.insert(0, str(BLOCKS_ROOT))
# Custom block files can `from pytorch_blocks._typecheck import typecheck` etc.
# We also expose CUSTOM_BLOCKS_ROOT on sys.path so user files can import each
# other by their bare module name.
if CUSTOM_BLOCKS_ROOT.is_dir():
    sys.path.insert(0, str(CUSTOM_BLOCKS_ROOT))


# ---------------------------------------------------------------------------
# Heuristics for binding constructor params to shape axes
# ---------------------------------------------------------------------------

# Maps constructor-param canonical names to the axis name(s) they typically
# control. The matching is suffix-aware: e.g. ``in_ch`` -> ``C_in``.
_AXIS_BY_PARAM: dict[str, tuple[str, ...]] = {
    "in_ch": ("C_in",),
    "in_channels": ("C_in",),
    "in_features": ("D_in", "I"),
    "in_dim": ("D_in",),
    "out_ch": ("C_out",),
    "out_channels": ("C_out",),
    "out_features": ("D_out", "O"),
    "out_dim": ("D_out",),
    "channels": ("C", "C_in"),
    "dim": ("D",),
    "embed_dim": ("D",),
    "hidden": ("D_h",),
    "hidden_dim": ("D_h",),
    "num_heads": ("H_attn",),
    "vocab_size": ("V",),
    "num_slots": ("K",),
    "num_latents": ("K",),
    "num_queries": ("Q",),
    "num_buckets": ("Buckets",),
    "max_len": ("T_max",),
    "z_dim": ("D_z",),
    "w_dim": ("D_w",),
    "code_dim": ("D_code",),
    "state_dim": ("N",),
    "rank": ("R",),
    "latent": ("Z",),
    "key_dim": ("D_k",),
    "value_dim": ("D_v",),
    "num_in": ("N_in",),
    "num_out": ("N_out",),
}


def infer_bindings(
    ctor_params: list[str], axes_used: set[str]
) -> dict[str, str]:
    """Return a mapping ``axis -> ctor_param`` based on common conventions."""
    bindings: dict[str, str] = {}
    for param in ctor_params:
        candidates = _AXIS_BY_PARAM.get(param, ())
        for axis in candidates:
            if axis in axes_used and axis not in bindings:
                bindings[axis] = param
                break
    return bindings


# ---------------------------------------------------------------------------
# Jaxtyping annotation extraction
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    shape: list[str]
    dtype: str
    optional: bool = False
    variadic: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "optional": self.optional,
            "variadic": self.variadic,
        }


def _classify_dtype(jt_cls: Any) -> str:
    """Map jaxtyping dtype tuple to a single short label."""
    dtypes = getattr(jt_cls, "dtypes", None)
    if dtypes is None or not isinstance(dtypes, tuple):
        return "any"
    if any(d.startswith("float") or d.startswith("bfloat") for d in dtypes):
        return "float"
    if any(d == "complex64" or d == "complex128" for d in dtypes):
        return "complex"
    if any(d.startswith("int") or d.startswith("uint") for d in dtypes):
        if dtypes == ("int8",):
            return "int8"
        return "int"
    if any(d == "bool" for d in dtypes):
        return "bool"
    return "any"


def _parse_shape(dim_str: str) -> list[str]:
    """Turn ``"B C H W"`` into ``["B", "C", "H", "W"]``."""
    return [tok for tok in dim_str.split() if tok]


def _is_jaxtyping(obj: Any) -> bool:
    return hasattr(obj, "dim_str") and hasattr(obj, "array_type")


def _extract_spec(annotation: Any) -> TensorSpec | None:
    """Pull a TensorSpec from an annotation, peeling Optional / Sequence wrappers."""
    if annotation is None or annotation is type(None):
        return None

    if _is_jaxtyping(annotation):
        return TensorSpec(
            shape=_parse_shape(annotation.dim_str),
            dtype=_classify_dtype(annotation),
        )

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[X]  /  Union[X, None]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            spec = _extract_spec(non_none[0])
            if spec is not None:
                spec.optional = True
            return spec
        return None

    # Sequence[X] / list[X] / tuple[X, ...]
    if origin in _SEQUENCE_ORIGINS or (
        isinstance(origin, type) and issubclass(origin, (list, tuple))
    ):
        # tuple[X, Y]  - fixed-length, returned separately by caller
        if origin is tuple and len(args) >= 2 and args[-1] is not Ellipsis:
            return None  # handled at the outer level for multi-output funcs
        inner_args = [a for a in args if a is not Ellipsis]
        if not inner_args:
            return None
        spec = _extract_spec(inner_args[0])
        if spec is None:
            return None
        spec.variadic = True
        return spec

    return None


def _extract_returns(annotation: Any) -> list[TensorSpec]:
    """Return one spec per output port. Handles ``tuple[X, Y, Z]`` returns."""
    if annotation is inspect.Signature.empty or annotation is None:
        return []

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # tuple[X, Y, Z]  - multi-output
    if origin is tuple and len(args) >= 2 and args[-1] is not Ellipsis:
        out: list[TensorSpec] = []
        for a in args:
            spec = _extract_spec(a)
            if spec is not None:
                out.append(spec)
        return out

    spec = _extract_spec(annotation)
    return [spec] if spec is not None else []


# ---------------------------------------------------------------------------
# Constructor params
# ---------------------------------------------------------------------------


def _classify_param_type(annotation: Any) -> str:
    """Best-effort string label for the JSON manifest."""
    if annotation is inspect.Signature.empty:
        return "any"
    if annotation in (int, float, bool, str):
        return annotation.__name__

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _classify_param_type(non_none[0])
    if origin in (list, tuple, typing.Sequence):
        return "list"
    if isinstance(annotation, type):
        return annotation.__name__
    return "any"


def _ctor_params(cls: type) -> list[dict[str, Any]]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    try:
        hints = typing.get_type_hints(cls.__init__)
    except Exception:
        hints = {}
    params: list[dict[str, Any]] = []
    for name, p in sig.parameters.items():
        if name in {"self", "args", "kwargs", "rngs"}:
            continue
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = p.default if p.default is not inspect.Parameter.empty else None
        annotation = hints.get(name, p.annotation)
        params.append(
            {
                "name": name,
                "type": _classify_param_type(annotation),
                "default": default if _is_json_safe(default) else None,
                "required": p.default is inspect.Parameter.empty,
            }
        )
    return params


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Walking the package
# ---------------------------------------------------------------------------


def _input_params(
    func: typing.Callable, hints: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the tensor-input port specs for ``func``."""
    sig = inspect.signature(func)
    ports: list[dict[str, Any]] = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, p.annotation)
        spec = _extract_spec(annotation)
        if spec is None:
            continue
        port = spec.to_json()
        port["name"] = name
        ports.append(port)
    return ports


def _output_ports(hints: dict[str, Any]) -> list[dict[str, Any]]:
    return_ann = hints.get("return", inspect.Signature.empty)
    specs = _extract_returns(return_ann)
    ports: list[dict[str, Any]] = []
    for i, s in enumerate(specs):
        port = s.to_json()
        port["name"] = f"out{i}" if len(specs) > 1 else "out"
        ports.append(port)
    return ports


def _entry_for_callable(
    obj: Any,
    name: str,
    module_path: str,
    framework: str,
    kind: str,
    func: typing.Callable,
    ctor: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        return None

    inputs = _input_params(func, hints)
    outputs = _output_ports(hints)
    if not inputs and not outputs:
        return None

    axes = {a for p in inputs + outputs for a in p["shape"] if a.isidentifier()}
    ctor_names = [p["name"] for p in (ctor or [])]
    bindings = infer_bindings(ctor_names, axes)

    return {
        "name": name,
        "module": module_path,
        "framework": framework,
        "kind": kind,
        "ctor": ctor or [],
        "inputs": inputs,
        "outputs": outputs,
        "bindings": bindings,
    }


def _is_block_class(cls: type, base_names: tuple[str, ...]) -> bool:
    if not inspect.isclass(cls):
        return False
    mro_names = {c.__name__ for c in cls.__mro__}
    return any(b in mro_names for b in base_names)


def _framework_targets(framework: str) -> tuple[tuple[str, ...], str]:
    """Base class names and forward-method name per framework."""
    if framework == "pytorch":
        return ("Module",), "forward"
    return ("Module", "nnx.Module"), "__call__"


def _entries_from_module(
    module: typing.Any,
    framework: str,
    module_label: str,
) -> list[dict[str, Any]]:
    """Extract block entries from an already-imported module."""
    base_names, method_name = _framework_targets(framework)
    entries: list[dict[str, Any]] = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue

        if _is_block_class(obj, base_names):
            method = obj.__dict__.get(method_name)
            if method is None:
                continue
            entry = _entry_for_callable(
                obj,
                name,
                module_label,
                framework,
                kind="module",
                func=method,
                ctor=_ctor_params(obj),
            )
            if entry is not None:
                entries.append(entry)

        elif inspect.isfunction(obj):
            entry = _entry_for_callable(
                obj,
                name,
                module_label,
                framework,
                kind="function",
                func=obj,
                ctor=None,
            )
            if entry is not None:
                entries.append(entry)
    return entries


def _walk(framework: str) -> list[dict[str, Any]]:
    pkg_name = f"{framework}_blocks"
    pkg = importlib.import_module(pkg_name)
    entries: list[dict[str, Any]] = []

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{pkg_name}.{info.name}")
        except Exception as exc:
            print(f"  ! skip {pkg_name}.{info.name}: {exc}", file=sys.stderr)
            continue
        entries.extend(_entries_from_module(module, framework, module.__name__))

    entries.sort(key=lambda e: (e["module"], e["name"]))
    return entries


def _walk_custom(framework: str) -> list[dict[str, Any]]:
    """Pick up user-defined blocks from ``models/customblocks/<framework>/*.py``.

    Each ``.py`` file is loaded as a standalone module (no ``__init__.py``
    required). Both ``nn.Module`` classes with annotated ``forward`` and
    jaxtyped module-level functions are picked up, just like the built-in
    walker. Entries are tagged with ``module = "custom.<filename_stem>"`` so
    they group together in the palette.
    """
    dir_path = CUSTOM_BLOCKS_ROOT / framework
    if not dir_path.is_dir():
        return []

    entries: list[dict[str, Any]] = []
    for py_file in sorted(dir_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        stem = py_file.stem
        # Unique module name so two files with the same stem in different
        # framework subdirs don't collide in sys.modules.
        module_name = f"_custom_{framework}_{stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                print(f"  ! skip custom {py_file.name}: no loader", file=sys.stderr)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"  ! skip custom {py_file.name}: {exc}", file=sys.stderr)
            continue

        entries.extend(_entries_from_module(module, framework, f"custom.{stem}"))

    entries.sort(key=lambda e: (e["module"], e["name"]))
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = REPO_ROOT / "web" / "public" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    for framework in ("pytorch", "flax"):
        print(f"-> {framework}_blocks")
        try:
            entries = _walk(framework)
        except ModuleNotFoundError as e:
            print(f"   skip {framework}: {e}", file=sys.stderr)
            entries = []

        custom_entries = _walk_custom(framework)
        if custom_entries:
            print(f"   + {len(custom_entries)} custom from customblocks/{framework}/")
            entries.extend(custom_entries)
            entries.sort(key=lambda e: (e["module"], e["name"]))

        if not entries:
            continue

        out = out_dir / f"{framework}.json"
        out.write_text(json.dumps(entries, indent=2))
        print(f"   wrote {len(entries):3d} entries -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
