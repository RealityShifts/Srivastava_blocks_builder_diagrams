#!/usr/bin/env python3
"""Generate color-coded Mermaid flowchart diagrams for every manifest block.

Output style matches the diagrams in
``RealityShifts/Srivastava-book-of-Blocks-diagrams``: one ``flowchart TD`` per
public block with the standard ``io/op/norm/act/attn/merge/emb/loss/ctrl/ref``
classes applied.

Usage (from repo root):

    python tools/generate_mermaid_blocks.py

Reads:
  - web/public/manifests/pytorch.json
  - web/public/manifests/flax.json

Writes (one Mermaid markdown per block):
  - diagrams/pytorch/<module_tail>/<Block>.md
  - diagrams/flax/<module_tail>/<Block>.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "web" / "public" / "manifests"
DEFAULT_OUT_DIR = REPO_ROOT / "diagrams"

MERMAID_INIT = "%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%"

# Standard color legend used in the reference diagrams repo.
CLASS_DEFS = [
    "classDef io fill:#f1f5f9,stroke:#334155,stroke-width:1.4px,color:#0f172a",
    "classDef op fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.4px,color:#1e3a8a",
    "classDef norm fill:#dcfce7,stroke:#15803d,stroke-width:1.4px,color:#14532d",
    "classDef act fill:#ffedd5,stroke:#c2410c,stroke-width:1.4px,color:#7c2d12",
    "classDef attn fill:#ede9fe,stroke:#6d28d9,stroke-width:1.4px,color:#4c1d95",
    "classDef merge fill:#fef3c7,stroke:#b45309,stroke-width:1.4px,color:#78350f",
    "classDef emb fill:#fef9c3,stroke:#a16207,stroke-width:1.4px,color:#713f12",
    "classDef loss fill:#fee2e2,stroke:#b91c1c,stroke-width:1.4px,color:#7f1d1d",
    "classDef ctrl fill:#f5f5f4,stroke:#52525b,stroke-width:1.4px,color:#27272a",
    "classDef ref fill:#e0f2fe,stroke:#0369a1,stroke-width:2px,color:#0c4a6e,stroke-dasharray: 4 2",
]


# ---------------------------------------------------------------------------
# Heuristic kind classification (matches the legend in the reference repo)
# ---------------------------------------------------------------------------


def _classify_kind(name: str, module: str) -> str:
    n = name.lower()
    m = module.lower()

    # Loss / objective
    if "loss" in n or n.startswith("info_nce") or n.endswith("loss"):
        return "loss"
    # Normalisation
    if "norm" in n or n.endswith("ln") or n.endswith("bn"):
        return "norm"
    # Activation
    if n in {"mish", "relu", "gelu", "silu", "tanh", "sigmoid", "softmax"}:
        return "act"
    # Attention
    if "attention" in n or n.endswith("attn") or "rotary" in n or "rope" in n:
        return "attn"
    # Embeddings / positional / token tables
    if "embedding" in n or "embed" in n or "patchembedding" in n or "clstoken" in n:
        return "emb"
    # Merge-style combiners (rare as a whole block, mostly internal)
    if n in {"add", "sum", "concat", "merge"}:
        return "merge"
    # Schedulers / optimisers / control / non-differentiable
    if any(
        s in n
        for s in (
            "scheduler",
            "checkpoint",
            "pruner",
            "trainer",
            "ema",
            "lion",
            "sophia",
            "replaybuffer",
            "targetnetwork",
            "pipelinestage",
        )
    ):
        return "ctrl"
    if "optimization_blocks" in m or "rl_blocks" in m:
        return "ctrl"
    # Memory / retrieval
    if "memory" in n or "retriev" in n or "kvcache" in n or "vectorstore" in n:
        return "ctrl"
    # Embedding categories
    if "embedding_blocks" in m and "loss" not in n:
        return "emb"

    return "op"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_filename(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    if not out:
        out = "block"
    if out[0].isdigit():
        out = f"n_{out}"
    return out


def _quote(label: str) -> str:
    """Escape a label so it works inside a Mermaid `["..."]` node."""
    safe = label.replace('"', "'")
    return safe


def _shape_to_paren(shape: list[Any]) -> str:
    """Convert ``["B", "C", "H", "W"]`` to ``(B, C, H, W)``.

    Empty shape becomes ``scalar``.
    """
    if not shape:
        return "scalar"
    return "(" + ", ".join(str(x) for x in shape) + ")"


def _shapes_line(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> str:
    in_parts: list[str] = []
    for p in inputs:
        s = _shape_to_paren(p.get("shape", []))
        name = p.get("name", "x")
        in_parts.append(f"{name}:{s}")
    in_s = ", ".join(in_parts) if in_parts else "·"

    out_parts: list[str] = []
    for p in outputs:
        s = _shape_to_paren(p.get("shape", []))
        out_parts.append(s)
    out_s = ", ".join(out_parts) if out_parts else "·"

    return f"`{in_s} → {out_s}`"


def _node_label_for_input(p: dict[str, Any]) -> str:
    name = p.get("name", "x")
    s = _shape_to_paren(p.get("shape", []))
    return f"{name}  {s}"


def _node_label_for_output(p: dict[str, Any]) -> str:
    name = p.get("name", "out")
    s = _shape_to_paren(p.get("shape", []))
    return f"{name}  {s}"


# ---------------------------------------------------------------------------
# Diagram rendering
# ---------------------------------------------------------------------------


def _render_block_md(entry: dict[str, Any]) -> str:
    name: str = entry.get("name", "Block")
    module: str = entry.get("module", "")
    framework: str = entry.get("framework", "")
    inputs: list[dict[str, Any]] = entry.get("inputs", []) or []
    outputs: list[dict[str, Any]] = entry.get("outputs", []) or []
    block_kind = _classify_kind(name, module)

    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"> `{framework}` block from `{module}`.")
    lines.append("")
    lines.append(f"**Shapes:** {_shapes_line(inputs, outputs)}")
    lines.append("")
    lines.append("```mermaid")
    lines.append(MERMAID_INIT)
    lines.append("flowchart TD")

    # Row 0: inputs (each ":::io")
    if not inputs:
        # Synthesize a generic input placeholder for blocks that take no tensors
        lines.append(f'    n0_0["(no input)"]:::io')
        input_ids = ["n0_0"]
    else:
        input_ids = []
        for i, p in enumerate(inputs):
            nid = f"n0_{i}"
            lines.append(f'    {nid}["{_quote(_node_label_for_input(p))}"]:::io')
            input_ids.append(nid)

    # Row 1: the block itself, kind picked by heuristic
    block_id = "n1_0"
    lines.append(f'    {block_id}["{_quote(name)}"]:::{block_kind}')

    # Row 2: outputs (each ":::io")
    if not outputs:
        lines.append(f'    n2_0["(no output)"]:::io')
        output_ids = ["n2_0"]
    else:
        output_ids = []
        for i, p in enumerate(outputs):
            nid = f"n2_{i}"
            lines.append(f'    {nid}["{_quote(_node_label_for_output(p))}"]:::io')
            output_ids.append(nid)

    # Wiring: every input -> block -> every output
    for nid in input_ids:
        lines.append(f"    {nid} --> {block_id}")
    for nid in output_ids:
        lines.append(f"    {block_id} --> {nid}")

    # Class defs (standard legend)
    for cd in CLASS_DEFS:
        lines.append(f"    {cd}")

    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write_framework(
    framework: str, manifest_path: Path, out_root: Path
) -> tuple[int, int]:
    if not manifest_path.exists():
        return 0, 0
    entries: list[dict[str, Any]] = json.loads(manifest_path.read_text())
    written = 0
    for entry in entries:
        module = str(entry.get("module", "unknown"))
        mod_tail = module.split(".")[-1] if module else "unknown"
        name = str(entry.get("name", "Block"))
        out_dir = out_root / framework / mod_tail
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{_sanitize_filename(name)}.md"
        out_file.write_text(_render_block_md(entry))
        written += 1
    return written, len(entries)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory containing pytorch.json / flax.json manifests.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the generated markdown diagrams.",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    p_count, p_total = _write_framework(
        "pytorch", args.manifest_dir / "pytorch.json", args.out
    )
    f_count, f_total = _write_framework(
        "flax", args.manifest_dir / "flax.json", args.out
    )

    print(f"pytorch: wrote {p_count}/{p_total} diagrams")
    print(f"flax:    wrote {f_count}/{f_total} diagrams")
    print(f"output:  {args.out}")


if __name__ == "__main__":
    main()
