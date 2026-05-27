#!/usr/bin/env python3
"""Fetch reference block diagrams from RealityShifts/Srivastava-book-of-Blocks-diagrams.

For every ``diagrams/<category>/<BlockName>.md`` in the reference repo, extract
the description blockquote, the ``**Shapes:**`` line, and the embedded
``mermaid`` flowchart. Write them all to ``web/public/block_info.json`` so the
Blocks Builder UI can show a per-block Info tab with the canonical diagram
without any runtime network calls.

Usage::

    python tools/fetch_block_diagrams.py

The recursive git-tree API gives us all 122+ filenames in one request; each
file is then pulled from raw.githubusercontent.com in a small thread pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "web" / "public" / "block_info.json"

UPSTREAM_REPO = "RealityShifts/Srivastava-book-of-Blocks-diagrams"
UPSTREAM_BRANCH = "main"
TREE_URL = f"https://api.github.com/repos/{UPSTREAM_REPO}/git/trees/{UPSTREAM_BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_BRANCH}"
HTML_BASE = f"https://github.com/{UPSTREAM_REPO}/blob/{UPSTREAM_BRANCH}"

USER_AGENT = "blocks-builder/0.1 (+https://github.com)"


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_diagram_paths() -> list[str]:
    """Return every ``diagrams/<category>/<Block>.md`` path in the upstream tree."""
    tree = json.loads(_http_get(TREE_URL).decode("utf-8"))
    if tree.get("truncated"):
        # Should not happen at this repo size, but bail loud if it ever does.
        raise RuntimeError("upstream git tree was truncated; switch to per-dir listing")
    return [
        item["path"]
        for item in tree["tree"]
        if item.get("type") == "blob"
        and item["path"].startswith("diagrams/")
        and item["path"].endswith(".md")
        and item["path"].count("/") == 2  # exclude diagrams/INDEX.md (no /)
        and not item["path"].endswith("/INDEX.md")
    ]


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_QUOTE_RE = re.compile(r"^>\s?(.*)$", re.MULTILINE)
_SHAPES_RE = re.compile(r"^\*\*Shapes:\*\*\s*(.+?)\s*$", re.MULTILINE)
_MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)
# Section heading: a bold line on its own, capitalised, no nested bold.
# Lookbehind ``(?<=^)`` is implicit via MULTILINE + start-of-line anchor.
_SECTION_HEADING_RE = re.compile(r"^\*\*([A-Z][^*\n]+?)\*\*\s*$")

# Section headings we already capture above; never re-emit as a section.
_RESERVED_HEADINGS = {"shapes"}


def _strip_backticks(text: str) -> str:
    t = text.strip()
    if t.startswith("`") and t.endswith("`"):
        return t.strip("`").strip()
    return t


def _parse_sections(post_mermaid: str) -> list[dict[str, Any]]:
    """Extract ``**Heading**`` / bulleted-body sections appearing after the mermaid.

    Used in / Tasks / Common pitfalls / See also are the four the upstream
    currently ships. Markdown links ``[label](url)`` are preserved verbatim
    inside each item; the UI converts them to anchor tags.
    """
    sections: list[dict[str, Any]] = []
    cur_heading: str | None = None
    cur_items: list[str] = []

    def flush() -> None:
        if cur_heading and cur_items:
            sections.append({"heading": cur_heading, "items": list(cur_items)})

    for raw in post_mermaid.splitlines():
        line = raw.rstrip()
        head = _SECTION_HEADING_RE.match(line)
        if head:
            flush()
            cur_heading = head.group(1).strip()
            cur_items = []
            if cur_heading.lower() in _RESERVED_HEADINGS:
                cur_heading = None
            continue
        if not line:
            continue
        if line.startswith("- "):
            cur_items.append(line[2:].strip())
        elif cur_heading and cur_items and not line.startswith("**"):
            # Continuation of the previous bullet (rare but supported).
            cur_items[-1] += " " + line.strip()
    flush()
    return sections


def parse_block_md(text: str) -> dict[str, Any]:
    """Pull title/description/shapes/mermaid/sections out of a single block markdown."""
    title_match = _TITLE_RE.search(text)
    title = title_match.group(1).strip() if title_match else ""

    # Description: collect the first contiguous run of `> ...` lines.
    desc_lines: list[str] = []
    for m in _QUOTE_RE.finditer(text):
        line = m.group(1).rstrip()
        if not line and desc_lines:
            break
        if line:
            desc_lines.append(line)
    description = " ".join(desc_lines).strip()

    shapes_match = _SHAPES_RE.search(text)
    shapes = _strip_backticks(shapes_match.group(1)) if shapes_match else ""

    mermaid_match = _MERMAID_RE.search(text)
    mermaid = mermaid_match.group(1).strip() if mermaid_match else ""

    post_mermaid = text[mermaid_match.end():] if mermaid_match else ""
    sections = _parse_sections(post_mermaid)

    return {
        "title": title,
        "description": description,
        "shapes": shapes,
        "mermaid": mermaid,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fetch_one(path: str) -> tuple[str, dict[str, Any]]:
    """Return ``(block_name, info_dict)`` for one upstream markdown path."""
    raw = _http_get(f"{RAW_BASE}/{path}").decode("utf-8")
    parsed = parse_block_md(raw)
    parts = path.split("/")  # ["diagrams", "<category>", "<Block>.md"]
    category = parts[1]
    name = parts[2][:-3]  # strip .md
    return name, {
        **parsed,
        "category": category,
        "source": f"{HTML_BASE}/{path}",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON (consumed by the UI as /block_info.json).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers.",
    )
    args = p.parse_args()

    try:
        paths = list_diagram_paths()
    except Exception as exc:
        print(f"failed to list upstream tree: {exc!r}", file=sys.stderr)
        sys.exit(1)

    print(f"upstream: {len(paths)} markdown files", file=sys.stderr)

    out: dict[str, dict[str, Any]] = {}
    errors: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, p): p for p in paths}
        for fut in concurrent.futures.as_completed(futures):
            path = futures[fut]
            try:
                name, info = fut.result()
                # On collision (e.g. `myarch/` example shadows a built-in),
                # keep whichever has a richer mermaid body — usually the one
                # from a "real" category beats the demo.
                prior = out.get(name)
                if prior is None or len(info["mermaid"]) > len(prior["mermaid"]):
                    out[name] = info
            except Exception as exc:
                errors.append((path, repr(exc)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"wrote {len(out)} entries -> {args.out}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} failed:", file=sys.stderr)
        for path, err in errors[:5]:
            print(f"  {path}: {err}", file=sys.stderr)
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more", file=sys.stderr)


if __name__ == "__main__":
    main()
