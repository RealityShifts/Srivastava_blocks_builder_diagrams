#!/usr/bin/env python3
"""Local HTTP server: execute traced GeneratedModel and return per-port shapes.

Start from repo root::

    python tools/shape_runner.py

Then POST JSON to http://127.0.0.1:8765/run from the Blocks Builder UI.

Expected payload::

    {
      "framework": "pytorch",
      "code": "... GeneratedModel with trace=true ...",
      "inputs": [
        {"arg": "x", "shape": [2, 3, 224, 224], "dtype": "float"}
      ]
    }

Response::

    {"ok": true, "shapes": {"nodeId/out": [2, 3, 224, 224], ...}, "num_params": 12345}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKS_ROOT = REPO_ROOT / "models" / "blocks"
CUSTOM_ROOT = REPO_ROOT / "models" / "customblocks"
sys.path.insert(0, str(BLOCKS_ROOT))


def _register_custom_namespace() -> None:
    """Make ``from custom.<stem> import ...`` importable.

    build_manifest tags user blocks under ``models/customblocks/<framework>/``
    with ``module = "custom.<stem>"``, so generated models import them via the
    ``custom`` package. That package doesn't exist on disk, so we synthesize it
    here and point its search path at the framework's customblocks directory
    (pytorch is the only framework the runner executes). Built-in blocks keep
    resolving through ``BLOCKS_ROOT`` on ``sys.path``.
    """
    import types

    pkg = sys.modules.get("custom")
    if pkg is None:
        pkg = types.ModuleType("custom")
        pkg.__path__ = []  # marks it as a (namespace) package
        sys.modules["custom"] = pkg
    fw_dir = str(CUSTOM_ROOT / "pytorch")
    if fw_dir not in pkg.__path__:
        pkg.__path__.append(fw_dir)


_register_custom_namespace()

HOST = "127.0.0.1"
PORT = int(os.environ.get("SHAPE_RUNNER_PORT", "8765"))


class NodeExecutionError(RuntimeError):
    def __init__(self, node_id: str, message: str):
        super().__init__(message)
        self.node_id = node_id


def _make_tensor(shape: tuple[int, ...], dtype: str):
    import torch

    if dtype == "int":
        return torch.randint(0, 10, shape)
    if dtype == "bool":
        return torch.ones(shape, dtype=torch.bool)
    return torch.randn(*shape)


def _instantiate_model(payload: dict[str, Any]):
    """Exec the posted code and instantiate ``GeneratedModel`` in eval mode.

    Shared by the shape runner and the (optional) torchvista view so both build
    the model the same way.
    """
    if payload.get("framework") != "pytorch":
        raise ValueError("only framework=pytorch is supported")

    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("missing code")

    ns: dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, ns)  # noqa: S102 - local dev tool only

    GeneratedModel = ns.get("GeneratedModel")
    if GeneratedModel is None:
        raise ValueError("code did not define GeneratedModel")

    model = GeneratedModel()
    model.eval()
    return model


def _input_specs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("missing inputs")
    return inputs


def _spec_tensor(spec: dict[str, Any]):
    """Build the example tensor for one input spec, validating its shape."""
    shape = spec.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"invalid input spec: {spec!r}")
    return _make_tensor(tuple(int(d) for d in shape), spec.get("dtype", "float"))


def _reraise_node_error(exc: Exception) -> None:
    """Translate a traced ``NODE_ERROR::`` marker into a NodeExecutionError."""
    msg = str(exc)
    if msg.startswith("NODE_ERROR::"):
        rest = msg[len("NODE_ERROR::") :]
        node_id, _, detail = rest.partition("::")
        raise NodeExecutionError(node_id=node_id or "unknown", message=detail or msg) from exc
    raise exc


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    model = _instantiate_model(payload)
    inputs = _input_specs(payload)

    kwargs: dict[str, Any] = {}
    for spec in inputs:
        arg = spec.get("arg")
        if not arg:
            raise ValueError(f"invalid input spec: {spec!r}")
        kwargs[str(arg)] = _spec_tensor(spec)

    try:
        with torch.no_grad():
            out = model(**kwargs)
    except Exception as exc:
        _reraise_node_error(exc)

    if not isinstance(out, dict):
        raise ValueError(f"traced forward must return dict, got {type(out).__name__}")

    shapes = {str(k): [int(d) for d in v] for k, v in out.items()}
    num_params = sum(p.numel() for p in model.parameters())  #cooment this if want to stop parameter calculation and save some compute
    return {"ok": True, "shapes": shapes, "num_params": int(num_params)}


def run_vista(payload: dict[str, Any]) -> dict[str, Any]:
    """Render the model's forward pass with torchvista and return the HTML.

    torchvista is an *optional* dependency: it is imported here, not at module
    load, so the rest of the runner (``/run`` shape checking) keeps working when
    it isn't installed. The caller (UI) inspects ``vista_available`` to show an
    install hint instead of a hard error.

    The posted ``code`` must be the **non-traced** GeneratedModel (a real
    forward returning tensors) - torchvista runs its own instrumented forward.
    Inputs are passed positionally in forward()-argument order, matching how the
    web side orders ``payload["inputs"]``.
    """
    try:
        import torchvista
    except ImportError as exc:
        return {
            "ok": False,
            "vista_available": False,
            "error": (
                "torchvista is not installed. Install it to enable the graph view:\n"
                "    pip install torchvista\n"
                f"(import error: {exc})"
            ),
        }

    import contextlib
    import io
    import tempfile

    model = _instantiate_model(payload)
    inputs = _input_specs(payload)
    # Positional tuple in forward() order; torchvista calls model(*inputs).
    example_inputs = tuple(_spec_tensor(spec) for spec in inputs)

    height = int(payload.get("height", 700) or 700)

    with tempfile.TemporaryDirectory() as tmp:
        out_html = Path(tmp) / "torchvista_graph.html"
        buf = io.StringIO()
        trace_exc: Exception | None = None
        # torchvista prints "Saved as ..." and emits an IPython display; swallow
        # both so they don't pollute the runner's stdout.
        with contextlib.redirect_stdout(buf):
            try:
                torchvista.trace_model(
                    model,
                    example_inputs,
                    export_format="html",
                    export_path=str(out_html),
                    height=height,
                )
            except Exception as exc:  # noqa: BLE001 - surface as a partial render
                trace_exc = exc

        # trace_model writes the HTML even when the forward pass fails partway
        # (plot_graph runs before it re-raises), so a partial graph is still
        # useful for debugging. Only hard-fail when nothing was produced.
        if not out_html.exists():
            if trace_exc is not None:
                _reraise_node_error(trace_exc)  # always raises
            raise RuntimeError("torchvista produced no output")
        html = out_html.read_text(encoding="utf-8")

    num_params = sum(p.numel() for p in model.parameters())
    result: dict[str, Any] = {
        "ok": True,
        "vista_available": True,
        "html": html,
        "num_params": int(num_params),
    }
    if trace_exc is not None:
        result["partial"] = True
        result["warning"] = str(trace_exc)
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[shape_runner] {self.address_string()} - {fmt % args}\n")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _send_json(self, status: int, result: dict[str, Any]) -> None:
        body = json.dumps(result).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        handlers = {"/run": run_payload, "/vista": run_vista}
        handler = handlers.get(self.path)
        if handler is None:
            self.send_error(404, "use POST /run or /vista")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            result = handler(payload)
            # A missing optional dependency (torchvista) is a normal,
            # recoverable state - return 200 so the UI can show an install
            # hint rather than treating it as a server error.
            status = 200 if result.get("ok") or result.get("vista_available") is False else 400
            self._send_json(status, result)
            return
        except NodeExecutionError as exc:
            self._send_json(400, {
                "ok": False,
                "error": str(exc),
                "node_id": exc.node_id,
                "trace": traceback.format_exc(),
            })
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc), "trace": traceback.format_exc()})


def main() -> None:
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        if exc.errno == 48:  # Address already in use
            pid_hint = _pid_on_port(PORT)
            print(
                f"Port {PORT} is already in use"
                + (f" (PID {pid_hint})" if pid_hint else "")
                + ".",
                file=sys.stderr,
            )
            print(
                "A shape runner is probably already running — use the UI as-is, or stop it:",
                file=sys.stderr,
            )
            if pid_hint:
                print(f"  kill {pid_hint}", file=sys.stderr)
            print(
                f"Or start on another port: SHAPE_RUNNER_PORT=8766 python tools/shape_runner.py",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    print(f"shape runner listening on http://{HOST}:{PORT}/run")
    print(f"PYTHONPATH includes {BLOCKS_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()


def _pid_on_port(port: int) -> str | None:
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out.splitlines()[0] if out else None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


if __name__ == "__main__":
    main()
