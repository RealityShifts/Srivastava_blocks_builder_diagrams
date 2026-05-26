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

    {"ok": true, "shapes": {"nodeId/out": [2, 3, 224, 224], ...}}
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
sys.path.insert(0, str(BLOCKS_ROOT))

HOST = "127.0.0.1"
PORT = int(os.environ.get("SHAPE_RUNNER_PORT", "8765"))


def _make_tensor(shape: tuple[int, ...], dtype: str):
    import torch

    if dtype == "int":
        return torch.randint(0, 10, shape)
    if dtype == "bool":
        return torch.ones(shape, dtype=torch.bool)
    return torch.randn(*shape)


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("framework") != "pytorch":
        raise ValueError("only framework=pytorch is supported")

    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("missing code")

    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("missing inputs")

    ns: dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, ns)  # noqa: S102 - local dev tool only

    GeneratedModel = ns.get("GeneratedModel")
    if GeneratedModel is None:
        raise ValueError("code did not define GeneratedModel")

    import torch

    model = GeneratedModel()
    model.eval()

    kwargs: dict[str, Any] = {}
    for spec in inputs:
        arg = spec.get("arg")
        shape = spec.get("shape")
        if not arg or not isinstance(shape, list) or not shape:
            raise ValueError(f"invalid input spec: {spec!r}")
        kwargs[str(arg)] = _make_tensor(tuple(int(d) for d in shape), spec.get("dtype", "float"))

    with torch.no_grad():
        out = model(**kwargs)

    if not isinstance(out, dict):
        raise ValueError(f"traced forward must return dict, got {type(out).__name__}")

    shapes = {str(k): [int(d) for d in v] for k, v in out.items()}
    return {"ok": True, "shapes": shapes}


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

    def do_POST(self) -> None:
        if self.path != "/run":
            self.send_error(404, "use POST /run")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            result = run_payload(payload)
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            err = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}
            body = json.dumps(err).encode("utf-8")
            self.send_response(400)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


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
