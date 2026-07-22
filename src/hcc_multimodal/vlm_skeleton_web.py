"""Local static server for the patch-based mock VLM browser demo."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def launch_demo(*, port: int = 7860, share: bool = False) -> None:
    """Serve the standalone browser demo without Gradio startup callbacks."""
    del share
    root = Path.cwd() / "demo-output" / "vlm-skeleton-static"
    index = root / "index.html"
    if not index.exists():
        raise RuntimeError(f"Static demo is missing: {index}")
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Running static VLM demo at http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["launch_demo"]
