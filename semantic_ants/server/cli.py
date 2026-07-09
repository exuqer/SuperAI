from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .service import ServerConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semantic-ants-web")
    parser.add_argument("--state-dir", default=".semantic_ants")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--static-dir", default="")
    args = parser.parse_args(argv)
    config = ServerConfig(
        state_dir=Path(args.state_dir),
        host=args.host,
        port=args.port,
        static_dir=Path(args.static_dir) if args.static_dir else None,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
