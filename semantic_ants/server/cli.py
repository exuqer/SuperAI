from __future__ import annotations

import argparse
from pathlib import Path

from semantic_ants.server.service import ServerConfig


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import uvicorn

        from semantic_ants.server.app import create_app
    except RuntimeError as exc:
        print(exc)
        return 1
    except ModuleNotFoundError as exc:
        print(f'Install web dependencies with: pip install -e ".[web]" ({exc.name} is missing)')
        return 1

    config = ServerConfig(
        state_dir=Path(args.state_dir),
        host=args.host,
        port=args.port,
        allow_network=not args.no_cache_refresh,
        autoload_builtin=not args.no_builtin,
        static_dir=Path(args.static_dir) if args.static_dir else None,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-ants-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", default=".semantic_ants")
    parser.add_argument("--no-cache-refresh", action="store_true")
    parser.add_argument("--no-builtin", action="store_true")
    parser.add_argument("--static-dir")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
