"""Small local runner; production deployment owns process supervision."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SuperAI local API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("superai.api:app", host=args.host, port=args.port, reload=args.reload)
