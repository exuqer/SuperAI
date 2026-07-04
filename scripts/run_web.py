#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    npm = _npm_command()
    if npm is None:
        print("npm не найден. Установите Node.js LTS: https://nodejs.org/", flush=True)
        return 1
    if not WEB_DIR.exists():
        print(f"Не найдена папка клиента: {WEB_DIR}", flush=True)
        return 1

    if not args.no_install and not (WEB_DIR / "node_modules").exists():
        print("web/node_modules не найден, запускаю npm install...", flush=True)
        install = subprocess.run([npm, "install"], cwd=WEB_DIR)
        if install.returncode != 0:
            return install.returncode

    backend = _start_backend(args)
    frontend = _start_frontend(args, npm)
    print(flush=True)
    print(f"API: http://{args.host}:{args.api_port}", flush=True)
    print(f"UI:  http://{args.ui_host}:{args.ui_port}", flush=True)
    print("Нажмите Ctrl+C, чтобы остановить клиент и сервер.", flush=True)

    processes = [backend, frontend]
    try:
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    _stop_all(processes)
                    return int(code)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nОстанавливаю клиент и сервер...", flush=True)
        _stop_all(processes)
        return 130


def _start_backend(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "semantic_ants.server.cli",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
        "--state-dir",
        args.state_dir,
    ]
    if args.no_cache_refresh:
        command.append("--no-cache-refresh")
    if args.no_builtin:
        command.append("--no-builtin")
    print("Запускаю backend:", " ".join(command), flush=True)
    return _popen(command, cwd=ROOT)


def _start_frontend(args: argparse.Namespace, npm: str) -> subprocess.Popen[bytes]:
    command = [
        npm,
        "run",
        "dev",
        "--",
        "--host",
        args.ui_host,
        "--port",
        str(args.ui_port),
    ]
    env = os.environ.copy()
    env["VITE_API_TARGET"] = f"http://{args.host}:{args.api_port}"
    print("Запускаю frontend:", " ".join(command), flush=True)
    return _popen(command, cwd=WEB_DIR, env=env)


def _popen(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    if os.name == "nt":
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)


def _stop_all(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        _terminate(process)
    deadline = time.time() + 6
    for process in processes:
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            process.kill()


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    except Exception:
        process.terminate()


def _npm_command() -> str | None:
    if os.name == "nt":
        return shutil.which("npm.cmd") or shutil.which("npm")
    return shutil.which("npm")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_web",
        description="Запускает semantic_ants backend и Vue dev server одной командой.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="host backend API")
    parser.add_argument("--api-port", type=int, default=8765, help="port backend API")
    parser.add_argument("--ui-host", default="127.0.0.1", help="host Vite dev server")
    parser.add_argument("--ui-port", type=int, default=5173, help="port Vite dev server")
    parser.add_argument("--state-dir", default=".semantic_ants", help="semantic_ants state directory")
    parser.add_argument("--no-cache-refresh", action="store_true", help="не обращаться к внешнему ConceptNet")
    parser.add_argument("--no-builtin", action="store_true", help="не загружать встроенную базу")
    parser.add_argument("--no-install", action="store_true", help="не запускать npm install автоматически")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
