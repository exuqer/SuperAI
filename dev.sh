#!/usr/bin/env bash
# ============================================================================
# dev.sh — запуск SuperAI (бекенд + фронтенд) одновременно
# Поддерживает: macOS, Linux, Windows (Git Bash / WSL)
# ============================================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_DIR="$ROOT_DIR/web"
export SUPERAI_ALLOW_TEST_RESET=true

cleanup() {
  echo ""
  echo "🛑 Останавливаю процессы..."
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
  wait 2>/dev/null
  echo "✅ Всё остановлено."
}

trap cleanup EXIT INT TERM

# ---- Бекенд (Python / uvicorn) ----
echo "🚀 Запускаю бекенд на порту $BACKEND_PORT..."
cd "$ROOT_DIR"
if command -v python3 &>/dev/null; then
  PYTHON_BIN=python3
elif command -v python &>/dev/null; then
  PYTHON_BIN=python
else
  echo "Python не найден. Установите Python 3 и зависимости проекта."
  exit 1
fi
# Один процесс backend надёжнее для локального SQLite, чем reload-parent.
"$PYTHON_BIN" -m uvicorn server.server:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

# ---- Фронтенд (Vite / Vue) ----
echo "🚀 Запускаю фронтенд..."
cd "$FRONTEND_DIR"
npm run dev:frontend &
FRONTEND_PID=$!

echo ""
echo "═══════════════════════════════════════════════"
echo "  Бекенд:  http://127.0.0.1:$BACKEND_PORT"
echo "  Фронтенд: http://localhost:3010"
echo "═══════════════════════════════════════════════"
echo "  Нажми Ctrl+C чтобы остановить всё."
echo ""

# Ждём любой из процессов
wait
