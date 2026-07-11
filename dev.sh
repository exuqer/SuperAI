#!/usr/bin/env bash
# ============================================================================
# dev.sh — запуск SuperAI (бекенд + фронтенд) одновременно
# Поддерживает: macOS, Linux, Windows (Git Bash / WSL)
# ============================================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_DIR="$ROOT_DIR/web"

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
if command -v uvicorn &>/dev/null; then
  uvicorn superai.api:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload &
elif command -v superai &>/dev/null; then
  superai --host 127.0.0.1 --port "$BACKEND_PORT" --reload &
else
  python -m uvicorn superai.api:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload &
fi
BACKEND_PID=$!

# ---- Фронтенд (Vite / Vue) ----
echo "🚀 Запускаю фронтенд..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "═══════════════════════════════════════════════"
echo "  Бекенд:  http://127.0.0.1:$BACKEND_PORT"
echo "  Фронтенд: http://localhost:5173"
echo "═══════════════════════════════════════════════"
echo "  Нажми Ctrl+C чтобы остановить всё."
echo ""

# Ждём любой из процессов
wait