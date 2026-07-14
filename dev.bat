@echo off
REM ============================================================================
REM dev.bat — запуск SuperAI (бекенд + фронтенд) одновременно
REM Для Windows (cmd)
REM ============================================================================

setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"
set "BACKEND_PORT=%BACKEND_PORT%"
if "%BACKEND_PORT%"=="" set "BACKEND_PORT=8000"
set "FRONTEND_DIR=%ROOT_DIR%web"

echo 🚀 Запускаю бекенд на порту %BACKEND_PORT%...
cd /d "%ROOT_DIR%"

start "superai-backend" cmd /c "python -m uvicorn server.server:app --host 127.0.0.1 --port %BACKEND_PORT%"

echo 🚀 Запускаю фронтенд...
cd /d "%FRONTEND_DIR%"
start "superai-frontend" cmd /c "npm run dev:frontend"

echo.
echo ═══════════════════════════════════════════════
echo   Бекенд:  http://127.0.0.1:%BACKEND_PORT%
echo   Фронтенд: http://localhost:3010
echo ═══════════════════════════════════════════════
echo   Закрой окна cmd чтобы остановить.
echo.

pause
