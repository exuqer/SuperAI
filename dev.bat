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

where uvicorn >nul 2>nul
if %ERRORLEVEL% equ 0 (
  start "superai-backend" cmd /c "uvicorn superai.api:app --host 127.0.0.1 --port %BACKEND_PORT% --reload"
) else (
  start "superai-backend" cmd /c "python -m uvicorn superai.api:app --host 127.0.0.1 --port %BACKEND_PORT% --reload"
)

echo 🚀 Запускаю фронтенд...
cd /d "%FRONTEND_DIR%"
start "superai-frontend" cmd /c "npm run dev"

echo.
echo ═══════════════════════════════════════════════
echo   Бекенд:  http://127.0.0.1:%BACKEND_PORT%
echo   Фронтенд: http://localhost:5173
echo ═══════════════════════════════════════════════
echo   Закрой окна cmd чтобы остановить.
echo.

pause