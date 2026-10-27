@echo off
rem Launch AI Punk Studio with the project .venv (created by "uv sync"). Arguments are passed through, e.g. launch.bat --headless
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Run "uv sync" first, see README.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m aipunk_studio %*
