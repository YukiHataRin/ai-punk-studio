@echo off
rem Launch Astra Studio with the project .conda environment. Arguments are passed through, e.g. launch.bat --headless
cd /d "%~dp0"
set PYTHONNOUSERSITE=1
".conda\python.exe" -m astra_studio %*
