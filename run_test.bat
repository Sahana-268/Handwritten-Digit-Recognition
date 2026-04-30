@echo off
set "PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%PY%" (
    "%PY%" test_project.py
) else (
    python test_project.py
)

