$ErrorActionPreference = "Stop"

$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $Python) {
    & $Python app.py
}
else {
    python app.py
}

