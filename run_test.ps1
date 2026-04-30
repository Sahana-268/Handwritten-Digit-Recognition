$ErrorActionPreference = "Stop"

$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $Python) {
    & $Python test_project.py
}
else {
    python test_project.py
}

