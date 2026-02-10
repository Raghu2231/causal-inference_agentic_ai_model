Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw "Virtual environment activation script not found at .venv\Scripts\Activate.ps1. Recreate venv with: python -m venv .venv"
}

. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
