<#
  One-time setup on Windows 11 (PowerShell).
  Run from the repo folder:   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

  - Creates .venv with the chosen Python (default 3.14; use -Python 3.13 if
    PyTorch has no CUDA wheel for 3.14 yet).
  - Installs PyTorch with CUDA (RTX 3050), then requirements.txt.
  - Copies settings.example.yaml to settings.yaml if it does not exist.
  - Creates videos_root / data_root folders. Never touches existing videos.
#>
param(
  [string]$Python = "3.14",
  [string]$CudaIndex = "https://download.pytorch.org/whl/cu128"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  Write-Error "Python launcher 'py' not found. Install Python $Python from python.org (tick 'Add to PATH'), then rerun."
}
py -$Python --version
if ($LASTEXITCODE -ne 0) { Write-Error "Python $Python is not installed. Install it or pass -Python <version>." }

if (-not (Test-Path .venv)) { py -$Python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url $CudaIndex
if ($LASTEXITCODE -ne 0) {
  Write-Error "PyTorch CUDA install failed for Python $Python. Delete .venv and rerun with -Python 3.13."
}
& $py -m pip install -r requirements.txt

if (-not (Test-Path settings.yaml)) {
  Copy-Item settings.example.yaml settings.yaml
  Write-Host "Created settings.yaml - check the paths in it."
}

foreach ($dir in @("D:\FullStack\BGMI\videos", "D:\FullStack\BGMI\data")) {
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null; Write-Host "Created $dir" }
}

Write-Host "`nSetup done. Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python cli.py doctor"
Write-Host "  python cli.py catalog"
