Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Starting AI Magic Invisibility Portal...         " -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".venv")) {
    Write-Host "[ERROR] Virtual environment (.venv) not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    Exit
}

Write-Host "Using virtual environment python..." -ForegroundColor Green
& .venv\Scripts\python.exe main.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Application exited with error code $LASTEXITCODE." -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
