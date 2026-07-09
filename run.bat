@echo off
title AI Magic Invisibility Portal Launcher
echo ===================================================
echo   Starting AI Magic Invisibility Portal...
echo ===================================================
echo.

:: Check if virtual environment exists
if not exist ".venv" (
    echo [ERROR] Virtual environment (.venv) not found.
    echo Please make sure you are in the correct directory.
    pause
    exit /b
)

:: Run using the virtual environment python executable
echo Using virtual environment python...
.venv\Scripts\python.exe main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
