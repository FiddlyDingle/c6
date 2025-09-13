@echo off
echo Starting Cerb AI Assistant - Phase 1
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the test suite first
echo Running system tests...
python test_system.py
if %errorlevel% neq 0 (
    echo.
    echo Tests failed. Please fix errors before running.
    pause
    exit /b 1
)

echo.
echo Tests passed! Starting Cerb AI Assistant...
echo.
echo Instructions:
echo - Type 'help' for available commands
echo - Type 'status' to check system status
echo - Type 'quit' to exit
echo.
echo Make sure LM Studio is running on localhost:1234
echo.

REM Start the main application
python main.py

echo.
echo Cerb AI Assistant has stopped.
pause
