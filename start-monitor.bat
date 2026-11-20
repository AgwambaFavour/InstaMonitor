@echo off
title Instagram Monitor
cd /d "%~dp0"
python monitor.py
echo.
echo Monitor stopped. Press any key to close.
pause >nul
