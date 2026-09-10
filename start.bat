@echo off
title Pediatric & Adult CCTV Counter
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run.py
pause
