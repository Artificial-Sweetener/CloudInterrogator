@echo off
REM Change directory to where this .bat is located
cd /d %~dp0

REM Optional: create venv if not already created
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Open a new command prompt with venv activated
start cmd /k ".venv\Scripts\activate"
