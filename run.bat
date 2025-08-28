@echo off
:: Change directory to where this .bat is located
cd /d %~dp0

:: Check if the virtual environment directory exists
if not exist .\venv (
    echo ---
    echo Hiii!~ It looks like this is your first time running the app.
    echo I'll set up a virtual environment for you. This might take a moment...
    echo ---
    
    :: Create the virtual environment
    python -m venv .venv
    
    echo ---
    echo Installing required packages...
    echo ---
    
    :: Activate venv and install packages from requirements.txt
    call .\venv\Scripts\activate.bat
    pip install -r requirements.txt
    
    echo ---
    echo Installing winaccent for that native Windows feel...
    echo ---
    pip install winaccent
    
    echo.
    echo ---
    echo All done! Starting the app now...
    echo ---
    echo.
)

:: Run the main application using the venv python
.\venv\Scripts\python.exe main.py
