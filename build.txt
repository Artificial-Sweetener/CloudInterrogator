@echo off

echo Activating virtual environment...
call .\.venv\Scripts\activate

echo Starting Nuitka compilation...

rem The command is split across multiple lines for readability using the ^ character.
nuitka --onefile ^
 --lto=yes ^
 --windows-console-mode=disable ^
 --windows-icon-from-ico=icon.ico ^
 --enable-plugin=pyside6 ^
 --include-module=winaccent ^
 --output-dir=build ^
 --output-filename=CloudInterrogator.exe ^
 --assume-yes-for-downloads ^
 main.py

echo.
echo Build complete. The executable can be found in the 'build' directory.
pause
