@echo off
rem One double-click: starts the demo stage minimized and opens the browser.
rem If the stage is already running, it just opens the browser tab.
start "Whetstone Stage" /min cmd /c "cd /d "%~dp0" && python src\bcv\demo_stage.py 8990 --open"
