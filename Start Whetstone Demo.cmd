@echo off
rem Public evidence demo uses port 8990. The entrypoint resolves its own repo.
start "Whetstone PUBLIC Stage" /min python "%~dp0src\bcv\demo_stage.py" 8990 --open
