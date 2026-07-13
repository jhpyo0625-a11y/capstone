@echo off
rem Hourly watch-folder check; fires the retrain pipeline at threshold (spec §6.6).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
uv run python -m coilvision.pipeline.watcher >> artifacts\watcher.log 2>&1
