@echo off
rem Runs the full retraining pipeline manually end-to-end (spec §6.6).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
uv run python -m coilvision.pipeline.retrain %*
