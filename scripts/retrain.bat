@echo off
rem Runs the full retraining pipeline manually end-to-end (spec §6.6).
cd /d "%~dp0.."
uv run python -m coilvision.pipeline.retrain %*
