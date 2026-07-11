@echo off
rem Batch inference: predict.bat <folder> [--out report.csv] [--overlays] (spec §6.5).
cd /d "%~dp0.."
uv run coil-predict %*
