# CLAUDE.md

Coil defect classifier: local, CPU-only CNN (EfficientNet-B0 via timm) that classifies wound-coil flex-PCB images as **Pass / Dent / Loose**, with Grad-CAM explanations and an automated gated retraining pipeline. Windows 11, no cloud — data never leaves this machine.

**Read `COIL_CLASSIFIER_SPEC.md` before any nontrivial work.** It is the source of truth for decisions, phases, and open questions.

## Commands

Python 3.14 is system default but PyTorch needs **Python 3.12** — always use the pinned venv (`uv venv --python 3.12` / `py -3.12`).

```bash
uv run pytest                    # run tests (must stay green)
uv run pytest tests/test_split.py   # leakage checks specifically
scripts/retrain.bat              # full pipeline: ingest→train→eval→gate→promote
coil-predict <folder> --overlays # batch inference → CSV + Grad-CAM overlays
```

## Core files

- `COIL_CLASSIFIER_SPEC.md` — spec, decisions log, task breakdown
- `configs/config.yaml` — every knob (paths, model, thresholds, promotion gate); never hardcode these
- `coilvision/data/manifest.py` — filename parser; manifest is the single source of truth downstream
- `coilvision/data/split.py` — grouped splits; `artifacts/manifests/test_v1.csv` is the frozen test set
- `coilvision/pipeline/retrain.py` — retraining orchestrator
- `models/production/POINTER.json` — which model is live, its threshold + preprocess version

## Hard rules (data integrity)

- `Coil-image-Dataset/` is **read-only raw data** — never modify, move, or write into it.
- **Never infer part identity or labels from filenames.** Confirmed 2026-07-11: `part#`+`shot#` does NOT identify a physical part — files like `…_6-1` and `…_6-3` in the same run can be completely different parts. Filename fields (run, part#, shot#, code) are provenance metadata only; the label folder is ground truth, and the classifier judges image content alone. (The old "C01–C30 label conflicts" were an artifact of this wrong assumption — resolved, labels are correct.)
- **Split by production run** (filename timestamp), never per-image — images within a run share panel/lighting/session conditions and leak.
- **Crop the bottom OSD strip** before anything touches a model: red `ErrorCount` text burned into every image correlates with the verdict (label leakage). A unit test must verify no red text survives.
- The frozen test set is never touched by retraining; refreshing it is a deliberate manual act (`test_v2` + gate re-baseline).
- The `[code]` filename field may encode the machine's verdict — keep it away from model inputs and split logic.
- Predictions are reported **per image only** — there is no filename key to roll up to physical parts.

## Style & conventions

- Config-driven: all paths/hyperparameters/thresholds from `configs/config.yaml` via `coilvision/config.py`.
- Preprocessing is one shared code path for train and predict — never duplicate it.
- Metrics priority: **fail-recall ≥ 95% first**, macro-F1 tiebreak. Always report per-class metrics and per-run breakdowns, never bare accuracy.
- Every training run writes to `artifacts/runs/<run_id>/` (config snapshot, metrics.json, plots, gallery); fixed seed for reproducibility.
- Bad/conflicted inputs go to `quarantine/` with a reason — never silently dropped.
- Tests with pytest; filename parsing, split leakage, and OSD crop each have dedicated tests that must pass before merging.
