# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Coil defect classifier: local, CPU-only classifier that labels wound-coil flex-PCB images **Pass / Dent / Loose**, with patch-level P(fail) heatmap explanations and an automated gated retraining pipeline. Windows 11, no cloud — data never leaves this machine. Production model is v1.0 (2026-07-13); see `README.md` for the operating manual (daily inference, retraining, rollback, troubleshooting).

**Read `COIL_CLASSIFIER_SPEC.md` before any nontrivial work.** It is the source of truth for decisions, phases, and open questions — including the architecture pivot below.

## Commands

Python 3.14 is system default but PyTorch needs **Python 3.12** — always use the pinned venv (`uv venv --python 3.12` / `py -3.12`).

```bash
uv run pytest                                    # run tests (must stay green)
uv run pytest tests/test_split.py                # leakage checks specifically
uv run pytest tests/test_split.py -k leakage      # a single test by name
scripts/retrain.bat [--force]                    # full pipeline: ingest→train→gate→promote/reject
scripts/predict.bat <folder> --overlays          # batch inference → CSV + P(fail) heatmaps
uv run python -m coilvision.train.patchclf       # retrain just the patch head
```

## Architecture — the pipeline, package by package

The production model is **not** the CNN the config's `train:` section describes —
that whole-image EfficientNet-B0/ConvNeXt path (Phase 3) and the `anomaly:`
PatchCore experiment were both superseded (2026-07-12; see decisions log).
Production is `patchclf:` in `configs/config.yaml`: a **logistic-regression
head on frozen ResNet50 patch features**, supervised by user-drawn defect
annotations. Data flows through `coilvision/` as:

1. **`data/`** — `manifest.py` builds the single source of truth (raw dataset
   + `data_accepted/`, run-grouped); `preprocess.py` crops the OSD strip and
   caches processed images keyed by a config fingerprint; `split.py` makes
   grouped train/val/test splits (`test_v1.csv` frozen); `validate.py`
   quarantines bad files before they reach a manifest.
2. **`annotate.py` / `annotations.py`** — serves the HTML annotation tool and
   parses the defect-region JSON it produces; this is the label source for
   patch supervision (`annotations_train*.json` only — `annotations_val.json`
   is diagnostic and must never train).
3. **`train/patchclf.py`** — THE model. Extracts frozen-ResNet50 patch
   features (`train/datamodule.py` loads/grids images), fits the logistic
   head against annotated positives + mined hard negatives, and exposes
   `score_processed`, the one scoring path shared by eval and serving.
   `train/trainer.py` holds the older two-stage CNN trainer (legacy path).
4. **`eval/`** — `metrics.py` computes fail-recall/macro-F1 at swept
   thresholds; `report.py` produces the formal evaluation + gallery;
   `gradcam.py` renders explanation overlays.
5. **`predict/cli.py`** — the `coil-predict` entry point (also driven by
   `scripts/predict.bat`): loads `models/production/POINTER.json`, checks its
   preprocess fingerprint, scores a folder, writes CSV + optional overlays.
6. **`pipeline/`** — `retrain.py` orchestrates a full cycle (validate → merge
   into `data_accepted/` → rebuild manifest/splits → train candidate → gate
   candidate vs. production on the frozen test set → `promote.py` swaps
   `POINTER.json` and archives the old model); `watcher.py` is the scheduled
   trigger (`scripts/register_task.ps1`) that fires `retrain.py` on new-image
   thresholds.

Every stage reads its knobs from `configs/config.yaml` via `coilvision/config.py` — never hardcode paths, thresholds, or the backbone choice.

## Core files

- `COIL_CLASSIFIER_SPEC.md` — spec, decisions log (READ THIS: every pivot is recorded with evidence)
- `configs/config.yaml` — every knob (paths, model, thresholds, promotion gate); never hardcode these
- `coilvision/train/patchclf.py` — THE model: logistic head on frozen resnet50 patch features; `score_processed` is the single scoring path shared by eval and serving
- `coilvision/data/manifest.py` — multi-root manifest (raw dataset + `data_accepted/`); single source of truth downstream
- `coilvision/data/split.py` — grouped splits; `artifacts/manifests/test_v1.csv` is the frozen test set
- `coilvision/pipeline/retrain.py` — retraining orchestrator + promotion gate
- `coilvision/annotations.py` + `annotate.py` — user defect-region annotations drive patch supervision
- `models/production/POINTER.json` — which model is live, its threshold + preprocess fingerprint

## Hard rules (data integrity)

- `Coil-image-Dataset/` is **read-only raw data** — never modify, move, or write into it.
- **Never infer part identity or labels from filenames.** Confirmed 2026-07-11: `part#`+`shot#` does NOT identify a physical part — files like `…_6-1` and `…_6-3` in the same run can be completely different parts. Filename fields (run, part#, shot#, code) are provenance metadata only; the label folder is ground truth, and the classifier judges image content alone. (The old "C01–C30 label conflicts" were an artifact of this wrong assumption — resolved, labels are correct.)
- **Split by production run** (filename timestamp), never per-image — images within a run share panel/lighting/session conditions and leak.
- **Crop the bottom OSD strip** before anything touches a model: red `ErrorCount` text burned into every image correlates with the verdict (label leakage). A unit test must verify no red text survives.
- The frozen test set is never touched by retraining; refreshing it is a deliberate manual act (`test_v2` + gate re-baseline).
- The `[code]` filename field may encode the machine's verdict — keep it away from model inputs and split logic.
- Predictions are reported **per image only** — there is no filename key to roll up to physical parts.
- **Val annotations never train.** `annotations_val.json` is diagnostic/eval only; training uses `annotations_train*.json` exclusively. The frozen test set has been evaluated exactly as recorded in the spec — never score it casually.
- Anything keyed by a **preprocess fingerprint** (caches, indexes, models) auto-invalidates on config change; never work around a fingerprint-mismatch error — it's the system refusing silent staleness.

## Style & conventions

- Config-driven: all paths/hyperparameters/thresholds from `configs/config.yaml` via `coilvision/config.py`.
- Preprocessing is one shared code path for train and predict — never duplicate it.
- Metrics priority: **fail-recall ≥ 95% first**, macro-F1 tiebreak. Always report per-class metrics and per-run breakdowns, never bare accuracy.
- Every training run writes to `artifacts/runs/<run_id>/` (config snapshot, metrics.json, plots, gallery); fixed seed for reproducibility.
- Bad/conflicted inputs go to `quarantine/` with a reason — never silently dropped.
- Tests with pytest; filename parsing, split leakage, and OSD crop each have dedicated tests that must pass before merging.
