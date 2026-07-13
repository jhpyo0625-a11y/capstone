# Coil Defect Classifier

Local, CPU-only visual inspection for wound-coil flex-PCB parts. Classifies
each image **Pass / Dent / Loose**, explains its verdicts with patch-level
P(fail) heatmaps, and retrains itself from a watch folder with a safety gate
that only promotes a new model when it beats the current one on a frozen test
set. Nothing leaves this machine.

**Production model** (promoted 2026-07-12, `models/production/POINTER.json`):
measured on the frozen test set — **fail-recall 93.9% at a 13.3% false-reject
rate** (threshold 0.9149, recall-first policy). Full history of every decision
and measurement lives in `COIL_CLASSIFIER_SPEC.md`.

---

## Setup (once per machine)

1. Install [uv](https://docs.astral.sh/uv/): `winget install astral-sh.uv` (then open a new terminal).
2. In this folder: `uv sync` — installs Python 3.12 and every dependency automatically.
3. Verify: `uv run pytest` → all tests green.

The first model run downloads pretrained backbone weights (~100 MB, one time —
needs internet once; everything after is fully offline).

## Daily use — inspecting a folder of images

```
scripts\predict.bat "C:\path\to\images" --out report.csv --overlays
```

- Scans the folder recursively for `.bmp` files (2448×2048, the camera's format).
- **Always quote paths** — these filenames contain `[...]`, which PowerShell
  otherwise treats as wildcards.
- `report.csv` has one row per image:

| column | meaning |
|---|---|
| `verdict` | **PASS**, **FAIL**, or **ERROR** (file unreadable / wrong size — reason in `issue`) |
| `predicted_class` | Pass, Dent, or Loose |
| `fail_score` | 0–1; FAIL when ≥ the production threshold |
| `dent_share` / `loose_share` | which defect type the hottest patches vote for |
| `run` `part` `shot` `code` | parsed from the filename — provenance only, never used for the verdict |
| `roi_confident` | False = coil location fell back to a fixed crop (rare; verdict still valid) |

- `--overlays` writes heatmap JPEGs next to the report showing *where* the
  model sees each problem — glance at these before trusting a surprising FAIL.

## Adding new data & retraining

Drop labeled images into the watch folder, keeping the class structure:

```
incoming\Pass\...            incoming\Fail\Dent\...            incoming\Fail\Loose\...
```

**Automatic:** register the hourly watcher once — `.\scripts\register_task.ps1`
(remove with `Unregister-ScheduledTask -TaskName CoilVisionWatcher`). It fires a
retrain cycle at ≥ 50 new images, or weekly if any are waiting, and logs to
`artifacts\watcher.log`.

**Manual:** `scripts\retrain.bat` (add `--force` to run below the 50-image threshold).

Each cycle (~30–45 min): validates and quarantines bad files → merges the good
ones into `data_accepted\` → rebuilds manifest/splits (the frozen test set is
never touched) → trains a candidate → compares candidate vs production **on the
frozen test set** → promotes only if fail-recall is not worse AND macro-F1 is
strictly better. Either way it writes
`artifacts\runs\retrain_<timestamp>\PIPELINE_REPORT.md` — read it after every cycle.

### ⚠ Annotate new defect images promptly

When a report says **ACTION NEEDED**, open the `annotate_*.html` page it names,
click/brush every defect, press **Download annotations**, and move the
downloaded JSON into `artifacts\annotation\` (keep its exact filename). Until
then, new defect images add no training signal — and worse, un-annotated
defect images entering the validation set can drag the auto-selected threshold
down (the gate will reject such candidates, but the cycle is wasted).

### Quarantine

Rejected files land in `quarantine\` with reasons in `quarantine_log.csv`:
`unreadable`, `unexpected_dims_WxH` (must be 2448×2048), `unparseable_filename`
(must match the machine's naming), `unknown_label_folder`, `duplicate_content`,
`name_collision` (same name as an existing image but different pixels — review
by hand).

### Rollback

Every promotion archives the previous model to `models\archive\<timestamp>\`.
To roll back: copy that folder's `head.joblib` + `POINTER.json` back into
`models\production\`.

## Rules that keep the numbers honest

- `Coil-image-Dataset\` is read-only, forever. New data goes through `incoming\`.
- The frozen test set (`artifacts\manifests\test_v1.csv`) is the referee for
  every promotion. Re-baselining it (`python -m coilvision.data.split --refreeze`)
  invalidates all past comparisons — deliberate act only.
- Don't hand-edit the `preprocess:` section of `configs\config.yaml`: every
  cache and the production model are fingerprinted against it, and
  `coil-predict` will refuse to run against a mismatched model (that refusal is
  a feature — retrain and re-promote instead).
- Filenames are provenance only. The model judges pixels.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `preprocess-version mismatch` | `configs\config.yaml` preprocess section changed since promotion. Revert the change, or run a retrain cycle and promote a matching model. |
| `another retrain is running (lock ...)` | A cycle is in progress. A crashed run's lock self-clears after 6 h; delete `artifacts\retrain.lock` to clear sooner. |
| `UnicodeEncodeError ... cp949` | Run via the `scripts\*.bat` entry points (they set UTF-8), or `set PYTHONIOENCODING=utf-8` first. |
| Watcher never fires | Is the task registered? (`Get-ScheduledTask CoilVisionWatcher`) Check `artifacts\watcher.log` and `artifacts\pipeline_state.json`. |
| Everything quarantined | Check `quarantine\quarantine_log.csv` reasons — usually wrong image size or renamed files. |
| First run very slow | One-time pretrained-weight download; needs internet once. |
| Disk fills up | `artifacts\cache\` holds preprocessed images (a few GB); safe to delete stale fingerprints (they rebuild on demand). |

## Repository map

```
coilvision\           the package: data\ (manifest, preprocess, splits, validate)
                      train\ (patch classifier), eval\ (metrics, formal report),
                      predict\ (CLI), pipeline\ (retrain, watcher, promote), annotate.py
configs\config.yaml   every knob (paths, model, thresholds, gate policy)
scripts\              predict.bat · retrain.bat · watcher.bat · register_task.ps1
models\production\    live model + POINTER.json;  models\archive\ = previous ones
artifacts\            manifests & splits, run reports, annotation pages, caches
COIL_CLASSIFIER_SPEC.md   the spec: every decision, measurement, and gap analysis
```
