# Coil Defect Classifier — Project Spec & Plan

**Status:** Draft v2 — label-conflict question **resolved 2026-07-11** (not conflicts; filename part numbers don't identify physical parts — see §2). Unblocked for Phase 0 completion.
**Owner:** jhpyo · **Lead dev:** Claude
**Last updated:** 2026-07-11

---

## 1. Goal

An end-to-end, locally-run image classification system that inspects wound-coil
flex-PCB parts and classifies each image as **Pass**, **Dent**, or **Loose**,
with visual explanations (Grad-CAM) of what the model keyed on, plus an
automated retraining pipeline covering data collection → cleaning → training →
evaluation → gated deployment.

**Success criteria**
- Fail-recall (Dent+Loose detected as fail) ≥ 95% on the frozen test set,
  at a false-reject rate the user accepts (measured & reported, not assumed).
- Macro-F1 reported per class; confusion matrix + Grad-CAM gallery per eval run.
- One command retrains and safely promotes (or rejects) a new model with zero
  manual steps.
- Everything runs on this Windows PC, CPU-only. No cloud, no data leaves the machine.

---

## 2. Dataset findings (EDA, 2026-07-07)

Source: `Coil-image-Dataset/` — 817 BMP images, all **2448×2048, 24-bit RGB**.

| Class | Images | Runs (panels) |
|---|---|---|
| Pass | 633 | 11 |
| Fail/Loose | 110 | 14 |
| Fail/Dent | 74 | 13 |

- **Filename schema:** `YYMMDD_HHMMSS_A35W_<part#>-<shot#> [<code>].bmp`
  - `YYMMDD_HHMMSS` — production run / panel timestamp (28 distinct runs).
  - `part#` — a per-image number 1–29. **Confirmed by user 2026-07-11: `part#`+`shot#`
    does NOT identify a physical part.** Two files sharing a run and `part#` but
    differing in `shot#` (e.g. `…_6-1` vs `…_6-3`) can be completely different
    physical parts. There is no known filename key for physical part identity.
  - `shot#` — 1, 2, or 3; exact meaning unconfirmed, but it is **not** "shot N of
    the same part".
  - `[code]` — always a power of 2 (1, 16, 1024, 16384, 65536, 262144); present in
    all classes so not a pure verdict flag — likely a lighting/exposure config
    bitmask (**meaning unconfirmed**).
  - **Rule: filename fields are provenance metadata only.** Never use them to
    infer part identity or labels. Ground truth is the label folder; the
    classifier judges image content alone.
- **Subject:** flat racetrack-wound fine copper coil on an orange flex PCB around
  a marked chip, inside a routed carrier panel.
- **Defect signatures:** *Loose* = spread/wavy winding turns, typically at the
  coil's rounded ends. *Dent* = locally crushed/kinked winding arc. Defects are
  small vs. the frame → ROI cropping matters.

### Data quality issues found

| # | Issue | Impact | Mitigation |
|---|---|---|---|
| 1 | Class imbalance ~7.5:1 (Pass vs each fail class) | Model biased to Pass | Class-weighted loss, stratified splits, per-class metrics |
| 2 | **Red OSD text burned into every image** (`Image acquired: N ErrorCount: N`, bottom-left) — ErrorCount correlates with the machine's verdict | **Label leakage** — model could read the answer off the image | Crop the bottom strip in preprocessing; verify no red OSD pixels survive |
| 3 | ~~30 parts have shots in two label folders~~ **RESOLVED 2026-07-11:** user reviewed all 74 flagged images — the "conflicts" were an artifact of assuming `part#`+`shot#` identifies one physical part; they are different parts and **all folder labels are correct** | None — no label noise, no quarantine needed | Lesson recorded as a rule in §2: never infer part identity/labels from filenames |
| 4 | Images within a production run share panel, lighting, and session conditions (and physical part identity cannot be established from filenames, so within-run duplicates can't be ruled out) | Random per-image split leaks correlated images train→test | **Group split by production run** (never per-image) |
| 5 | Only 28 distinct runs | High split variance; test set is small at the group level | Report per-run breakdown; optional grouped K-fold for confidence intervals |
| 6 | ≥2 distinct board layouts (e.g. run `251014_141452` differs) | ROI/model must not assume one layout | ROI detection must be layout-agnostic; track layout in manifest; stratify splits so both layouts appear in train |

---

## 3. Decisions log (agreed 2026-07-07)

| Topic | Decision |
|---|---|
| Approach | **Transfer-learning CNN** (pretrained small backbone, fine-tuned on ROI crops) with Grad-CAM visualization |
| Task framing | **3-class single model** (Pass / Dent / Loose); pass-vs-fail derived as `P(Dent)+P(Loose)` |
| Compute | **All local, this PC, CPU-only** — training and inference |
| Eval priority | **Defect recall first** — tune decision threshold for fail-recall ≥ 95%, accept more false rejects; primary gate metric = fail-recall, tiebreak macro-F1 |
| Retraining trigger | **Watch folder + threshold**: new labeled images accumulate in `incoming/`; pipeline auto-runs at ≥ 50 new images or weekly, whichever first |
| Serving | **CLI batch tool**: `predict <folder>` → CSV report + Grad-CAM overlays |
| Promotion | **Auto-promote with gate**: new model replaces production only if it beats it on frozen test (fail-recall must not degrade); all runs archived as versioned JSON + plots |
| Label conflicts | **Resolved 2026-07-11** — not conflicts; folder labels confirmed correct (see §2). Training data unfrozen |
| Model selection metric | **val fail-AUC, macro-F1 tiebreak** (2026-07-12): early-stopping on fail-recall selected the degenerate all-fail predictor (recall 1.0 @ FRR 0.98, run `20260711_231842`); AUC is threshold-free, operating threshold tuned in Phase 4 |
| Input representation | **768×288 rectangular letterbox + per-image normalization** (2026-07-12): the original 384² square downscaled the winding pitch to ~1.2px — below Nyquist, defect texture aliased away — and ImageNet normalization left per-session exposure shortcuts; observed as val fail-AUC *below* 0.5 while train loss fell (runs `231842`, `234510`) |
| Filename semantics | **Filenames are provenance only** (2026-07-11): `part#`/`shot#` do not identify a physical part; never use filename fields to infer labels or identity. The classifier decides Pass/Dent/Loose from image content alone |

---

## 4. Open questions

1. ~~C01–C30 adjudication~~ **RESOLVED 2026-07-11**: user reviewed the images —
   same run+`part#` with different `shot#` are different physical parts, so there
   were never any label conflicts; all folder labels are correct as-is.
2. What do `shot#` (-1/-2/-3) and `[code]` actually mean on the inspection
   machine? Still unconfirmed — but now moot for modeling: both are treated as
   opaque provenance metadata and are never inputs to the model or label logic.
   If `[code]` encodes the machine's verdict in any way, it must additionally be
   excluded from anything the model or splits can see.
3. ~~Per-image vs per-part reporting~~ **RESOLVED by #1's finding**: since
   physical part identity cannot be derived from filenames, there is no key to
   aggregate on — predictions are reported **per image only**. A part-level
   rollup can be added later only if the inspection machine provides a real
   part ID.

---

## 5. Architecture

```
MyProject1/
├── Coil-image-Dataset/          # RAW data — read-only, never modified
├── incoming/                    # drop zone for new labeled data (same Pass/Fail structure)
│   ├── Pass/  ├── Fail/Dent/  └── Fail/Loose/
├── quarantine/                  # rejected/conflicted images + reason log
├── coilvision/                  # Python package (src)
│   ├── config.py                # loads configs/config.yaml
│   ├── data/
│   │   ├── validate.py          # schema/integrity checks on incoming files
│   │   ├── manifest.py          # parse filenames → manifest.parquet/csv
│   │   ├── preprocess.py        # OSD crop, ROI detect, resize, cache
│   │   └── split.py             # grouped, stratified split; frozen test manifest
│   ├── train/
│   │   ├── datamodule.py        # datasets, augmentation, class weights
│   │   └── trainer.py           # fine-tune loop, early stopping, checkpoints
│   ├── eval/
│   │   ├── metrics.py           # per-class + fail-recall @ threshold, per-run breakdown
│   │   ├── gradcam.py           # Grad-CAM overlays + HTML gallery
│   │   └── report.py            # metrics.json, confusion matrix png, gallery
│   ├── predict/
│   │   └── cli.py               # `coil-predict <folder>` → CSV + overlays
│   └── pipeline/
│       ├── retrain.py           # orchestrator: ingest→preprocess→train→eval→gate→promote
│       └── watcher.py           # counts incoming/, fires retrain at threshold
├── configs/config.yaml          # every knob lives here (paths, model, thresholds, gate)
├── artifacts/
│   ├── manifests/               # manifest.csv, splits (train/val/test file lists)
│   ├── eda/                     # EDA report + plots
│   └── runs/<run_id>/           # per-training-run: config snapshot, metrics, plots, gallery
├── models/
│   ├── <run_id>/model.pt        # every trained candidate
│   └── production/              # promoted model + its metrics + POINTER.json (which run_id, when, why)
├── scripts/                     # retrain.bat, predict.bat, register_task.ps1 (Task Scheduler)
├── tests/                       # pytest: filename parsing, split leakage check, OSD crop check
└── COIL_CLASSIFIER_SPEC.md      # this file
```

**Environment:** system Python is 3.14 — PyTorch wheels may lag it. Use `uv` (or
`py -3.12`) to pin a **Python 3.12 virtualenv**. Core deps: `torch` (CPU),
`timm`, `opencv-python`, `pandas`, `scikit-learn`, `grad-cam`, `matplotlib`,
`pyyaml`, `pytest`.

---

## 6. Technical spec

### 6.1 Preprocessing (Phase 1)

1. **Validate** — readable BMP, expected 2448×2048×24bpp, filename parses to
   schema; failures → `quarantine/` with reason.
2. **Manifest** — one row per image: path, class, run, part#, shot#, code, layout
   (auto-clustered), file hash. Single source of truth for everything downstream.
   run/part#/shot#/code are provenance only — never model inputs or label logic
   (run is additionally used for split grouping).
3. **OSD removal** — crop the bottom **12.5%** (keep rows 0–1791). Measured
   2026-07-11: the red text occupies rows 1822–1858 across the dataset, so the
   originally assumed ~8% crop (keep 0–1884) would NOT have removed it. A unit
   test asserts no saturated-red text pixels remain in any processed image.
4. **ROI extraction** — layout-agnostic coil localization via texture density
   (revised 2026-07-11 after a color+gradient approach grabbed the whole board
   strip): band-pass energy |gray − blur| restricted to copper hues, pooled into
   a density map; the connected component containing the density peak is the
   winding band → bbox padded ~15%. Sanity checks (peak centrality, area, aspect,
   width fraction) gate confidence; fallback = fixed central crop (5–95% W,
   20–72% H), flagged in the cache index.
5. **Resize** to 768×288 (letterboxed, rectangular — see decisions log
   2026-07-12), cache as PNG in `artifacts/cache/` keyed by file hash +
   preprocess-config fingerprint (any parameter change invalidates the cache).
6. **EDA report** (once, and refreshed on retrain): class/run/shot/code
   distributions, brightness histograms per code value, layout cluster counts,
   duplicate detection (perceptual hash).

### 6.2 Splitting (Phase 2)

- **Group = production run** (timestamp). Shots and parts of one run never span splits.
- Frozen **test**: ~6 runs (≥ 2 with Dent, ≥ 2 with Loose, both layouts if possible),
  written to `artifacts/manifests/test_v1.csv` and **never touched by retraining**.
- Remaining runs → train/val ≈ 80/20 by run, stratified by class presence.
- A leakage unit test asserts zero run overlap between splits.
- New `incoming/` data joins train/val only; test refresh is a deliberate manual act (creates `test_v2`, re-baselines the gate).

### 6.3 Training (Phase 3)

- Backbone: **EfficientNet-B0** via `timm`, ImageNet weights (fallback candidate: `convnext_tiny`).
- Input **768×288** (rectangular letterbox matched to coil aspect; see decisions
  log 2026-07-12), per-image normalization. Two-stage fine-tune: (1) head only,
  lr 3e-3, 3 epochs; (2) full network, lr 1e-4, cosine decay, up to 30 epochs,
  early stop on val **fail-AUC** (macro-F1 tiebreak), patience 5.
- Loss: class-weighted cross-entropy (weights ∝ inverse class frequency, from train split only).
- Augmentation (train only): h/v flip, rotation ±5°, brightness/contrast ±15%,
  slight random-resized-crop (0.9–1.0). No hue shifts (copper color is signal).
- Batch 16, AdamW, seed fixed; config snapshot saved per run.
- Expected CPU wall time: ~30–90 min per retrain (measure in Phase 3, record in config as a budget).

### 6.4 Evaluation & visualization (Phase 4)

- Per run: confusion matrix, per-class precision/recall/F1, macro-F1,
  **fail-recall vs false-reject-rate curve**; operating threshold chosen as the
  lowest false-reject point achieving fail-recall ≥ 95% on val, then reported on test.
- Per-run (production-run) breakdown table to spot bad sessions/layouts.
- **Grad-CAM gallery**: HTML page with overlays for every test fail, every
  misclassification, and a sample of correct passes — this is the "what is it
  looking at" deliverable.
- All outputs to `artifacts/runs/<run_id>/`.

### 6.5 Deployment (Phase 5)

- `coil-predict <folder> [--out report.csv] [--overlays]`:
  loads `models/production/`, preprocesses identically (shared code path),
  writes CSV **one row per image** (file, run, part#, shot#, predicted class,
  probs, pass/fail verdict at production threshold) + optional Grad-CAM
  overlays. No part-level rollup — filenames don't identify physical parts
  (see §2); run/part#/shot# columns are provenance only.
- Model packaged with its preprocess version + threshold in `POINTER.json`;
  predict refuses to run on a version mismatch.

### 6.6 Retraining pipeline (Phase 6)

```
watcher (Task Scheduler, hourly count + weekly forced)
  └─ if new images in incoming/ ≥ 50 OR 7 days elapsed with any new:
       1. validate + quarantine bad files (unreadable, wrong size, unparseable name)
          — NOTE: no filename-based label-conflict check; same run+part# across
          labels is normal (different physical parts, see §2)
       2. merge into dataset copy, rebuild manifest, preprocess cache
       3. re-split train/val (test_v1 frozen)
       4. train candidate
       5. evaluate on frozen test
       6. GATE: promote iff fail-recall(candidate) ≥ fail-recall(production)
                AND macro-F1(candidate) > macro-F1(production)
       7. promote → swap models/production/ (previous kept); reject → archive with report
       8. write artifacts/runs/<run_id>/PIPELINE_REPORT.md either way
```

- Idempotent and resumable; a lock file prevents concurrent runs.
- `scripts/retrain.bat` runs the same pipeline manually end-to-end.

---

## 7. Task breakdown

**Phase 0 — Setup & label audit** *(in progress)*
- [x] EDA: structure, counts, dimensions, filename schema, leakage risks
- [x] Conflict contact sheet (C01–C30) delivered for adjudication
- [x] User adjudicated C01–C30 (2026-07-11): not conflicts — labels correct as-is; open question 3 resolved (per-image reporting only)
- [x] Python 3.12 venv + dependencies + project skeleton + pytest wiring (2026-07-11: uv-managed CPython 3.12.13 in `.venv/`, torch 2.13.0+cpu, 4 smoke tests green)
- *Acceptance: `pytest` green on a trivial test; decisions recorded here.* ✅

**Phase 1 — Data pipeline** *(built 2026-07-11)*
- [x] Filename parser + manifest builder — 817/817 parsed & valid; 4 layout clusters
- [x] Validation + quarantine flow (report-only on raw dataset; move+log for incoming/)
- [x] OSD crop (12.5%, measured) + red-text leakage test over the full cache
- [x] ROI detector (texture-density, both layouts) + fallback + spot-check sheet — 815/817 confident, 2 flagged fallbacks (coil fully inside fallback crop)
- [x] Preprocess cache (`artifacts/cache/`, 384² PNG) + EDA report (`artifacts/eda/eda_report.md`)
- *Acceptance: manifest covers 100% of files ✅; leakage test green ✅; ROI spot-check sheet → `artifacts/eda/roi_spotcheck.html`, **awaiting user approval**.*

**Phase 2 — Splits** *(built 2026-07-11)*
- [x] Grouped stratified split — deterministic exhaustive search over run combos
  (no RNG; constraints + target fractions in `configs/config.yaml`); `test_v1`
  written and freeze mechanism verified (rerun refuses to rewrite it; `--refreeze` = deliberate re-baseline)
- [x] Leakage unit test (zero run overlap between splits; every valid image in exactly one split)
- *Acceptance: split summary table (runs/images/classes per split) **approved by user 2026-07-11**; `test_v1` (6 runs / 146 images, ~18% per class) is now the frozen baseline.* ✅

**Phase 3 — Training**
- [ ] Datamodule + augmentation + class weights
- [ ] Trainer with early stopping, checkpoints, config snapshots
- [ ] First trained candidate + measured CPU wall time
- *Acceptance: training reproducible from one command; val fail-recall reported.*

**Phase 4 — Evaluation & Grad-CAM**
- [ ] Metrics module + threshold selection + per-run breakdown
- [ ] Grad-CAM gallery generator
- [ ] Baseline report on test_v1
- *Acceptance: fail-recall ≥ 95% at an accepted false-reject rate, or a documented gap analysis + iteration plan.*

**Phase 5 — Deployment (CLI)**
- [ ] `coil-predict` CLI + CSV/rollup/overlays + version-pinned production loading
- *Acceptance: user runs it on a fresh folder and the report is correct/readable.*

**Phase 6 — Automated retraining**
- [ ] Orchestrator (steps 1–8 above) + lock + reports
- [ ] Watcher + Task Scheduler registration script
- [ ] Dry-run drill: drop 50 images into `incoming/`, watch full auto cycle promote/reject correctly
- *Acceptance: the drill passes untouched, including a deliberate bad-model rejection test.*

**Phase 7 — Docs & handoff**
- [ ] README: setup, daily use, retraining ops, troubleshooting
- [ ] Final walkthrough
- *Acceptance: user can operate everything without Claude.*

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| 28 runs is few groups → noisy test estimates | Per-run breakdown; optional grouped 5-fold CV for error bars before trusting the gate |
| PyTorch vs Python 3.14 | Pinned 3.12 venv from day one |
| ROI detector fails on a future new layout | Fallback central crop + low-confidence flag in manifest; flagged images surfaced in pipeline report |
| Incoming data arrives mislabeled | No automated filename-based detection possible (filenames carry no identity/label info) — rely on file validation, per-run metric breakdowns to spot bad sessions, and periodic manual spot-checks; contact-sheet tooling reusable |
| CPU retrain too slow as data grows | Cache preprocessed images; if wall time exceeds budget, drop input to 320² or freeze more layers before considering cloud |
