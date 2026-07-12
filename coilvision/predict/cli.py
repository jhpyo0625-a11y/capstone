"""`coil-predict <folder>` → per-image CSV + P(fail) heatmap overlays (spec §6.5).

Loads models/production/ (head.joblib + POINTER.json), preprocesses raw BMPs
through the identical pipeline (OSD crop → ROI → letterbox → patch features),
and writes one CSV row per image: provenance fields (run/part/shot/code —
never used for the verdict), fail score, PASS/FAIL at the production
threshold, and predicted class. No part-level rollup — filenames don't
identify physical parts (spec §2). Refuses to run if the current preprocess
config fingerprint differs from the one the model was promoted under.

Usage:
    coil-predict <folder> [--out report.csv] [--overlays] [--model models/production]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from coilvision.anomaly import PatchExtractor, anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.data.manifest import parse_filename
from coilvision.data.preprocess import preprocess_fingerprint, preprocess_image
from coilvision.train.patchclf import score_processed


def load_production(model_dir: Path, cfg: dict):
    pointer_path = model_dir / "POINTER.json"
    if not pointer_path.exists():
        raise FileNotFoundError(f"no production model: {pointer_path} missing (run coilvision.pipeline.promote)")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    current_fp = preprocess_fingerprint(cfg)
    if pointer["preprocess_fingerprint"] != current_fp:
        raise RuntimeError(
            "preprocess-version mismatch: model was promoted under fingerprint "
            f"{pointer['preprocess_fingerprint']} (v{pointer['preprocess_version']}) but the current "
            f"config produces {current_fp}. Re-promote a matching model or restore the config."
        )
    head = joblib.load(model_dir / "head.joblib")["head"]
    return head, pointer


def predict_folder(folder: Path, cfg: dict, model_dir: Path, overlays_dir: Path | None = None) -> pd.DataFrame:
    head, pointer = load_production(model_dir, cfg)
    extractor = PatchExtractor(cfg)
    top_k = int(pointer["aggregation"].removeprefix("top"))
    threshold = pointer["threshold"]
    pattern = cfg["data"]["filename_pattern"]

    files = sorted(folder.rglob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"no .bmp files under {folder}")
    if overlays_dir:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, path in enumerate(files, 1):
        parsed = parse_filename(path.name, pattern) or {"run": "", "part": -1, "shot": -1, "code": -1}
        row = {"file": path.name, "path": str(path), **parsed}
        img = cv2.imread(str(path))
        if img is None:
            rows.append({**row, "verdict": "ERROR", "predicted_class": "", "fail_score": np.nan,
                         "dent_share": np.nan, "loose_share": np.nan, "roi_confident": "", "issue": "unreadable"})
            continue
        h, w = img.shape[:2]
        if (w, h) != (cfg["data"]["expected_width"], cfg["data"]["expected_height"]):
            # wrong-format input would get a confident verdict on garbage — refuse per image
            rows.append({**row, "verdict": "ERROR", "predicted_class": "", "fail_score": np.nan,
                         "dent_share": np.nan, "loose_share": np.nan, "roi_confident": "",
                         "issue": f"unexpected_dims_{w}x{h}"})
            continue
        processed, meta = preprocess_image(img, cfg)
        s = score_processed(extractor, head, processed, top_k)
        verdict = "FAIL" if s["score"] >= threshold else "PASS"
        rows.append({
            **row,
            "verdict": verdict,
            "predicted_class": s["vote"] if verdict == "FAIL" else "Pass",
            "fail_score": round(s["score"], 4),
            "dent_share": s["dent_share"],
            "loose_share": s["loose_share"],
            "roi_confident": meta["roi_confident"],
            "issue": "",
        })
        if overlays_dir:
            heat = cv2.applyColorMap(
                cv2.resize((np.clip(s["pfail_map"], 0, 1) * 255).astype(np.uint8),
                           (processed.shape[1], processed.shape[0])), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(processed, 0.6, heat, 0.4, 0)
            cv2.putText(overlay, f"{verdict} {s['score']:.3f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # flatten the relative path into the name so same-stem files in
            # different subfolders can't overwrite each other's overlays
            safe = str(path.relative_to(folder).with_suffix("")).replace("\\", "_").replace("/", "_")
            cv2.imwrite(str(overlays_dir / f"{safe}_overlay.jpg"), overlay)
        if n % 20 == 0:
            print(f"  {n}/{len(files)}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(prog="coil-predict", description=__doc__)
    ap.add_argument("folder", help="folder of .bmp images to inspect (searched recursively)")
    ap.add_argument("--out", default="report.csv", help="output CSV path (default: report.csv)")
    ap.add_argument("--overlays", action="store_true", help="write P(fail) heatmap overlays next to the report")
    ap.add_argument("--model", default=None, help="model dir (default: models/production)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = anomaly_cfg(load_config())
    model_dir = Path(args.model) if args.model else resolve_path(cfg, "production_dir")
    out_csv = Path(args.out)
    overlays_dir = out_csv.parent / (out_csv.stem + "_overlays") if args.overlays else None

    df = predict_folder(Path(args.folder), cfg, model_dir, overlays_dir)
    df.to_csv(out_csv, index=False)

    n_fail = int((df["verdict"] == "FAIL").sum())
    n_err = int((df["verdict"] == "ERROR").sum())
    print(f"\n{len(df)} images: {len(df) - n_fail - n_err} PASS, {n_fail} FAIL, {n_err} ERROR "
          f"({(time.time() - t0) / 60:.1f} min)")
    if n_fail:
        print(df[df["verdict"] == "FAIL"][["file", "predicted_class", "fail_score"]].to_string(index=False))
    print(f"report -> {out_csv}" + (f", overlays -> {overlays_dir}" if overlays_dir else ""))


if __name__ == "__main__":
    main()
