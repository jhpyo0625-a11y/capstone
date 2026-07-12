"""Patch-level supervised classifier on frozen backbone features (2026-07-12).

Diagnosis that led here (spec decisions log): defects occupy 1-2 wire pitches;
image-level CNN training memorized run appearance (val fail-AUC <= 0.60) and
unsupervised PatchCore stalled at 0.73 because benign-but-unusual winding
patches score as hot as defects. The user's brush annotations give patch-level
labels, so a light supervised head on the SAME frozen patch features can learn
defect-vs-benign directly. Nothing is fine-tuned; there is nothing to memorize.

Data:
  positives = annotated patch cells from train defect images (class = Dent/Loose)
  negatives = winding cells sampled from train PASS images only
  (unannotated cells of defect images are excluded — annotation may not be
  exhaustive, so they are neither positive nor trusted negative)

Image score = top-k mean of P(fail) over winding patches; fail = Dent + Loose.
Selection metric: image-level val fail-AUC. The frozen test set stays untouched.

Run:  uv run python -m coilvision.train.patchclf
Writes artifacts/runs/patchclf_<ts>/: head.joblib, val_scores.csv, summary.json,
probability overlays.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from coilvision.annotations import defect_mask, load_annotations, mask_to_grid
from coilvision.anomaly import PatchExtractor, _load_batch, anomaly_cfg, save_heatmaps, winding_mask
from coilvision.config import load_config, resolve_path
from coilvision.data.preprocess import build_cache, preprocess_fingerprint
from coilvision.train.datamodule import load_split_frame

CLASSES = ["normal", "Dent", "Loose"]  # head label space; fail = Dent + Loose


def _features_for(extractor: PatchExtractor, cache_dir: Path, cache_file: str) -> np.ndarray:
    return extractor(_load_batch([cache_file], cache_dir))[0].numpy()


def build_patch_dataset(cfg: dict, extractor: PatchExtractor, ann: dict) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) patch features/labels from the train split. Cached to disk keyed by
    preprocess fingerprint + annotation content."""
    p = cfg["patchclf"]
    cache_dir = resolve_path(cfg, "cache_dir")
    ann_blob = json.dumps(ann, sort_keys=True).encode()
    key = hashlib.blake2b(ann_blob + preprocess_fingerprint(cfg).encode(), digest_size=6).hexdigest()
    npz_path = cache_dir / f"patch_dataset_{key}.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        print(f"patch dataset loaded from cache: X={d['X'].shape}")
        return d["X"], d["y"]

    train = load_split_frame("train", cfg)
    rng = np.random.default_rng(p["seed"])
    xs, ys = [], []

    defects = train[train["class"] != "Pass"]
    for n, (_, r) in enumerate(defects.iterrows(), 1):
        entry = ann.get(r["relpath"])
        if entry is None or (not entry["strokes"] and not entry["boxes"]):
            continue
        img = cv2.imread(str(cache_dir / r["cache_file"]))
        feats = _features_for(extractor, cache_dir, r["cache_file"])
        gmask = mask_to_grid(defect_mask(entry, img.shape[1], img.shape[0]), extractor.grid, p["min_annot_frac"])
        cells = feats[gmask.flatten()]
        xs.append(cells)
        ys.append(np.full(len(cells), CLASSES.index(r["class"])))
        if n % 25 == 0:
            print(f"  defect images {n}/{len(defects)}")

    passes = train[train["class"] == "Pass"]
    for n, (_, r) in enumerate(passes.iterrows(), 1):
        img = cv2.imread(str(cache_dir / r["cache_file"]))
        feats = _features_for(extractor, cache_dir, r["cache_file"])
        wmask = winding_mask(img, extractor.grid).flatten()
        idx = np.flatnonzero(wmask)
        take = rng.choice(idx, size=min(p["negatives_per_image"], len(idx)), replace=False)
        xs.append(feats[take])
        ys.append(np.zeros(len(take), dtype=int))
        if n % 50 == 0:
            print(f"  pass images {n}/{len(passes)}")

    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    np.savez_compressed(npz_path, X=X, y=y)
    print(f"patch dataset: X={X.shape}, labels {np.bincount(y).tolist()} (normal/Dent/Loose) -> {npz_path.name}")
    return X, y


def score_images(cfg: dict, extractor: PatchExtractor, head, frame: pd.DataFrame):
    """Per-image fail scores (several top-k variants) + P(fail) grids for overlays."""
    p = cfg["patchclf"]
    cache_dir = resolve_path(cfg, "cache_dir")
    ks = (1, 5, 10, 20)
    scores = {f"top{k}": [] for k in ks}
    dent_vs_loose, maps = [], []
    for n, (_, r) in enumerate(frame.iterrows(), 1):
        img = cv2.imread(str(cache_dir / r["cache_file"]))
        feats = _features_for(extractor, cache_dir, r["cache_file"])
        probs = head.predict_proba(feats)
        pfail = (probs[:, 1] + probs[:, 2]).reshape(extractor.grid)
        wmask = winding_mask(img, extractor.grid)
        vals = np.sort(pfail[wmask])
        for k in ks:
            kk = min(k, len(vals))
            scores[f"top{k}"].append(float(vals[-kk:].mean()))
        # among the hottest fail patches, which defect class dominates?
        flat_idx = np.argsort(pfail[wmask])[-min(p["top_k"], wmask.sum()):]
        top_probs = probs[np.flatnonzero(wmask.flatten())][flat_idx]
        dent_vs_loose.append("Dent" if top_probs[:, 1].sum() >= top_probs[:, 2].sum() else "Loose")
        maps.append(np.where(wmask, pfail, 0.0))
        if n % 50 == 0:
            print(f"  scored {n}/{len(frame)}")
    return {k: np.array(v) for k, v in scores.items()}, dent_vs_loose, maps


def main() -> None:
    t0 = time.time()
    cfg = anomaly_cfg(load_config())
    out_dir = resolve_path(cfg, "artifacts_dir") / "runs" / time.strftime("patchclf_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg["patchclf"]

    manifest = pd.read_csv(resolve_path(cfg, "manifests_dir") / "manifest.csv", keep_default_na=False)
    build_cache(manifest, cfg)  # ensure hi-res cache (reused if present)

    ann = load_annotations(resolve_path(cfg, "artifacts_dir") / "annotation" / "annotations_train.json")
    extractor = PatchExtractor(cfg)
    # prime extractor.grid with one image so mask_to_grid has dimensions
    first = load_split_frame("train", cfg).iloc[0]
    _features_for(extractor, resolve_path(cfg, "cache_dir"), first["cache_file"])

    X, y = build_patch_dataset(cfg, extractor, ann)
    head = LogisticRegression(max_iter=3000, C=p["C"], class_weight="balanced")
    head.fit(X, y)
    train_patch_auc = roc_auc_score(y > 0, head.predict_proba(X)[:, 1:].sum(axis=1))
    print(f"patch-level TRAIN AUC (fit sanity): {train_patch_auc:.4f}")

    val = load_split_frame("val", cfg)
    score_variants, dl_vote, maps = score_images(cfg, extractor, head, val)
    is_fail = (val["class"] != "Pass").to_numpy()
    aucs = {}
    for name, s in score_variants.items():
        val[f"score_{name}"] = s
        aucs[name] = float(roc_auc_score(is_fail, s))
        print(f"VAL IMAGE FAIL-AUC [{name:5s}]: {aucs[name]:.4f}")
    best_name = max(aucs, key=aucs.get)

    val["dent_vs_loose_vote"] = dl_vote
    defects = val[val["class"] != "Pass"]
    dl_acc = float((defects["dent_vs_loose_vote"] == defects["class"]).mean())
    print(f"dent-vs-loose vote accuracy on val defects: {dl_acc:.3f}")

    joblib.dump({"head": head, "classes": CLASSES, "preprocess_fingerprint": preprocess_fingerprint(cfg),
                 "anomaly_cfg": cfg["anomaly"], "patchclf_cfg": p}, out_dir / "head.joblib")
    val_cols = ["relpath", "class", "run", "dent_vs_loose_vote"] + [f"score_{n}" for n in score_variants]
    val[val_cols].to_csv(out_dir / "val_scores.csv", index=False)
    save_heatmaps(val, maps, score_variants[best_name], cfg, out_dir)
    (out_dir / "summary.json").write_text(
        json.dumps({"val_image_fail_auc": aucs, "best_variant": best_name,
                    "dent_vs_loose_val_acc": dl_acc, "train_patch_auc": float(train_patch_auc),
                    "n_patches": int(len(y)), "label_counts": np.bincount(y).tolist(),
                    "wall_time_min": round((time.time() - t0) / 60, 1)}, indent=2),
        encoding="utf-8",
    )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
