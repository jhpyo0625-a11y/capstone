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


def score_processed(extractor: PatchExtractor, head, img_bgr: np.ndarray, top_k: int) -> dict:
    """Score ONE preprocessed (cache-format) BGR image.

    The single scoring code path shared by training-time evaluation and the
    predict CLI (spec §6.5: identical preprocessing/scoring in train and serve).
    Returns fail score (top-k mean of P(fail) over winding patches), the
    Dent-vs-Loose vote among the hottest patches, and the P(fail) grid.
    """
    import torch

    from coilvision.train.datamodule import IMAGENET_MEAN, IMAGENET_STD

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    feats = extractor(torch.from_numpy(np.ascontiguousarray(x)))[0].numpy()
    probs = head.predict_proba(feats)
    pfail = (probs[:, 1] + probs[:, 2]).reshape(extractor.grid)
    wmask = winding_mask(img_bgr, extractor.grid)
    vals = np.sort(pfail[wmask])
    kk = min(top_k, len(vals))
    score = float(vals[-kk:].mean())
    flat_idx = np.argsort(pfail[wmask])[-kk:]
    top_probs = probs[np.flatnonzero(wmask.flatten())][flat_idx]
    dent, loose = float(top_probs[:, 1].sum()), float(top_probs[:, 2].sum())
    return {
        "score": score,
        "vote": "Dent" if dent >= loose else "Loose",
        "dent_share": round(dent / max(dent + loose, 1e-9), 4),
        "loose_share": round(loose / max(dent + loose, 1e-9), 4),
        "pfail_map": np.where(wmask, pfail, 0.0),
        "winding_vals": vals,  # sorted P(fail) over winding patches, for aggregation variants
    }


def load_train_annotations(cfg: dict) -> dict:
    """Merge every annotations_train*.json — the retrain pipeline generates a
    new page (and thus a new JSON) whenever un-annotated defect images arrive."""
    ann_dir = resolve_path(cfg, "artifacts_dir") / "annotation"
    files = sorted(ann_dir.glob("annotations_train*.json"))
    if not files:
        raise FileNotFoundError(f"no annotations_train*.json in {ann_dir}")
    merged: dict = {}
    for f in files:
        merged.update(load_annotations(f))
    return merged


def dataset_key(cfg: dict, ann: dict, train_hashes: list[str]) -> str:
    """Cache key for the patch dataset: annotation content + preprocess
    fingerprint + sampling params + the TRAIN SPLIT CONTENT (retraining with
    new incoming images must invalidate). Mining params excluded — mining runs
    after this cache."""
    p = cfg["patchclf"]
    blob = json.dumps(
        {
            "ann": ann,
            "min_annot_frac": p["min_annot_frac"],
            "negatives_per_image": p["negatives_per_image"],
            "seed": p["seed"],
            "train_hashes": sorted(train_hashes),
        },
        sort_keys=True,
    ).encode()
    return hashlib.blake2b(blob + preprocess_fingerprint(cfg).encode(), digest_size=6).hexdigest()


def build_patch_dataset(cfg: dict, extractor: PatchExtractor, ann: dict) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) patch features/labels from the train split. Cached to disk keyed
    by dataset_key (annotations + preprocess fingerprint + sampling params +
    train split content)."""
    p = cfg["patchclf"]
    cache_dir = resolve_path(cfg, "cache_dir")
    key = dataset_key(cfg, ann, list(load_split_frame("train", cfg)["hash"]))
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
        if img is None:
            raise FileNotFoundError(f"cache image missing or unreadable: {cache_dir / r['cache_file']}")
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
        if img is None:
            raise FileNotFoundError(f"cache image missing or unreadable: {cache_dir / r['cache_file']}")
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
    scores.update({"mean": [], "hotfrac": [], "logodds_top20": []})
    dent_vs_loose, maps = [], []
    for n, (_, r) in enumerate(frame.iterrows(), 1):
        img = cv2.imread(str(cache_dir / r["cache_file"]))
        if img is None:
            raise FileNotFoundError(f"cache image missing or unreadable: {cache_dir / r['cache_file']}")
        # score_processed is THE scoring path (shared with the predict CLI);
        # everything here derives from its outputs so eval and serving can't diverge
        s = score_processed(extractor, head, img, p["top_k"])
        vals = s["winding_vals"]
        for k in ks:
            kk = min(k, len(vals))
            scores[f"top{k}"].append(float(vals[-kk:].mean()))
        # breadth-aware variants: falsified on val 2026-07-12 (kept for monitoring)
        scores["mean"].append(float(vals.mean()))
        scores["hotfrac"].append(float((vals > 0.5).mean()))
        top = np.clip(vals[-min(20, len(vals)):], 1e-6, 1 - 1e-6)
        scores["logodds_top20"].append(float(np.log(top / (1 - top)).mean()))
        dent_vs_loose.append(s["vote"])
        maps.append(s["pfail_map"])
        if n % 50 == 0:
            print(f"  scored {n}/{len(frame)}")
    return {k: np.array(v) for k, v in scores.items()}, dent_vs_loose, maps


def train_candidate(cfg: dict) -> dict:
    """Full candidate training + val evaluation. Returns run metadata for the
    retrain pipeline (head path, val scores path, metrics, annotation coverage)."""
    t0 = time.time()
    run_id = time.strftime("patchclf_%Y%m%d_%H%M%S")
    out_dir = resolve_path(cfg, "artifacts_dir") / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg["patchclf"]

    manifest = pd.read_csv(resolve_path(cfg, "manifests_dir") / "manifest.csv", keep_default_na=False)
    build_cache(manifest, cfg)  # ensure hi-res cache (reused if present)

    ann = load_train_annotations(cfg)
    train = load_split_frame("train", cfg)
    train_defects = train[train["class"] != "Pass"]
    unannotated = [r for r in train_defects["relpath"] if r not in ann or not (ann[r]["strokes"] or ann[r]["boxes"])]
    if unannotated:
        print(f"NOTE: {len(unannotated)} train defect images have no annotations yet "
              f"(they contribute to val/eval only, not patch supervision)")

    extractor = PatchExtractor(cfg)
    X, y = build_patch_dataset(cfg, extractor, ann)
    head = LogisticRegression(max_iter=3000, C=p["C"], class_weight="balanced")
    head.fit(X, y)
    assert list(head.classes_) == [0, 1, 2], f"head classes misaligned: {head.classes_}"  # P(fail)=cols 1+2

    # Hard-negative mining (train split only): the first head saturates on rare
    # benign winding textures it never sampled — mine the hottest Pass patches
    # and refit so "unusual but fine" stops looking like a defect.
    for rnd in range(p.get("hard_negative_rounds", 0)):
        cache_dir = resolve_path(cfg, "cache_dir")
        passes = load_split_frame("train", cfg)
        passes = passes[passes["class"] == "Pass"]
        hard = []
        for n, (_, r) in enumerate(passes.iterrows(), 1):
            img = cv2.imread(str(cache_dir / r["cache_file"]))
            if img is None:
                raise FileNotFoundError(f"cache image missing or unreadable: {cache_dir / r['cache_file']}")
            feats = _features_for(extractor, cache_dir, r["cache_file"])
            pfail = head.predict_proba(feats)[:, 1:].sum(axis=1)
            wmask = winding_mask(img, extractor.grid).flatten()
            idx = np.flatnonzero(wmask)
            take = idx[np.argsort(pfail[idx])[-p["hard_negatives_per_image"]:]]
            hard.append(feats[take])
            if n % 100 == 0:
                print(f"  mining round {rnd + 1}: {n}/{len(passes)}")
        X = np.concatenate([X, np.concatenate(hard).astype(np.float32)])
        y = np.concatenate([y, np.zeros(sum(len(h) for h in hard), dtype=int)])
        head = LogisticRegression(max_iter=3000, C=p["C"], class_weight="balanced")
        head.fit(X, y)
        print(f"  round {rnd + 1}: +{sum(len(h) for h in hard)} hard negatives, labels {np.bincount(y).tolist()}")

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
    summary = {"run_id": run_id, "val_image_fail_auc": aucs, "best_variant": best_name,
               "dent_vs_loose_val_acc": dl_acc, "train_patch_auc": float(train_patch_auc),
               "n_patches": int(len(y)), "label_counts": np.bincount(y).tolist(),
               "n_unannotated_train_defects": len(unannotated),
               "wall_time_min": round((time.time() - t0) / 60, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_dir}")
    return {**summary, "run_dir": out_dir, "head_path": out_dir / "head.joblib",
            "val_scores_path": out_dir / "val_scores.csv", "unannotated_relpaths": unannotated}


def main() -> None:
    train_candidate(anomaly_cfg(load_config()))


if __name__ == "__main__":
    main()
