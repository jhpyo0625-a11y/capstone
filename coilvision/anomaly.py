"""PatchCore-style anomaly scoring experiment (2026-07-12).

Motivation: the 3-class classifier stalls at val fail-AUC ~0.6 — the defect
signal occupies ~1-2% of pixels and the CNN memorizes run appearance instead
(spec decisions log). This scores an image by the distance of its worst patches
to a memory bank of patch features from train-split PASS images only, using a
frozen pretrained backbone. No training loop, so nothing can be memorized.

Patch features: stride-8 and stride-16 maps, 3x3 local smoothing, channel
concat, fixed-seed random projection, L2 norm (cosine distance is robust to
exposure). Image score = mean of the top-k nearest-neighbor patch distances.

Run:  uv run python -m coilvision.anomaly
Writes artifacts/runs/anomaly_<ts>/: val_scores.csv, summary.json, heatmaps.
"""

from __future__ import annotations

import copy
import json
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from coilvision.config import load_config, resolve_path
from coilvision.data.preprocess import build_cache
from coilvision.train.datamodule import IMAGENET_MEAN, IMAGENET_STD, load_split_frame


def anomaly_cfg(cfg: dict) -> dict:
    """Config clone whose preprocess.resize is the anomaly resolution.

    The cache fingerprint keys both PNGs and index, so this cache coexists
    with the classifier's — nothing is clobbered.
    """
    cfg2 = copy.deepcopy(cfg)
    cfg2["preprocess"]["resize"] = list(cfg["anomaly"]["resize"])
    return cfg2


def _load_batch(paths: list, cache_dir) -> torch.Tensor:
    imgs = []
    for p in paths:
        bgr = cv2.imread(str(cache_dir / p))
        if bgr is None:
            raise FileNotFoundError(cache_dir / p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append((rgb - IMAGENET_MEAN) / IMAGENET_STD)
    x = np.stack(imgs).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(x))


class PatchExtractor:
    def __init__(self, cfg: dict):
        a = cfg["anomaly"]
        self.model = timm.create_model(
            a["backbone"], pretrained=True, features_only=True, out_indices=tuple(a["out_indices"])
        )
        self.model.eval()
        feat_dim = sum(self.model.feature_info.channels())
        g = torch.Generator().manual_seed(a["seed"])
        self.proj = torch.randn(feat_dim, a["proj_dim"], generator=g) / (a["proj_dim"] ** 0.5)
        self.grid: tuple[int, int] | None = None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, n_patches, proj_dim), L2-normalized."""
        maps = self.model(x)
        base = maps[0].shape[-2:]
        self.grid = (int(base[0]), int(base[1]))
        pooled = [F.avg_pool2d(m, 3, stride=1, padding=1) for m in maps]
        aligned = [p if p.shape[-2:] == base else F.interpolate(p, size=base, mode="bilinear", align_corners=False) for p in pooled]
        f = torch.cat(aligned, dim=1)  # (B, C, h, w)
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(b, h * w, c) @ self.proj
        return F.normalize(f, dim=-1)


def build_bank(extractor: PatchExtractor, frame: pd.DataFrame, cfg: dict) -> torch.Tensor:
    a = cfg["anomaly"]
    cache_dir = resolve_path(cfg, "cache_dir")
    rng = np.random.default_rng(a["seed"])
    chunks = []
    files = list(frame["cache_file"])
    for i in range(0, len(files), a["batch_size"]):
        feats = extractor(_load_batch(files[i : i + a["batch_size"]], cache_dir))
        for f in feats:  # (n_patches, D)
            take = rng.choice(len(f), size=min(a["patches_per_image"], len(f)), replace=False)
            chunks.append(f[take])
        if (i // a["batch_size"]) % 25 == 0:
            print(f"  bank {min(i + a['batch_size'], len(files))}/{len(files)}")
    bank = torch.cat(chunks)
    if len(bank) > a["bank_size"]:
        keep = rng.choice(len(bank), size=a["bank_size"], replace=False)
        bank = bank[torch.from_numpy(keep)]
    print(f"  bank: {tuple(bank.shape)}")
    return bank


def winding_mask(img_bgr: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Boolean mask on the patch grid covering the winding band (+margin).

    Same texture-density trick as the ROI detector, run on the cached crop:
    off-coil board patches score high on unfamiliar runs for the wrong reason
    (observed 2026-07-12, run anomaly_20260712_073206) — only winding patches
    should participate in the anomaly score.
    """
    s = img_bgr.shape[1] / 768.0  # sigmas were tuned at 768 wide
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    copper = ((hsv[..., 0] <= 25) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 90)).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.3 * s)) * copper
    dens = cv2.GaussianBlur(hp, (0, 0), 11 * s)
    if dens.max() <= 1e-6:
        return np.ones((grid[0], grid[1]), dtype=bool)  # degenerate image: don't mask
    mask = (dens > 0.25 * dens.max()).astype(np.uint8)
    ks = max(3, int(13 * s) | 1)
    mask = cv2.dilate(mask, np.ones((ks, ks), np.uint8))
    small = cv2.resize(mask.astype(np.float32), (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
    out = small > 0.2
    if out.sum() < 10:  # mask collapsed: fall back to everything
        return np.ones((grid[0], grid[1]), dtype=bool)
    return out


def score_frame(extractor: PatchExtractor, frame: pd.DataFrame, bank: torch.Tensor, cfg: dict):
    """Returns dict of score variants per image + masked patch-distance maps.

    raw_top10        = top-10 over all patches (v1 behavior)
    masked_top{1,5,10} = top-k over winding patches only
    """
    a = cfg["anomaly"]
    cache_dir = resolve_path(cfg, "cache_dir")
    files = list(frame["cache_file"])
    scores = {"raw_top10": [], "masked_top1": [], "masked_top5": [], "masked_top10": []}
    maps = []
    for i in range(0, len(files), a["batch_size"]):
        batch_files = files[i : i + a["batch_size"]]
        feats = extractor(_load_batch(batch_files, cache_dir))
        for f, fname in zip(feats, batch_files):
            dmin = (1.0 - (f @ bank.T).max(dim=1).values).reshape(extractor.grid).numpy()
            k = min(10, dmin.size)
            scores["raw_top10"].append(float(np.sort(dmin, axis=None)[-k:].mean()))
            m = winding_mask(cv2.imread(str(cache_dir / fname)), extractor.grid)
            wind = np.sort(dmin[m])
            for k2 in (1, 5, 10):
                kk = min(k2, len(wind))
                scores[f"masked_top{k2}"].append(float(wind[-kk:].mean()))
            shown = np.where(m, dmin, np.nan)
            maps.append(np.nan_to_num(shown, nan=float(np.nanmin(shown))))
        if (i // a["batch_size"]) % 25 == 0:
            print(f"  score {min(i + a['batch_size'], len(files))}/{len(files)}")
    return {k: np.array(v) for k, v in scores.items()}, maps


def save_heatmaps(frame: pd.DataFrame, maps: list, scores: np.ndarray, cfg: dict, out_dir, n_top: int = 8) -> None:
    cache_dir = resolve_path(cfg, "cache_dir")
    order = np.argsort(-scores)
    chosen = list(order[:n_top]) + list(order[-3:])  # most anomalous + most normal
    lo, hi = float(np.min([m.min() for m in maps])), float(np.max([m.max() for m in maps]))
    for rank, idx in enumerate(chosen):
        row = frame.iloc[idx]
        img = cv2.imread(str(cache_dir / row["cache_file"]))
        heat = (np.clip((maps[idx] - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(cv2.resize(heat, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
        label = f"{row['class']} score={scores[idx]:.4f}"
        cv2.putText(overlay, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{rank:02d}_{row['class']}_{scores[idx]:.4f}.png"), overlay)


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    out_dir = resolve_path(cfg, "artifacts_dir") / "runs" / time.strftime("anomaly_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = anomaly_cfg(cfg)  # hi-res preprocess for the anomaly path
    manifest = pd.read_csv(resolve_path(cfg, "manifests_dir") / "manifest.csv", keep_default_na=False)
    print(f"ensuring anomaly cache at {cfg['preprocess']['resize']} ...")
    build_cache(manifest, cfg)

    train = load_split_frame("train", cfg)
    val = load_split_frame("val", cfg)
    train_pass = train[train["class"] == "Pass"]
    print(f"bank source: {len(train_pass)} train Pass images; scoring {len(val)} val images")

    extractor = PatchExtractor(cfg)
    bank = build_bank(extractor, train_pass, cfg)
    score_variants, maps = score_frame(extractor, val, bank, cfg)

    val = val.copy()
    is_fail = (val["class"] != "Pass").to_numpy()
    aucs = {}
    for name, s in score_variants.items():
        val[f"score_{name}"] = s
        aucs[name] = float(roc_auc_score(is_fail, s))
        print(f"VAL FAIL-AUC [{name:16s}]: {aucs[name]:.4f}")

    best_name = max(aucs, key=aucs.get)
    best = score_variants[best_name]
    by_class = val.groupby("class")[f"score_{best_name}"].describe()[["count", "mean", "std", "min", "max"]].round(4)

    val[["relpath", "class", "run"] + [f"score_{n}" for n in score_variants]].to_csv(out_dir / "val_scores.csv", index=False)
    save_heatmaps(val, maps, best, cfg, out_dir)
    (out_dir / "summary.json").write_text(
        json.dumps({"val_fail_auc": aucs, "best_variant": best_name,
                    "wall_time_min": round((time.time() - t0) / 60, 1),
                    "bank_shape": list(bank.shape), "config": cfg["anomaly"]}, indent=2),
        encoding="utf-8",
    )

    print(f"\nval scores by class [{best_name}]:")
    print(by_class.to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
