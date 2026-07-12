"""Datasets, augmentation, class weights (spec §6.3).

Reads pre-processed 384px PNGs from the cache (never raw BMPs — one shared
preprocess path). Images are preloaded into RAM (~250 MB for the full set) so
CPU epochs aren't I/O-bound. Augmentation is train-only and hue-preserving:
copper color is signal.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from coilvision.config import resolve_path
from coilvision.data.preprocess import cache_path_for

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_split_frame(name: str, cfg: dict) -> pd.DataFrame:
    """Split rows joined with their cache filenames and integer labels."""
    manifests_dir = resolve_path(cfg, "manifests_dir")
    frame = pd.read_csv(manifests_dir / f"{name}_v{cfg['split']['version']}.csv", keep_default_na=False)
    class_to_idx = {c: i for i, c in enumerate(cfg["data"]["classes"])}
    frame["label"] = frame["class"].map(class_to_idx)
    if frame["label"].isna().any():
        bad = frame.loc[frame["label"].isna(), "class"].unique()
        raise ValueError(f"split '{name}' has classes not in config: {bad}")
    frame["cache_file"] = frame["hash"].map(lambda h: cache_path_for(h, cfg).name)
    return frame


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (mean ~1), computed from the train split only."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"class(es) missing from train split: counts={counts.tolist()}")
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def augment(img: np.ndarray, rng: np.random.Generator, aug: dict) -> np.ndarray:
    """h/v flip, rotation, slight random-resized-crop, brightness/contrast. No hue shifts."""
    h, w = img.shape[:2]
    if rng.random() < aug["hflip_p"]:
        img = img[:, ::-1]
    if rng.random() < aug["vflip_p"]:
        img = img[::-1]
    deg = rng.uniform(-aug["rotate_deg"], aug["rotate_deg"])
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    s = rng.uniform(aug["rrc_scale"][0], aug["rrc_scale"][1])
    ch, cw = int(round(h * s)), int(round(w * s))
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    img = cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
    alpha = 1.0 + rng.uniform(-aug["contrast"], aug["contrast"])
    beta = 255.0 * rng.uniform(-aug["brightness"], aug["brightness"])
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


class CoilDataset(Dataset):
    """Yields (CHW float32 ImageNet-normalized tensor, int label)."""

    def __init__(self, frame: pd.DataFrame, cfg: dict, train: bool, seed: int = 0):
        cache_dir = resolve_path(cfg, "cache_dir")
        self.frame = frame.reset_index(drop=True)
        self.train = train
        self.aug = cfg["train"]["augment"]
        self.normalize = cfg["train"].get("normalize", "imagenet")
        self.rng = np.random.default_rng(seed)
        self.images: list[np.ndarray] = []
        for f in self.frame["cache_file"]:
            bgr = cv2.imread(str(cache_dir / f))
            if bgr is None:
                raise FileNotFoundError(f"cache image missing or unreadable: {cache_dir / f}")
            self.images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.labels = self.frame["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = self.images[i]
        if self.train:
            img = augment(img, self.rng, self.aug)
        if self.normalize == "per_image":
            # channels joint: hue relationships preserved, per-session exposure removed
            f = img.astype(np.float32)
            x = (f - f.mean()) / (f.std() + 1e-6)
        else:
            x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), int(self.labels[i])


def make_loaders(cfg: dict, limit_train: int | None = None, limit_val: int | None = None):
    """(train_loader, val_loader, class_weights). Limits are for smoke runs only."""
    seed = cfg["train"]["seed"]
    train_frame = load_split_frame("train", cfg)
    val_frame = load_split_frame("val", cfg)
    if limit_train:
        train_frame = train_frame.groupby("class", group_keys=False).head(max(limit_train // 3, 4))
    if limit_val:
        val_frame = val_frame.groupby("class", group_keys=False).head(max(limit_val // 3, 4))

    train_ds = CoilDataset(train_frame, cfg, train=True, seed=seed)
    val_ds = CoilDataset(val_frame, cfg, train=False, seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    weights = class_weights(train_ds.labels, len(cfg["data"]["classes"]))
    return train_loader, val_loader, weights
