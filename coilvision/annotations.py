"""Load and rasterize user defect-region annotations (annotate.py schema v2).

Strokes/boxes are normalized to the annotation image (the hi-res anomaly cache);
`defect_mask` rasterizes them into a binary mask at any target resolution, which
drives patch sampling, heatmap verification, and metric masking.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_annotations(path: str | Path) -> dict[str, dict]:
    """relpath -> {class, no_defect_visible, boxes, strokes}. Only version 2."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 2:
        raise ValueError(f"unsupported annotation version: {data.get('version')}")
    return {im["relpath"]: im for im in data["images"]}


def defect_mask(entry: dict, width: int, height: int) -> np.ndarray:
    """Rasterize one image's boxes+strokes into a uint8 {0,1} mask of (height, width)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in entry.get("boxes", []):
        x0, y0 = int(b["x0"] * width), int(b["y0"] * height)
        x1, y1 = int(np.ceil(b["x1"] * width)), int(np.ceil(b["y1"] * height))
        cv2.rectangle(mask, (x0, y0), (max(x1, x0 + 1), max(y1, y0 + 1)), 1, thickness=-1)
    for s in entry.get("strokes", []):
        r = max(1, int(round(s["r"] * width)))
        pts = [(int(p["x"] * width), int(p["y"] * height)) for p in s["pts"]]
        for p in pts:
            cv2.circle(mask, p, r, 1, thickness=-1)
        for a, b in zip(pts, pts[1:]):
            cv2.line(mask, a, b, 1, thickness=2 * r)
    return mask


def mask_to_grid(mask: np.ndarray, grid: tuple[int, int], min_frac: float = 0.10) -> np.ndarray:
    """Downsample a pixel mask to a patch grid (gh, gw): patch is positive if
    >= min_frac of its pixels are annotated."""
    gh, gw = grid
    small = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return small >= min_frac
