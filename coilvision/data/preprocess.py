"""OSD crop, ROI detection, resize, cache (spec §6.1 steps 3-5).

The bottom OSD strip MUST be cropped before anything touches a model: the burned-in
red `ErrorCount` text (rows 1822-1858, measured 2026-07-11) correlates with the
machine's verdict — label leakage. One shared code path for train and predict.

Run as a script to build the preprocess cache + ROI spot-check sheet:
    uv run python -m coilvision.data.preprocess
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coilvision.config import load_config, resolve_path


def crop_osd(img: np.ndarray, bottom_frac: float) -> np.ndarray:
    h = img.shape[0]
    return img[: h - int(round(h * bottom_frac))]


def count_red_text_pixels(img: np.ndarray, red_cfg: dict) -> int:
    """Count pure-red OSD pixels (the leakage signature). Copper wire does not trigger this."""
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    mask = (r > red_cfg["r_min"]) & (g < red_cfg["g_max"]) & (b < red_cfg["b_max"])
    return int(mask.sum())


def _fallback_bbox(w: int, h: int, roi_cfg: dict) -> tuple[int, int, int, int]:
    fb = roi_cfg["fallback_crop"]
    return (int(fb["x0"] * w), int(fb["y0"] * h), int(fb["x1"] * w), int(fb["y1"] * h))


def detect_roi(img: np.ndarray, roi_cfg: dict) -> tuple[tuple[int, int, int, int], bool]:
    """Layout-agnostic coil localization on an OSD-cropped image.

    The winding is the only region with dense fine stripe texture: band-pass energy
    (|gray − blur|) restricted to copper hues, blurred into a density map. The
    connected component containing the density peak is the winding band → padded
    bbox. Validated visually on both layouts (2026-07-11). Returns
    (bbox x0,y0,x1,y1 in full-res coords, confident).
    """
    H, W = img.shape[:2]
    scale = roi_cfg["det_width"] / W
    small = cv2.resize(img, (roi_cfg["det_width"], int(round(H * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    copper = (
        (hsv[..., 0] <= roi_cfg["hue_max"])
        & (hsv[..., 1] >= roi_cfg["sat_min"])
        & (hsv[..., 2] >= roi_cfg["val_min"])
    ).astype(np.float32)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    highpass = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), roi_cfg["highpass_sigma"])) * copper
    density = cv2.GaussianBlur(highpass, (0, 0), roi_cfg["density_sigma"])

    py, px = np.unravel_index(np.argmax(density), density.shape)
    peak = float(density[py, px])
    cr = roi_cfg["center_region"]
    peak_central = cr["x0"] * sw < px < cr["x1"] * sw and cr["y0"] * sh < py < cr["y1"] * sh
    if peak < 1e-3 or not peak_central:
        return _fallback_bbox(W, H, roi_cfg), False

    mask = (density > roi_cfg["peak_frac"] * peak).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    lab = labels[py, px]
    x, y, w, h = (stats[lab, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))

    area_frac = (w * h) / (sw * sh)
    aspect = w / max(h, 1)
    confident = (
        roi_cfg["min_area_frac"] <= area_frac <= roi_cfg["max_area_frac"]
        and roi_cfg["min_aspect"] <= aspect <= roi_cfg["max_aspect"]
        and w / sw <= roi_cfg["max_width_frac"]
    )
    if not confident:
        return _fallback_bbox(W, H, roi_cfg), False

    pad_w, pad_h = w * roi_cfg["pad_frac"], h * roi_cfg["pad_frac"]
    x0 = max(0, int((x - pad_w) / scale))
    y0 = max(0, int((y - pad_h) / scale))
    x1 = min(W, int((x + w + pad_w) / scale))
    y1 = min(H, int((y + h + pad_h) / scale))
    return (x0, y0, x1, y1), True


def letterbox(img: np.ndarray, size: int | list[int] | tuple[int, int]) -> np.ndarray:
    """Aspect-preserving resize into a (width, height) canvas, black-padded."""
    tw, th = (size, size) if isinstance(size, int) else (size[0], size[1])
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    resized = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), dtype=img.dtype)
    rh, rw = resized.shape[:2]
    top, left = (th - rh) // 2, (tw - rw) // 2
    out[top : top + rh, left : left + rw] = resized
    return out


def preprocess_image(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """The one shared train/predict path: OSD crop → ROI crop → letterbox resize."""
    p = cfg["preprocess"]
    cropped = crop_osd(img, p["osd_crop_bottom_frac"])
    (x0, y0, x1, y1), confident = detect_roi(cropped, p["roi"])
    roi = cropped[y0:y1, x0:x1]
    out = letterbox(roi, p["resize"])
    return out, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "roi_confident": confident}


def preprocess_fingerprint(cfg: dict) -> str:
    """Short hash of the whole preprocess config section.

    Baked into every cache filename so that ANY parameter change (even without a
    version bump) invalidates the cache — the index and the PNG pixels can never
    silently disagree.
    """
    blob = json.dumps(cfg["preprocess"], sort_keys=True)
    return hashlib.blake2b(blob.encode(), digest_size=4).hexdigest()


def cache_path_for(file_hash: str, cfg: dict) -> Path:
    name = f"{file_hash}_v{cfg['preprocess']['version']}_{preprocess_fingerprint(cfg)}.png"
    return resolve_path(cfg, "cache_dir") / name


def cache_index_path(cfg: dict) -> Path:
    """Fingerprint-keyed like the PNGs, so caches for different preprocess
    configs (e.g. the hi-res anomaly cache) never clobber each other's index."""
    name = f"cache_index_v{cfg['preprocess']['version']}_{preprocess_fingerprint(cfg)}.csv"
    return resolve_path(cfg, "manifests_dir") / name


def build_cache(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Preprocess every valid manifest row into the PNG cache. Returns the cache index.

    Resumable: rows whose cache file (hash + config fingerprint) already exists are
    reused from the previous index without recomputation. Per-image failures are
    reported and skipped, never aborting the build.
    """
    dataset_dir = resolve_path(cfg, "dataset_dir")
    cache_dir = resolve_path(cfg, "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_index_path(cfg)

    prev_by_hash: dict[str, dict] = {}
    if index_path.exists():
        prev = pd.read_csv(index_path, keep_default_na=False)
        prev_by_hash = {r["hash"]: r for _, r in prev.iterrows()}

    records = []
    failures = []
    reused = 0
    # itertuples() mangles the reserved-word column name "class" — rename for iteration
    todo = manifest[manifest["valid"]].rename(columns={"class": "cls"})
    for i, row in enumerate(todo.itertuples(), 1):
        out_path = cache_path_for(row.hash, cfg)
        prev_row = prev_by_hash.get(row.hash)
        if prev_row is not None and prev_row["cache_file"] == out_path.name and out_path.exists():
            records.append(
                {
                    "relpath": row.relpath,
                    "hash": row.hash,
                    "class": row.cls,
                    "run": row.run,
                    "cache_file": out_path.name,
                    "x0": prev_row["x0"],
                    "y0": prev_row["y0"],
                    "x1": prev_row["x1"],
                    "y1": prev_row["y1"],
                    "roi_confident": prev_row["roi_confident"],
                }
            )
            reused += 1
            continue
        try:
            img = cv2.imread(str(dataset_dir / row.relpath))
            if img is None:
                raise ValueError("unreadable image")
            processed, meta = preprocess_image(img, cfg)
            if not cv2.imwrite(str(out_path), processed):
                raise IOError(f"failed to write {out_path}")
        except Exception as e:  # one bad file must never abort the whole build
            failures.append({"relpath": row.relpath, "error": f"{type(e).__name__}: {e}"})
            continue
        records.append(
            {
                "relpath": row.relpath,
                "hash": row.hash,
                "class": row.cls,
                "run": row.run,
                "cache_file": out_path.name,
                **meta,
            }
        )
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}")

    index = pd.DataFrame(records)
    index.to_csv(index_path, index=False)
    print(f"cache index: {len(index)} rows ({reused} reused) -> {index_path}")
    if failures:
        print(f"WARNING: {len(failures)} file(s) failed preprocessing and were skipped:")
        for f in failures:
            print(f"  {f['relpath']}: {f['error']}")
    return index


def write_spotcheck_sheet(index: pd.DataFrame, cfg: dict, out_html: Path, thumb_width: int = 440) -> None:
    """HTML contact sheet: ROI bbox drawn on one image per run + every low-confidence detection."""
    dataset_dir = resolve_path(cfg, "dataset_dir")
    sample = index.groupby("run", group_keys=False).head(1)
    lowconf = index[~index["roi_confident"]]
    chosen = (
        pd.concat([sample, lowconf])
        .drop_duplicates(subset="relpath")
        .sort_values(["run", "relpath"])
        .rename(columns={"class": "cls"})
    )

    cells = []
    for row in chosen.itertuples():
        img = cv2.imread(str(dataset_dir / row.relpath))
        if img is None:
            print(f"  WARN: could not read {row.relpath}, skipped in spot-check sheet")
            continue
        cropped = crop_osd(img, cfg["preprocess"]["osd_crop_bottom_frac"])
        color = (0, 255, 0) if row.roi_confident else (0, 0, 255)
        cv2.rectangle(cropped, (row.x0, row.y0), (row.x1, row.y1), color, 12)
        s = thumb_width / cropped.shape[1]
        thumb = cv2.resize(cropped, (thumb_width, int(cropped.shape[0] * s)))
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            print(f"  WARN: JPEG encode failed for {row.relpath}, skipped in spot-check sheet")
            continue
        b64 = base64.b64encode(buf).decode()
        tag = "" if row.roi_confident else " — <b style='color:red'>FALLBACK</b>"
        cells.append(
            f"<div style='display:inline-block;margin:4px;text-align:center'>"
            f"<img src='data:image/jpeg;base64,{b64}' width='{thumb_width}'><br>"
            f"<small>{row.run} · {row.cls} · {Path(row.relpath).name}{tag}</small></div>"
        )
    n_low = len(lowconf)
    html = (
        f"<html><body style='font-family:sans-serif;background:#222;color:#eee'>"
        f"<h2>ROI spot-check — one per run + all fallbacks ({len(chosen)} shown, {n_low} fallback)</h2>"
        f"<p>Green box = detected ROI (padded). Red box = low-confidence fallback central crop.</p>"
        + "".join(cells)
        + "</body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"spot-check sheet: {len(chosen)} images ({n_low} fallback) -> {out_html}")


def main() -> None:
    cfg = load_config()
    manifest_path = resolve_path(cfg, "manifests_dir") / "manifest.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    index = build_cache(manifest, cfg)
    n_conf = int(index["roi_confident"].sum())
    print(f"ROI confident: {n_conf}/{len(index)} ({len(index) - n_conf} fallback)")
    write_spotcheck_sheet(index, cfg, resolve_path(cfg, "artifacts_dir") / "eda" / "roi_spotcheck.html")


if __name__ == "__main__":
    main()
