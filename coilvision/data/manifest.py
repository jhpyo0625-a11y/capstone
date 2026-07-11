"""Parse filenames into the manifest — single source of truth downstream (spec §6.1 step 2).

One row per image: path, class, run, part#, shot#, code, dims, file hash, layout, ahash.
run/part#/shot#/code are provenance only — never model inputs or label logic
(run is additionally used for split grouping).

Run as a script to (re)build artifacts/manifests/manifest.csv:
    uv run python -m coilvision.data.manifest
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coilvision.config import load_config, resolve_path

THUMB_SIZE = (32, 24)  # (w, h) grayscale thumbnail for layout clustering
AHASH_SIZE = 8  # 8x8 -> 64-bit average hash for near-duplicate detection


def parse_filename(name: str, pattern: str) -> dict | None:
    """Parse one BMP filename into provenance fields, or None if it doesn't match."""
    m = re.match(pattern, name)
    if m is None:
        return None
    d = m.groupdict()
    return {"run": d["run"], "part": int(d["part"]), "shot": int(d["shot"]), "code": int(d["code"])}


def class_from_relpath(relpath: Path) -> str | None:
    """Label folder is ground truth: Pass/, Fail/Dent/, Fail/Loose/."""
    parts = relpath.parts
    if parts[0] == "Pass":
        return "Pass"
    if parts[0] == "Fail" and len(parts) > 2 and parts[1] in ("Dent", "Loose"):
        return parts[1]
    return None


def ahash(gray_thumb: np.ndarray) -> str:
    small = cv2.resize(gray_thumb, (AHASH_SIZE, AHASH_SIZE), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def cluster_layouts(run_signatures: dict[str, np.ndarray], corr_threshold: float = 0.90) -> dict[str, str]:
    """Greedy clustering of per-run thumbnail signatures by normalized correlation."""
    centroids: list[np.ndarray] = []
    assignment: dict[str, str] = {}
    for run in sorted(run_signatures):
        sig = run_signatures[run]
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        best_i, best_corr = None, corr_threshold
        for i, c in enumerate(centroids):
            corr = float(np.dot(sig, c) / len(sig))
            if corr > best_corr:
                best_i, best_corr = i, corr
        if best_i is None:
            centroids.append(sig)
            best_i = len(centroids) - 1
        assignment[run] = f"L{best_i}"
    return assignment


def build_manifest(cfg: dict) -> pd.DataFrame:
    dataset_dir = resolve_path(cfg, "dataset_dir")
    pattern = cfg["data"]["filename_pattern"]
    exp_w, exp_h = cfg["data"]["expected_width"], cfg["data"]["expected_height"]

    rows = []
    thumbs: dict[int, np.ndarray] = {}  # row index -> gray thumb, for layout clustering
    for path in sorted(dataset_dir.rglob("*.bmp")):
        relpath = path.relative_to(dataset_dir)
        issues = []
        cls = class_from_relpath(relpath)
        if cls is None:
            issues.append("unknown_label_folder")
        parsed = parse_filename(path.name, pattern)
        if parsed is None:
            issues.append("unparseable_filename")
            parsed = {"run": "", "part": -1, "shot": -1, "code": -1}

        data = path.read_bytes()
        file_hash = hashlib.blake2b(data, digest_size=16).hexdigest()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            issues.append("unreadable")
            w = h = -1
            hash_a = ""
        else:
            h, w = img.shape[:2]
            if (w, h) != (exp_w, exp_h):
                issues.append(f"unexpected_dims_{w}x{h}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA)
            # normalize per image so exposure differences (the [code] variants)
            # don't fragment layout clusters — structure is what matters
            t = thumb.astype(np.float32)
            thumbs[len(rows)] = (t - t.mean()) / (t.std() + 1e-9)
            hash_a = ahash(thumb)

        rows.append(
            {
                "relpath": str(relpath),
                "filename": path.name,
                "class": cls or "",
                **parsed,
                "width": w,
                "height": h,
                "hash": file_hash,
                "ahash": hash_a,
                "issues": ";".join(issues),
                "valid": not issues,
            }
        )

    df = pd.DataFrame(rows)

    # Layout cluster per run, from the average thumbnail of that run's images
    run_sigs: dict[str, np.ndarray] = {}
    for run, group in df[df["run"] != ""].groupby("run"):
        sigs = [thumbs[i].flatten() for i in group.index if i in thumbs]
        if sigs:
            run_sigs[run] = np.mean(sigs, axis=0)
    layout_by_run = cluster_layouts(run_sigs)
    df["layout"] = df["run"].map(layout_by_run).fillna("")
    return df


def main() -> None:
    cfg = load_config()
    df = build_manifest(cfg)
    out_dir = resolve_path(cfg, "manifests_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.csv"
    df.to_csv(out_path, index=False)

    print(f"manifest: {len(df)} rows -> {out_path}")
    print(f"valid: {int(df['valid'].sum())} / {len(df)}")
    if not df["valid"].all():
        print(df.loc[~df["valid"], ["relpath", "issues"]].to_string(index=False))
    print("\nclass counts:")
    print(df["class"].value_counts().to_string())
    print(f"\nruns: {df.loc[df['run'] != '', 'run'].nunique()}")
    print("\nlayout clusters (runs / images):")
    by_layout = df[df["layout"] != ""].groupby("layout").agg(runs=("run", "nunique"), images=("relpath", "count"))
    print(by_layout.to_string())


if __name__ == "__main__":
    main()
