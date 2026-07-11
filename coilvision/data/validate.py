"""Schema/integrity checks + quarantine flow (spec §6.1 step 1).

Quarantines unreadable BMPs, wrong dimensions, unparseable filenames.
No filename-based label-conflict check — filenames carry no part identity (spec §2).

Raw `Coil-image-Dataset/` is read-only: validation there is report-only (issues land
in the manifest). Moving files into quarantine/ applies to `incoming/` ingest.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from coilvision.data.manifest import class_from_relpath, parse_filename


def validate_file(path: Path, cfg: dict, relpath: Path | None = None) -> list[str]:
    """Return a list of issues (empty = valid). Does not move anything."""
    issues = []
    if relpath is not None and class_from_relpath(relpath) is None:
        issues.append("unknown_label_folder")
    if parse_filename(path.name, cfg["data"]["filename_pattern"]) is None:
        issues.append("unparseable_filename")
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        issues.append("unreadable")
    else:
        h, w = img.shape[:2]
        if (w, h) != (cfg["data"]["expected_width"], cfg["data"]["expected_height"]):
            issues.append(f"unexpected_dims_{w}x{h}")
    return issues


def quarantine_file(path: Path, reasons: list[str], quarantine_dir: Path) -> Path:
    """Move a bad incoming file into quarantine/ and append to the reason log."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / path.name
    n = 1
    while dest.exists():
        dest = quarantine_dir / f"{path.stem}_{n}{path.suffix}"
        n += 1
    shutil.move(str(path), str(dest))
    log = quarantine_dir / "quarantine_log.csv"
    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp", "original_path", "quarantined_as", "reasons"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), str(path), dest.name, ";".join(reasons)])
    return dest
