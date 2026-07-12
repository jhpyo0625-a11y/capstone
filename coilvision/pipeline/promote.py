"""Package a trained patch head as the production model (spec §6.5).

Copies head.joblib into models/production/ and writes POINTER.json recording
which run, when, why, the operating threshold, and the preprocess fingerprint
that predict must match. Phase 6's auto-promotion gate will call promote()
after comparing candidate vs production on the frozen test set.

Run:  uv run python -m coilvision.pipeline.promote --head <path> \
          --from-val <val_scores.csv> --recall-target 1.0 --why "..."
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import pandas as pd

from coilvision.anomaly import anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.data.preprocess import preprocess_fingerprint
from coilvision.eval.report import select_threshold


def promote(head_path: Path, threshold_info: dict, why: str, cfg: dict) -> Path:
    bundle = joblib.load(head_path)
    current_fp = preprocess_fingerprint(cfg)
    if bundle.get("preprocess_fingerprint") != current_fp:
        raise RuntimeError(
            f"refusing to promote: head was trained under preprocess fingerprint "
            f"{bundle.get('preprocess_fingerprint')} but current config is {current_fp}"
        )
    prod = resolve_path(cfg, "production_dir")
    prod.mkdir(parents=True, exist_ok=True)
    # spec §6.6: promote swaps production, PREVIOUS KEPT — archive before overwrite
    old_pointer = prod / "POINTER.json"
    if old_pointer.exists():
        old = json.loads(old_pointer.read_text(encoding="utf-8"))
        stamp = old.get("promoted_at", "unknown").replace(":", "").replace(" ", "_")
        archive = resolve_path(cfg, "models_dir") / "archive" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("head.joblib", "POINTER.json"):
            if (prod / name).exists():
                shutil.move(str(prod / name), str(archive / name))
        print(f"previous production archived -> {archive}")
    shutil.copy2(head_path, prod / "head.joblib")
    pointer = {
        "source_head": str(head_path),
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "why": why,
        "threshold": threshold_info["threshold"],
        "threshold_policy": threshold_info.get("policy", ""),
        "val_fail_recall": threshold_info.get("fail_recall"),
        "val_false_reject_rate": threshold_info.get("false_reject_rate"),
        "aggregation": f"top{cfg['patchclf']['top_k']}",
        "classes": bundle["classes"],
        "preprocess_version": cfg["preprocess"]["version"],
        "preprocess_fingerprint": current_fp,
        "input_size": list(cfg["preprocess"]["resize"]),
    }
    (prod / "POINTER.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    print(f"promoted {head_path.name} -> {prod}")
    print(json.dumps(pointer, indent=2))
    return prod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", required=True, help="path to head.joblib of the candidate")
    ap.add_argument("--from-val", required=True, help="val_scores.csv of the same run (threshold source)")
    ap.add_argument("--recall-target", type=float, default=None,
                    help="default: eval.production_recall_target from config")
    ap.add_argument("--why", required=True)
    args = ap.parse_args()

    cfg = anomaly_cfg(load_config())
    target = args.recall_target if args.recall_target is not None else cfg["eval"]["production_recall_target"]
    val = pd.read_csv(args.from_val, keep_default_na=False)
    col = f"score_top{cfg['patchclf']['top_k']}"
    op = select_threshold(val[col].to_numpy(), (val["class"] != "Pass").to_numpy(), target)
    op["policy"] = f"val fail-recall >= {target} on {col}"
    promote(Path(args.head), op, args.why, cfg)


if __name__ == "__main__":
    main()
