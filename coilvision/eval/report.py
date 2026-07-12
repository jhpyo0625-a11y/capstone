"""Formal evaluation report (spec §6.4): threshold from val, one-shot test report.

Protocol: the operating threshold is the highest score threshold achieving
fail-recall >= eval.fail_recall_target on VAL (i.e. lowest false-reject rate at
target recall). The frozen test set is then evaluated ONCE at that threshold.

Outputs to artifacts/runs/eval_<ts>/:
  metrics.json         thresholds + val/test metrics + config/head provenance
  per_run.csv          per-production-run breakdown (both splits)
  curve.png            fail-recall vs false-reject-rate curves, operating point
  confusion_test.png   3-class confusion matrix on test
  gallery.html         P(fail)-heatmap gallery: every test defect, every
                       misclassification, sampled correct passes

Run:  uv run python -m coilvision.eval.report
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from coilvision.anomaly import PatchExtractor, anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.data.preprocess import preprocess_fingerprint
from coilvision.train.datamodule import load_split_frame
from coilvision.train.patchclf import score_images

CLASS_ORDER = ["Pass", "Dent", "Loose"]


def select_threshold(scores: np.ndarray, is_fail: np.ndarray, target: float) -> dict:
    """Highest threshold with fail-recall >= target == lowest FRR at target recall."""
    if not is_fail.any():
        raise ValueError("no defect images in the threshold-selection split")
    best = None
    for thr in np.sort(np.unique(scores[is_fail])):
        recall = float((scores[is_fail] >= thr).mean())
        if recall >= target:
            best = {
                "threshold": float(thr),
                "fail_recall": recall,
                "false_reject_rate": float((scores[~is_fail] >= thr).mean()),
            }
    if best is None:
        raise ValueError(f"no threshold achieves fail-recall >= {target}")
    return best


def classify(scores: np.ndarray, votes: list[str], threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, votes, "Pass")


def split_metrics(df: pd.DataFrame, threshold: float) -> dict:
    is_fail = (df["class"] != "Pass").to_numpy()
    scores = df["score"].to_numpy()
    pred = df["predicted"].to_numpy()
    rep = classification_report(df["class"], pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    return {
        "images": int(len(df)),
        "fail_auc": float(roc_auc_score(is_fail, scores)),
        "fail_recall": float((scores[is_fail] >= threshold).mean()),
        "false_reject_rate": float((scores[~is_fail] >= threshold).mean()),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "per_class": {c: {k: round(rep[c][k], 4) for k in ("precision", "recall", "f1-score", "support")} for c in CLASS_ORDER},
        "confusion": confusion_matrix(df["class"], pred, labels=CLASS_ORDER).tolist(),
    }


def per_run_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (split, run), g in df.groupby(["split", "run"]):
        defects = g[g["class"] != "Pass"]
        passes = g[g["class"] == "Pass"]
        rows.append(
            {
                "split": split,
                "run": run,
                "images": len(g),
                "defects": len(defects),
                "missed_defects": int((defects["score"] < threshold).sum()),
                "passes": len(passes),
                "false_rejects": int((passes["score"] >= threshold).sum()),
                "mean_score": round(float(g["score"].mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "run"])


def recall_frr_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    is_fail = (df["class"] != "Pass").to_numpy()
    scores = df["score"].to_numpy()
    thrs = np.sort(np.unique(scores))
    recall = np.array([(scores[is_fail] >= t).mean() for t in thrs])
    frr = np.array([(scores[~is_fail] >= t).mean() for t in thrs])
    return frr, recall


def plot_curves(val: pd.DataFrame, test: pd.DataFrame, op: dict, target: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, df, color in (("val", val, "#2a7"), ("test", test, "#d33")):
        frr, rec = recall_frr_curve(df)
        ax.plot(frr, rec, label=name, color=color)
    ax.axhline(target, color="#888", ls="--", lw=1, label=f"recall target {target}")
    ax.plot(op["false_reject_rate"], op["fail_recall"], "k*", ms=14, label="operating point (val)")
    ax.set_xlabel("false-reject rate")
    ax.set_ylabel("fail-recall")
    ax.set_title("fail-recall vs false-reject rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_confusion(cm: list[list[int]], out: Path) -> None:
    m = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(3), CLASS_ORDER)
    ax.set_yticks(range(3), CLASS_ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_title("test confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def write_gallery(df: pd.DataFrame, maps: list[np.ndarray], cfg: dict, out_html: Path,
                  n_correct_pass: int = 10, seed: int = 0) -> None:
    """Every defect + every misclassification + sampled correct passes, with P(fail) overlays."""
    cache_dir = resolve_path(cfg, "cache_dir")
    df = df.reset_index(drop=True)
    wrong = df["class"] != df["predicted"]
    is_defect = df["class"] != "Pass"
    correct_pass = df.index[(~is_defect) & (~wrong)]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(correct_pass, size=min(n_correct_pass, len(correct_pass)), replace=False)
    chosen = sorted(set(df.index[is_defect]) | set(df.index[wrong]) | set(sampled))

    cells = []
    for i in chosen:
        r = df.loc[i]
        img = cv2.imread(str(cache_dir / r["cache_file"]))
        if img is None:
            print(f"  WARN: unreadable {r['cache_file']}, skipped in gallery")
            continue
        heat = (np.clip(maps[i], 0, 1) * 255).astype(np.uint8)  # absolute P(fail) scale
        heat = cv2.applyColorMap(cv2.resize(heat, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
        overlay = cv2.resize(overlay, (900, int(900 * img.shape[0] / img.shape[1])))
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        verdict = "CORRECT" if r["class"] == r["predicted"] else "WRONG"
        color = "#7f7" if verdict == "CORRECT" else "#f77"
        cells.append(
            f"<div style='display:inline-block;margin:5px;text-align:center'>"
            f"<img src='data:image/jpeg;base64,{base64.b64encode(buf).decode()}' width='900'><br>"
            f"<small>true <b>{r['class']}</b> · predicted <b style='color:{color}'>{r['predicted']}</b>"
            f" · score {r['score']:.4f} · {Path(r['relpath']).name}</small></div>"
        )
    html = ("<html><body style='font-family:sans-serif;background:#1b1b1b;color:#eee'>"
            f"<h2>Test gallery — P(fail) heatmaps ({len(cells)} images: all defects, all errors, "
            f"{len(sampled)} correct passes)</h2>" + "".join(cells) + "</body></html>")
    out_html.write_text(html, encoding="utf-8")
    print(f"gallery: {len(cells)} images -> {out_html}")


def main() -> None:
    t0 = time.time()
    base_cfg = load_config()
    cfg = anomaly_cfg(base_cfg)
    gate_target = base_cfg["eval"]["fail_recall_target"]
    policy_target = base_cfg["eval"]["production_recall_target"]
    top_k = base_cfg["patchclf"]["top_k"]
    out_dir = resolve_path(cfg, "artifacts_dir") / "runs" / time.strftime("eval_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    heads = sorted((resolve_path(cfg, "artifacts_dir") / "runs").glob("patchclf_*/head.joblib"))
    if not heads:
        raise FileNotFoundError("no trained patchclf head found — run coilvision.train.patchclf first")
    head_path = heads[-1]
    bundle = joblib.load(head_path)
    current_fp = preprocess_fingerprint(cfg)
    if bundle.get("preprocess_fingerprint") != current_fp:
        raise RuntimeError(
            f"preprocess-version mismatch: head {head_path} was trained under fingerprint "
            f"{bundle.get('preprocess_fingerprint')} but current config produces {current_fp}"
        )
    head = bundle["head"]
    print(f"head: {head_path}")

    extractor = PatchExtractor(cfg)
    frames, all_maps = {}, {}
    for split in ("val", "test"):
        frame = load_split_frame(split, cfg)
        print(f"scoring {split} ({len(frame)} images) ...")
        variants, votes, maps = score_images(cfg, extractor, head, frame)
        frame["score"] = variants[f"top{top_k}"]
        frame["vote"] = votes
        frame["split"] = split
        frames[split] = frame
        all_maps[split] = maps

    # operating point = the PRODUCTION policy (val-recall >= production_recall_target),
    # so this report and the promotion gate measure at the threshold production actually runs
    op = select_threshold(frames["val"]["score"].to_numpy(),
                          (frames["val"]["class"] != "Pass").to_numpy(), policy_target)
    print(f"\noperating point from VAL (policy: recall>={policy_target}): thr={op['threshold']:.4f} "
          f"recall={op['fail_recall']:.3f} frr={op['false_reject_rate']:.3f}")

    for split in ("val", "test"):
        frames[split]["predicted"] = classify(frames[split]["score"].to_numpy(),
                                              list(frames[split]["vote"]), op["threshold"])
    metrics = {split: split_metrics(frames[split], op["threshold"]) for split in ("val", "test")}

    both = pd.concat([frames["val"], frames["test"]])
    runs = per_run_table(both, op["threshold"])
    runs.to_csv(out_dir / "per_run.csv", index=False)
    plot_curves(frames["val"], frames["test"], op, gate_target, out_dir / "curve.png")
    plot_confusion(metrics["test"]["confusion"], out_dir / "confusion_test.png")
    write_gallery(frames["test"], all_maps["test"], cfg, out_dir / "gallery.html")
    both[["relpath", "class", "run", "split", "score", "vote", "predicted"]].to_csv(
        out_dir / "image_scores.csv", index=False)

    result = {
        "head": str(head_path),
        "preprocess_fingerprint": current_fp,
        "top_k": top_k,
        "operating_point_val": op,
        "threshold_policy_recall_target": policy_target,
        "gate_recall_target": gate_target,
        "gate_met": metrics["test"]["fail_recall"] >= gate_target,
        "accepted_frr": base_cfg["eval"].get("accepted_frr"),
        "metrics": metrics,
        "wall_time_min": round((time.time() - t0) / 60, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== TEST (one-shot, frozen) ===")
    t = metrics["test"]
    print(f"fail-AUC {t['fail_auc']:.4f} | fail-recall {t['fail_recall']:.3f} "
          f"| FRR {t['false_reject_rate']:.3f} | macro-F1 {t['macro_f1']:.3f}")
    print("confusion (rows=true Pass/Dent/Loose):")
    for row, name in zip(t["confusion"], CLASS_ORDER):
        print(f"  {name:6s} {row}")
    print("\nper-run breakdown:")
    print(runs.to_string(index=False))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
