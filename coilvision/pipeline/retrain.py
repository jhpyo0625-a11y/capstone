"""Retraining orchestrator: ingest → merge → split → train → gate → promote (spec §6.6).

Steps:
  1. validate + quarantine bad incoming files (unreadable, wrong dims,
     unparseable name, unknown folder, exact-duplicate content)
  2. merge accepted files into data_accepted/, rebuild manifest + preprocess cache
  3. re-split train/val around the FROZEN test manifest (new images whose run
     belongs to a frozen test run are excluded entirely and reported)
  4. train a candidate patch head (annotated defects + Pass negatives + mining)
  5. evaluate candidate AND current production on the frozen test set, each at
     its own operating threshold (candidate: fresh val by production policy;
     production: its POINTER threshold)
  6. GATE: promote iff fail-recall(candidate) >= fail-recall(production)
           AND macro-F1(candidate) > macro-F1(production)
  7. promote (previous archived) or reject (candidate stays in its run dir)
  8. write PIPELINE_REPORT.md either way; update pipeline state

Idempotent and resumable: every step reuses caches; a crashed run can simply be
rerun. A lock file prevents concurrent runs (stale locks are broken after
retrain.lock_stale_hours).

Run:  uv run python -m coilvision.pipeline.retrain [--force]
      (--force runs even when the incoming threshold isn't met)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

from coilvision.annotate import build_page
from coilvision.anomaly import PatchExtractor, anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.data import manifest as manifest_mod
from coilvision.data import split as split_mod
from coilvision.data.preprocess import build_cache
from coilvision.data.validate import quarantine_file, validate_file
from coilvision.eval.report import classify, select_threshold, split_metrics
from coilvision.pipeline.promote import promote
from coilvision.train.datamodule import load_split_frame
from coilvision.train.patchclf import score_images, train_candidate


# ---------- lock & state ----------

def acquire_lock(cfg: dict) -> Path:
    lock = resolve_path(cfg, "artifacts_dir") / "retrain.lock"
    if lock.exists():
        age_h = (time.time() - lock.stat().st_mtime) / 3600
        if age_h < cfg["retrain"]["lock_stale_hours"]:
            raise RuntimeError(f"another retrain is running (lock {lock}, {age_h:.1f}h old)")
        print(f"WARNING: breaking stale lock ({age_h:.1f}h old -- crashed run?)")
        lock.unlink()
    lock.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    return lock


def state_path(cfg: dict) -> Path:
    return resolve_path(cfg, "artifacts_dir") / "pipeline_state.json"


def read_state(cfg: dict) -> dict:
    p = state_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------- step 1-2: ingest ----------

def ingest_incoming(cfg: dict, known_hashes: set[str]) -> dict:
    incoming = resolve_path(cfg, "incoming_dir")
    accepted = resolve_path(cfg, "accepted_dir")
    dataset_root = resolve_path(cfg, "dataset_dir")
    quarantine = resolve_path(cfg, "quarantine_dir")
    stats = {"accepted": 0, "quarantined": 0, "duplicates": 0, "by_class": Counter(), "reasons": Counter()}
    for path in sorted(incoming.rglob("*.bmp")):
        rel = path.relative_to(incoming)
        issues = validate_file(path, cfg, relpath=rel)
        if issues:
            quarantine_file(path, issues, quarantine)
            stats["quarantined"] += 1
            stats["reasons"].update(issues)
            continue
        file_hash = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        if file_hash in known_hashes:
            quarantine_file(path, ["duplicate_content"], quarantine)
            stats["duplicates"] += 1
            continue
        # relpath must stay globally unique across BOTH roots: annotations and
        # joins key on it. Same name + different bytes (dedupe already passed)
        # is suspicious — renaming would break the filename schema, so a human
        # should look at it instead.
        dest = accepted / rel
        if dest.exists() or (dataset_root / rel).exists():
            quarantine_file(path, ["name_collision"], quarantine)
            stats["quarantined"] += 1
            stats["reasons"]["name_collision"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        known_hashes.add(file_hash)
        stats["accepted"] += 1
        stats["by_class"][str(rel.parts[0] if rel.parts[0] != "Fail" else "/".join(rel.parts[:2]))] += 1
    return stats


def unmanifested_accepted(cfg: dict, manifest: pd.DataFrame) -> int:
    """Files sitting in data_accepted/ that the manifest doesn't know yet —
    the signature of a run that crashed after ingest. A rerun must proceed."""
    accepted = resolve_path(cfg, "accepted_dir")
    on_disk = sum(1 for _ in accepted.rglob("*.bmp")) if accepted.exists() else 0
    in_manifest = int((manifest["root"] == "accepted").sum()) if "root" in manifest.columns else 0
    return on_disk - in_manifest


# ---------- step 5-6: gate ----------

def evaluate_head(cfg: dict, extractor: PatchExtractor, head, frame: pd.DataFrame, threshold: float) -> dict:
    variants, votes, _ = score_images(cfg, extractor, head, frame)
    df = frame.copy()
    df["score"] = variants[f"top{cfg['patchclf']['top_k']}"]
    df["predicted"] = classify(df["score"].to_numpy(), votes, threshold)
    return split_metrics(df, threshold)


def gate_decision(candidate: dict, production: dict) -> tuple[bool, str]:
    """Spec §6.6: promote iff fail-recall not worse AND macro-F1 strictly better."""
    recall_ok = candidate["fail_recall"] >= production["fail_recall"]
    f1_ok = candidate["macro_f1"] > production["macro_f1"]
    reason = (f"candidate recall {candidate['fail_recall']:.3f} vs production {production['fail_recall']:.3f} "
              f"({'ok' if recall_ok else 'WORSE'}); "
              f"candidate macro-F1 {candidate['macro_f1']:.3f} vs {production['macro_f1']:.3f} "
              f"({'better' if f1_ok else 'not better'})")
    return recall_ok and f1_ok, reason


# ---------- orchestrator ----------

def run(force: bool = False, trigger: str = "manual") -> dict:
    base_cfg = load_config()
    cfg = anomaly_cfg(base_cfg)
    lock = acquire_lock(cfg)
    t0 = time.time()
    run_id = time.strftime("retrain_%Y%m%d_%H%M%S")
    report_dir = resolve_path(cfg, "artifacts_dir") / "runs" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Pipeline report — {run_id}", "", f"Trigger: {trigger}", ""]
    try:
        manifests_dir = resolve_path(cfg, "manifests_dir")
        old_manifest = pd.read_csv(manifests_dir / "manifest.csv", keep_default_na=False)

        # 1: ingest
        stats = ingest_incoming(cfg, set(old_manifest["hash"]))
        lines += ["## Ingest", "",
                  f"accepted {stats['accepted']} (by class: {dict(stats['by_class'])}), "
                  f"quarantined {stats['quarantined']} (reasons: {dict(stats['reasons'])}), "
                  f"duplicates {stats['duplicates']}", ""]
        print(f"ingest: +{stats['accepted']} accepted, {stats['quarantined']} quarantined, "
              f"{stats['duplicates']} duplicates")
        drift = unmanifested_accepted(cfg, old_manifest)
        if drift > 0:
            lines += [f"Resuming unfinished merge: {drift} accepted file(s) not yet in the manifest.", ""]
            print(f"resuming unfinished merge: {drift} accepted file(s) not yet manifested")
        if stats["accepted"] == 0 and drift <= 0 and not force:
            raise RuntimeError("nothing accepted from incoming/, no unfinished merge, and --force not given")

        # 2: manifest + cache
        manifest = manifest_mod.build_manifest(cfg)
        manifest.to_csv(manifests_dir / "manifest.csv", index=False)
        lines += ["## Dataset", "", f"{len(manifest)} images "
                  f"({(manifest['root'] == 'accepted').sum()} accepted-incoming); "
                  f"classes: {manifest['class'].value_counts().to_dict()}", ""]
        build_cache(manifest, cfg)

        # 3: splits around frozen test
        v = cfg["split"]["version"]
        test_file = pd.read_csv(manifests_dir / f"test_v{v}.csv", keep_default_na=False)
        frozen_runs = sorted(test_file["run"].unique())
        splits = split_mod.build_splits(manifest, cfg, frozen_test_runs=frozen_runs)
        for name in ("train", "val"):
            splits[splits["split"] == name].to_csv(manifests_dir / f"{name}_v{v}.csv", index=False)
        orphans = set(splits.loc[splits["split"] == "test", "relpath"]) - set(test_file["relpath"])
        if orphans:
            lines += [f"**WARNING**: {len(orphans)} new image(s) belong to frozen-test runs and are "
                      "EXCLUDED from train/val (test_v1 is frozen):", ""]
            lines += [f"- `{o}`" for o in sorted(orphans)] + [""]
        summ = split_mod.summary(splits)
        lines += ["## Splits", "", "```", summ.to_string(), "```", ""]

        # 4: train candidate
        cand = train_candidate(cfg)
        best_auc = cand["val_image_fail_auc"][f"top{cfg['patchclf']['top_k']}"]
        lines += ["## Candidate", "",
                  f"run `{cand['run_id']}`; val fail-AUC (top{cfg['patchclf']['top_k']}): {best_auc:.4f}; "
                  f"dent/loose vote acc: {cand['dent_vs_loose_val_acc']:.3f}", ""]
        if cand["n_unannotated_train_defects"]:
            page = resolve_path(cfg, "artifacts_dir") / "annotation" / f"annotate_{run_id}.html"
            frame = load_split_frame("train", cfg).set_index("relpath")
            rows = [{"relpath": r, "class": frame.loc[r, "class"], "cache_file": frame.loc[r, "cache_file"]}
                    for r in cand["unannotated_relpaths"]]
            build_page(rows, resolve_path(cfg, "cache_dir"), page,
                       f"Coil defect annotation — new images ({run_id})", f"annotations_train_{run_id}.json")
            lines += [f"**ACTION NEEDED**: {len(rows)} un-annotated train defect image(s) — "
                      f"annotate via `{page}` (they currently add no patch supervision)", ""]

        # 5: evaluate candidate + production on frozen test
        extractor = PatchExtractor(cfg)
        test_frame = load_split_frame("test", cfg)
        val_scores = pd.read_csv(cand["val_scores_path"], keep_default_na=False)
        col = f"score_top{cfg['patchclf']['top_k']}"
        op = select_threshold(val_scores[col].to_numpy(), (val_scores["class"] != "Pass").to_numpy(),
                              base_cfg["eval"]["production_recall_target"])
        cand_head = joblib.load(cand["head_path"])["head"]
        print("evaluating candidate on frozen test ...")
        cand_metrics = evaluate_head(cfg, extractor, cand_head, test_frame, op["threshold"])

        prod_dir = resolve_path(cfg, "production_dir")
        pointer = json.loads((prod_dir / "POINTER.json").read_text(encoding="utf-8"))
        prod_head = joblib.load(prod_dir / "head.joblib")["head"]
        print("evaluating production on frozen test ...")
        prod_metrics = evaluate_head(cfg, extractor, prod_head, test_frame, pointer["threshold"])

        # 6-7: gate
        promoted, reason = gate_decision(cand_metrics, prod_metrics)
        lines += ["## Gate (frozen test)", "",
                  "| model | threshold | fail-recall | FRR | macro-F1 | fail-AUC |",
                  "|---|---|---|---|---|---|",
                  f"| candidate | {op['threshold']:.4f} | {cand_metrics['fail_recall']:.3f} | "
                  f"{cand_metrics['false_reject_rate']:.3f} | {cand_metrics['macro_f1']:.3f} | {cand_metrics['fail_auc']:.4f} |",
                  f"| production | {pointer['threshold']:.4f} | {prod_metrics['fail_recall']:.3f} | "
                  f"{prod_metrics['false_reject_rate']:.3f} | {prod_metrics['macro_f1']:.3f} | {prod_metrics['fail_auc']:.4f} |",
                  "", f"Decision: **{'PROMOTE' if promoted else 'REJECT'}** — {reason}", ""]
        print(f"gate: {'PROMOTE' if promoted else 'REJECT'} -- {reason}")
        if promoted:
            op["policy"] = f"val fail-recall >= {base_cfg['eval']['production_recall_target']} on {col}"
            promote(Path(cand["head_path"]), op, f"auto-promotion by {run_id}: {reason}", cfg)
        else:
            lines += [f"Candidate kept (not promoted) at `{cand['run_dir']}`", ""]

        # 8: state + report
        state = read_state(cfg)
        state.update({"last_success": time.strftime("%Y-%m-%d %H:%M:%S"), "last_run_id": run_id,
                      "last_decision": "PROMOTE" if promoted else "REJECT"})
        state_path(cfg).write_text(json.dumps(state, indent=2), encoding="utf-8")
        lines += [f"Wall time: {(time.time() - t0) / 60:.1f} min"]
        result = {"run_id": run_id, "promoted": promoted, "reason": reason,
                  "candidate": cand_metrics, "production": prod_metrics, "ingest": {k: v for k, v in stats.items() if k != "by_class"}}
        (report_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result
    except Exception as e:
        lines += ["", f"## FAILED", "", f"`{type(e).__name__}: {e}`"]
        raise
    finally:
        (report_dir / "PIPELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"report -> {report_dir / 'PIPELINE_REPORT.md'}")
        lock.unlink(missing_ok=True)


def main() -> None:
    run(force="--force" in sys.argv[1:])


if __name__ == "__main__":
    main()
