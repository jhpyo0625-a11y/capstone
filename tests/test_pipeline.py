"""Retrain pipeline units: ingest/quarantine/dedupe, gate decision, watcher trigger, lock."""

import copy
import hashlib
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytest

from coilvision.anomaly import anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.pipeline.retrain import acquire_lock, gate_decision, ingest_incoming, unmanifested_accepted
from coilvision.pipeline.watcher import should_run

CFG = anomaly_cfg(load_config())
GOOD_NAME = "260401_120000_A35W_3-1 [1024].bmp"


def tmp_cfg(tmp_path):
    cfg = copy.deepcopy(CFG)
    for key, sub in (("incoming_dir", "incoming"), ("accepted_dir", "accepted"),
                     ("quarantine_dir", "quarantine"), ("artifacts_dir", "artifacts"),
                     ("dataset_dir", "dataset")):
        cfg["paths"][key] = str(tmp_path / sub)
    (tmp_path / "incoming" / "Pass").mkdir(parents=True)
    (tmp_path / "incoming" / "Fail" / "Dent").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "dataset").mkdir()
    return cfg


def write_frame(path, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (CFG["data"]["expected_height"], CFG["data"]["expected_width"], 3), dtype=np.uint8)
    cv2.imwrite(str(path), img.astype(np.uint8))
    return path


class TestIngest:
    def test_good_file_accepted_into_class_structure(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        write_frame(tmp_path / "incoming" / "Pass" / GOOD_NAME)
        stats = ingest_incoming(cfg, known_hashes=set())
        assert stats["accepted"] == 1 and stats["quarantined"] == 0
        assert (tmp_path / "accepted" / "Pass" / GOOD_NAME).exists()
        assert not (tmp_path / "incoming" / "Pass" / GOOD_NAME).exists()

    def test_bad_files_quarantined_with_reasons(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        (tmp_path / "incoming" / "Pass" / GOOD_NAME).write_bytes(b"junk")  # unreadable
        write_frame(tmp_path / "incoming" / "Fail" / "Dent" / "badname.bmp")  # unparseable
        stats = ingest_incoming(cfg, known_hashes=set())
        assert stats["accepted"] == 0 and stats["quarantined"] == 2
        assert stats["reasons"]["unreadable"] == 1
        assert stats["reasons"]["unparseable_filename"] == 1
        assert (tmp_path / "quarantine" / "quarantine_log.csv").exists()

    def test_duplicate_content_quarantined(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        p = write_frame(tmp_path / "incoming" / "Pass" / GOOD_NAME, seed=7)
        h = hashlib.blake2b(p.read_bytes(), digest_size=16).hexdigest()
        stats = ingest_incoming(cfg, known_hashes={h})
        assert stats["duplicates"] == 1 and stats["accepted"] == 0

    def test_wrong_folder_quarantined(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        (tmp_path / "incoming" / "Mystery").mkdir()
        write_frame(tmp_path / "incoming" / "Mystery" / GOOD_NAME)
        stats = ingest_incoming(cfg, known_hashes=set())
        assert stats["quarantined"] == 1
        assert stats["reasons"]["unknown_label_folder"] == 1

    def test_name_collision_with_dataset_root_quarantined(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        (tmp_path / "dataset" / "Pass").mkdir(parents=True)
        write_frame(tmp_path / "dataset" / "Pass" / GOOD_NAME, seed=1)  # existing dataset image
        write_frame(tmp_path / "incoming" / "Pass" / GOOD_NAME, seed=2)  # same name, different bytes
        stats = ingest_incoming(cfg, known_hashes=set())
        assert stats["accepted"] == 0
        assert stats["reasons"]["name_collision"] == 1
        assert not (tmp_path / "accepted" / "Pass" / GOOD_NAME).exists()

    def test_name_collision_with_prior_accepted_quarantined(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        (tmp_path / "accepted" / "Pass").mkdir(parents=True)
        write_frame(tmp_path / "accepted" / "Pass" / GOOD_NAME, seed=1)
        write_frame(tmp_path / "incoming" / "Pass" / GOOD_NAME, seed=2)
        stats = ingest_incoming(cfg, known_hashes=set())
        assert stats["reasons"]["name_collision"] == 1


class TestResume:
    def test_unmanifested_accepted_counts_drift(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        manifest = pd.DataFrame({"root": ["dataset", "accepted"], "hash": ["a", "b"]})
        assert unmanifested_accepted(cfg, manifest) == -1  # nothing on disk, 1 in manifest
        (tmp_path / "accepted" / "Pass").mkdir(parents=True)
        write_frame(tmp_path / "accepted" / "Pass" / GOOD_NAME)
        write_frame(tmp_path / "accepted" / "Pass" / "260401_130000_A35W_4-1 [16].bmp", seed=3)
        assert unmanifested_accepted(cfg, manifest) == 1  # 2 on disk, 1 manifested

    def test_old_manifest_without_root_column(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        (tmp_path / "accepted" / "Pass").mkdir(parents=True)
        write_frame(tmp_path / "accepted" / "Pass" / GOOD_NAME)
        manifest = pd.DataFrame({"hash": ["a"]})  # pre-Phase-6 manifest
        assert unmanifested_accepted(cfg, manifest) == 1


class TestGate:
    P = {"fail_recall": 0.90, "macro_f1": 0.80}

    def test_promotes_when_strictly_better(self):
        ok, reason = gate_decision({"fail_recall": 0.93, "macro_f1": 0.82}, self.P)
        assert ok and "better" in reason

    def test_rejects_equal_macro_f1(self):
        ok, _ = gate_decision({"fail_recall": 0.95, "macro_f1": 0.80}, self.P)
        assert not ok  # macro-F1 must be STRICTLY better

    def test_rejects_worse_recall_even_with_better_f1(self):
        ok, reason = gate_decision({"fail_recall": 0.85, "macro_f1": 0.95}, self.P)
        assert not ok and "WORSE" in reason

    def test_equal_recall_better_f1_promotes(self):
        ok, _ = gate_decision({"fail_recall": 0.90, "macro_f1": 0.81}, self.P)
        assert ok


class TestWatcher:
    RC = {"min_new_images": 50, "max_days_between": 7}
    NOW = datetime(2026, 7, 12, 12, 0, 0)

    def test_no_new_images(self):
        assert should_run(0, "2026-07-01 00:00:00", self.NOW, self.RC) == (False, "no new images")

    def test_threshold_reached(self):
        fire, reason = should_run(50, "2026-07-12 00:00:00", self.NOW, self.RC)
        assert fire and "threshold" in reason

    def test_below_threshold_recent_cycle(self):
        fire, _ = should_run(10, "2026-07-11 00:00:00", self.NOW, self.RC)
        assert not fire

    def test_weekly_force_with_any_new(self):
        fire, reason = should_run(3, "2026-07-01 00:00:00", self.NOW, self.RC)
        assert fire and "waited" in reason

    def test_first_ever_cycle(self):
        fire, _ = should_run(3, None, self.NOW, self.RC)
        assert fire


class TestRunOrchestration:
    """Mocked end-to-end: promote/reject wiring, state file, report-on-failure."""

    def _setup(self, tmp_path, monkeypatch, cand_metrics, prod_metrics, fail_at_gate=False):
        import json

        import joblib

        from coilvision.pipeline import retrain as rt

        cfg = tmp_cfg(tmp_path)
        cfg["paths"]["manifests_dir"] = str(tmp_path / "manifests")
        cfg["paths"]["production_dir"] = str(tmp_path / "production")
        cfg["paths"]["models_dir"] = str(tmp_path / "models")
        cfg["paths"]["cache_dir"] = str(tmp_path / "cache")
        for d in ("manifests", "production", "cache"):
            (tmp_path / d).mkdir()

        manifest = pd.DataFrame({"relpath": ["Pass/a.bmp"], "root": ["dataset"], "hash": ["h1"],
                                 "class": ["Pass"], "run": ["r1"], "valid": [True]})
        manifest.to_csv(tmp_path / "manifests" / "manifest.csv", index=False)
        pd.DataFrame({"relpath": ["Fail/Dent/t.bmp"], "run": ["rt"], "class": ["Dent"]}).to_csv(
            tmp_path / "manifests" / f"test_v{cfg['split']['version']}.csv", index=False)
        (tmp_path / "production" / "POINTER.json").write_text(json.dumps(
            {"threshold": 0.9, "preprocess_fingerprint": "x", "preprocess_version": 1,
             "aggregation": "top20", "promoted_at": "2026-01-01 00:00:00"}), encoding="utf-8")
        joblib.dump({"head": "prod"}, tmp_path / "production" / "head.joblib")

        cand_dir = tmp_path / "cand"
        cand_dir.mkdir()
        joblib.dump({"head": "cand"}, cand_dir / "head.joblib")
        pd.DataFrame({"class": ["Dent", "Pass"], "score_top20": [0.95, 0.2]}).to_csv(
            cand_dir / "val_scores.csv", index=False)

        splits = pd.DataFrame({"relpath": ["Pass/a.bmp", "Fail/Dent/t.bmp"], "split": ["train", "test"],
                               "run": ["r1", "rt"], "class": ["Pass", "Dent"],
                               "filename": ["a", "t"], "hash": ["h1", "h2"], "layout": ["L0", "L0"]})

        monkeypatch.setattr(rt, "load_config", lambda: cfg)
        monkeypatch.setattr(rt, "anomaly_cfg", lambda c: c)
        monkeypatch.setattr(rt, "ingest_incoming", lambda c, k: {"accepted": 1, "quarantined": 0,
                                                                 "duplicates": 0, "by_class": {}, "reasons": {}})
        monkeypatch.setattr(rt.manifest_mod, "build_manifest", lambda c: manifest)
        monkeypatch.setattr(rt, "build_cache", lambda m, c: None)
        monkeypatch.setattr(rt.split_mod, "build_splits", lambda m, c, frozen_test_runs: splits)
        monkeypatch.setattr(rt.split_mod, "summary", lambda s: pd.DataFrame({"x": [1]}))
        monkeypatch.setattr(rt, "train_candidate", lambda c: {
            "run_id": "cand", "run_dir": cand_dir, "head_path": cand_dir / "head.joblib",
            "val_scores_path": cand_dir / "val_scores.csv", "val_image_fail_auc": {"top20": 0.99},
            "dent_vs_loose_val_acc": 0.9, "n_unannotated_train_defects": 0, "unannotated_relpaths": []})
        monkeypatch.setattr(rt, "PatchExtractor", lambda c: object())
        monkeypatch.setattr(rt, "load_split_frame", lambda name, c: splits[splits["split"] == name])

        calls = iter([cand_metrics, prod_metrics])

        def fake_eval(c, e, h, f, t):
            if fail_at_gate:
                raise RuntimeError("boom")
            return next(calls)

        monkeypatch.setattr(rt, "evaluate_head", fake_eval)
        promoted = []
        monkeypatch.setattr(rt, "promote", lambda *a, **k: promoted.append(a))
        return rt, cfg, promoted

    GOOD = {"fail_recall": 0.95, "macro_f1": 0.85, "false_reject_rate": 0.1, "fail_auc": 0.99}
    BASE = {"fail_recall": 0.94, "macro_f1": 0.80, "false_reject_rate": 0.1, "fail_auc": 0.98}

    def test_promote_branch(self, tmp_path, monkeypatch):
        rt, cfg, promoted = self._setup(tmp_path, monkeypatch, self.GOOD, self.BASE)
        result = rt.run(force=True)
        assert result["promoted"] and len(promoted) == 1
        assert rt.read_state(cfg)["last_decision"] == "PROMOTE"

    def test_reject_branch(self, tmp_path, monkeypatch):
        rt, cfg, promoted = self._setup(tmp_path, monkeypatch, self.BASE, self.GOOD)
        result = rt.run(force=True)
        assert not result["promoted"] and promoted == []
        assert rt.read_state(cfg)["last_decision"] == "REJECT"
        report = next((tmp_path / "artifacts" / "runs").rglob("PIPELINE_REPORT.md")).read_text(encoding="utf-8")
        assert "REJECT" in report

    def test_failure_still_writes_report_and_releases_lock(self, tmp_path, monkeypatch):
        rt, cfg, _ = self._setup(tmp_path, monkeypatch, self.GOOD, self.BASE, fail_at_gate=True)
        with pytest.raises(RuntimeError, match="boom"):
            rt.run(force=True)
        report = next((tmp_path / "artifacts" / "runs").rglob("PIPELINE_REPORT.md")).read_text(encoding="utf-8")
        assert "FAILED" in report and "boom" in report
        assert not (tmp_path / "artifacts" / "retrain.lock").exists()
        assert rt.read_state(cfg) == {}  # failed run must not claim success


class TestLock:
    def test_lock_blocks_second_run(self, tmp_path):
        cfg = tmp_cfg(tmp_path)
        lock = acquire_lock(cfg)
        with pytest.raises(RuntimeError, match="another retrain"):
            acquire_lock(cfg)
        lock.unlink()

    def test_stale_lock_is_broken(self, tmp_path):
        import os
        import time

        cfg = tmp_cfg(tmp_path)
        lock = acquire_lock(cfg)
        stale = time.time() - (cfg["retrain"]["lock_stale_hours"] + 1) * 3600
        os.utime(lock, (stale, stale))
        lock2 = acquire_lock(cfg)  # breaks the stale lock instead of raising
        assert lock2.exists()
        lock2.unlink()
