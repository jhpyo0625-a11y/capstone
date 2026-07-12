"""Retrain pipeline units: ingest/quarantine/dedupe, gate decision, watcher trigger, lock."""

import copy
import hashlib
from datetime import datetime

import cv2
import numpy as np
import pytest

from coilvision.anomaly import anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.pipeline.retrain import acquire_lock, gate_decision, ingest_incoming
from coilvision.pipeline.watcher import should_run

CFG = anomaly_cfg(load_config())
GOOD_NAME = "260401_120000_A35W_3-1 [1024].bmp"


def tmp_cfg(tmp_path):
    cfg = copy.deepcopy(CFG)
    for key, sub in (("incoming_dir", "incoming"), ("accepted_dir", "accepted"),
                     ("quarantine_dir", "quarantine"), ("artifacts_dir", "artifacts")):
        cfg["paths"][key] = str(tmp_path / sub)
    (tmp_path / "incoming" / "Pass").mkdir(parents=True)
    (tmp_path / "incoming" / "Fail" / "Dent").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
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
