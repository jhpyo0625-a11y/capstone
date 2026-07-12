"""Promotion: fingerprint refusal, POINTER contents, previous-model archiving."""

import copy
import json

import joblib
import pytest

from coilvision.anomaly import anomaly_cfg
from coilvision.config import load_config
from coilvision.data.preprocess import preprocess_fingerprint
from coilvision.pipeline.promote import promote

CFG = anomaly_cfg(load_config())
OP = {"threshold": 0.91, "fail_recall": 1.0, "false_reject_rate": 0.07, "policy": "test"}


def tmp_cfg(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["paths"]["production_dir"] = str(tmp_path / "models" / "production")
    return cfg


def make_head(tmp_path, name, fingerprint):
    p = tmp_path / name
    joblib.dump({"head": name, "classes": ["normal", "Dent", "Loose"], "preprocess_fingerprint": fingerprint}, p)
    return p


def test_promote_writes_pointer(tmp_path):
    cfg = tmp_cfg(tmp_path)
    head = make_head(tmp_path, "h1.joblib", preprocess_fingerprint(cfg))
    prod = promote(head, OP, "initial", cfg)
    pointer = json.loads((prod / "POINTER.json").read_text(encoding="utf-8"))
    assert pointer["threshold"] == 0.91
    assert pointer["preprocess_fingerprint"] == preprocess_fingerprint(cfg)
    assert (prod / "head.joblib").exists()


def test_promote_refuses_mismatched_fingerprint(tmp_path):
    cfg = tmp_cfg(tmp_path)
    head = make_head(tmp_path, "h1.joblib", "deadbeef")
    with pytest.raises(RuntimeError, match="fingerprint"):
        promote(head, OP, "bad", cfg)


def test_promote_archives_previous_production(tmp_path):
    cfg = tmp_cfg(tmp_path)
    fp = preprocess_fingerprint(cfg)
    promote(make_head(tmp_path, "h1.joblib", fp), OP, "first", cfg)
    promote(make_head(tmp_path, "h2.joblib", fp), dict(OP, threshold=0.88), "second", cfg)

    prod_pointer = json.loads((tmp_path / "models" / "production" / "POINTER.json").read_text(encoding="utf-8"))
    assert prod_pointer["threshold"] == 0.88  # new model live
    archives = list((tmp_path / "models" / "archive").iterdir())
    assert len(archives) == 1  # previous kept
    old_pointer = json.loads((archives[0] / "POINTER.json").read_text(encoding="utf-8"))
    assert old_pointer["threshold"] == 0.91
    assert joblib.load(archives[0] / "head.joblib")["head"] == "h1.joblib"
