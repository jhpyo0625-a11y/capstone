"""Predict CLI: production loading, version-mismatch refusal, verdict wiring."""

import copy
import json

import joblib
import pytest

from coilvision.anomaly import anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.data.preprocess import preprocess_fingerprint
from coilvision.predict.cli import load_production

CFG = anomaly_cfg(load_config())


def make_model_dir(tmp_path, fingerprint):
    joblib.dump({"head": "stub", "classes": ["normal", "Dent", "Loose"]}, tmp_path / "head.joblib")
    (tmp_path / "POINTER.json").write_text(
        json.dumps({"preprocess_fingerprint": fingerprint, "preprocess_version": 1,
                    "threshold": 0.9, "aggregation": "top20"}), encoding="utf-8")
    return tmp_path


def test_load_production_accepts_matching_fingerprint(tmp_path):
    model_dir = make_model_dir(tmp_path, preprocess_fingerprint(CFG))
    head, pointer = load_production(model_dir, CFG)
    assert head == "stub"
    assert pointer["threshold"] == 0.9


def test_load_production_refuses_mismatched_fingerprint(tmp_path):
    model_dir = make_model_dir(tmp_path, "deadbeef")
    with pytest.raises(RuntimeError, match="mismatch"):
        load_production(model_dir, CFG)


def test_load_production_missing_pointer(tmp_path):
    with pytest.raises(FileNotFoundError, match="POINTER"):
        load_production(tmp_path, CFG)


def test_config_change_would_be_refused(tmp_path):
    # the same guarantee end to end: tweaking any preprocess knob flips the fingerprint
    model_dir = make_model_dir(tmp_path, preprocess_fingerprint(CFG))
    tweaked = copy.deepcopy(CFG)
    tweaked["preprocess"]["osd_crop_bottom_frac"] = 0.10
    with pytest.raises(RuntimeError, match="mismatch"):
        load_production(model_dir, tweaked)


PROD = resolve_path(CFG, "production_dir")


@pytest.mark.skipif(not (PROD / "POINTER.json").exists(), reason="no production model promoted")
def test_real_production_pointer_is_consistent():
    pointer = json.loads((PROD / "POINTER.json").read_text(encoding="utf-8"))
    assert pointer["preprocess_fingerprint"] == preprocess_fingerprint(CFG)
    assert 0 < pointer["threshold"] < 1
    assert pointer["aggregation"] == f"top{CFG['patchclf']['top_k']}"
    assert (PROD / "head.joblib").exists()
