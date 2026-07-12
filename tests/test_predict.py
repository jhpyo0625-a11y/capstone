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


@pytest.mark.skipif(not (PROD / "POINTER.json").exists(), reason="no production model promoted")
def test_predict_folder_end_to_end(tmp_path):
    import cv2
    import numpy as np

    from coilvision.predict.cli import predict_folder

    w, h = CFG["data"]["expected_width"], CFG["data"]["expected_height"]
    cv2.imwrite(str(tmp_path / "250825_152739_A35W_2-1 [1024].bmp"), np.full((h, w, 3), 60, np.uint8))
    (tmp_path / "corrupt.bmp").write_bytes(b"junk")
    cv2.imwrite(str(tmp_path / "small.bmp"), np.zeros((480, 640, 3), np.uint8))

    df = predict_folder(tmp_path, CFG, PROD).set_index("file")

    assert df.loc["corrupt.bmp", "verdict"] == "ERROR"
    assert df.loc["corrupt.bmp", "issue"] == "unreadable"
    assert df.loc["small.bmp", "verdict"] == "ERROR"
    assert df.loc["small.bmp", "issue"] == "unexpected_dims_640x480"
    good = df.loc["250825_152739_A35W_2-1 [1024].bmp"]
    assert good["verdict"] in ("PASS", "FAIL")
    assert 0 <= good["fail_score"] <= 1
    assert good["run"] == "250825_152739" and good["part"] == 2  # provenance parsed
    assert set(df.columns) >= {"verdict", "predicted_class", "fail_score", "dent_share",
                               "loose_share", "roi_confident", "issue"}
