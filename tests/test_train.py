"""Datamodule, metrics, and early-stopping unit tests (fast — no actual training)."""

import copy

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from coilvision.config import load_config
from coilvision.data.split import CLASSES  # noqa: F401  (import guards package wiring)
from coilvision.eval.metrics import fail_auc, fail_recall, false_reject_rate, fail_scores, macro_f1, per_class_recall
from coilvision.train.datamodule import CoilDataset, augment, class_weights
from coilvision.train.trainer import EarlyStopper

CFG = load_config()
FAIL_IDX = [1, 2]  # classes = [Pass, Dent, Loose]


# ---- metrics ----

def test_fail_recall_and_frr_known_values():
    y = np.array([0, 0, 1, 2])  # Pass, Pass, Dent, Loose
    probs = np.array(
        [
            [0.9, 0.05, 0.05],  # pass, predicted pass
            [0.3, 0.4, 0.3],    # pass, predicted fail (false reject)
            [0.6, 0.3, 0.1],    # dent, predicted pass (missed!)
            [0.1, 0.2, 0.7],    # loose, predicted fail
        ]
    )
    assert fail_recall(y, probs, FAIL_IDX) == 0.5
    assert false_reject_rate(y, probs, FAIL_IDX) == 0.5
    np.testing.assert_allclose(fail_scores(probs, FAIL_IDX), [0.1, 0.7, 0.4, 0.9])


def test_fail_recall_threshold_dependence():
    y = np.array([1])
    probs = np.array([[0.55, 0.25, 0.20]])  # P(fail)=0.45
    assert fail_recall(y, probs, FAIL_IDX, threshold=0.5) == 0.0
    assert fail_recall(y, probs, FAIL_IDX, threshold=0.4) == 1.0


def test_macro_f1_and_per_class_recall():
    y = np.array([0, 1, 2, 0])
    assert macro_f1(y, y.copy()) == 1.0
    recalls = per_class_recall(y, np.array([0, 1, 1, 0]), 3)
    assert recalls == [1.0, 1.0, 0.0]


def test_fail_recall_nan_when_no_defects():
    y = np.array([0, 0])
    probs = np.full((2, 3), 1 / 3)
    assert np.isnan(fail_recall(y, probs, FAIL_IDX))


def test_fail_auc_rewards_separation_not_all_fail():
    y = np.array([0, 0, 1, 2])
    perfect = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    assert fail_auc(y, perfect, FAIL_IDX) == 1.0
    # the degenerate all-fail predictor: recall 1.0 but NO separation -> AUC 0.5
    all_fail = np.tile([0.02, 0.49, 0.49], (4, 1))
    assert fail_recall(y, all_fail, FAIL_IDX) == 1.0
    assert fail_auc(y, all_fail, FAIL_IDX) == 0.5
    assert np.isnan(fail_auc(np.array([0, 0]), perfect[:2], FAIL_IDX))  # single class -> nan


# ---- class weights ----

def test_class_weights_inverse_frequency():
    labels = np.array([0] * 8 + [1] * 2 + [2] * 2)
    w = class_weights(labels, 3)
    assert w[1] == w[2] and w[1] / w[0] == 4.0  # 4x rarer -> 4x weight
    assert pytest.approx(float(w.mean() * 3), rel=1e-6) == float(w.sum())


def test_class_weights_missing_class_raises():
    with pytest.raises(ValueError, match="missing"):
        class_weights(np.array([0, 0, 1]), 3)


# ---- augmentation ----

AUG = CFG["train"]["augment"]


def test_augment_preserves_shape_dtype_and_is_seed_deterministic():
    img = np.random.default_rng(0).integers(0, 255, (384, 384, 3), dtype=np.uint8)
    a = augment(img.copy(), np.random.default_rng(7), AUG)
    b = augment(img.copy(), np.random.default_rng(7), AUG)
    c = augment(img.copy(), np.random.default_rng(8), AUG)
    assert a.shape == img.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_augment_never_swaps_channels():
    # red-dominant image must stay red-dominant (hue is signal)
    img = np.zeros((384, 384, 3), dtype=np.uint8)
    img[..., 0] = 200  # R in RGB layout
    out = augment(img, np.random.default_rng(0), AUG)
    center = out[100:284, 100:284]
    assert center[..., 0].mean() > center[..., 1].mean() + 50
    assert center[..., 0].mean() > center[..., 2].mean() + 50


# ---- dataset ----

def test_coil_dataset_items(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["cache_dir"] = str(tmp_path)
    size = cfg["preprocess"]["resize"]
    tw, th = (size, size) if isinstance(size, int) else (size[0], size[1])
    files, labels = [], [0, 1, 2, 0]
    rng = np.random.default_rng(0)
    for i in range(4):
        name = f"img{i}.png"
        cv2.imwrite(str(tmp_path / name), rng.integers(0, 255, (th, tw, 3), dtype=np.uint8).astype(np.uint8))
        files.append(name)
    frame = pd.DataFrame({"cache_file": files, "label": labels})

    for train in (False, True):
        ds = CoilDataset(frame, cfg, train=train, seed=1)
        x, y = ds[1]
        assert isinstance(x, torch.Tensor) and x.shape == (3, th, tw) and x.dtype == torch.float32
        assert y == 1
        assert len(ds) == 4
        if cfg["train"].get("normalize") == "per_image" and not train:
            assert abs(float(x.mean())) < 0.05  # per-image normalization centers the tensor


def test_coil_dataset_missing_cache_file_raises(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["cache_dir"] = str(tmp_path)
    frame = pd.DataFrame({"cache_file": ["nope.png"], "label": [0]})
    with pytest.raises(FileNotFoundError):
        CoilDataset(frame, cfg, train=False)


# ---- early stopping ----

def test_early_stopper_improvement_and_patience():
    es = EarlyStopper(patience=2)
    assert es.update((0.80, 0.5), 0)
    assert es.update((0.80, 0.6), 1)  # macro-F1 tiebreak counts as improvement
    assert not es.update((0.80, 0.6), 2)  # equal is NOT improvement
    assert not es.should_stop
    assert not es.update((0.79, 0.9), 3)
    assert es.should_stop
    assert es.best_epoch == 1


def test_early_stopper_resets_counter_on_improvement():
    es = EarlyStopper(patience=2)
    es.update((0.5, 0.5), 0)
    es.update((0.4, 0.4), 1)
    assert es.update((0.9, 0.9), 2)
    assert es.bad == 0 and es.best_epoch == 2
