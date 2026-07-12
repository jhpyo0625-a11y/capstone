"""Annotation loading + mask rasterization (these drive patch training labels)."""

import json

import numpy as np
import pytest

from coilvision.annotations import defect_mask, load_annotations, mask_to_grid


def entry(boxes=(), strokes=()):
    return {"relpath": "x.bmp", "class": "Dent", "no_defect_visible": False,
            "boxes": list(boxes), "strokes": list(strokes)}


def test_box_rasterizes_to_filled_rect():
    m = defect_mask(entry(boxes=[{"x0": 0.25, "y0": 0.5, "x1": 0.5, "y1": 0.75}]), 400, 200)
    assert m.shape == (200, 400)
    assert m[125, 150] == 1  # inside
    assert m[125, 90] == 0 and m[40, 150] == 0  # outside
    assert m.sum() == pytest.approx((0.25 * 400 + 1) * (0.25 * 200 + 1), rel=0.05)


def test_stroke_rasterizes_along_polyline_with_radius():
    s = {"r": 0.01, "pts": [{"x": 0.1, "y": 0.5}, {"x": 0.4, "y": 0.5}]}  # r = 4px of 400
    m = defect_mask(entry(strokes=[s]), 400, 200)
    assert m[100, 100] == 1  # on the line
    assert m[100, 42] == 1  # round cap at start
    assert m[100, 250] == 0  # beyond end
    assert m[80, 100] == 0  # outside radius
    assert m[102, 100] == 1  # within radius


def test_single_point_stroke_is_a_dot():
    s = {"r": 0.02, "pts": [{"x": 0.5, "y": 0.5}]}
    m = defect_mask(entry(strokes=[s]), 400, 200)
    assert m[100, 200] == 1
    assert 100 < m.sum() < 350  # ~pi*8^2 = 201


def test_mask_to_grid_thresholds_by_coverage():
    m = np.zeros((80, 160), dtype=np.uint8)
    m[0:8, 0:8] = 1  # exactly one full 8px cell
    g = mask_to_grid(m, (10, 20), min_frac=0.10)
    assert g.shape == (10, 20)
    assert g[0, 0] and g.sum() == 1
    g2 = mask_to_grid(m, (10, 20), min_frac=1.01)
    assert g2.sum() == 0


def test_load_annotations_rejects_wrong_version(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"version": 1, "images": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_annotations(p)


def test_load_annotations_roundtrip(tmp_path):
    p = tmp_path / "a.json"
    data = {"version": 2, "images": [entry(strokes=[{"r": 0.01, "pts": [{"x": 0.2, "y": 0.3}]}])]}
    p.write_text(json.dumps(data), encoding="utf-8")
    ann = load_annotations(p)
    assert "x.bmp" in ann and ann["x.bmp"]["strokes"][0]["r"] == 0.01
