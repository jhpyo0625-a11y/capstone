"""OSD crop / red-text leakage / ROI / letterbox unit tests (synthetic images — fast)."""

import cv2
import numpy as np

from coilvision.config import load_config
from coilvision.data.preprocess import (
    count_red_text_pixels,
    crop_osd,
    detect_roi,
    letterbox,
    preprocess_image,
)

CFG = load_config()
P = CFG["preprocess"]


def synthetic_frame() -> np.ndarray:
    """Gray frame with red OSD text at the measured position (rows ~1822-1858)."""
    img = np.full((2048, 2448, 3), 60, dtype=np.uint8)
    cv2.putText(img, "Image acquired : 150 ErrorCount : 0", (10, 1850),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)  # BGR pure red
    return img


def test_synthetic_red_text_is_detected_before_crop():
    assert count_red_text_pixels(synthetic_frame(), P["red_text"]) > 100


def test_osd_crop_removes_red_text():
    cropped = crop_osd(synthetic_frame(), P["osd_crop_bottom_frac"])
    assert count_red_text_pixels(cropped, P["red_text"]) == 0


def test_osd_crop_keeps_margin_above_measured_text():
    # text starts at row 1822 (measured); the crop must remove everything from there down
    kept = 2048 - int(round(2048 * P["osd_crop_bottom_frac"]))
    assert kept <= 1822 - 20, f"crop keeps rows 0..{kept}, too close to text at 1822"


def test_copper_colored_pixels_do_not_trigger_red_detector():
    # copper wire ≈ BGR (140, 120, 190) — reddish but not the OSD's pure red
    img = np.full((100, 100, 3), (140, 120, 190), dtype=np.uint8)
    assert count_red_text_pixels(img, P["red_text"]) == 0


def test_roi_falls_back_on_featureless_image():
    img = np.full((1792, 2448, 3), 60, dtype=np.uint8)
    (x0, y0, x1, y1), confident = detect_roi(img, P["roi"])
    assert not confident
    fb = P["roi"]["fallback_crop"]
    assert (x0, y0) == (int(fb["x0"] * 2448), int(fb["y0"] * 1792))
    assert (x1, y1) == (int(fb["x1"] * 2448), int(fb["y1"] * 1792))


def test_letterbox_shape_and_content_centered():
    img = np.full((100, 400, 3), 200, dtype=np.uint8)
    out = letterbox(img, 384)
    assert out.shape == (384, 384, 3)
    assert out[192, 192].tolist() == [200, 200, 200]  # content at center
    assert out[10, 192].tolist() == [0, 0, 0]  # padding above


def test_preprocess_image_output_is_model_ready_and_leak_free():
    out, meta = preprocess_image(synthetic_frame(), CFG)
    assert out.shape == (P["resize"], P["resize"], 3)
    assert count_red_text_pixels(out, P["red_text"]) == 0
    assert meta["roi_confident"] is False  # featureless synthetic -> fallback
