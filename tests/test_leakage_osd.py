"""Integration leakage gate: NO cached (processed) image may contain red OSD pixels.

Spec §6.1: 'a unit test asserts no saturated-red text pixels remain in any
processed image'. Runs over the full preprocess cache when it exists.
"""

import cv2
import pytest

from coilvision.config import load_config, resolve_path

from coilvision.data.preprocess import preprocess_fingerprint

CFG = load_config()
CACHE_DIR = resolve_path(CFG, "cache_dir")
_SUFFIX = f"_v{CFG['preprocess']['version']}_{preprocess_fingerprint(CFG)}.png"
CACHED = sorted(CACHE_DIR.glob(f"*{_SUFFIX}")) if CACHE_DIR.exists() else []


@pytest.mark.skipif(not CACHED, reason="preprocess cache not built yet")
def test_no_red_osd_pixels_in_any_cached_image():
    from coilvision.data.preprocess import count_red_text_pixels

    offenders = []
    for p in CACHED:
        img = cv2.imread(str(p))
        n = count_red_text_pixels(img, CFG["preprocess"]["red_text"])
        if n:
            offenders.append((p.name, n))
    assert not offenders, f"red OSD pixels survived preprocessing: {offenders[:10]}"
