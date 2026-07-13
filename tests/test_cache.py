"""build_cache behavior: failure resilience, reuse, and index↔PNG consistency."""

import copy

import cv2
import numpy as np
import pandas as pd
import pytest

from coilvision.config import load_config, resolve_path
from coilvision.data.preprocess import build_cache, cache_index_path, cache_path_for, preprocess_fingerprint

CFG = load_config()


def tmp_cfg(tmp_path):
    """Config clone whose data/cache/manifest paths all live under tmp_path."""
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["dataset_dir"] = str(tmp_path / "ds")
    cfg["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg["paths"]["manifests_dir"] = str(tmp_path / "manifests")
    (tmp_path / "ds" / "Pass").mkdir(parents=True)
    (tmp_path / "manifests").mkdir()
    return cfg


def make_manifest(rows):
    return pd.DataFrame(
        [
            {"relpath": r, "hash": f"hash{i}", "class": "Pass", "run": "250825_152739", "valid": True}
            for i, r in enumerate(rows)
        ]
    )


def write_frame(path):
    img = np.full((2048, 2448, 3), 60, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_bad_file_is_skipped_not_fatal(tmp_path, capsys):
    cfg = tmp_cfg(tmp_path)
    write_frame(tmp_path / "ds" / "Pass" / "good.bmp")
    (tmp_path / "ds" / "Pass" / "corrupt.bmp").write_bytes(b"not a bmp")
    manifest = make_manifest([r"Pass\good.bmp", r"Pass\corrupt.bmp", r"Pass\missing.bmp"])

    index = build_cache(manifest, cfg)

    assert len(index) == 1  # only the good file
    assert index.iloc[0]["relpath"] == r"Pass\good.bmp"
    out = capsys.readouterr().out
    assert "2 file(s) failed" in out
    assert "corrupt.bmp" in out and "missing.bmp" in out


def test_rebuild_reuses_cache_without_recompute(tmp_path, capsys):
    cfg = tmp_cfg(tmp_path)
    write_frame(tmp_path / "ds" / "Pass" / "good.bmp")
    manifest = make_manifest([r"Pass\good.bmp"])

    first = build_cache(manifest, cfg)
    png = resolve_path(cfg, "cache_dir") / first.iloc[0]["cache_file"]
    mtime = png.stat().st_mtime_ns
    capsys.readouterr()

    second = build_cache(manifest, cfg)

    assert "(1 reused)" in capsys.readouterr().out
    assert png.stat().st_mtime_ns == mtime  # PNG untouched
    assert second.iloc[0]["cache_file"] == first.iloc[0]["cache_file"]
    assert [second.iloc[0][k] for k in ("x0", "y0", "x1", "y1")] == [first.iloc[0][k] for k in ("x0", "y0", "x1", "y1")]


def test_config_change_invalidates_cache(tmp_path):
    cfg = tmp_cfg(tmp_path)
    write_frame(tmp_path / "ds" / "Pass" / "good.bmp")
    manifest = make_manifest([r"Pass\good.bmp"])
    build_cache(manifest, cfg)

    tweaked = copy.deepcopy(cfg)
    tweaked["preprocess"]["roi"]["pad_frac"] = 0.20  # param change, no version bump
    assert preprocess_fingerprint(tweaked) != preprocess_fingerprint(cfg)
    assert cache_path_for("hash0", tweaked) != cache_path_for("hash0", cfg)

    index = build_cache(manifest, tweaked)
    assert (resolve_path(cfg, "cache_dir") / index.iloc[0]["cache_file"]).exists()  # regenerated under new key


from coilvision.anomaly import anomaly_cfg

MANIFEST_PATH = resolve_path(CFG, "manifests_dir") / "manifest.csv"
PROD_CFG = anomaly_cfg(CFG)  # the production/pipeline path maintains the hi-res cache
PROD_INDEX = cache_index_path(PROD_CFG)


@pytest.mark.skipif(not (MANIFEST_PATH.exists() and PROD_INDEX.exists()), reason="cache not built yet")
def test_production_cache_consistent_with_manifest():
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    index = pd.read_csv(PROD_INDEX, keep_default_na=False)
    assert len(index) == int(manifest["valid"].sum())
    cache_dir = resolve_path(PROD_CFG, "cache_dir")
    missing = [f for f in index["cache_file"] if not (cache_dir / f).exists()]
    assert not missing, f"{len(missing)} index rows point at missing PNGs"
    size = PROD_CFG["preprocess"]["resize"]
    tw, th = (size, size) if isinstance(size, int) else (size[0], size[1])
    sample = cv2.imread(str(cache_dir / index.iloc[0]["cache_file"]))
    assert sample.shape == (th, tw, 3)
    assert set(index["hash"]) <= set(manifest["hash"])
