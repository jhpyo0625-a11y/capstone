"""Phase 0 acceptance: package imports, config loads, key knobs sane."""

import re

from coilvision.config import load_config, resolve_path


def test_package_imports():
    import coilvision

    assert coilvision.__version__


def test_config_loads_with_sane_knobs():
    cfg = load_config()
    assert cfg["data"]["classes"] == ["Pass", "Dent", "Loose"]
    assert cfg["train"]["backbone"] == "efficientnet_b0"
    assert 0 < cfg["preprocess"]["osd_crop_bottom_frac"] < 0.2
    assert cfg["split"]["group_by"] == "run"
    assert cfg["eval"]["fail_recall_target"] >= 0.95


def test_dataset_dir_exists():
    cfg = load_config()
    assert resolve_path(cfg, "dataset_dir").is_dir()


def test_filename_pattern_parses_known_name():
    cfg = load_config()
    m = re.match(cfg["data"]["filename_pattern"], "250825_152739_A35W_2-1 [1024].bmp")
    assert m is not None
    assert m.group("run") == "250825_152739"
    assert m.group("part") == "2"
    assert m.group("shot") == "1"
    assert m.group("code") == "1024"
