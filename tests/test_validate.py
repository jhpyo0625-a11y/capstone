"""Validation + quarantine flow unit tests (temp files — never touches the raw dataset)."""

from pathlib import Path

import cv2
import numpy as np

from coilvision.config import load_config
from coilvision.data.validate import quarantine_file, validate_file

CFG = load_config()
GOOD_NAME = "250825_152739_A35W_2-1 [1024].bmp"


def write_bmp(path: Path, w: int, h: int) -> Path:
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))
    return path


def test_correct_file_passes(tmp_path):
    p = write_bmp(tmp_path / GOOD_NAME, CFG["data"]["expected_width"], CFG["data"]["expected_height"])
    assert validate_file(p, CFG) == []


def test_wrong_dims_flagged(tmp_path):
    p = write_bmp(tmp_path / GOOD_NAME, 640, 480)
    assert "unexpected_dims_640x480" in validate_file(p, CFG)


def test_bad_filename_flagged(tmp_path):
    p = write_bmp(tmp_path / "random_name.bmp", CFG["data"]["expected_width"], CFG["data"]["expected_height"])
    assert "unparseable_filename" in validate_file(p, CFG)


def test_unreadable_flagged(tmp_path):
    p = tmp_path / GOOD_NAME
    p.write_bytes(b"this is not a bmp")
    assert "unreadable" in validate_file(p, CFG)


def test_quarantine_moves_and_logs(tmp_path):
    p = write_bmp(tmp_path / "bad.bmp", 10, 10)
    qdir = tmp_path / "quarantine"
    dest = quarantine_file(p, ["unexpected_dims_10x10"], qdir)
    assert not p.exists()
    assert dest.exists()
    log = (qdir / "quarantine_log.csv").read_text()
    assert "unexpected_dims_10x10" in log
    # second file with the same name gets a distinct quarantine name
    p2 = write_bmp(tmp_path / "bad.bmp", 10, 10)
    dest2 = quarantine_file(p2, ["unreadable"], qdir)
    assert dest2 != dest and dest2.exists()
