"""Filename parser units + manifest integration (spec Phase 1: covers 100% of files)."""

from pathlib import Path

import pandas as pd
import pytest

from coilvision.config import load_config, resolve_path
from coilvision.data.manifest import class_from_relpath, parse_filename

CFG = load_config()
PATTERN = CFG["data"]["filename_pattern"]


def test_parse_valid_names():
    p = parse_filename("250825_152739_A35W_2-1 [1024].bmp", PATTERN)
    assert p == {"run": "250825_152739", "part": 2, "shot": 1, "code": 1024}
    p = parse_filename("251014_141452_A35W_28-3 [262144].bmp", PATTERN)
    assert p == {"run": "251014_141452", "part": 28, "shot": 3, "code": 262144}


@pytest.mark.parametrize(
    "bad",
    [
        "250825_152739_A35W_2-1.bmp",  # missing code
        "250825_152739_B22X_2-1 [1024].bmp",  # wrong product token
        "note.txt",
        "250825_A35W_2-1 [1024].bmp",  # truncated timestamp
    ],
)
def test_parse_rejects_bad_names(bad):
    assert parse_filename(bad, PATTERN) is None


def test_class_from_relpath():
    assert class_from_relpath(Path("Pass/x.bmp")) == "Pass"
    assert class_from_relpath(Path("Fail/Dent/x.bmp")) == "Dent"
    assert class_from_relpath(Path("Fail/Loose/x.bmp")) == "Loose"
    assert class_from_relpath(Path("Other/x.bmp")) is None


MANIFEST = resolve_path(CFG, "manifests_dir") / "manifest.csv"


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built yet")
class TestManifestIntegration:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(MANIFEST, keep_default_na=False)

    def test_covers_every_dataset_file(self, df):
        n_files = sum(1 for _ in resolve_path(CFG, "dataset_dir").rglob("*.bmp"))
        assert len(df) == n_files

    def test_all_valid(self, df):
        assert df["valid"].all(), df.loc[~df["valid"], ["relpath", "issues"]].to_string()

    def test_expected_class_counts(self, df):
        counts = df["class"].value_counts()
        assert counts["Pass"] == 633
        assert counts["Loose"] == 110
        assert counts["Dent"] == 74

    def test_expected_run_count(self, df):
        assert df["run"].nunique() == 28

    def test_layout_clusters_found(self, df):
        # spec §2: at least 2 distinct board layouts
        n = df.loc[df["layout"] != "", "layout"].nunique()
        assert 2 <= n <= 4, f"unexpected layout cluster count {n}"

    def test_hashes_unique_or_reported(self, df):
        # exact duplicate files are possible but should be rare; fail loudly if rampant
        dupes = df[df.duplicated(subset="hash", keep=False)]
        assert len(dupes) < 20, f"{len(dupes)} exact-duplicate files:\n{dupes['relpath'].to_string()}"
