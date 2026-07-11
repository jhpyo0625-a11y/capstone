"""Split constraints + the Phase 2 leakage gate: zero run overlap between splits."""

import pandas as pd
import pytest

from coilvision.config import load_config, resolve_path
from coilvision.data.split import CLASSES, build_splits, run_table, select_runs

CFG = load_config()


def synthetic_manifest() -> pd.DataFrame:
    """12 runs: 4 dent-ish, 4 loose-ish, 4 pass-only; two layouts, one singleton layout."""
    rows = []
    spec = [
        ("run01", {"Dent": 10}, "L0"),
        ("run02", {"Dent": 8}, "L0"),
        ("run03", {"Dent": 6, "Loose": 2}, "L1"),
        ("run04", {"Dent": 4}, "L0"),
        ("run05", {"Loose": 12}, "L0"),
        ("run06", {"Loose": 10}, "L1"),
        ("run07", {"Loose": 8, "Pass": 5}, "L0"),
        ("run08", {"Loose": 6}, "L1"),
        ("run09", {"Pass": 60}, "L0"),
        ("run10", {"Pass": 50}, "L1"),
        ("run11", {"Pass": 40}, "L0"),
        ("run12", {"Pass": 30}, "L2"),  # singleton layout — must stay in train
    ]
    for run, classes, layout in spec:
        for cls, n in classes.items():
            for i in range(n):
                rows.append(
                    {
                        "relpath": f"{cls}/{run}_{i}.bmp",
                        "filename": f"{run}_{i}.bmp",
                        "hash": f"{run}_{cls}_{i}",
                        "class": cls,
                        "run": run,
                        "layout": layout,
                        "valid": True,
                    }
                )
    return pd.DataFrame(rows)


def small_cfg() -> dict:
    return {
        "split": {
            "version": 1,
            "test": {
                "n_runs": [4],
                "min_dent_runs": 1,
                "min_loose_runs": 1,
                "min_pass_images": 20,
                "min_layouts": 2,
                "class_frac_target": 0.25,
                "class_frac_window": [0.08, 0.45],
            },
            "val": {
                "n_runs": [3],
                "min_dent_runs": 1,
                "min_loose_runs": 1,
                "min_pass_images": 10,
                "min_layouts": 1,
                "class_frac_target": 0.25,
                "class_frac_window": [0.08, 0.50],
            },
        }
    }


class TestSyntheticSplits:
    @pytest.fixture(scope="class")
    def splits(self):
        return build_splits(synthetic_manifest(), small_cfg())

    def test_every_image_assigned_exactly_once(self, splits):
        assert len(splits) == len(synthetic_manifest())
        assert set(splits["split"].unique()) == {"train", "val", "test"}

    def test_zero_run_overlap(self, splits):
        by_split = {s: set(splits.loc[splits["split"] == s, "run"]) for s in ("train", "val", "test")}
        assert not by_split["train"] & by_split["val"]
        assert not by_split["train"] & by_split["test"]
        assert not by_split["val"] & by_split["test"]

    def test_constraints_hold(self, splits):
        test = splits[splits["split"] == "test"]
        assert test.loc[test["class"] == "Dent", "run"].nunique() >= 1
        assert test.loc[test["class"] == "Loose", "run"].nunique() >= 1
        assert (test["class"] == "Pass").sum() >= 20
        assert test["layout"].nunique() >= 2

    def test_singleton_layout_stays_in_train(self, splits):
        assert set(splits.loc[splits["layout"] == "L2", "split"]) == {"train"}

    def test_deterministic(self, splits):
        again = build_splits(synthetic_manifest(), small_cfg())
        pd.testing.assert_frame_equal(splits.reset_index(drop=True), again.reset_index(drop=True))

    def test_frozen_test_runs_respected(self, splits):
        frozen = sorted(splits.loc[splits["split"] == "test", "run"].unique())
        again = build_splits(synthetic_manifest(), small_cfg(), frozen_test_runs=frozen)
        assert sorted(again.loc[again["split"] == "test", "run"].unique()) == frozen


def test_select_runs_raises_when_infeasible():
    m = synthetic_manifest()
    table = run_table(m)
    impossible = dict(small_cfg()["split"]["test"], min_dent_runs=99)
    with pytest.raises(RuntimeError, match="no feasible"):
        select_runs(table, list(table.index), impossible, {c: int(table[c].sum()) for c in CLASSES})


# ---- integration over the real, written split files (the Phase 2 acceptance gate) ----

V = CFG["split"]["version"]
MANIFESTS = resolve_path(CFG, "manifests_dir")
SPLIT_FILES = {s: MANIFESTS / f"{s}_v{V}.csv" for s in ("train", "val", "test")}


@pytest.mark.skipif(not all(p.exists() for p in SPLIT_FILES.values()), reason="splits not built yet")
class TestRealSplits:
    @pytest.fixture(scope="class")
    def parts(self):
        return {s: pd.read_csv(p, keep_default_na=False) for s, p in SPLIT_FILES.items()}

    def test_leakage_zero_run_overlap(self, parts):
        runs = {s: set(df["run"]) for s, df in parts.items()}
        assert not runs["train"] & runs["val"]
        assert not runs["train"] & runs["test"]
        assert not runs["val"] & runs["test"]

    def test_covers_every_valid_image_exactly_once(self, parts):
        manifest = pd.read_csv(MANIFESTS / "manifest.csv", keep_default_na=False)
        all_rel = pd.concat([df["relpath"] for df in parts.values()])
        assert len(all_rel) == int(manifest["valid"].sum())
        assert all_rel.is_unique

    def test_spec_constraints_on_frozen_test(self, parts):
        test = parts["test"]
        assert test.loc[test["class"] == "Dent", "run"].nunique() >= 2
        assert test.loc[test["class"] == "Loose", "run"].nunique() >= 2
        assert test["layout"].nunique() >= 2

    def test_val_has_defects_for_early_stopping(self, parts):
        val = parts["val"]
        assert (val["class"] == "Dent").sum() > 0
        assert (val["class"] == "Loose").sum() > 0
