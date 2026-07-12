"""Grouped, stratified splits; frozen test manifest (spec §6.2).

Group = production run — shots and parts of one run never span splits (images
within a run share panel/lighting/session conditions). Selection is a
deterministic exhaustive search over run combinations (no RNG): feasible combos
satisfy the config's min-runs/min-images/layout constraints and keep every
class's image fraction inside a window; the best combo minimizes distance to
the target fraction. Fully reproducible from manifest + config.

test_vN.csv is FROZEN once written: rerunning refuses to touch it and only
regenerates train/val around it. Re-freezing (new test baseline) is a deliberate
manual act:  uv run python -m coilvision.data.split --refreeze

Run:  uv run python -m coilvision.data.split
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter

import pandas as pd

from coilvision.config import load_config, resolve_path

CLASSES = ("Dent", "Loose", "Pass")
SPLIT_COLUMNS = ["relpath", "filename", "hash", "class", "run", "layout", "split"]


def run_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-run class image counts + layout."""
    valid = manifest[manifest["valid"]]
    t = valid.pivot_table(index="run", columns="class", values="relpath", aggfunc="count").fillna(0).astype(int)
    for c in CLASSES:
        if c not in t.columns:
            t[c] = 0
    t["layout"] = valid.groupby("run")["layout"].first()
    return t[[*CLASSES, "layout"]]


def select_runs(
    table: pd.DataFrame,
    pool: list[str],
    rules: dict,
    totals: dict[str, int],
) -> list[str]:
    """Deterministic constrained search for the best run combination in `pool`.

    `totals` is the denominator for class fractions (whole dataset for test,
    remaining pool for val). Layout-exhaustion check is relative to `pool`:
    a combo may never take every pool run of any layout, so train always keeps
    at least one run of each layout present in the pool.
    """
    runs = sorted(pool)
    counts = {c: [int(table.at[r, c]) for r in runs] for c in CLASSES}
    layouts = [table.at[r, "layout"] for r in runs]
    pool_layout_counts = Counter(layouts)
    lo, hi = rules["class_frac_window"]
    target = rules["class_frac_target"]

    best_key, best_combo = None, None
    for n in rules["n_runs"]:
        for combo in itertools.combinations(range(len(runs)), n):
            if sum(1 for i in combo if counts["Dent"][i] > 0) < rules["min_dent_runs"]:
                continue
            if sum(1 for i in combo if counts["Loose"][i] > 0) < rules["min_loose_runs"]:
                continue
            if sum(counts["Pass"][i] for i in combo) < rules["min_pass_images"]:
                continue
            fracs = {c: sum(counts[c][i] for i in combo) / max(totals[c], 1) for c in CLASSES}
            if any(not (lo <= fracs[c] <= hi) for c in CLASSES):
                continue
            combo_layouts = Counter(layouts[i] for i in combo)
            if len(combo_layouts) < rules["min_layouts"]:
                continue
            if any(combo_layouts[l] >= pool_layout_counts[l] for l in combo_layouts):
                continue  # would exhaust a layout from the remaining pool
            score = sum(abs(fracs[c] - target) for c in CLASSES)
            key = (round(score, 9), combo)  # combo tuple = stable deterministic tiebreak
            if best_key is None or key < best_key:
                best_key, best_combo = key, combo
    if best_combo is None:
        raise RuntimeError(f"no feasible run combination for constraints: {rules}")
    return [runs[i] for i in best_combo]


def build_splits(manifest: pd.DataFrame, cfg: dict, frozen_test_runs: list[str] | None = None) -> pd.DataFrame:
    """Assign every valid manifest row to train/val/test. Returns rows with a `split` column."""
    table = run_table(manifest)
    all_runs = list(table.index)

    if frozen_test_runs is None:
        totals = {c: int(table[c].sum()) for c in CLASSES}
        test_runs = select_runs(table, all_runs, cfg["split"]["test"], totals)
    else:
        test_runs = frozen_test_runs

    pool = [r for r in all_runs if r not in set(test_runs)]
    pool_totals = {c: int(table.loc[pool, c].sum()) for c in CLASSES}
    val_runs = select_runs(table, pool, cfg["split"]["val"], pool_totals)

    out = manifest[manifest["valid"]].copy()
    test_set, val_set = set(test_runs), set(val_runs)
    out["split"] = out["run"].map(lambda r: "test" if r in test_set else "val" if r in val_set else "train")
    return out[SPLIT_COLUMNS]


def summary(splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ("train", "val", "test"):
        part = splits[splits["split"] == name]
        row = {"split": name, "runs": part["run"].nunique(), "images": len(part)}
        for c in CLASSES:
            n = int((part["class"] == c).sum())
            row[c] = n
            row[f"{c}%"] = round(100 * n / max(int((splits["class"] == c).sum()), 1), 1)
        row["layouts"] = ",".join(sorted(part["layout"].unique()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def main(refreeze: bool = False) -> None:
    cfg = load_config()
    v = cfg["split"]["version"]
    manifests_dir = resolve_path(cfg, "manifests_dir")
    test_path = manifests_dir / f"test_v{v}.csv"

    frozen_runs = None
    if test_path.exists() and not refreeze:
        frozen_runs = sorted(pd.read_csv(test_path, keep_default_na=False)["run"].unique())
        print(f"test_v{v} is FROZEN ({len(frozen_runs)} runs) -- regenerating train/val around it.")
        print("(re-freezing the test set is a deliberate act: rerun with --refreeze)")

    manifest = pd.read_csv(manifests_dir / "manifest.csv", keep_default_na=False)
    splits = build_splits(manifest, cfg, frozen_test_runs=frozen_runs)

    for name in ("train", "val", "test"):
        path = manifests_dir / f"{name}_v{v}.csv"
        if name == "test" and frozen_runs is not None:
            continue  # never rewrite a frozen test manifest
        splits[splits["split"] == name].to_csv(path, index=False)
        print(f"wrote {path.name}: {int((splits['split'] == name).sum())} images")

    s = summary(splits)
    print("\n" + s.to_string())
    print("\ntest runs: " + ", ".join(sorted(splits.loc[splits['split'] == 'test', 'run'].unique())))
    print("val runs:  " + ", ".join(sorted(splits.loc[splits['split'] == 'val', 'run'].unique())))


if __name__ == "__main__":
    main(refreeze="--refreeze" in sys.argv[1:])
