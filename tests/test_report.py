"""Threshold selection + classification logic for the formal eval (pure functions)."""

import numpy as np
import pandas as pd

from coilvision.eval.report import classify, per_run_table, select_threshold


def test_select_threshold_lowest_frr_at_target():
    #  defects: 0.9, 0.8, 0.7, 0.3   passes: 0.85, 0.5, 0.2, 0.1
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.85, 0.5, 0.2, 0.1])
    is_fail = np.array([True, True, True, True, False, False, False, False])
    op = select_threshold(scores, is_fail, target=0.75)
    # thr=0.7 keeps 3/4 defects (0.75) and rejects 1/4 passes (0.85)
    assert op["threshold"] == 0.7
    assert op["fail_recall"] == 0.75
    assert op["false_reject_rate"] == 0.25


def test_select_threshold_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    is_fail = np.array([True, True, False, False])
    op = select_threshold(scores, is_fail, target=1.0)
    assert op["threshold"] == 0.8
    assert op["fail_recall"] == 1.0
    assert op["false_reject_rate"] == 0.0


def test_classify_threshold_and_vote():
    pred = classify(np.array([0.99, 0.5, 0.99]), ["Dent", "Loose", "Loose"], threshold=0.9)
    assert pred.tolist() == ["Dent", "Pass", "Loose"]


def test_per_run_table_counts():
    df = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "run": ["r1", "r1", "r1", "r2"],
            "class": ["Dent", "Pass", "Pass", "Loose"],
            "score": [0.95, 0.96, 0.10, 0.20],  # defect caught; one pass falsely rejected; loose missed
        }
    )
    t = per_run_table(df, threshold=0.9).set_index("run")
    assert t.loc["r1", "defects"] == 1 and t.loc["r1", "missed_defects"] == 0
    assert t.loc["r1", "false_rejects"] == 1 and t.loc["r1", "passes"] == 2
    assert t.loc["r2", "missed_defects"] == 1
