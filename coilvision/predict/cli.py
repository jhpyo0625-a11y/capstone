"""`coil-predict <folder>` → per-image CSV + Grad-CAM overlays (spec §6.5).

Reports one row per image — no part-level rollup (filenames don't identify
physical parts, spec §2). Refuses to run on a preprocess-version mismatch.
Phase 5 — not yet implemented.
"""


def main() -> None:
    raise SystemExit("coil-predict is not implemented yet (Phase 5 — see COIL_CLASSIFIER_SPEC.md)")
