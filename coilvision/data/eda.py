"""EDA report (spec §6.1 step 6): distributions, layout clusters, brightness by
code, duplicate detection. Refreshed on every retrain.

Run:  uv run python -m coilvision.data.eda
Writes artifacts/eda/eda_report.md + plots.
"""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from coilvision.config import load_config, resolve_path


def brightness_by_code(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mean brightness of cached ROI crops, grouped by the [code] filename field."""
    cache_dir = resolve_path(cfg, "cache_dir")
    manifest = pd.read_csv(resolve_path(cfg, "manifests_dir") / "manifest.csv", keep_default_na=False)
    code_by_hash = manifest.set_index("hash")["code"]
    rows = []
    for r in index.itertuples():
        img = cv2.imread(str(cache_dir / r.cache_file), cv2.IMREAD_GRAYSCALE)
        rows.append({"code": int(code_by_hash.get(r.hash, -1)), "brightness": float(img[img > 0].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out_dir = resolve_path(cfg, "artifacts_dir") / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve_path(cfg, "manifests_dir") / "manifest.csv", keep_default_na=False)
    idx_path = resolve_path(cfg, "manifests_dir") / f"cache_index_v{cfg['preprocess']['version']}.csv"
    index = pd.read_csv(idx_path, keep_default_na=False) if idx_path.exists() else None

    lines = ["# EDA report", "", f"Images: {len(manifest)}, valid: {int(manifest['valid'].sum())}", ""]

    lines += ["## Class counts", "", manifest["class"].value_counts().to_markdown(), ""]

    lines += ["## Runs × class", "", pd.crosstab(manifest["run"], manifest["class"]).to_markdown(), ""]

    lines += ["## Code × class", "", pd.crosstab(manifest["code"], manifest["class"]).to_markdown(), ""]

    lines += ["## Shot × class", "", pd.crosstab(manifest["shot"], manifest["class"]).to_markdown(), ""]

    by_layout = manifest.groupby("layout").agg(runs=("run", "nunique"), images=("relpath", "count"))
    lines += ["## Layout clusters", "", by_layout.to_markdown(), ""]

    exact = manifest[manifest.duplicated(subset="hash", keep=False)].sort_values("hash")
    lines += ["## Exact duplicate files (same content hash)", ""]
    lines += [exact[["relpath", "hash"]].to_markdown(index=False) if len(exact) else "None found.", ""]

    near = manifest[manifest.duplicated(subset="ahash", keep=False)]
    n_groups = near.groupby("ahash").ngroups if len(near) else 0
    lines += [
        "## Near-duplicates (same 64-bit average hash)",
        "",
        f"{len(near)} images in {n_groups} groups (visually near-identical frames; "
        f"expected within a run — this is why splits group by run).",
        "",
    ]

    # plots
    fig, ax = plt.subplots(figsize=(5, 3))
    manifest["class"].value_counts().plot.bar(ax=ax, color=["#4a4", "#d33", "#e80"])
    ax.set_title("Class counts")
    fig.tight_layout()
    fig.savefig(out_dir / "class_counts.png", dpi=120)

    if index is not None:
        bb = brightness_by_code(index, cfg)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bb.boxplot(column="brightness", by="code", ax=ax)
        ax.set_title("ROI brightness by [code]")
        ax.set_ylabel("mean gray value")
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(out_dir / "brightness_by_code.png", dpi=120)
        lines += [
            "## Brightness by [code]",
            "",
            "![brightness](brightness_by_code.png)",
            "",
            bb.groupby("code")["brightness"].describe()[["count", "mean", "std"]].round(1).to_markdown(),
            "",
        ]
        n_fallback = int((~index["roi_confident"]).sum())
        lines += ["## ROI detection", "", f"confident: {len(index) - n_fallback}/{len(index)}, fallback: {n_fallback}", ""]

    (out_dir / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA report -> {out_dir / 'eda_report.md'}")


if __name__ == "__main__":
    main()
