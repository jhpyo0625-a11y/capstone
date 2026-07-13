"""Watch-folder trigger (spec §6.6): counts incoming/, fires retrain at threshold.

Fires when new images >= retrain.min_new_images, OR when ANY new images have
waited retrain.max_days_between days since the last successful cycle.
Designed to run hourly via Task Scheduler (scripts/register_task.ps1);
the retrain lock makes overlapping fires harmless.

Run:  uv run python -m coilvision.pipeline.watcher
"""

from __future__ import annotations

import time
from datetime import datetime

from coilvision.anomaly import anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.pipeline import retrain


def should_run(n_new: int, last_success: str | None, now: datetime, retrain_cfg: dict) -> tuple[bool, str]:
    if n_new == 0:
        return False, "no new images"
    if n_new >= retrain_cfg["min_new_images"]:
        return True, f"{n_new} new images >= threshold {retrain_cfg['min_new_images']}"
    if last_success is None:
        return True, f"{n_new} new images and no previous successful cycle"
    days = (now - datetime.strptime(last_success, "%Y-%m-%d %H:%M:%S")).total_seconds() / 86400
    if days >= retrain_cfg["max_days_between"]:
        return True, f"{n_new} new images waited {days:.1f} days >= {retrain_cfg['max_days_between']}"
    return False, f"{n_new} new images below threshold; {days:.1f} days since last cycle"


def main() -> None:
    import pandas as pd

    cfg = anomaly_cfg(load_config())
    n_incoming = sum(1 for _ in resolve_path(cfg, "incoming_dir").rglob("*.bmp"))
    # a crashed run may have merged files without processing them — count those as new
    manifest_path = resolve_path(cfg, "manifests_dir") / "manifest.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False) if manifest_path.exists() else pd.DataFrame()
    drift = max(0, retrain.unmanifested_accepted(cfg, manifest))
    state = retrain.read_state(cfg)
    fire, reason = should_run(n_incoming + drift, state.get("last_success"), datetime.now(), cfg["retrain"])
    if drift:
        reason += f" (incl. {drift} unprocessed from a previous run)"
    print(f"watcher {time.strftime('%Y-%m-%d %H:%M:%S')}: {reason} -> {'RUN' if fire else 'skip'}")
    if fire:
        retrain.run(force=drift > 0 and n_incoming == 0, trigger=f"watcher: {reason}")


if __name__ == "__main__":
    main()
