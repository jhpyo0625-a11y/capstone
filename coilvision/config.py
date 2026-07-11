"""Load configs/config.yaml — the single source for every knob (spec §5)."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a paths.* entry relative to the project root."""
    return PROJECT_ROOT / cfg["paths"][key]
