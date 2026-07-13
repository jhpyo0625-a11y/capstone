"""Pin README.md's load-bearing claims to reality so the manual can't rot silently."""

from pathlib import Path

from coilvision.config import PROJECT_ROOT, load_config

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
CFG = load_config()


def test_referenced_scripts_exist_and_are_documented():
    for script in ("predict.bat", "retrain.bat", "register_task.ps1"):
        assert (PROJECT_ROOT / "scripts" / script).exists(), f"scripts/{script} missing"
        assert script in README, f"README no longer mentions scripts/{script}"
    assert (PROJECT_ROOT / "scripts" / "watcher.bat").exists()


def test_documented_quarantine_reasons_match_code():
    validate_src = (PROJECT_ROOT / "coilvision" / "data" / "validate.py").read_text(encoding="utf-8")
    retrain_src = (PROJECT_ROOT / "coilvision" / "pipeline" / "retrain.py").read_text(encoding="utf-8")
    code = validate_src + retrain_src
    for reason in ("unreadable", "unexpected_dims_", "unparseable_filename",
                   "unknown_label_folder", "duplicate_content", "name_collision"):
        assert reason in README, f"README missing quarantine reason {reason}"
        assert reason in code, f"documented reason {reason} no longer produced by code"


def test_documented_thresholds_match_config():
    n = CFG["retrain"]["min_new_images"]
    assert f"{n} new images" in README or f"{n} images" in README, \
        f"README's trigger threshold no longer matches config ({n})"
    assert CFG["retrain"]["max_days_between"] == 7 and "weekly" in README
    assert "production_recall_target" in CFG["eval"]  # the documented threshold policy knob


def test_documented_commands_use_uv():
    # bare `python -m` would run the system 3.14 without deps — every command must go through uv or a .bat
    for line in README.splitlines():
        if "python -m coilvision" in line:
            assert "uv run python -m coilvision" in line, f"command missing uv prefix: {line.strip()}"


def test_out_of_git_folders_are_documented_for_migration():
    for folder in ("Coil-image-Dataset", "models", "data_accepted"):
        assert folder in README, f"README migration note missing {folder}"
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("Coil-image-Dataset/", "models/", "data_accepted/"):
        assert pattern in gitignore


def test_production_pointer_referenced_paths():
    assert "models/production/POINTER.json" in README.replace("\\", "/")
    assert (Path(PROJECT_ROOT) / "configs" / "config.yaml").exists()
