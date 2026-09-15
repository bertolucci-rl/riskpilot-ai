from pathlib import Path

from riskpilot import config


def test_project_root_points_at_repository():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "src" / "riskpilot" / "config.py").is_file()


def test_directories_are_nested_under_project_root():
    for path in (
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.ARTIFACTS_DIR,
        config.FIGURES_DIR,
        config.METRICS_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
        config.APPLICATION_TRAIN_PATH,
    ):
        assert isinstance(path, Path)
        assert config.PROJECT_ROOT in path.parents, path


def test_constants():
    assert config.RANDOM_STATE == 42
    assert 0.0 < config.TEST_SIZE < 0.5
    assert config.TARGET_COL == "TARGET"
    assert config.ID_COL == "SK_ID_CURR"
    assert set(config.ALLOWED_TARGET_VALUES) == {0, 1}
    assert config.APPLICATION_TRAIN_PATH.name == "application_train.csv"


def test_no_user_specific_absolute_paths_in_source():
    source_root = config.PROJECT_ROOT / "src"
    offenders = []
    for py_file in source_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "C:\\Users" in text or "C:/Users" in text or "/home/" in text:
            offenders.append(py_file)
    assert not offenders, f"Hard-coded absolute paths found in {offenders}"
