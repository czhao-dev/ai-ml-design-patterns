"""Tests for src/config.py."""

from pathlib import Path

from src import config


def test_project_root_is_repo_root():
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_derived_paths_nest_under_project_root():
    assert config.RAW_DATA_DIR == config.DATA_DIR / "raw"
    assert config.TRAINED_MODELS_DIR == config.MODELS_DIR / "trained"
    assert config.FIGURES_DIR == config.REPORTS_DIR / "figures"
    for path in (config.DATA_DIR, config.MODELS_DIR, config.REPORTS_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_dataset_paths_derive_from_raw_data_dir():
    assert config.DATASET_ARCHIVE == config.RAW_DATA_DIR / "images_dataSAT.tar"
    assert config.EXTRACTED_DATASET_DIR == config.RAW_DATA_DIR / "images_dataSAT"


def test_class_names_cover_both_binary_labels():
    assert config.CLASS_NAMES == {
        "class_0_non_agri": "non-agricultural",
        "class_1_agri": "agricultural",
    }
