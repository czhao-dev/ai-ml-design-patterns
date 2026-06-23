"""Shared project paths and class labels."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_ARCHIVE = RAW_DATA_DIR / "images_dataSAT.tar"
EXTRACTED_DATASET_DIR = RAW_DATA_DIR / "images_dataSAT"

CLASS_NAMES = {
    "class_0_non_agri": "non-agricultural",
    "class_1_agri": "agricultural",
}

