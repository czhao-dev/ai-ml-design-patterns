"""Shared project paths, class labels, and training constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
DATASET_ARCHIVE = RAW_DATA_DIR / "o-vs-r-split-reduced-1200.zip"
EXTRACTED_DATASET_DIR = RAW_DATA_DIR / "o-vs-r-split"
TRAIN_DIR = EXTRACTED_DATASET_DIR / "train"
TEST_DIR = EXTRACTED_DATASET_DIR / "test"

# Alphabetical class order ("O" < "R") matches Keras's flow_from_directory
# label indexing, so O -> 0 and R -> 1 throughout this project.
CLASS_NAMES = {
    "O": "organic material",
    "R": "recyclable material",
}

IMG_SIZE = (150, 150)
INPUT_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42

FEATURE_EXTRACTION_MODEL_PATH = TRAINED_MODELS_DIR / "vgg16_feature_extraction_recyclable_classifier.keras"
FINE_TUNED_MODEL_PATH = TRAINED_MODELS_DIR / "vgg16_fine_tuned_recyclable_classifier.keras"
