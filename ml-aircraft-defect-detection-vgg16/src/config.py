"""Shared project paths, class labels, and training constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
DATASET_ARCHIVE = RAW_DATA_DIR / "aircraft-damage-dataset-v1.tar"
EXTRACTED_DATASET_DIR = RAW_DATA_DIR / "aircraft_damage_dataset_v1"
TRAIN_DIR = EXTRACTED_DATASET_DIR / "train"
VALID_DIR = EXTRACTED_DATASET_DIR / "valid"
TEST_DIR = EXTRACTED_DATASET_DIR / "test"

# Alphabetical class order ("crack" < "dent") matches Keras's flow_from_directory
# label indexing, so crack -> 0 and dent -> 1 throughout this project.
CLASS_NAMES = {
    "crack": "crack damage",
    "dent": "dent damage",
}

IMG_SIZE = (224, 224)
INPUT_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 32
SEED = 42

FEATURE_EXTRACTION_MODEL_PATH = TRAINED_MODELS_DIR / "vgg16_feature_extraction_aircraft_damage_classifier.keras"
FINE_TUNED_MODEL_PATH = TRAINED_MODELS_DIR / "vgg16_fine_tuned_aircraft_damage_classifier.keras"
