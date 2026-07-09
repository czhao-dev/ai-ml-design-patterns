"""Shared path constants and sibling-project imports.

ml-model-compression is a sibling directory of cnn-vit-satellite-image-classifier
under the same applied-ml-projects working tree (both live in the same git
repo as plain subdirectories, not separate installed packages). Both projects
happen to name their local package `src`, so a plain `sys.path` insert would
have the sibling's `src.config` resolve against *this* project's already-
imported `src` package instead. To avoid that collision, the handful of
self-contained sibling modules we need are loaded directly by file path via
importlib, under distinct module names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# --- Local project paths -----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ml-model-compression/
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_SUMMARY_MD = REPORTS_DIR / "results_summary.md"
RESULTS_CACHE_JSON = REPORTS_DIR / "results.json"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"  # gitignored — compressed checkpoints

# --- Sibling project location --------------------------------------------
SIBLING_PROJECT_ROOT = (PROJECT_ROOT.parent / "cnn-vit-satellite-image-classifier").resolve()

if not SIBLING_PROJECT_ROOT.is_dir():
    raise FileNotFoundError(
        f"Expected sibling project at {SIBLING_PROJECT_ROOT}; "
        "ml-model-compression assumes it sits alongside cnn-vit-satellite-image-classifier."
    )


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_satellite_config = _load_module(
    "satellite_config", SIBLING_PROJECT_ROOT / "src" / "config.py"
)
_satellite_metrics = _load_module(
    "satellite_metrics", SIBLING_PROJECT_ROOT / "src" / "metrics.py"
)
_satellite_pytorch_models = _load_module(
    "satellite_pytorch_models", SIBLING_PROJECT_ROOT / "serve" / "pytorch_models.py"
)

CNN_ViT_Hybrid = _satellite_pytorch_models.CNN_ViT_Hybrid
build_satellite_cnn = _satellite_pytorch_models.build_satellite_cnn
SATELLITE_CLASS_NAMES = _satellite_config.CLASS_NAMES
EXTRACTED_DATASET_DIR = _satellite_config.EXTRACTED_DATASET_DIR
SATELLITE_TRAINED_MODELS_DIR = _satellite_config.TRAINED_MODELS_DIR
binary_classification_metrics = _satellite_metrics.binary_classification_metrics

# --- Checkpoint locations (staged from the parent project's original run) --
PYTORCH_CNN_CHECKPOINT = SATELLITE_TRAINED_MODELS_DIR / "ai_capstone_pytorch_state_dict.pth"
PYTORCH_CNN_VIT_CHECKPOINT = SATELLITE_TRAINED_MODELS_DIR / "pytorch_cnn_vit_ai_capstone_model_state_dict.pth"

# --- Canonical split / CNN-ViT hyperparameters --------------------------
# SPLIT_SEED matches scripts/05_pytorch_cnn_classifier.py's original CNN
# training split (SEED=42), reused here as the single canonical split for
# every technique in this project (pruning/quantization target the CNN
# directly, so this gives a zero-leakage held-out set for it; the student
# also trains/evaluates on it). The CNN-ViT teacher was originally trained
# on a *different* split (SEED=7331, script 08) -- see the loader in
# eval_utils.py for the resulting caveat when it's used as a frozen
# soft-label source during distillation.
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8

# CNN_ViT_Hybrid's constructor defaults (depth=6, heads=8) do NOT match how
# the real checkpoint was trained -- these must be passed explicitly.
CNN_VIT_EMBED_DIM = 768
CNN_VIT_DEPTH = 3
CNN_VIT_HEADS = 6

for _dir in (REPORTS_DIR, FIGURES_DIR, TRAINED_MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
