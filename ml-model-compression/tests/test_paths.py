"""Tests for src/paths.py.

paths.py loads a handful of modules from the sibling
cnn-vit-satellite-image-classifier project by file path (to avoid a `src`
package name collision -- see the module docstring). These tests confirm
that cross-project wiring resolves correctly.
"""

from src import paths


def test_sibling_project_root_exists_and_is_the_expected_directory():
    assert paths.SIBLING_PROJECT_ROOT.is_dir()
    assert paths.SIBLING_PROJECT_ROOT.name == "cnn-vit-satellite-image-classifier"


def test_local_paths_nest_under_project_root():
    assert paths.FIGURES_DIR == paths.REPORTS_DIR / "figures"
    assert paths.TRAINED_MODELS_DIR == paths.MODELS_DIR / "trained"
    for path in (paths.REPORTS_DIR, paths.FIGURES_DIR, paths.TRAINED_MODELS_DIR):
        assert paths.PROJECT_ROOT in path.parents


def test_local_directories_are_created_on_import():
    assert paths.REPORTS_DIR.is_dir()
    assert paths.FIGURES_DIR.is_dir()
    assert paths.TRAINED_MODELS_DIR.is_dir()


def test_checkpoint_paths_point_inside_the_sibling_project():
    assert paths.SIBLING_PROJECT_ROOT in paths.PYTORCH_CNN_CHECKPOINT.parents
    assert paths.SIBLING_PROJECT_ROOT in paths.PYTORCH_CNN_VIT_CHECKPOINT.parents


def test_sibling_modules_are_loaded_and_usable():
    assert paths.SATELLITE_CLASS_NAMES == {
        "class_0_non_agri": "non-agricultural",
        "class_1_agri": "agricultural",
    }
    assert callable(paths.binary_classification_metrics)
    assert callable(paths.build_satellite_cnn)
    assert callable(paths.CNN_ViT_Hybrid)


def test_cnn_vit_hyperparameters_override_the_class_defaults():
    # The trained checkpoint used depth=3, heads=6 -- not CNN_ViT_Hybrid's
    # own defaults (6, 8) -- so callers must always pass these explicitly.
    model = paths.CNN_ViT_Hybrid(
        num_classes=2, embed_dim=paths.CNN_VIT_EMBED_DIM, depth=paths.CNN_VIT_DEPTH, heads=paths.CNN_VIT_HEADS
    )
    assert len(model.vit.blocks) == paths.CNN_VIT_DEPTH
