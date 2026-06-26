# Python Scripts

These files contain the project workflow as cleaned Python source code.

They are useful for:

- Reviewing the project as source code on GitHub
- Searching model, data, and evaluation logic
- Reusing code in future scripts or apps

## Script Order

| Order | Script | Purpose |
|---:|---|---|
| 1 | `01_data_pipeline.py` | Download and extract the dataset, build train/validation/test image generators, visualize augmented samples |
| 2 | `02_vgg16_feature_extraction_model.py` | Train a VGG16 model with a frozen convolutional base (feature extraction) |
| 3 | `03_vgg16_fine_tuned_model.py` | Fine-tune a VGG16 model from `block5_conv3` onward |
| 4 | `04_evaluate_models.py` | Evaluate both saved models on the full held-out test set |

## Notes

The source course notebook this project was built from had two issues fixed here:

- Its test evaluation called `model.evaluate(test_generator, steps=test_generator.samples // batch_size)`, which floors to `steps=1` for a 50-image test set with `batch_size=32` — so only a single 32-image batch was ever actually scored, despite being reported as the "test accuracy." `04_evaluate_models.py` loads and scores all 50 held-out test images directly.
- The notebook also bolted on an unrelated BLIP image-captioning demo (generic "describe this image" captions, not tied to the crack/dent prediction). It has been dropped entirely to keep this project focused on the defect classifier; see [`reports/results_summary.md`](../reports/results_summary.md) for details.

This project also adds a fine-tuning comparison (`03_vgg16_fine_tuned_model.py`) beyond the original notebook's single frozen-base model, mirroring the structure of the sibling [`ml-recyclable-material-classifier-vgg16`](../../ml-recyclable-material-classifier-vgg16/) project.

Scripts assume they are run with the project's `.venv` activated, from anywhere inside the repository (paths are resolved via `src/config.py`, not the working directory).
