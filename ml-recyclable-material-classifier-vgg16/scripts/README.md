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

The source course notebook this project was built from had a bug where the cell meant to create a `test_generator` actually reassigned `train_generator` to the test directory, so training silently ran on test images. `01_data_pipeline.py` and the training scripts build three separate, correctly named generators to avoid this. See [`reports/results_summary.md`](../reports/results_summary.md) for details.

Scripts assume they are run with the project's `.venv` activated, from anywhere inside the repository (paths are resolved via `src/config.py`, not the working directory).
