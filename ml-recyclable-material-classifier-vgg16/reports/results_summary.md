# Results Summary

## Objective

Classify images into organic and recyclable material categories using VGG16 transfer learning in two configurations: a frozen feature-extraction model and a model fine-tuned from `block5_conv3` onward.

## Dataset

- 1,200 total images (500 O / 500 R training pool, 100 O / 100 R held-out test set)
- 800 training / 200 validation images (20% validation split of the training pool)
- Binary image classification task

## Model Results

Evaluated on the full 200-image held-out test set (`scripts/04_evaluate_models.py`):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| VGG16 Feature-Extraction | 0.8650 | 0.9195 | 0.8000 | 0.8556 | 0.9332 |
| VGG16 Fine-Tuned | 0.8650 | 0.8842 | 0.8400 | 0.8615 | 0.9534 |

> **Methodology note:** The source course notebook had a variable-naming bug where the cell intended to create a `test_generator` instead reassigned `train_generator` to point at the test directory — every training run in the original notebook was therefore fit on test-directory images, masked by a toy `steps_per_epoch=5` run that nobody scrutinized closely. `scripts/01_data_pipeline.py` and the training scripts (`02`, `03`) build three separate, correctly named generators (`train_generator`, `val_generator`, `test_generator`) so training only ever sees the training split. The original notebook's evaluation step also only sampled 50+50 of the 200 test images; the numbers above evaluate all 200 held-out test images. Both models were trained for real (not the original's 10-epoch toy run): the feature-extraction model ran 30 epochs and the fine-tuned model 20 epochs, both with early stopping on validation loss.

## Interpretation

Both models reached the same overall test accuracy (0.865), but fine-tuning the top convolutional block measurably improved ranking quality (ROC-AUC 0.9534 vs. 0.9332) and recall on recyclable material (0.84 vs. 0.80), at the cost of some precision on that class. This is the expected trade-off from fine-tuning: allowing the last convolutional block to adapt to this dataset's textures gives the model more discriminative power, while the frozen feature-extraction model leans on generic ImageNet features and stays more conservative about predicting the recyclable class.

## Key Takeaways

- A frozen, ImageNet-pretrained VGG16 backbone already gets to respectable accuracy (86.5%) on this dataset with only a small dense head trained from scratch.
- Fine-tuning the last convolutional block improves the model's ranking ability (ROC-AUC) and balances precision/recall more evenly across classes, even when overall accuracy is unchanged.
- Catching and fixing the train/test generator bug from the source notebook mattered: training on the correct split is what makes these numbers trustworthy.

## Future Work

- Fine-tune additional convolutional blocks (e.g. from `block4_conv1`) to see if ranking quality improves further.
- Add Grad-CAM visualizations to inspect which image regions drive each model's predictions.
- Package the best model behind a small Streamlit or Gradio demo for interactive classification.
- Track experiments with a reproducible configuration file instead of hardcoded script constants.
