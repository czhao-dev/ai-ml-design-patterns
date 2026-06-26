# Results Summary

## Objective

Classify aircraft surface damage into crack and dent categories using VGG16 transfer learning in two configurations: a frozen feature-extraction model and a model fine-tuned from `block5_conv3` onward.

## Dataset

- 446 total images (150 crack / 150 dent training, 48 crack / 48 dent validation, 25 crack / 25 dent held-out test)
- Binary image classification task

## Model Results

Evaluated on the full 50-image held-out test set (`scripts/04_evaluate_models.py`):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| VGG16 Feature-Extraction | 0.8600 | 0.8462 | 0.8800 | 0.8627 | 0.8880 |
| VGG16 Fine-Tuned | 0.8200 | 0.7857 | 0.8800 | 0.8302 | 0.8960 |

> **Methodology note:** The source course notebook evaluated the test set with `model.evaluate(test_generator, steps=test_generator.samples // batch_size)`, which floors to `steps=1` for 50 test images at `batch_size=32` — so only a single 32-image batch was ever actually scored, despite being reported as the overall "test accuracy." `04_evaluate_models.py` loads and scores all 50 held-out test images directly. The original notebook also trained for a fixed 5 epochs with no early stopping or model checkpointing; both models here train for up to 30 epochs with early stopping on validation loss, restoring the best weights. The notebook's separate, unrelated BLIP image-captioning demo (generic image-to-text, not tied to the crack/dent prediction) was dropped entirely, and a `block5_conv3`-onward fine-tuning comparison was added beyond the original's single frozen-base model.

## Interpretation

Both models recall dent damage equally well (0.88), but the frozen feature-extraction model outperforms the fine-tuned model on accuracy (0.86 vs. 0.82), precision (0.85 vs. 0.79), and F1 (0.86 vs. 0.83). The fine-tuned model does edge ahead on ROC-AUC (0.896 vs. 0.888), meaning its raw scores rank images slightly better even though its default 0.5-threshold predictions are less accurate overall. This differs from the sibling recyclable-material-classifier project, where fine-tuning improved every metric except accuracy — likely because this dataset's training split is much smaller (300 images vs. 800), so unfreezing `block5_conv3` onward gives the model more capacity to overfit before it generalizes.

## Key Takeaways

- A frozen, ImageNet-pretrained VGG16 backbone reaches 86% accuracy on this 300-image training set with only a small dense head trained from scratch — a strong baseline for a dataset this size.
- Fine-tuning the last convolutional block did not improve accuracy here, unlike on the larger recyclable-material dataset; with only 300 training images, the extra trainable capacity appears to trade some precision for a marginal ROC-AUC gain.
- Catching and fixing the test-evaluation bug from the source notebook mattered: the original's `steps=1` truncation only ever scored one batch, so its reported "test accuracy" wasn't measuring the full 50-image held-out set.

## Future Work

- Fine-tune additional convolutional blocks (e.g. from `block4_conv1`) to see if performance improves further.
- Add Grad-CAM visualizations to inspect which image regions drive each model's predictions.
- Extend beyond binary crack/dent classification to a multi-class or multi-label damage taxonomy.
- Package the best model behind a small Streamlit or Gradio demo for interactive classification.
