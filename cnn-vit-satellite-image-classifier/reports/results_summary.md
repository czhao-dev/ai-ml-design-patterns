# Results Summary

## Objective

Classify satellite image tiles into agricultural and non-agricultural land categories using deep learning models in Keras/TensorFlow and PyTorch.

## Dataset

- 6,000 total satellite image tiles
- 3,000 agricultural tiles
- 3,000 non-agricultural tiles
- Binary image classification task

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Loss |
|---|---:|---:|---:|---:|---:|---:|
| Keras CNN | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 0.9999 | 0.0259 |
| PyTorch CNN | 0.9992 | 0.9983 | 1.0000 | 0.9991 | 1.0000 | 0.0031 |
| Keras CNN-ViT Hybrid | 0.9917 | 1.0000 | 0.9833 | 0.9916 | 0.9834 | 0.1362 |
| PyTorch CNN-ViT Hybrid | 0.9750 | 0.9539 | 0.9983 | 0.9756 | 0.9997 | 0.0662 |

> **Methodology note:** These numbers come from evaluating each model on its held-out validation split only (1,200 images, 20% of `images_dataSAT`) — the same split reserved during training and never seen by that model's weights. `scripts/06_keras_vs_pytorch_cnn_comparison.py` and `scripts/09_final_cnn_vit_evaluation.py` reconstruct this split using the exact seed/`validation_split` parameters from the corresponding training script (`04`, `05`, `07`, `08`), so no training image is re-scored. All four checkpoints were trained from scratch end-to-end on a GCP T4 GPU VM (20 epochs for each CNN baseline, then a frozen-backbone transformer head trained for 3 epochs Keras / 5 epochs PyTorch on top) — no pretrained weights were downloaded from IBM's course storage at any stage, unlike earlier runs of this table.

## Interpretation

The CNN baselines were already highly effective on their own — local visual patterns (color, texture) in the satellite tiles are strong indicators of agricultural land, and both frameworks' from-scratch CNNs cleared 99% accuracy on the held-out split (PyTorch CNN reached 99.92%, the strongest single model here).

Genuinely retraining the CNN-ViT hybrids (instead of fine-tuning IBM's already-tuned checkpoint) tells a more interesting story than the CNN baselines alone: the Keras hybrid held nearly steady with its CNN counterpart (99.17% accuracy), but the PyTorch hybrid dropped to 97.50% accuracy with a lower precision (95.39%) than any other model in this table. Both hybrids freeze the CNN backbone and only train a randomly-initialized transformer head, for a handful of epochs (3 Keras, 5 PyTorch) — a useful, honest data point that a from-scratch hybrid isn't automatically an improvement over the CNN baseline within this small training budget, and that the two frameworks' hybrids don't converge equally fast under it.

## Key Takeaways

- Generator and framework-native data loaders are better suited than loading all images into memory.
- Keras provides a compact high-level workflow for rapid experimentation.
- PyTorch provides explicit control over model architecture, training loops, and evaluation.
- CNN-ViT hybrids are a natural extension when both local texture and global layout are relevant.

## Future Work

- Add a standalone inference script for new satellite tiles.
- Validate on a geographically distinct holdout set.
- Add Grad-CAM or attention visualizations for interpretability.
- Package the best model behind a small Streamlit app.
