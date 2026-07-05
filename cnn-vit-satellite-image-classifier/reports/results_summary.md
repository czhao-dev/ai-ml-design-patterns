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
| Keras CNN | 0.9933 | 1.0000 | 0.9867 | 0.9933 | 1.0000 | 0.0257 |
| PyTorch CNN | 0.9983 | 0.9965 | 1.0000 | 0.9983 | 1.0000 | 0.0041 |
| Keras CNN-ViT Hybrid | 0.9942 | 0.9966 | 0.9917 | 0.9942 | 0.9991 | 0.1138 |
| PyTorch CNN-ViT Hybrid | 0.9967 | 0.9983 | 0.9950 | 0.9967 | 0.9999 | 0.0104 |

> **Methodology note:** These numbers come from evaluating each model on its held-out validation split only (1,200 images, 20% of `images_dataSAT`) — the same split reserved during training and never seen by that model's weights. `scripts/06_keras_vs_pytorch_cnn_comparison.py` and `scripts/09_final_cnn_vit_evaluation.py` reconstruct this split using the exact seed/`validation_split` parameters from the corresponding training script (`04`, `05`, `07`, `08`), so no training image is re-scored. Earlier versions of this table evaluated over the full 6,000-image dataset, which leaked training data into the metrics — the corrected numbers above are close to the old ones because the held-out split was already used for checkpoint selection during training, not because the leak didn't matter.

## Interpretation

All models performed strongly on the balanced satellite image dataset. The PyTorch CNN and PyTorch CNN-ViT hybrid produced the highest overall scores in these recorded runs.

The CNN baselines were already highly effective, suggesting that local visual patterns in the satellite tiles are strong indicators of agricultural land. The CNN-ViT hybrids add a transformer component that can model broader spatial relationships, which is useful for land-use imagery where texture, field boundaries, and larger spatial patterns can matter together.

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
