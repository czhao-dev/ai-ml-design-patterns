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
| Keras CNN | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 1.0000 | 0.0247 |
| PyTorch CNN | 0.9988 | 0.9983 | 0.9993 | 0.9988 | 1.0000 | 0.0024 |
| Keras CNN-ViT Hybrid | 0.9958 | 0.9990 | 0.9927 | 0.9958 | 0.9998 | 0.0530 |
| PyTorch CNN-ViT Hybrid | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 1.0000 | 0.0047 |

> **Methodology note:** These metrics are computed by evaluating each model over the full `images_dataSAT` dataset (6,000 images), which overlaps with the data used to train that model. They reflect fit quality, not held-out generalization performance. A held-out evaluation is listed under Future Work.

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
