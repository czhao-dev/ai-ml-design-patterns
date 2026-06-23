# Python Scripts

These files contain the project workflow as cleaned Python source code.

They are useful for:

- Reviewing the project as source code on GitHub
- Searching model, data, and evaluation logic
- Comparing framework implementations
- Reusing code in future scripts or apps

## Script Order

| Order | Script | Purpose |
|---:|---|---|
| 1 | `01_data_loading_memory_vs_generator.py` | Compare memory-based and generator-based image loading |
| 2 | `02_keras_data_pipeline.py` | Build Keras data loading and augmentation workflows |
| 3 | `03_pytorch_data_pipeline.py` | Build PyTorch Dataset and DataLoader workflows |
| 4 | `04_keras_cnn_classifier.py` | Train and evaluate a Keras CNN baseline |
| 5 | `05_pytorch_cnn_classifier.py` | Train and evaluate a PyTorch CNN baseline |
| 6 | `06_keras_vs_pytorch_cnn_comparison.py` | Compare CNN results across frameworks |
| 7 | `07_keras_cnn_vit_hybrid.py` | Build and train a Keras CNN-ViT hybrid |
| 8 | `08_pytorch_cnn_vit_hybrid.py` | Build and train a PyTorch CNN-ViT hybrid |
| 9 | `09_final_cnn_vit_evaluation.py` | Compare final hybrid models |

## Notes

Some scripts still depend on local data, downloaded files, and trained model artifacts. For a production-style command-line workflow, the next step would be to refactor repeated code into reusable modules under `src/`.
