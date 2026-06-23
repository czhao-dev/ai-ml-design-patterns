# Models

Large trained model artifacts are stored locally under:

```text
models/trained/
```

That folder is ignored by Git. For public sharing, upload large model files to one of these places instead:

- GitHub Releases
- Hugging Face Hub
- Google Drive
- Git LFS, if model versioning is important

## Local Artifacts

The local trained artifacts from the completed model runs are:

| File | Model |
|---|---|
| `ai_capstone_keras_best_model.model.keras` | Keras CNN baseline |
| `ai_capstone_pytorch_state_dict.pth` | PyTorch CNN baseline |
| `keras_cnn_vit_ai_capstone.keras` | Keras CNN-ViT hybrid |
| `pytorch_cnn_vit_ai_capstone_model_state_dict.pth` | PyTorch CNN-ViT hybrid |
