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
| `vgg16_feature_extraction_recyclable_classifier.keras` | VGG16 with a frozen convolutional base (feature extraction) |
| `vgg16_fine_tuned_recyclable_classifier.keras` | VGG16 fine-tuned from `block5_conv3` onward |
