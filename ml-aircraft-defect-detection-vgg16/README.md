# Aircraft Damage Classification with VGG16 Transfer Learning

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
![Python](https://img.shields.io/badge/python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)

Binary classification of aircraft surface damage into **crack** and **dent** categories using VGG16 transfer learning, implemented in two configurations: a frozen feature-extraction model and a model fine-tuned from `block5_conv3` onward, with both evaluated on a held-out test set.

## Table of Contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Project Background](#project-background)
- [Future Work](#future-work)
- [License](#license)

## Highlights

- Binary aircraft damage classification (crack vs. dent) on a balanced, 446-image dataset with dedicated train/validation/test splits
- Two VGG16 transfer-learning configurations compared head-to-head: frozen feature extraction vs. fine-tuning the top convolutional block
- Full evaluation suite: accuracy, precision, recall, F1, ROC-AUC, and confusion matrices on the complete held-out test set
- Caught and fixed a test-evaluation bug in the original course notebook: floor-division on `steps` meant only one 32-image batch of the 50-image test set was ever actually scored (see [Results](#results))

## Dataset

The dataset contains 446 JPG images of aircraft surface damage, balanced across two classes:

| Class | Meaning | Train | Valid | Test |
|---|---|---:|---:|---:|
| `crack` | Crack damage | 150 | 48 | 25 |
| `dent` | Dent damage | 150 | 48 | 25 |

Originally published as the [Aircraft Damage Detection](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) dataset on Roboflow (CC BY 4.0), re-hosted by IBM Skills Network. The data pipeline downloads the archive automatically:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar
```

Large local data files are kept out of Git. See [`data/data.md`](data/data.md) for setup notes.

## Approach

**Feature extraction** ([`02_vgg16_feature_extraction_model.py`](scripts/02_vgg16_feature_extraction_model.py)): an ImageNet-pretrained VGG16 convolutional base, fully frozen, feeding a small dense classification head (`Dense(512)` → `Dropout(0.3)` ×2 → `Dense(1, sigmoid)`). Trained with the Adam optimizer and early stopping on validation loss.

**Fine-tuning** ([`03_vgg16_fine_tuned_model.py`](scripts/03_vgg16_fine_tuned_model.py)): the same architecture, but with the convolutional base unfrozen from `block5_conv3` onward, letting the top of the backbone adapt to aircraft surface textures while keeping earlier, more generic layers frozen.

Both models train on the 300-image training split with on-the-fly augmentation (width/height shift, horizontal flip) and the 96-image validation split, then get evaluated together in [`04_evaluate_models.py`](scripts/04_evaluate_models.py) on the full 50-image held-out test set.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| VGG16 Feature-Extraction | 0.8600 | 0.8462 | 0.8800 | 0.8627 | 0.8880 |
| VGG16 Fine-Tuned | 0.8200 | 0.7857 | 0.8800 | 0.8302 | 0.8960 |

Both models reach a similar recall on dent damage (0.88), but the frozen feature-extraction model edges out the fine-tuned model on accuracy, precision, and F1, while fine-tuning the top convolutional block gives a small bump in ranking quality (ROC-AUC). On this small, 300-image training set, letting `block5_conv3` onward adapt doesn't pay off as clearly as it does on the larger recyclable-material dataset — likely because there's less data here to fine-tune on before overfitting risk increases.

> **Methodology note:** The source course notebook evaluated the test set with `model.evaluate(test_generator, steps=test_generator.samples // batch_size)`, which floors to `steps=1` for 50 test images at `batch_size=32` — so only a single 32-image batch was ever actually scored, despite being reported as the overall "test accuracy." This project's evaluation script scores all 50 held-out test images directly, and also adds a fine-tuning comparison beyond the original notebook's single frozen-base model. See [`reports/results_summary.md`](reports/results_summary.md) for the full writeup.

## Repository Structure

```text
.
├── data/
│   └── data.md
├── models/
│   └── models.md
├── reports/
│   ├── figures/
│   └── results_summary.md
├── scripts/
│   ├── 01_data_pipeline.py
│   ├── 02_vgg16_feature_extraction_model.py
│   ├── 03_vgg16_fine_tuned_model.py
│   └── 04_evaluate_models.py
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── visualization.py
├── README.md
└── requirements.txt
```

`scripts/` contains the full project workflow as self-contained Python source files, numbered in execution order — see [`scripts/README.md`](scripts/README.md) for what each one does. `src/` holds small reusable helpers (paths, metrics, plotting) shared across scripts.

## Getting Started

### Requirements

- Python 3.13
- ~1 GB of free disk space for TensorFlow and its dependencies

### Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Workflow

Each script downloads any data or pretrained weights it needs on first run. Scripts resolve all paths through `src/config.py`, so they can be run from anywhere inside the repository:

```bash
python scripts/01_data_pipeline.py
python scripts/02_vgg16_feature_extraction_model.py
python scripts/03_vgg16_fine_tuned_model.py
python scripts/04_evaluate_models.py
```

Scripts are numbered in execution order; `04` expects the trained model artifacts from `02` and `03` to exist first.

## Project Background

This project began as an IBM/Coursera "Classification and Captioning" course final-project notebook. It has since been reorganized into a clear Python workflow: a fine-tuning comparison was added beyond the original single frozen-base model, a test-evaluation bug was fixed (see [Results](#results)), reusable helper modules and documented results were added, and the notebook's unrelated BLIP image-captioning demo was dropped to keep the project focused on the defect classifier.

## Future Work

- Fine-tune additional convolutional blocks (e.g. from `block4_conv1`) to see if performance improves further
- Add Grad-CAM visualizations to inspect which image regions drive each model's predictions
- Extend beyond binary crack/dent classification to a multi-class or multi-label damage taxonomy
- Package the best model behind a small Streamlit or Gradio demo for interactive classification

## License

This project is licensed under the MIT License. See the root [LICENSE](../LICENSE) file for details.
