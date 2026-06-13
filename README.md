# Satellite Land Classification with CNNs and Vision Transformers

This project classifies satellite image tiles as **agricultural** or **non-agricultural** land using deep learning models built in both Keras/TensorFlow and PyTorch.

The work progresses from data-loading experiments to CNN baselines, then to CNN-Vision Transformer hybrid models that combine local feature extraction with global attention.

## Project Highlights

- Binary land-use classification from satellite image tiles
- 6,000 image dataset with balanced classes
- Keras and PyTorch data pipelines
- CNN baselines in both frameworks
- CNN-ViT hybrid models in both frameworks
- Cross-framework evaluation using accuracy, precision, recall, F1, ROC-AUC, loss, and confusion matrices

## Dataset

The dataset contains 6,000 JPG satellite tiles:

| Class | Meaning | Images |
|---|---|---:|
| `class_0_non_agri` | Non-agricultural land | 3,000 |
| `class_1_agri` | Agricultural land | 3,000 |

The project data pipeline downloads the dataset from IBM Skills Network cloud storage:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
```

Large local data files are kept out of Git. See `data/data.md` for setup notes.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Keras CNN | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 1.0000 |
| PyTorch CNN | 0.9988 | 0.9983 | 0.9993 | 0.9988 | 1.0000 |
| Keras CNN-ViT Hybrid | 0.9958 | 0.9990 | 0.9927 | 0.9958 | 0.9998 |
| PyTorch CNN-ViT Hybrid | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 1.0000 |

The PyTorch models achieved the strongest scores in these runs, with the final PyTorch CNN-ViT hybrid reaching 99.90% accuracy.

> **Methodology note:** These numbers come from evaluating each model over the full `images_dataSAT` dataset, which overlaps with the data used to train that model. They reflect fit quality on the full dataset rather than held-out generalization performance. See [Next Improvements](#next-improvements) for a planned held-out evaluation.

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
│   ├── 01_data_loading_memory_vs_generator.py
│   ├── 02_keras_data_pipeline.py
│   ├── 03_pytorch_data_pipeline.py
│   ├── 04_keras_cnn_classifier.py
│   ├── 05_pytorch_cnn_classifier.py
│   ├── 06_keras_vs_pytorch_cnn_comparison.py
│   ├── 07_keras_cnn_vit_hybrid.py
│   ├── 08_pytorch_cnn_vit_hybrid.py
│   └── 09_final_cnn_vit_evaluation.py
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── visualization.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Python Scripts

The `scripts/` folder contains the project workflow as Python source code, organized from data loading through final model evaluation.

The Python scripts are:

1. `01_data_loading_memory_vs_generator.py`
2. `02_keras_data_pipeline.py`
3. `03_pytorch_data_pipeline.py`
4. `04_keras_cnn_classifier.py`
5. `05_pytorch_cnn_classifier.py`
6. `06_keras_vs_pytorch_cnn_comparison.py`
7. `07_keras_cnn_vit_hybrid.py`
8. `08_pytorch_cnn_vit_hybrid.py`
9. `09_final_cnn_vit_evaluation.py`

These scripts are intended for code review, search, and future refactoring.

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run or inspect the Python scripts in `scripts/` for the source workflow.

## Project Background

This project began as an IBM/Coursera deep learning capstone sequence. It has since been reorganized to include a clear Python workflow, documented results, reusable helper modules, and handling for large data and model artifacts.

## Next Improvements

- Add a small inference script for classifying a new satellite tile.
- Export selected plots from model runs into `reports/figures/`.
- Add a lightweight Streamlit demo for interactive predictions.
- Track experiments with a reproducible configuration file.

## License

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full license text.

