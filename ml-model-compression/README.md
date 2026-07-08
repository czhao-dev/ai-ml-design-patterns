# Model Compression: Pruning, Quantization, and Knowledge Distillation

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)

A focused study in production-oriented model compression applied to the four trained models from [`ml-satellite-image-classifier`](../ml-satellite-image-classifier). The project benchmarks three orthogonal compression families — **magnitude and structured pruning**, **post-training quantization (PTQ)**, and **knowledge distillation** — measuring the tradeoff between model size, inference latency, and classification accuracy on a held-out satellite image test set.

The goal is to answer a concrete engineering question: _for a fixed accuracy budget, what is the smallest, fastest model we can ship?_

## Table of Contents

- [Highlights](#highlights)
- [Background](#background)
- [Techniques](#techniques)
  - [Pruning](#pruning)
  - [Post-Training Quantization](#post-training-quantization)
  - [Knowledge Distillation](#knowledge-distillation)
- [Benchmark Results](#benchmark-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Design Notes](#design-notes)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Highlights

- Three compression techniques applied to the same base architecture and dataset, enabling a controlled, apples-to-apples benchmark
- Unstructured magnitude pruning and structured L1 channel pruning sweeping sparsity from 20% to 80%; structured pruning produces real latency gains without sparse BLAS support
- Static INT8 quantization via `torch.ao.quantization` (observer calibration on 200 training images) and dynamic INT8 for the ViT's linear layers
- Knowledge distillation: PyTorch CNN-ViT (99.67% accuracy, teacher) → lightweight 3-block student CNN, trained with a temperature-scaled soft-label loss; student is ~150× smaller than the teacher (measured: 1.0 MB vs. 150.95 MB FP32)
- Unified benchmark table: every model variant is scored on accuracy, F1, model size (MB), CPU inference latency (ms/image), and throughput (images/s)
- Compressed models are drop-in replacements for the inference server in `ml-satellite-image-classifier/serve/` — `ModelRegistry` requires no changes

## Background

`ml-satellite-image-classifier` trains and evaluates four models for binary land-use classification (agricultural vs. non-agricultural, 64×64 satellite tiles):

| Model | Accuracy | F1 | Size (FP32) |
|---|---:|---:|---:|
| PyTorch CNN | 99.83% | 0.9983 | ~34 MB |
| PyTorch CNN-ViT | 99.67% | 0.9967 | ~90 MB |
| Keras CNN | 99.33% | 0.9933 | ~35 MB |
| Keras CNN-ViT | 99.42% | 0.9942 | ~91 MB |

The PyTorch models are used as compression targets throughout this project. Keras models are included in the benchmark table for reference but are not the focus of the compression scripts.

The dataset, data pipeline, and evaluation split are identical to the parent project: 6,000 JPG satellite tiles (3,000 per class), evaluated on a fixed 1,200-image held-out validation split that no model saw during training. The trained FP32 weights (`models/trained/*.pth`) are loaded from `ml-satellite-image-classifier/models/trained/` via a relative path; see [Getting Started](#getting-started).

## Techniques

### Pruning

**Script:** [`scripts/01_pruning.py`](scripts/01_pruning.py)

Pruning removes weights from a trained network, reducing parameter count and (for structured pruning) the number of active channels.

**Unstructured magnitude pruning** (`torch.nn.utils.prune.l1_unstructured`) zeroes out the lowest-magnitude weights globally across all convolutional and linear layers. Because the surviving weights are distributed sparsely across the weight tensors, this does not reduce wall-clock latency on standard hardware without sparse BLAS support — but it reduces storage size after compression and is the canonical first step before fine-tuning recovery.

**Structured L1 channel pruning** (`torch.nn.utils.prune.ln_structured`, `n=1`, `dim=0`) removes entire output channels (filters) from convolutional layers ranked by L1 norm, then rebuilds the network with the surviving channels hardcoded. This produces a physically smaller model with real latency improvements on any CPU or GPU.

Both variants are swept across sparsity levels {20%, 40%, 60%, 80%}. For each level:
1. Prune the FP32 PyTorch CNN
2. Evaluate on the 1,200-image held-out split (no fine-tuning) to measure zero-shot accuracy drop
3. Apply `prune.remove()` to make masks permanent, then save the pruned checkpoint
4. Record model size and CPU latency

The structured variant additionally rebuilds the model with pruned dimensions so the saved checkpoint requires no sparsity mask and can be loaded with a standard `nn.Sequential`.

### Post-Training Quantization

**Script:** [`scripts/02_quantization.py`](scripts/02_quantization.py)

Quantization maps 32-bit floating-point weights and activations to lower-precision integers, reducing memory footprint and enabling integer arithmetic on CPUs that support it.

**Static INT8 PTQ** uses `torch.ao.quantization.prepare` + `torch.ao.quantization.convert` with `torch.ao.quantization.get_default_qconfig(engine)` (`'x86'`/`fbgemm` on x86 hosts, `'qnnpack'` on ARM — the engine is auto-detected). A calibration pass of 200 training images (never seen at test time) is run through the prepared model to collect activation statistics for scale and zero-point computation. The quantized model uses `torch.float32` inputs but internally operates in INT8; `BatchNorm` layers are excluded from quantization (see [Design Notes](#design-notes)) and run in FP32.

**Dynamic INT8 PTQ** (`torch.ao.quantization.quantize_dynamic`) quantizes only the linear layers at runtime, without a calibration pass. It is applied to the ViT's `nn.Linear` modules and is appropriate when activation ranges are input-dependent.

For each quantized model:
1. Load the FP32 PyTorch checkpoint
2. Apply the quantization scheme
3. Evaluate accuracy on the held-out split
4. Measure model size (serialized) and CPU inference latency

No fine-tuning or quantization-aware training (QAT) is applied. PTQ-only results reflect what can be achieved at deployment time with a trained checkpoint and no access to GPU training infrastructure.

### Knowledge Distillation

**Script:** [`scripts/03_distillation.py`](scripts/03_distillation.py)

Knowledge distillation trains a smaller **student** network to match the output distribution of a larger, pre-trained **teacher**, rather than training from hard (one-hot) labels alone. The teacher's soft probability vectors carry more information than a binary ground-truth label — they encode the model's uncertainty across classes and serve as a richer training signal for the student.

**Teacher:** PyTorch CNN-ViT (99.67% accuracy per the source project; 99.83% measured on this project's canonical split, ~151 MB FP32 measured). The teacher is frozen throughout distillation; only the student's weights are updated.

**Student:** A lightweight 3-block CNN (`StudentCNN`), roughly following the first three blocks of the satellite CNN backbone (32 → 64 → 128 channels), followed by global average pooling and a two-class head. ~259K parameters (~1.0 MB FP32) — measured at ~150× fewer parameters than the CNN-ViT teacher, well beyond a rough "order of magnitude smaller" framing.

**Loss function:**

```
L = α · CE(student_logits, hard_labels) + (1 - α) · T² · KL(softmax(student_logits / T), softmax(teacher_logits / T))
```

where `T` is the temperature (default 4.0) and `α` balances the hard-label cross-entropy against the soft-label KL divergence (default 0.3). Higher temperature softens the teacher's distribution, exposing more inter-class similarity signal. The `T²` factor is the standard Hinton et al. (2015) correction: temperature-softened KL gradients shrink by ~1/T², so multiplying back by `T²` keeps the soft-loss gradient magnitude comparable to the hard-label loss across different temperature settings.

Training runs on the 4,800-image training split (the same split used in `ml-satellite-image-classifier`) for 30 epochs with Adam and cosine learning-rate decay. The best checkpoint by validation accuracy is saved.

A **baseline student** (same `StudentCNN` architecture, trained from scratch on hard labels only) is included as a control to isolate the distillation benefit.

## Benchmark Results

Real run output — see [`reports/results_summary.md`](reports/results_summary.md) for the machine-generated version of this table plus notes, and [`reports/figures/`](reports/figures/) for the plots.

| Model | Compression | Accuracy | F1 | Size (MB) | Latency (ms/img) | Throughput (img/s) |
|---|---|---:|---:|---:|---:|---:|
| PyTorch CNN (FP32 baseline) | — | 99.92% | 0.9991 | 74.72 | 4.074 | 245.5 |
| PyTorch CNN-ViT (FP32 baseline) | — | 99.83% | 0.9983 | 150.95 | 8.814 | 113.5 |
| PyTorch CNN — pruned 20% (unstructured) | Pruning | 99.92% | 0.9991 | 74.72 | 4.143 | 241.3 |
| PyTorch CNN — pruned 40% (unstructured) | Pruning | 99.83% | 0.9983 | 74.72 | 4.226 | 236.6 |
| PyTorch CNN — pruned 60% (unstructured) | Pruning | 99.08% | 0.9904 | 74.72 | 4.116 | 242.9 |
| PyTorch CNN — pruned 80% (unstructured) | Pruning | 55.25% | 0.1268 | 74.72 | 4.121 | 242.7 |
| PyTorch CNN — pruned 20% (structured) | Pruning | 77.67% | 0.6968 | 49.16 | 3.146 | 317.9 |
| PyTorch CNN — pruned 40% (structured) | Pruning | 52.00% | 0.0000 | 28.86 | 2.126 | 470.3 |
| PyTorch CNN — pruned 60% (structured) | Pruning | 52.00% | 0.0000 | 13.96 | 1.039 | 962.3 |
| PyTorch CNN — pruned 80% (structured) | Pruning | 52.00% | 0.0000 | 4.34 | 0.506 | 1974.9 |
| PyTorch CNN — INT8 static PTQ | Quantization | 100.00% | 1.0000 | 18.76 | 2.234 | 447.6 |
| PyTorch CNN-ViT — INT8 dynamic PTQ | Quantization | 99.83% | 0.9983 | 90.21 | 5.656 | 176.8 |
| StudentCNN (hard labels only) | Distillation | 100.00% | 1.0000 | 1.00 | 0.612 | 1633.9 |
| StudentCNN (distilled from CNN-ViT) | Distillation | 99.92% | 0.9991 | 1.00 | 0.591 | 1691.6 |

Latency is measured as the mean of 500 single-image CPU inference calls (batch size 1, no GPU) after 50 warmup steps, using `time.perf_counter`. All measurements on Apple Silicon (arm64), CPU only.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/size_vs_accuracy_dark.png">
  <img src="reports/size_vs_accuracy_light.png" alt="Scatter plot of model size in MB (log scale) versus accuracy, colored by compression technique, showing distillation and quantization variants near the top-left (small and accurate) and low-sparsity pruning variants collapsed near the class prior.">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/pruning_accuracy_vs_sparsity_dark.png">
  <img src="reports/pruning_accuracy_vs_sparsity_light.png" alt="Line chart of accuracy versus sparsity for unstructured and structured pruning, showing unstructured pruning holding accuracy until 80% sparsity while structured pruning collapses to the class prior by 40% sparsity.">
</picture>

Both charts are generated directly from [`reports/results.json`](reports/results.json) by
[`reports/generate_plots.py`](reports/generate_plots.py). The size-vs-accuracy scatter makes the
overall shape of the study legible in one view: the distillation and INT8-quantization points
sit in the top-left (small *and* accurate), while unstructured pruning barely moves off the
74.72 MB baseline column until it collapses at 80% sparsity, and structured pruning trades size
for accuracy almost linearly. The sparsity line chart isolates that same pruning story — unstructured
pruning tracks the FP32 baseline closely through 60% sparsity before falling off a cliff at 80%,
while structured pruning starts losing accuracy immediately and hits the ~52% class-prior floor
by 40% sparsity.

**Reading the table honestly:**
- **Unstructured pruning** holds up to 60% sparsity (accuracy barely moves) but collapses at 80% — expected, since masked-but-still-dense weights don't reduce serialized size (all four unstructured rows report the same 74.72 MB) or latency; the benefit is storage compression after a sparse-aware format, not measured here.
- **Structured pruning** is far more aggressive: real size and latency drop immediately, but so does accuracy — by 40% sparsity the model has collapsed to predicting a single class (52.00% ≈ the class prior), exactly the "raw accuracy cliff" this project evaluates without fine-tuning recovery (see [Design Notes](#design-notes)).
- **Quantization** is the best accuracy/compression tradeoff in this table: static INT8 PTQ on the CNN cuts size 4× and latency ~1.8× with no measurable accuracy loss; dynamic PTQ on the CNN-ViT's linear layers is more modest (only the Transformer's `nn.Linear` layers are touched, not the CNN backbone or patch-embed conv) but still meaningfully smaller and faster than FP32.
- **Distillation** produces the smallest, fastest models by a wide margin (~150× smaller than the CNN-ViT teacher, sub-millisecond latency) — but on this dataset the hard-label-only baseline student matches or slightly exceeds the distilled student's accuracy. The task is easy enough for a 3-block CNN that the teacher's soft labels don't add measurable signal here; the distilled student's real advantage is smoother, more stable training (see `reports/figures/distillation_curves.png` — the baseline dips to 79% val accuracy at epoch 5, the distilled run never does).

## Repository Structure

```text
ml-model-compression/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_pruning.py              # Magnitude and structured pruning sweep
│   ├── 02_quantization.py         # Static and dynamic INT8 PTQ
│   └── 03_distillation.py         # Teacher–student knowledge distillation
├── src/
│   ├── __init__.py
│   ├── student_model.py           # StudentCNN architecture definition
│   ├── benchmark.py               # Latency/throughput measurement + results cache/figures
│   ├── eval_utils.py              # Canonical split, checkpoint loading, accuracy/F1 evaluation
│   └── paths.py                   # Shared path constants (points into satellite classifier)
├── models/
│   └── trained/                   # gitignored — pruned/quantized/student checkpoints
└── reports/
    ├── results_summary.md         # Full benchmark table and commentary (generated)
    ├── results.json                # Structured results cache backing the table + figures (generated)
    ├── generate_plots.py           # README chart generator (reads results.json only, no retraining)
    ├── size_vs_accuracy_*.png      # Generated size-vs-accuracy scatter (light + dark)
    ├── pruning_accuracy_vs_sparsity_*.png  # Generated pruning sparsity curve (light + dark)
    └── figures/
        ├── accuracy_vs_sparsity.png    # Pruning accuracy curve
        ├── size_vs_latency.png         # Pareto plot: size vs. latency across all variants
        └── distillation_curves.png     # Train/val loss for student and baseline
```

## Getting Started

### Prerequisites

- The trained FP32 PyTorch checkpoints from `ml-satellite-image-classifier` must exist at:
  ```
  ../ml-satellite-image-classifier/models/trained/ai_capstone_pytorch_state_dict.pth
  ../ml-satellite-image-classifier/models/trained/pytorch_cnn_vit_ai_capstone_model_state_dict.pth
  ```
  See [`ml-satellite-image-classifier/models/models.md`](../ml-satellite-image-classifier/models/models.md) for how to produce them.

- The satellite image dataset must be downloaded. Running any script in `ml-satellite-image-classifier/scripts/` on first run will download it automatically.

### Install

```bash
cd ml-model-compression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Compression Scripts

Each script is self-contained and writes its outputs to `reports/`.

```bash
# Pruning sweep (unstructured + structured, 4 sparsity levels each)
python scripts/01_pruning.py

# Post-training quantization (static CNN + dynamic ViT)
python scripts/02_quantization.py

# Knowledge distillation (trains student for ~30 epochs on CPU)
python scripts/03_distillation.py
```

Each script upserts its rows into `reports/results.json` (a small structured cache, keyed by variant name) and regenerates `reports/results_summary.md` plus the relevant figures from the full accumulated set — so `size_vs_latency.png` stays a complete cross-technique Pareto plot no matter which scripts have been run so far, while each script remains independently runnable.

### requirements.txt

```
torch>=2.3.0
torchvision>=0.18.0
pillow>=10.0.0
numpy>=1.26.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
```

`scikit-learn` is required transitively — it backs `binary_classification_metrics`, reused directly from `ml-satellite-image-classifier/src/metrics.py`.

## Design Notes

**Why PyTorch only?** TensorFlow's quantization API (`tf.lite.TFLiteConverter`) targets mobile/edge deployment and outputs a `.tflite` binary rather than a Keras model — it would require a separate serving path and complicates the benchmark. `torch.ao.quantization` produces a standard `nn.Module` that integrates cleanly with the existing `ModelRegistry`.

**Why no QAT?** Quantization-aware training requires a full training loop with fake-quantization nodes inserted. PTQ is the realistic baseline for teams deploying a model they did not train themselves. QAT is flagged in [Future Work](#future-work) as the natural next step.

**Why evaluate without fine-tuning after pruning?** Fine-tuning a pruned model typically recovers most accuracy loss, but it obscures how much damage pruning itself does. Evaluating the pruned-but-not-recovered model shows the raw accuracy cliff, which is the more honest engineering baseline (visible in the structured-pruning rows above, which collapse to the class prior by 40% sparsity). Fine-tuning recovery is flagged in [Future Work](#future-work).

**Static quantization needs explicit `QuantStub`/`DeQuantStub`, and the CNN's `BatchNorm` layers can't be fused.** `build_satellite_cnn()` returns a bare `nn.Sequential` — eager-mode static PTQ requires quant/dequant boundaries inserted manually. More notably, its block order is `Conv2d → ReLU → MaxPool2d → BatchNorm2d` (BN after pooling, not the usual `Conv → BN → ReLU`), which `torch.ao.quantization.fuse_modules` doesn't recognize, and there is no standalone quantized `BatchNorm` kernel. `scripts/02_quantization.py` works around this by running every `BatchNorm1d`/`BatchNorm2d` in FP32, sandwiched between a `DeQuantStub`/`QuantStub` pair, while everything else (convs, linears) runs genuinely as INT8.

**The CNN-ViT teacher's real hyperparameters don't match `CNN_ViT_Hybrid`'s constructor defaults.** The class defaults to `depth=6, heads=8`, but the checkpoint was actually trained with `depth=3, heads=6` (see `ml-satellite-image-classifier/scripts/08_pytorch_cnn_vit_hybrid.py`). `src/paths.py` pins the correct values as `CNN_VIT_DEPTH`/`CNN_VIT_HEADS` constants; loading with the class defaults would silently fail to align the ViT block weights.

**Canonical split.** Every accuracy/F1 number in this project uses the standalone CNN's original SEED=42 80/20 split (`src/paths.py: SPLIT_SEED`), not the CNN-ViT teacher's own training split (SEED=7331). This gives one consistent, zero-leakage held-out set for pruning/quantization (which target the CNN directly) and for the student (which trains and is evaluated on it). The one caveat: the frozen CNN-ViT teacher was originally trained on a different split, so a handful of its own past training images may reappear in this project's "training" 4,800 when it's used as a soft-label source during distillation — acceptable since the teacher itself is never the object being evaluated here.

**Inference server compatibility.** The compressed PyTorch models expose the same `forward(x: Tensor) -> Tensor` interface as the originals. Swapping them into `ml-satellite-image-classifier/serve/model_registry.py` requires only pointing `_MODEL_FILES` at the new checkpoint paths.

## Future Work

- **Fine-tuning recovery after pruning**: a short 5-epoch fine-tune pass typically recovers 1–3% accuracy at 60% sparsity; adds a more complete picture of the pruning budget
- **Quantization-aware training (QAT)**: `torch.ao.quantization.prepare_qat` + training loop with fake-quantization nodes; expected to close the accuracy gap vs. PTQ at high compression ratios
- **TorchScript / ONNX export**: export the best compressed model to ONNX and benchmark with ONNX Runtime (`onnxruntime-cpu`), which often yields additional latency gains through graph-level optimizations
- **4-bit NF4 quantization with `bitsandbytes`**: relevant for the ViT variant; extends the benchmark to sub-8-bit regimes
- **End-to-end inference server benchmark**: replace the FP32 model in `ml-satellite-image-classifier/serve/` with each compressed variant and report `/predict` endpoint latency under load with `locust` or `wrk`

## References

- Han, S., Pool, J., Tran, J., and Dally, W.J. "Learning Both Weights and Connections for Efficient Neural Networks." *NeurIPS*, 2015. [arxiv.org/abs/1506.02626](https://arxiv.org/abs/1506.02626)
- Molchanov, P., Tyree, S., Karras, T., Aila, T., and Kautz, J. "Pruning Convolutional Neural Networks for Resource Efficient Inference." *ICLR*, 2017. [arxiv.org/abs/1611.06440](https://arxiv.org/abs/1611.06440)
- Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*, 2018. [arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." *NeurIPS Workshop*, 2015. [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
- PyTorch documentation: [Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html), [Static Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

## License

Apache License 2.0. See [LICENSE](../LICENSE) for the full license text.
