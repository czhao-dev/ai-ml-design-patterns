# TinyLLM Lab

TinyLLM Lab is an educational project for training a small GPT-style language model from scratch. The goal is to understand the complete language-modeling pipeline, including tokenizer training, dataset preprocessing, Transformer implementation, model training, evaluation, text generation, and performance benchmarking.

This project is intentionally small enough to run on consumer hardware while still demonstrating the core ideas behind modern decoder-only large language models.

## Project Goals

The main goals of this project are to:

* Build a small decoder-only Transformer language model from scratch.
* Train a custom tokenizer on a text dataset.
* Preprocess and batch tokenized text for causal language modeling.
* Implement a training loop with checkpointing and validation.
* Generate text using sampling strategies such as temperature, top-k, and top-p sampling.
* Evaluate model quality using validation loss and perplexity.
* Benchmark training and inference performance.
* Document design tradeoffs clearly for learning and portfolio demonstration.

This is not intended to compete with production LLMs. Instead, it is a hands-on systems and machine learning project for understanding how small language models work end to end.

## Contents

* [Features](#features)
* [High-Level Architecture](#high-level-architecture)
* [Repository Structure](#repository-structure)
* [Model Variants](#model-variants)
* [Installation](#installation)
* [Dataset](#dataset)
* [Training a Tokenizer](#training-a-tokenizer)
* [Preparing the Dataset](#preparing-the-dataset)
* [Training](#training)
* [Text Generation](#text-generation)
* [Evaluation](#evaluation)
* [Benchmarking](#benchmarking)
* [Experiments](#experiments)
* [Example Training Curve](#example-training-curve)
* [Implementation Notes](#implementation-notes)
* [Skills Demonstrated](#skills-demonstrated)
* [Future Work](#future-work)
* [Limitations](#limitations)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Features

* Custom GPT-style Transformer implementation
* Tokenizer training and dataset encoding
* Causal language modeling objective
* Configurable model sizes
* Training and validation loops
* Checkpoint save/load support
* Text generation script
* Perplexity evaluation
* Training loss logging
* Throughput and memory benchmarking
* Reproducible experiment configuration

## High-Level Architecture

```text
Input Text
   |
   v
Tokenizer Training
   |
   v
Tokenized Dataset
   |
   v
Mini-batch Loader
   |
   v
Decoder-only Transformer
   |
   v
Next-token Prediction
   |
   v
Training / Evaluation / Generation
```

The model follows a standard decoder-only Transformer architecture:

```text
Token IDs
   |
   v
Token Embedding + Positional Embedding
   |
   v
Transformer Block x N
   |   - Causal Self-Attention
   |   - Feed-Forward Network
   |   - LayerNorm
   |   - Residual Connections
   |
   v
Language Modeling Head
   |
   v
Next-token Logits
```

## Repository Structure

```text
ml-tiny-llm-gpt/
├── README.md
├── requirements.txt
├── configs/
│   ├── tiny.yaml
│   ├── small.yaml
│   └── medium.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── tokenizer/
├── tinyllm/
│   ├── __init__.py
│   ├── model.py
│   ├── attention.py
│   ├── transformer.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── generation.py
│   └── utils.py
├── scripts/
│   ├── download_tinystories.py
│   ├── train_tokenizer.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── plot_loss.py
├── experiments/
│   ├── runs/
│   └── results/
└── tests/
    ├── test_model.py
    ├── test_attention.py
    └── test_tokenizer.py
```

## Model Variants

The project supports multiple model sizes for experimentation.

| Model  | Layers | Hidden Size | Attention Heads | Context Length | Approx. Parameters |
| ------ | -----: | ----------: | --------------: | -------------: | -----------------: |
| Tiny   |      4 |         256 |               4 |            256 |             ~4.3M |
| Small  |      6 |         384 |               6 |            512 |               ~15M |
| Medium |      8 |         512 |               8 |            512 |               ~30M |

All three variants share a vocabulary size of 4096 (see [Training a Tokenizer](#training-a-tokenizer)), so the Tiny model's parameter count is a bit below the commonly-cited "~5M" estimate that assumes a larger vocabulary.

These configurations are intentionally modest: all three have been trained for the full 20,000 steps used throughout this README (see [Evaluation](#evaluation) and [Example Training Curve](#example-training-curve) for results). Tiny trains comfortably on a laptop CPU or Apple Silicon GPU in a few hours; Small and Medium are still small enough to experiment with locally at reduced step counts, but a full 20,000-step run benefits from a discrete GPU — the Small/Medium results in this README were trained on a rented cloud GPU rather than the author's laptop (see the note under [Benchmarking](#benchmarking)).

## Installation

Clone the repository:

```bash
git clone https://github.com/czhao-dev/ml-tiny-llm-gpt.git
cd ml-tiny-llm-gpt
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
numpy
tqdm
pyyaml
tokenizers
matplotlib
huggingface_hub
pytest
```

## Dataset

This project trains on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a synthetic dataset of short, simple stories generated by GPT-3.5/4 specifically for training small language models. Its deliberately narrow vocabulary (a few thousand words) makes it a good fit for the small models this project trains. The project also works with any other plain-text dataset following the same format.

Download a subset of TinyStories (fast iteration — default ~75k train / 3k valid stories) or the full ~2M-story dataset:

```bash
python scripts/download_tinystories.py                 # subset, default sizes
python scripts/download_tinystories.py --full           # full dataset
```

Expected raw data format:

```text
data/raw/train.txt
data/raw/valid.txt
```

Each file contains plain text with stories delimited by `<|endoftext|>`, which also serves as the tokenizer's special/EOS token. The preprocessing script tokenizes the text and creates training sequences for next-token prediction.

## Training a Tokenizer

Train a Byte Pair Encoding tokenizer:

```bash
python scripts/train_tokenizer.py \
  --input data/raw/train.txt \
  --vocab-size 4096 \
  --output data/tokenizer/tokenizer.json
```

Example output:

```text
Tokenizer saved to data/tokenizer/tokenizer.json
Vocabulary size: 4096
```

A vocabulary size of 4096 (rather than a larger general-purpose vocabulary like 8000+) is recommended for TinyStories: its text uses a deliberately small, simple vocabulary, so a smaller BPE vocabulary wastes less of the model's parameter budget on the embedding table and leaves more capacity for the Transformer body itself.

## Preparing the Dataset

Encode the raw text dataset:

```bash
python scripts/prepare_dataset.py \
  --tokenizer data/tokenizer/tokenizer.json \
  --train data/raw/train.txt \
  --valid data/raw/valid.txt \
  --output data/processed/
```

This writes flat `uint16` token-ID arrays to `train.bin` and `valid.bin`, which the training loop memory-maps for efficient random-access batching without loading the full dataset into RAM.

## Training

Train the Tiny model:

```bash
python scripts/train.py \
  --config configs/tiny.yaml
```

Swap in `configs/small.yaml` or `configs/medium.yaml` to train one of the larger variants instead (see [Model Variants](#model-variants)); all three configs share the same schema and write checkpoints/logs to their own subdirectory under `experiments/runs/`.

Example configuration (`configs/tiny.yaml`):

```yaml
model:
  vocab_size: 4096
  context_length: 256
  n_layers: 4
  n_heads: 4
  hidden_size: 256
  dropout: 0.1

training:
  batch_size: 32
  max_steps: 20000
  learning_rate: 0.0003
  weight_decay: 0.1
  warmup_steps: 1000
  eval_interval: 500
  eval_iters: 100
  checkpoint_interval: 1000
  grad_clip: 1.0
  dtype: fp32
  seed: 1337

data:
  train_path: data/processed/train.bin
  valid_path: data/processed/valid.bin

output:
  checkpoint_dir: experiments/runs/tiny/checkpoints
  log_dir: experiments/runs/tiny/logs
```

## Text Generation

Generate text from a trained checkpoint:

```bash
python scripts/generate.py \
  --checkpoint experiments/runs/tiny/checkpoints/best.pt \
  --tokenizer data/tokenizer/tokenizer.json \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

Example output (Tiny model, trained for 20,000 steps on the TinyStories subset):

```text
Once upon a time there was a little girl named Lily. She had a cozy chair in her room. One day,
Lily's mommy brought her a new chair to sit on. Lily was so happy and started to sit on the chair.
After a while, Lily's mommy came in and saw the chair. She asked her mommy if she wanted to sit on
the chair. Her mommy said yes and they sat on the chair together.
```

## Evaluation

Evaluate validation loss and perplexity:

```bash
python scripts/evaluate.py \
  --checkpoint experiments/runs/tiny/checkpoints/best.pt \
  --tokenizer data/tokenizer/tokenizer.json \
  --valid data/processed/valid.bin
```

Example output (TinyStories subset, 20,000 steps):

| Model  | Validation Loss | Perplexity |
| ------ | ---------------: | ----------: |
| Tiny   |             1.96 |       7.11 |
| Small  |             1.68 |       5.36 |
| Medium |             1.63 |       5.09 |

Perplexity drops monotonically with model size on this dataset, as expected — Small and Medium were trained with the same tokenizer, dataset, step count, and optimizer schedule as Tiny (see [Model scaling](#experiments) below), so the comparison isolates the effect of capacity.

## Benchmarking

Run a basic training and inference benchmark:

```bash
python scripts/benchmark.py \
  --checkpoint experiments/runs/tiny/checkpoints/best.pt \
  --batch-size 1 \
  --context-length 256
```

Example benchmark table (Apple M3, batch size 1, no CUDA available so "GPU" is MPS; each model benchmarked at its own trained context length):

| Model  | Device | Context Length | Inference Tok/s | Train Tok/s | Peak Memory (MB) |
| ------ | ------ | --------------: | ---------------: | -----------: | ----------------: |
| Tiny   | CPU    |             256 |            69205 |       21871 |              302.7 |
| Tiny   | MPS    |             256 |            54378 |       43218 |               25.5 |
| Small  | CPU    |             512 |            28931 |        9886 |              513.7 |
| Small  | MPS    |             512 |            25350 |       16762 |               66.5 |
| Medium | CPU    |             512 |            13838 |        4978 |              665.2 |
| Medium | MPS    |             512 |            23776 |        8533 |              127.1 |

At this small batch size, inference is actually *faster on CPU* than MPS for Tiny and Small — per-op dispatch overhead on the MPS backend dominates when there isn't enough work per kernel launch to amortize it. Medium is large enough that MPS overtakes CPU on inference too. Training (forward+backward, which does more compute per step) favors MPS across all three sizes, and MPS peak memory is far lower than the CPU process's RSS figure. Larger batch sizes are expected to favor MPS more consistently across the board.

This table measures single-sample (batch size 1) inference/training speed on local Apple Silicon, which is the right regime for interactive/low-latency use but not for bulk training throughput. The Small and Medium *training runs* behind the results in this README (20,000 steps each, batch size 32) were run on a rented AWS g4dn.xlarge (Tesla T4) instead of locally: at that batch size and step count, the two runs would have taken roughly 34 hours combined on the M3 versus about 2.9h (Small) + 6.9h (Medium) on the T4.

## Experiments

| Experiment           | Description                                                      | Status  |
| -------------------- | ----------------------------------------------------------------- | ------- |
| Model scaling        | Compare Tiny, Small, and Medium parameter counts                  | Done — see [Evaluation](#evaluation) and [Example Training Curve](#example-training-curve) |
| Inference speed      | Measure tokens/sec across different model sizes                   | Done — see [Benchmarking](#benchmarking) |
| Tokenizer comparison | Compare character-level tokenization and BPE tokenization         | Planned |
| Context length       | Compare 128, 256, and 512 token context windows                   | Planned |
| Sampling methods     | Compare greedy decoding, temperature sampling, top-k, and top-p   | Planned |
| Training duration    | Analyze validation loss as training steps increase                | Planned |

## Example Training Curve

Training and validation loss curves can be generated from the training logs:

```bash
python scripts/plot_loss.py \
  --log experiments/runs/tiny/logs/train_log.jsonl
```

Example reports (TinyStories subset, 20,000 steps):

```text
Tiny
Final train loss: 1.99
Final validation loss: 1.96
Final perplexity: 7.10
Best checkpoint: step 19999 (val_loss 1.96)

Small
Final train loss: 1.48
Final validation loss: 1.68
Final perplexity: 5.38
Best checkpoint: step 19500 (val_loss 1.68)

Medium
Final train loss: 1.24
Final validation loss: 1.62
Final perplexity: 5.05
Best checkpoint: step 19500 (val_loss 1.61)
```

## Implementation Notes

### Causal Self-Attention

The model uses causal self-attention so that each token can only attend to previous tokens and itself. This prevents the model from seeing future tokens during training.

### Positional Embeddings

Since the Transformer architecture does not have recurrence, positional embeddings are added to token embeddings so the model can learn token order.

### Language Modeling Head

The final hidden states are projected back to the vocabulary size to produce next-token logits.

### Loss Function

The model is trained using cross-entropy loss between predicted next-token logits and the actual next tokens.

## Skills Demonstrated

This project demonstrates practical experience with:

* Transformer architecture
* Tokenization
* Language model pretraining
* PyTorch model implementation
* Training loop design
* GPU/MPS/CPU device handling
* Cloud GPU training (AWS EC2)
* Checkpointing and reproducibility
* Evaluation and perplexity measurement
* Text generation algorithms
* Performance benchmarking
* Clean project structure and documentation

## Future Work

Possible extensions:

* Add LoRA fine-tuning support
* Add instruction fine-tuning on a small custom dataset
* Add a C++ inference runtime
* Add int8 or int4 quantization
* Export model weights to a simple custom binary format
* Add a small web demo
* Compare PyTorch, MLX, and C++ inference performance
* Add attention visualization
* Add unit tests for each model component

## Limitations

This project trains small models for educational purposes. The generated text quality will be limited by model size, dataset size, training time, and available hardware. The project is intended to demonstrate understanding of LLM training mechanics rather than to produce a production-quality assistant.

## License

This project is released under the MIT License.

## Acknowledgments

This project is inspired by modern decoder-only Transformer language models and educational small-scale LLM training projects. The goal is to make the core ideas behind language model training understandable, reproducible, and practical on local hardware.