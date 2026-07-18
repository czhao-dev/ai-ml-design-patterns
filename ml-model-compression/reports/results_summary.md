# Benchmark Results

Full run log for all compression techniques. Regenerated automatically each time a script in scripts/ is run.

| Model | Compression | Accuracy | F1 | Size (MB) | Latency (ms/img) | Throughput (img/s) |
|---|---|---:|---:|---:|---:|---:|
| PyTorch CNN (FP32 baseline) | Baseline | 99.92% | 0.9991 | 74.72 | 28.586 | 35.0 |
| PyTorch CNN-ViT (FP32 baseline) | Baseline | 97.58% | 0.9754 | 150.95 | 17.999 | 55.6 |
| StudentCNN (distilled from CNN-ViT) | Distillation | 99.17% | 0.9914 | 1.00 | 2.555 | 391.5 |
| StudentCNN (hard labels only) | Distillation | 99.92% | 0.9991 | 1.00 | 2.548 | 392.5 |
| PyTorch CNN — pruned 20% (structured) | Pruning | 97.58% | 0.9744 | 49.16 | 19.646 | 50.9 |
| PyTorch CNN — pruned 20% (unstructured) | Pruning | 99.92% | 0.9991 | 74.72 | 28.020 | 35.7 |
| PyTorch CNN — pruned 40% (structured) | Pruning | 53.42% | 0.0573 | 28.86 | 6.653 | 150.3 |
| PyTorch CNN — pruned 40% (unstructured) | Pruning | 100.00% | 1.0000 | 74.72 | 28.125 | 35.6 |
| PyTorch CNN — pruned 60% (structured) | Pruning | 52.00% | 0.0000 | 13.96 | 3.774 | 265.0 |
| PyTorch CNN — pruned 60% (unstructured) | Pruning | 99.92% | 0.9991 | 74.72 | 28.341 | 35.3 |
| PyTorch CNN — pruned 80% (structured) | Pruning | 52.00% | 0.0000 | 4.34 | 2.124 | 470.9 |
| PyTorch CNN — pruned 80% (unstructured) | Pruning | 94.25% | 0.9363 | 74.72 | 12.921 | 77.4 |
| PyTorch CNN — INT8 static PTQ | Quantization | 99.92% | 0.9991 | 18.83 | 3.295 | 303.5 |
| PyTorch CNN-ViT — INT8 dynamic PTQ | Quantization | 97.50% | 0.9746 | 90.21 | 31.137 | 32.1 |

## Notes

- Latency: mean of 500 single-image CPU forward passes after 50 warmup steps (`time.perf_counter`).
- Accuracy/F1: fixed SEED=42 held-out validation split (1,200 images), identical for every row.
