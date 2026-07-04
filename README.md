# Applied Machine Learning Projects

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A monorepo of ten end-to-end machine learning projects spanning computer vision, large language models, graph learning, time-series forecasting, model compression, and production inference serving. Each project lives in its own subdirectory with independent dependencies, tests, documented results, and a full README.

## Projects at a Glance

| Project | Area | Key Technologies | Standout |
| --- | --- | --- | --- |
| [ml-satellite-image-classifier](#ml-satellite-image-classifier) | Computer Vision | PyTorch, Keras/TF, FastAPI, Docker | 99.83% accuracy; FastAPI server serving all four models |
| [ml-model-compression](#ml-model-compression) | Model Compression | PyTorch, `torch.ao.quantization` | Distilled student ~150× smaller than its teacher at 99.9%+ accuracy |
| [ml-llm-alignment-fine-tuning](#ml-llm-alignment-fine-tuning) | LLM Alignment | PyTorch, TRL, HuggingFace, LoRA | Full SFT → RM → PPO RLHF → DPO pipeline, all trained locally |
| [ml-tiny-llm-gpt](#ml-tiny-llm-gpt) | Language Modeling | PyTorch | GPT decoder-only Transformer built from scratch |
| [ml-gcp-vertex-rag-chatbot](#ml-gcp-vertex-rag-chatbot) | RAG / GenAI | LangChain, Vertex AI, Chroma, Cloud Run | Document Q&A app deployed to GCP Cloud Run |
| [ml-movie-recommender](#ml-movie-recommender) | Graph ML | PyTorch Geometric, igraph | Heterogeneous GNN over IMDb graphs; top-N recommendation on MovieLens |
| [ml-social-network-predictor](#ml-social-network-predictor) | Graph ML | igraph, PyTorch | DeepWalk embeddings reach 0.986 ROC-AUC on 4,039-node Facebook graph |
| [ml-wearable-motion-classifier](#ml-wearable-motion-classifier) | Classical ML | scikit-learn, NumPy, SciPy | IMU → trajectory → ensemble classifier for clinical rehabilitation |
| [ml-recyclable-material-classifier-vgg16](#ml-recyclable-material-classifier-vgg16) | Computer Vision | Keras/TF, VGG16 | Transfer learning; caught and fixed training-on-test bug from source notebook |
| [ml-boston-climate-modeler](#ml-boston-climate-modeler) | Time-Series | TensorFlow, Python | Pure-TF LSTM + Transformer from scratch (no Keras); 7-day multi-step forecasting; 56 unit tests |

---

## Project Details

### ml-satellite-image-classifier

Binary classification of 64×64 satellite image tiles as agricultural vs. non-agricultural land.

- **Models:** Six-block CNN (32→1024 channels, 5×5 kernels) and CNN–Vision Transformer hybrid (CNN feature map tokenized and fed through multi-head self-attention blocks), each implemented independently in both Keras/TensorFlow and PyTorch — four models total, trained and evaluated in parallel across frameworks.
- **Results:** PyTorch CNN 99.83%, Keras CNN 99.33%, PyTorch CNN-ViT 99.67%, Keras CNN-ViT 99.42% — all on a 1,200-image held-out split never seen during training.
- **Inference server:** FastAPI app (`serve/`) loads all four models at startup and exposes `/health`, `/models`, and `POST /predict?model=` endpoints. Model backend is selectable per request. Deployed with Docker Compose; model weights mounted read-only at runtime to keep the image small.
- **Notable:** Caught and fixed a data-leakage bug in the original evaluation methodology that scored models against the full training set rather than a held-out split.

**Stack:** Python · PyTorch · Keras/TensorFlow · FastAPI · Uvicorn · Docker Compose

---

### ml-model-compression

Three orthogonal compression techniques — pruning, post-training quantization, and knowledge distillation — benchmarked against the PyTorch CNN and CNN-ViT models trained in `ml-satellite-image-classifier`, all scored on the same fixed held-out split for a controlled comparison.

- **Pruning:** Unstructured global L1 magnitude pruning and structured L1 channel pruning (`torch.nn.utils.prune`), swept across 20–80% sparsity. Unstructured holds accuracy to 60% sparsity but doesn't reduce size/latency without sparse BLAS; structured produces real size/latency drops but collapses to the class prior by 40% sparsity without fine-tuning recovery — an intentionally honest "raw accuracy cliff" measurement.
- **Quantization:** Static INT8 PTQ on the CNN (4× smaller, ~1.8× faster, no measurable accuracy loss) and dynamic INT8 PTQ on the CNN-ViT's linear layers. Required working around two undocumented gaps in the source architecture: eager-mode static quantization needs manual `QuantStub`/`DeQuantStub` insertion, and the CNN's `BatchNorm` layers (positioned after pooling, not fusable with the preceding conv) have no quantized kernel and must run in FP32 sandwiched between quant/dequant boundaries.
- **Knowledge distillation:** A 3-block, ~259K-parameter `StudentCNN` distilled from the frozen CNN-ViT teacher (temperature-scaled KL + hard-label CE, with the standard T² gradient-scaling term) reaches 99.9%+ accuracy at ~150× smaller than the teacher and sub-millisecond CPU latency — one of several places results diverged from the initial write-up (~12× was the original estimate) once actually measured.
- **Reproducibility groundwork:** Corrected the CNN-ViT's constructor defaults (`depth=6, heads=8`) against its real trained hyperparameters (`depth=3, heads=6`) before any of the above would load correctly, and standardized every technique on a single canonical held-out split.

**Stack:** Python · PyTorch · `torch.ao.quantization` · torchvision · scikit-learn

---

### ml-llm-alignment-fine-tuning

Four LLM alignment techniques implemented end-to-end, each with its own training objective and evaluation metric — all locally runnable on a laptop CPU.

- **Supervised fine-tuning (SFT):** LoRA-adapts `facebook/opt-350m` on CodeAlpaca-20k with TRL's `SFTTrainer`; evaluated with SacreBLEU before and after fine-tuning.
- **Reward modeling:** GPT-2 + LoRA trained on chosen/rejected response pairs with `RewardTrainer`; reaches **0.96 pairwise ranking accuracy** on a held-out preference set.
- **PPO RLHF:** `gpt2-imdb` steered toward positive and negative sentiment with `PPOTrainer` against a sentiment-classifier reward; mean reward improves from 0.24 → 1.27 (positive policy) and −0.32 → 0.56 (negative policy).
- **DPO:** GPT-2 + LoRA fine-tuned directly on preference pairs with `DPOTrainer` (no separate reward model or RL loop); reaches **0.70 reward accuracy** on held-out pairs.
- Every script replaces commented-out or pre-downloaded training from the source notebooks with real, locally-runnable training — every number above comes from an actual training run.

**Stack:** Python · PyTorch · HuggingFace Transformers · TRL · LoRA (PEFT)

---

### ml-tiny-llm-gpt

A from-scratch GPT-style language model covering the complete pipeline from raw text to generated output.

- **Architecture:** Decoder-only Transformer with configurable depth, heads, and embedding dimension; causal self-attention, learned positional embeddings, and layer normalization — no HuggingFace model code in the loop.
- **Pipeline:** Custom BPE tokenizer training → dataset preprocessing and sequence packing → training loop with gradient clipping and validation checkpointing → top-k / top-p text generation → perplexity evaluation → throughput and memory benchmarking.
- Intentionally sized to run on consumer hardware while demonstrating every component of a modern LLM training stack.

**Stack:** Python · PyTorch

---

### ml-gcp-vertex-rag-chatbot

A deployed RAG document Q&A app: upload a PDF, TXT, Markdown, CSV, or DOCX file and ask questions grounded in its content.

**Live demo:** https://rag-pdf-chatbot-715060982814.us-central1.run.app

- **RAG pipeline:** LangChain document loaders → `RecursiveCharacterTextSplitter` → Vertex AI `text-embedding-004` embeddings → Chroma vector store → `RetrievalQA` chain → Gemini 2.5 Flash answer with source grounding.
- **Deployment:** Containerized with Docker and deployed to GCP Cloud Run with scale-to-zero cost controls; credentials handled via Application Default Credentials for local development.
- **Interface:** Gradio web UI; standalone annotated scripts for each RAG concept (loading, splitting, embedding, retrieval) as reference implementations.

**Stack:** Python · LangChain · Google Vertex AI (Gemini + text-embedding-004) · Chroma · Gradio · Docker · Cloud Run

---

### ml-movie-recommender

Graph feature engineering pipeline extended with a heterogeneous GNN, evaluated on two separate tasks.

- **Feature engineering (igraph):** Actor/movie networks built from IMDb data; actors ranked by PageRank, movie communities detected with Fast Greedy Newman, Jaccard movie-movie similarity computed — all kept exactly as in the original coursework pipeline.
- **Heterogeneous GNN (PyTorch Geometric):** `HeteroConv` encoder with `GraphConv` for weighted relations and `SAGEConv` for unweighted bipartite edges; graph-derived features (PageRank, community ID, Jaccard similarity) used as node features.
- **IMDb track:** Movie rating prediction benchmarked against neighborhood averaging, linear regression, and bipartite graph averaging baselines.
- **MovieLens track:** Genuine personalized top-N recommendation on `ml-latest-small` (943 users, ~9,700 movies), evaluated with Precision/Recall/NDCG@{5,10,20} against the full catalog — not sampled negatives.

**Stack:** Python · PyTorch · PyTorch Geometric · igraph · scikit-learn

---

### ml-social-network-predictor

Structural analysis of large social graphs extended into a link-prediction task comparing hand-engineered heuristics against learned node embeddings.

- **Graph analysis (igraph):** Degree distribution, ego-network extraction, Fast-Greedy / Edge-Betweenness / Infomap / Walktrap community detection, embeddedness and dispersion scoring on 4,039-node Facebook and Google+ graphs.
- **Node embeddings:** Hand-rolled DeepWalk-style skip-gram model (PyTorch) trained on random walks from the training graph only — test edges withheld before any graph operation.
- **Link prediction results** (full Facebook graph, 1,000 held-out test edges): heuristics-only **0.974 ROC-AUC**, embeddings-only **0.986 ROC-AUC**, combined 0.953 ROC-AUC (threshold sensitivity; see case study writeup).
- Proper holdout methodology documented: test edges excluded before all graph analysis and embedding training to prevent leakage.

**Stack:** Python · PyTorch · igraph · scikit-learn

---

### ml-wearable-motion-classifier

A signal-processing and ML pipeline classifying upper-body movements from wrist-worn IMU data in a clinical rehabilitation context (Wolf Motor Function Test).

- **Preprocessing:** Sensor-frame alignment, gravity subtraction, zero-velocity updates (ZUPT), and wrist trajectory reconstruction from raw accelerometer/gyroscope/quaternion captures of a wrist-worn MPU-9150.
- **Features:** Vertical power, azimuth rotation, peak counts, path length, variance, and acceleration statistics extracted from reconstructed trajectories.
- **Models:** Deterministic rule-based baseline + four scikit-learn classifiers (SVM, random forest, histogram gradient boosting, soft-voting ensemble). Data augmentation pipeline with source-trial-aware grouped cross-validation to prevent leakage from augmented variants back into the test fold.
- **Results:** ~74% accuracy (grouped CV, 15 of 17 classes, real + augmented data) — see README for an honest discussion of dataset limitations.
- **Testing:** 6-module test suite covering the CLI, sensor parser, preprocessing, quaternion math, feature extraction, and the rule-based classifier. Not approved for clinical use.

**Stack:** Python · scikit-learn · NumPy · SciPy · pandas

---

### ml-recyclable-material-classifier-vgg16

Binary classification of organic vs. recyclable material images using VGG16 transfer learning, with two configurations compared head-to-head.

- **Configurations:** Frozen feature-extraction head vs. fine-tuning from `block5_conv3` onward; both trained on 800 images with augmentation, evaluated on a 200-image held-out test set.
- **Results:** Both reach 86.5% test accuracy; fine-tuning improves ROC-AUC (0.9534 vs. 0.9332) and produces more balanced precision/recall across classes.
- **Notable:** Caught and fixed a data-handling bug in the source notebook — a mislabeled generator meant every original training run was training on test-directory images, not training-directory images.

**Stack:** Python · Keras/TensorFlow · VGG16

---

### ml-boston-climate-modeler

Daily weather forecasting for Reading, MA (Boston suburb) from NOAA station data — two complete pipelines in one repo: a stdlib-only Ridge baseline and a pure-TensorFlow deep learning stack.

- **Ridge baseline (v0.1):** NOAA CSV parsing → missing-value handling → seasonal, lag, and rolling-window feature engineering → Ridge regression from scratch (Gauss-Jordan solver, no NumPy) → model serialization to JSON. Temperature R² = 0.747 on 2016 test set.
- **LSTM forecaster (v0.2):** 2-layer stacked `LSTMCell` unrolled with `tf.unstack`; all gate weights as raw `tf.Variable`; trained with `tf.GradientTape` and a hand-built Adam optimiser. Predicts PRCP, SNOW, and TOBS jointly 7 days ahead.
- **Transformer forecaster (v0.2):** Pre-norm encoder with sinusoidal positional encoding, `MultiHeadAttention` (Q/K/V projections as `tf.Variable`, scaled dot-product via `tf.linalg.matmul`), GELU feed-forward blocks, and mean pooling. **No `tf.keras` API used anywhere.**
- **Training infrastructure:** `tf.GradientTape` mini-batch loop, `tf.clip_by_global_norm` gradient clipping, temporal validation split, early stopping. Trained on **58 years of data** (1960–2017, 20 974 windows); the Transformer reaches R² = **0.693** on 7-day-ahead temperature over a 2018–2019 test set, outperforming the LSTM (R² = 0.639). PRCP/SNOW R² near zero is expected given event sparsity.
- **Notebook:** `notebooks/climate_exploration.ipynb` covers EDA, training loss curves, attention heatmap visualisation for each encoder block, and a four-model comparison.
- **Testing:** 56 unit tests across data cleaning, feature engineering, sequence windowing, scaler round-trips, TF primitive shapes, and weight serialisation.

**Stack:** Python · TensorFlow 2.14+ (`tf.Module` / `tf.Variable` / `tf.GradientTape`) · NumPy · Jupyter · Matplotlib

---

## References

Papers and resources that directly informed the techniques used across these projects.

**Transformers and attention**
- Vaswani, A., et al. "Attention Is All You Need." *NeurIPS*, 2017. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021. [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929) *(ml-satellite-image-classifier)*

**Recurrent networks and time-series forecasting**
- Hochreiter, S., and Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 9(8):1735–1780, 1997. [doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) *(ml-boston-climate-modeler)*
- Jozefowicz, R., et al. "An Empirical Exploration of Recurrent Network Architectures." *ICML*, 2015. [proceedings.mlr.press/v37/jozefowicz15.html](https://proceedings.mlr.press/v37/jozefowicz15.html) *(ml-boston-climate-modeler — forget-gate bias initialisation)*
- Kingma, D.P., and Ba, J. "Adam: A Method for Stochastic Optimization." *ICLR*, 2015. [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980) *(ml-boston-climate-modeler)*
- Ba, J.L., et al. "Layer Normalization." 2016. [arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450) *(ml-boston-climate-modeler)*

**Language modeling**
- Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019. [openai.com/research/language-unsupervised](https://openai.com/research/language-unsupervised) *(ml-tiny-llm-gpt)*
- Eldan, R., and Li, Y. "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" 2023. [arxiv.org/abs/2305.07759](https://arxiv.org/abs/2305.07759) *(ml-tiny-llm-gpt)*
- Karpathy, A. nanoGPT. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) *(ml-tiny-llm-gpt)*

**LLM alignment and fine-tuning**
- Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS*, 2022. [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) *(ml-llm-alignment-fine-tuning)*
- Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS*, 2023. [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) *(ml-llm-alignment-fine-tuning)*
- Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022. [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) *(ml-llm-alignment-fine-tuning)*
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." 2017. [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) *(ml-llm-alignment-fine-tuning)*

**Retrieval-augmented generation**
- Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020. [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) *(ml-gcp-vertex-rag-chatbot)*

**Graph learning**
- Fey, M., and Lenssen, J.E. "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop*, 2019. [arxiv.org/abs/1903.02428](https://arxiv.org/abs/1903.02428) *(ml-movie-recommender)*
- Perozzi, B., Al-Rfou, R., and Skiena, S. "DeepWalk: Online Learning of Social Representations." *KDD*, 2014. [arxiv.org/abs/1403.6652](https://arxiv.org/abs/1403.6652) *(ml-social-network-predictor)*
- Backstrom, L., and Kleinberg, J. "Romantic Partnerships and the Dispersion of Social Ties." *CSCW*, 2014. [arxiv.org/abs/1310.6753](https://arxiv.org/abs/1310.6753) *(ml-social-network-predictor)*

**Transfer learning and convolutional networks**
- Simonyan, K., and Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*, 2015. [arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556) *(ml-recyclable-material-classifier-vgg16)*
- Russakovsky, O., et al. "ImageNet Large Scale Visual Recognition Challenge." *IJCV*, 2015. [arxiv.org/abs/1409.0575](https://arxiv.org/abs/1409.0575) *(ml-satellite-image-classifier, ml-recyclable-material-classifier-vgg16)*

**Model compression**
- Han, S., Pool, J., Tran, J., and Dally, W.J. "Learning Both Weights and Connections for Efficient Neural Networks." *NeurIPS*, 2015. [arxiv.org/abs/1506.02626](https://arxiv.org/abs/1506.02626) *(ml-model-compression)*
- Molchanov, P., Tyree, S., Karras, T., Aila, T., and Kautz, J. "Pruning Convolutional Neural Networks for Resource Efficient Inference." *ICLR*, 2017. [arxiv.org/abs/1611.06440](https://arxiv.org/abs/1611.06440) *(ml-model-compression)*
- Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*, 2018. [arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877) *(ml-model-compression)*
- Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." *NeurIPS Workshop*, 2015. [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531) *(ml-model-compression)*

**Clinical assessment**
- Wolf, S.L., et al. "Assessing Wolf Motor Function Test as Outcome Measure for Research in Patients After Stroke." *Stroke*, 32(7):1635–1639, 2001. [doi.org/10.1161/01.STR.32.7.1635](https://doi.org/10.1161/01.STR.32.7.1635) *(ml-wearable-motion-classifier)*

**Datasets**
- NOAA Global Historical Climatology Network Daily (GHCN-D). [ncei.noaa.gov](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) *(ml-boston-climate-modeler)*
- Harper, F.M., and Konstan, J.A. "The MovieLens Datasets: History and Context." *ACM TIIS*, 5(4):1–19, 2015. [doi.org/10.1145/2827872](https://doi.org/10.1145/2827872) *(ml-movie-recommender)*
- Leskovec, J., and Mcauley, J. "Learning to Discover Social Circles in Ego Networks." *NeurIPS*, 2012. [snap.stanford.edu/data/ego-Facebook.html](https://snap.stanford.edu/data/ego-Facebook.html) *(ml-social-network-predictor)*

## License

This repository is licensed under the [MIT License](LICENSE).
