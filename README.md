# Applied AI/ML Projects

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A monorepo of ten end-to-end machine learning projects spanning computer vision, large language models, agentic AI and tool use, graph learning, time-series forecasting, model compression, production inference serving, causal inference, and systems-level ML inference engineering. Each project lives in its own subdirectory with independent dependencies, tests, documented results, and a full README.

## Projects at a Glance

| Project | Area | Key Technologies | Standout |
| --- | --- | --- | --- |
| [GenAI RAG Chatbot](gen-ai-rag-chatbot/README.md) | RAG / GenAI | LangChain, Vertex AI, Chroma, Cloud Run | Document Q&A app deployed to GCP Cloud Run |
| [Agentic AI Tool Use](agentic-ai-tool-use/README.md) | Agentic AI / Tool Use | OpenAI API, Python | Live gpt-4.1 benchmark on 35 tasks — Reflexion 94.29% beats Plan-and-Execute 88.57% and ReAct 85.71% for $0.39 total; 44 tests, zero real API calls in CI |
| [Churn Predictor](churn-predictor/README.md) | Causal Inference / Uplift Modeling | scikit-learn, EconML | Causal forest is the only model to beat random targeting (Qini +369); targeted policy cuts net losses 98.5% vs. blanket offers |
| [Tiny LLM GPT](tiny-llm-gpt/README.md) | Language Modeling | PyTorch, AWS EC2 | Tiny/Small/Medium scaling sweep — perplexity 7.11 → 5.09; Small/Medium trained on a rented cloud GPU |
| [LLM Alignment Fine-Tuning](llm-alignment-fine-tuning/README.md) | LLM Alignment | PyTorch, TRL, HuggingFace, LoRA | Full SFT → RM → PPO RLHF → DPO pipeline, all trained locally |
| [CNN-ViT Satellite Image Classifier](cnn-vit-satellite-image-classifier/README.md) | Computer Vision | PyTorch, Keras/TF, FastAPI, Docker | 99.83% accuracy; FastAPI server serving all four models |
| [ML Model Compression](ml-model-compression/README.md) | Model Compression | PyTorch, `torch.ao.quantization` | Distilled student ~150× smaller than its teacher at 99.9%+ accuracy |
| [GNN Movie Recommender](gnn-movie-recommender/README.md) | Graph ML | PyTorch Geometric, igraph | Heterogeneous GNN benchmarked at full IMDb scale (240K movies) as well as a small sample; top-N recommendation on MovieLens |
| [LSTM Transformer Climate Modeler](lstm-transformer-climate-modeler/README.md) | Time-Series | TensorFlow, Python | Pure-TF LSTM + Transformer from scratch (no Keras); 7-day multi-step forecasting; 56 unit tests |
| [Tensor Graph Inference Engine](tensor-graph-inference-engine/README.md) | Systems / Inference Engine | Python, NumPy | Static-graph INT8 engine with a hand-rolled greedy arena planner cutting activation memory ~37% (39,904 vs. 63,072 bytes), fully regression-pinned |

---

## Project Details

### [GenAI RAG Chatbot](gen-ai-rag-chatbot/README.md)

A deployed RAG document Q&A app: upload a PDF, TXT, Markdown, CSV, or DOCX file and ask questions grounded in its content.

**Live demo:** https://rag-pdf-chatbot-715060982814.us-central1.run.app

- **RAG pipeline:** LangChain document loaders → `RecursiveCharacterTextSplitter` → Vertex AI `text-embedding-004` embeddings → Chroma vector store → `RetrievalQA` chain → Gemini 2.5 Flash answer with source grounding.
- **Deployment:** Containerized with Docker and deployed to GCP Cloud Run with scale-to-zero cost controls; credentials handled via Application Default Credentials for local development.
- **Interface:** Gradio web UI; standalone annotated scripts for each RAG concept (loading, splitting, embedding, retrieval) as reference implementations.

**Stack:** Python · LangChain · Google Vertex AI (Gemini + text-embedding-004) · Chroma · Gradio · Docker · Cloud Run

---

### [Agentic AI Tool Use](agentic-ai-tool-use/README.md)

Three agent architectures — ReAct, Plan-and-Execute, and Reflexion — implemented from scratch directly against the OpenAI Chat Completions tool-use API (no LangChain `AgentExecutor`), benchmarked head-to-head on the same hand-built 35-task suite.

- **Architectures:** ReAct is a single flat tool-use loop with tools bound from turn one; Plan-and-Execute runs an upfront planning call with tools deliberately unbound (so nothing can execute before a plan exists), then one bounded sub-loop per subtask, then a synthesis call; Reflexion retries (up to 3 attempts) on explicit failure signals — an unresolved tool error, exhausted step budget, or the model admitting failure — never the ground-truth answer — with a self-critique call between attempts that sees only the failed transcript.
- **Hand-rolled tools:** an AST-whitelist calculator (no `eval()`), a sandboxed subprocess Python executor (static import/builtin pre-check, minimal env with no API key, timeout + POSIX resource limits), and an Okapi BM25 retrieval index built from scratch — no scikit-learn, no embeddings.
- **Benchmark:** 35 hand-written tasks (arithmetic, multi-hop QA, code execution, injected-error recovery) with unambiguous ground truth, evaluated over a fully original synthetic knowledge base (17 documents describing fictional companies/people/products with deliberately chained facts) — no real documents, no external search API, no licensing risk.
- **Live results (gpt-4.1, full 35-task run, $0.39 total):** Reflexion 94.29% success beats Plan-and-Execute 88.57% and ReAct 85.71%, at roughly a third of Plan-and-Execute's cost — Plan-and-Execute pays a ~3x LLM-call overhead per task (planning + per-subtask loop + synthesis) that isn't recovered by materially better accuracy on this task set. The first full run scored only 71-74% for ReAct/Reflexion; that gap turned out to be a scoring-harness bug, not a capability gap — the system prompts didn't constrain answers to a bare value, so `gpt-4.1` correctly solving a problem and answering in a full sentence was marked wrong by exact-match scoring. Fixing the prompts to require a bare fact recovered the missing 15-20 points.
- **Testing:** 44 tests against a hand-written fake OpenAI client verify each architecture's control flow (stopping conditions, tool dispatch, retry-after-failure logic) with zero real API calls and zero cost, including a regression guard that ground truth never leaks into Reflexion's self-critique prompt.

**Stack:** Python · OpenAI API · pytest · Matplotlib

---

### [Churn Predictor: Uplift Modeling for Retention Targeting](churn-predictor/README.md)

Uplift modeling (not classification) for a retention-offer targeting decision: which customers should actually receive an offer, versus who would convert/stay anyway.

- **Real RCT, not a simulation**: Kevin Hillstrom's MineThatData email challenge — 64,000 customers genuinely randomized into email/no-email arms — so every causal estimate below is computed from real counterfactual data, not an invented ground-truth treatment effect. Randomization is verified (all covariate SMDs < 0.02) before any downstream estimate is trusted.
- **Three CATE estimators compared head-to-head**: a hand-rolled T-learner, a hand-rolled X-learner (Künzel et al., 2019, with propensity-weighted blending), and a causal forest (`econml`'s `CausalForestDML`) — evaluated against a naive "predict conversion, ignore treatment" baseline on identical features and splits.
- **Causal-inference metrics, not classification metrics**: Qini curves/coefficients, AUUC, uplift@k, and per-decile calibration, all built from group-level treated-vs-control comparisons on held-out data since individual treatment effects are fundamentally unobservable. Only the causal forest beats random targeting (Qini +369); the hand-rolled T-/X-learners underperform random ranking on this dataset — reported as a real finding, not adjusted away.
- **A genuine $ number**: targeting the top 5% of customers by causal-forest-predicted uplift reduces net losses by 98.5% versus blanket-targeting everyone (-$198 vs. -$13,170 at an assumed $2/offer cost), while still capturing 12.6% of the total achievable incremental revenue for 2.5% of the campaign cost.

**Stack:** Python · scikit-learn · EconML (`CausalForestDML`) · pandas · Matplotlib

---

### [Tiny LLM GPT](tiny-llm-gpt/README.md)

A from-scratch GPT-style language model covering the complete pipeline from raw text to generated output.

- **Architecture:** Decoder-only Transformer with configurable depth, heads, and embedding dimension; causal self-attention, learned positional embeddings, and layer normalization — no HuggingFace model code in the loop.
- **Pipeline:** Custom BPE tokenizer training → dataset preprocessing and sequence packing → training loop with gradient clipping and validation checkpointing → top-k / top-p text generation → perplexity evaluation → throughput and memory benchmarking.
- **Model scaling experiment:** Tiny (~4.3M), Small (~15M), and Medium (~30M) variants all trained for the full 20,000 steps on the same TinyStories tokenizer/dataset — validation perplexity improves monotonically (7.11 → 5.36 → 5.09) with capacity.
- **Cloud training:** Small/Medium's full-scale runs were provisioned on a rented AWS g4dn.xlarge (Tesla T4) rather than the author's laptop, cutting combined training time from an estimated ~34 hours (Apple M3) to under 10 hours.
- Intentionally sized to run on consumer hardware while demonstrating every component of a modern LLM training stack.

**Stack:** Python · PyTorch · AWS EC2

---

### [LLM Alignment Fine-Tuning](llm-alignment-fine-tuning/README.md)

Four LLM alignment techniques implemented end-to-end, each with its own training objective and evaluation metric — all locally runnable on a laptop CPU.

- **Supervised fine-tuning (SFT):** LoRA-adapts `facebook/opt-350m` on CodeAlpaca-20k with TRL's `SFTTrainer`; evaluated with SacreBLEU before and after fine-tuning.
- **Reward modeling:** GPT-2 + LoRA trained on chosen/rejected response pairs with `RewardTrainer`; reaches **0.96 pairwise ranking accuracy** on a held-out preference set.
- **PPO RLHF:** `gpt2-imdb` steered toward positive and negative sentiment with `PPOTrainer` against a sentiment-classifier reward; mean reward improves from 0.24 → 1.27 (positive policy) and −0.32 → 0.56 (negative policy).
- **DPO:** GPT-2 + LoRA fine-tuned directly on preference pairs with `DPOTrainer` (no separate reward model or RL loop); reaches **0.70 reward accuracy** on held-out pairs.
- Every script replaces commented-out or pre-downloaded training from the source notebooks with real, locally-runnable training — every number above comes from an actual training run.

**Stack:** Python · PyTorch · HuggingFace Transformers · TRL · LoRA (PEFT)

---

### [CNN-ViT Satellite Image Classifier](cnn-vit-satellite-image-classifier/README.md)

Binary classification of 64×64 satellite image tiles as agricultural vs. non-agricultural land.

- **Models:** Six-block CNN (32→1024 channels, 5×5 kernels) and CNN–Vision Transformer hybrid (CNN feature map tokenized and fed through multi-head self-attention blocks), each implemented independently in both Keras/TensorFlow and PyTorch — four models total, trained and evaluated in parallel across frameworks.
- **Results:** PyTorch CNN 99.83%, Keras CNN 99.33%, PyTorch CNN-ViT 99.67%, Keras CNN-ViT 99.42% — all on a 1,200-image held-out split never seen during training.
- **Inference server:** FastAPI app (`serve/`) loads all four models at startup and exposes `/health`, `/models`, and `POST /predict?model=` endpoints. Model backend is selectable per request. Deployed with Docker Compose; model weights mounted read-only at runtime to keep the image small.
- **Notable:** Caught and fixed a data-leakage bug in the original evaluation methodology that scored models against the full training set rather than a held-out split.

**Stack:** Python · PyTorch · Keras/TensorFlow · FastAPI · Uvicorn · Docker Compose

---

### [ML Model Compression](ml-model-compression/README.md)

Three orthogonal compression techniques — pruning, post-training quantization, and knowledge distillation — benchmarked against the PyTorch CNN and CNN-ViT models trained in `ml-satellite-image-classifier`, all scored on the same fixed held-out split for a controlled comparison.

- **Pruning:** Unstructured global L1 magnitude pruning and structured L1 channel pruning (`torch.nn.utils.prune`), swept across 20–80% sparsity. Unstructured holds accuracy to 60% sparsity but doesn't reduce size/latency without sparse BLAS; structured produces real size/latency drops but collapses to the class prior by 40% sparsity without fine-tuning recovery — an intentionally honest "raw accuracy cliff" measurement.
- **Quantization:** Static INT8 PTQ on the CNN (4× smaller, ~1.8× faster, no measurable accuracy loss) and dynamic INT8 PTQ on the CNN-ViT's linear layers. Required working around two undocumented gaps in the source architecture: eager-mode static quantization needs manual `QuantStub`/`DeQuantStub` insertion, and the CNN's `BatchNorm` layers (positioned after pooling, not fusable with the preceding conv) have no quantized kernel and must run in FP32 sandwiched between quant/dequant boundaries.
- **Knowledge distillation:** A 3-block, ~259K-parameter `StudentCNN` distilled from the frozen CNN-ViT teacher (temperature-scaled KL + hard-label CE, with the standard T² gradient-scaling term) reaches 99.9%+ accuracy at ~150× smaller than the teacher and sub-millisecond CPU latency — one of several places results diverged from the initial write-up (~12× was the original estimate) once actually measured.
- **Reproducibility groundwork:** Corrected the CNN-ViT's constructor defaults (`depth=6, heads=8`) against its real trained hyperparameters (`depth=3, heads=6`) before any of the above would load correctly, and standardized every technique on a single canonical held-out split.

**Stack:** Python · PyTorch · `torch.ao.quantization` · torchvision · scikit-learn

---

### [GNN Movie Recommender](gnn-movie-recommender/README.md)

Graph feature engineering pipeline extended with a heterogeneous GNN, evaluated on two separate tasks at two different scales.

- **Feature engineering (igraph):** Actor/movie networks built from IMDb data; actors ranked by PageRank, movie communities detected (Fast Greedy Newman by default, Louvain at full scale), Jaccard movie-movie similarity computed — all kept exactly as in the original coursework pipeline.
- **Heterogeneous GNN (PyTorch Geometric):** `HeteroConv` encoder with `GraphConv` for weighted relations and `SAGEConv` for unweighted bipartite edges; graph-derived features (PageRank, community ID, Jaccard similarity) used as node features. Mini-batch `NeighborLoader` training for the full-scale graph, full-batch for the small sample and MovieLens.
- **IMDb track, two scales:** a 7-movie hand-curated sample (leave-one-out CV, useful only to prove the pipeline runs) and a real full-scale run built from IMDb's official Non-Commercial Datasets (129,720 labeled movies, random holdout split) — benchmarked against neighborhood averaging, linear regression, and bipartite graph averaging baselines. **Honest full-scale finding:** the GNN (RMSE 1.35) does not beat a trivial train-mean baseline (RMSE 1.13) as configured here, consistent with the linear-regression heuristic's own R² ≈ 0.02 across all 129,720 movies — real evidence about generalization at scale, not just proof the pipeline runs.
- **MovieLens track:** Genuine personalized top-N recommendation on `ml-latest-small` (943 users, ~9,700 movies), evaluated with Precision/Recall/NDCG@{5,10,20} against the full catalog — not sampled negatives.

**Stack:** Python · PyTorch · PyTorch Geometric · igraph · scikit-learn

---

### [LSTM Transformer Climate Modeler](lstm-transformer-climate-modeler/README.md)

Daily weather forecasting for Reading, MA (Boston suburb) from NOAA station data — two complete pipelines in one repo: a stdlib-only Ridge baseline and a pure-TensorFlow deep learning stack.

- **Ridge baseline (v0.1):** NOAA CSV parsing → missing-value handling → seasonal, lag, and rolling-window feature engineering → Ridge regression from scratch (Gauss-Jordan solver, no NumPy) → model serialization to JSON. Temperature R² = 0.747 on 2016 test set.
- **LSTM forecaster (v0.2):** 2-layer stacked `LSTMCell` unrolled with `tf.unstack`; all gate weights as raw `tf.Variable`; trained with `tf.GradientTape` and a hand-built Adam optimiser. Predicts PRCP, SNOW, and TOBS jointly 7 days ahead.
- **Transformer forecaster (v0.2):** Pre-norm encoder with sinusoidal positional encoding, `MultiHeadAttention` (Q/K/V projections as `tf.Variable`, scaled dot-product via `tf.linalg.matmul`), GELU feed-forward blocks, and mean pooling. **No `tf.keras` API used anywhere.**
- **Training infrastructure:** `tf.GradientTape` mini-batch loop, `tf.clip_by_global_norm` gradient clipping, temporal validation split, early stopping. Trained on **58 years of data** (1960–2017, 20 974 windows); the Transformer reaches R² = **0.693** on 7-day-ahead temperature over a 2018–2019 test set, outperforming the LSTM (R² = 0.639). PRCP/SNOW R² near zero is expected given event sparsity.
- **Notebook:** `notebooks/climate_exploration.ipynb` covers EDA, training loss curves, attention heatmap visualisation for each encoder block, and a four-model comparison.
- **Testing:** 56 unit tests across data cleaning, feature engineering, sequence windowing, scaler round-trips, TF primitive shapes, and weight serialisation.

**Stack:** Python · TensorFlow 2.14+ (`tf.Module` / `tf.Variable` / `tf.GradientTape`) · NumPy · Jupyter · Matplotlib

---

### [Tensor Graph Inference Engine](tensor-graph-inference-engine/README.md)

A from-scratch, static-graph INT8 neural-network inference engine: an offline compiler that quantizes and memory-plans a fixed demo topology, and a minimal runtime that executes it via pre-planned buffer views with no additional allocation.

- **Static graph + INT8 quantization:** builds a DAG for a 3-layer MLP with a residual/skip connection (14 nodes, 21 tensors), quantizing weights and activations to per-tensor symmetric INT8 (scale = max(abs)/127, zero_point = 0).
- **Greedy arena memory planner:** a first-fit interval/lifetime allocator packs all activation tensors into a single pre-sized arena — 39,904 bytes versus a naive 63,072-byte allocation (~37% reduction), regression-pinned in the test suite.
- **Zero-allocation-spirit runtime:** the engine performs exactly two allocations at load time (file buffer + arena buffer); `forward()` writes results directly into pre-planned buffer views (NumPy basic slices reinterpreted via `.view(dtype)`) rather than allocating per step.
- **Full regression coverage:** pytest suite pins exact tensor/node counts and arena byte size, verifies quantization round-trip error bounds, and validates end-to-end prediction agreement against an fp32 reference implementation.
- **Originally built in C++** (header-only, CPU-only) as a systems exercise, then fully migrated to a pure-Python/NumPy implementation preserving the identical binary artifact format, algorithm, and regression-pinned constants.

**Stack:** Python · NumPy · pytest

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

**Agentic AI and tool use**
- Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*, 2023. [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) *(ml-agentic-tool-use-bakeoff)*
- Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS*, 2023. [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366) *(ml-agentic-tool-use-bakeoff)*
- Wang, L., et al. "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models." *ACL*, 2023. [arxiv.org/abs/2305.04091](https://arxiv.org/abs/2305.04091) *(ml-agentic-tool-use-bakeoff)*
- Robertson, S., and Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 2009. [doi.org/10.1561/1500000019](https://doi.org/10.1561/1500000019) *(ml-agentic-tool-use-bakeoff)*

**Graph learning**
- Fey, M., and Lenssen, J.E. "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop*, 2019. [arxiv.org/abs/1903.02428](https://arxiv.org/abs/1903.02428) *(ml-movie-recommender)*

**Causal inference and uplift modeling**
- Künzel, S.R., Sekhon, J.S., Bickel, P.J., and Yu, B. "Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning." *PNAS*, 2019. [arxiv.org/abs/1706.03461](https://arxiv.org/abs/1706.03461) *(churn-predictor)*
- Athey, S., and Wager, S. "Estimating Treatment Effects with Causal Forests: An Application." *Observational Studies*, 2019. [arxiv.org/abs/1902.07409](https://arxiv.org/abs/1902.07409) *(churn-predictor)*
- Radcliffe, N.J., and Surry, P.D. "Real-World Uplift Modelling with Significance-Based Uplift Trees." *Portrait Technical Report*, 2011. *(churn-predictor)*
- Gutierrez, P., and Gérardy, J-Y. "Causal Inference and Uplift Modelling: A Review of the Literature." *JMLR Workshop and Conference Proceedings*, 2017. *(churn-predictor)*

**Transfer learning and convolutional networks**
- Simonyan, K., and Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*, 2015. [arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556) *(ml-satellite-image-classifier)*
- Russakovsky, O., et al. "ImageNet Large Scale Visual Recognition Challenge." *IJCV*, 2015. [arxiv.org/abs/1409.0575](https://arxiv.org/abs/1409.0575) *(ml-satellite-image-classifier)*

**Model compression**
- Han, S., Pool, J., Tran, J., and Dally, W.J. "Learning Both Weights and Connections for Efficient Neural Networks." *NeurIPS*, 2015. [arxiv.org/abs/1506.02626](https://arxiv.org/abs/1506.02626) *(ml-model-compression)*
- Molchanov, P., Tyree, S., Karras, T., Aila, T., and Kautz, J. "Pruning Convolutional Neural Networks for Resource Efficient Inference." *ICLR*, 2017. [arxiv.org/abs/1611.06440](https://arxiv.org/abs/1611.06440) *(ml-model-compression)*
- Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*, 2018. [arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877) *(ml-model-compression)*
- Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." *NeurIPS Workshop*, 2015. [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531) *(ml-model-compression)*

**Datasets**
- NOAA Global Historical Climatology Network Daily (GHCN-D). [ncei.noaa.gov](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) *(ml-boston-climate-modeler)*
- Harper, F.M., and Konstan, J.A. "The MovieLens Datasets: History and Context." *ACM TIIS*, 5(4):1–19, 2015. [doi.org/10.1145/2827872](https://doi.org/10.1145/2827872) *(ml-movie-recommender)*
- Hillstrom, K. "The MineThatData E-Mail Analytics And Data Mining Challenge." 2008. [minethatdata.com](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv) *(churn-predictor)*

## License

This repository is licensed under the [MIT License](LICENSE).
