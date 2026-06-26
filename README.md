# Applied Machine Learning Projects

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A monorepo of independent, end-to-end machine learning projects, each in its own subdirectory with its own README, dependencies, and tests. Licensed as a whole under the [MIT License](LICENSE).

## Projects

| Project | Description |
| --- | --- |
| [ml-wearable-motion-classifier](ml-wearable-motion-classifier/) | Classifies upper-body movements from wrist-worn IMU recordings collected during the Wolf Motor Function Test (WMFT), turning raw accelerometer/gyroscope/quaternion data into wrist trajectories and motion-shape features, classified with a rule-based baseline or trained scikit-learn models. |
| [ml-satellite-image-classifier](ml-satellite-image-classifier/) | Binary classification of satellite image tiles as agricultural or non-agricultural land, implemented end-to-end in both Keras/TensorFlow and PyTorch, progressing from CNN baselines to CNN–Vision Transformer hybrids. |
| [ml-boston-climate-modeler](ml-boston-climate-modeler/) | Forecasts Boston-area (Reading, MA) daily precipitation, snowfall, and temperature from historical NOAA station data with a dependency-free, reproducible Python pipeline. |
| [ml-tiny-llm-gpt](ml-tiny-llm-gpt/) | An educational, from-scratch implementation of a small GPT-style decoder-only Transformer, covering tokenizer training, dataset preprocessing, training/checkpointing, and text generation. |
| [ml-movie-recommender](ml-movie-recommender/) | Builds actor/movie graphs from IMDb filmography data (PageRank, Jaccard similarity, community detection) as engineered features for a heterogeneous GNN (PyTorch Geometric), benchmarked against the original heuristic baselines on movie-rating prediction and, on MovieLens, genuine personalized top-N recommendation. |
| [ml-gcp-vertex-rag-chatbot](ml-gcp-vertex-rag-chatbot/) | A retrieval-augmented generation document Q&A app that lets users upload a PDF, TXT, Markdown, CSV, or DOCX file and ask questions answered from its content, built with LangChain, Google Vertex AI (Gemini, `text-embedding-004`), and Chroma, served through Gradio and deployable to Cloud Run. |
| [ml-social-network-predictor](ml-social-network-predictor/) | Analyzes Facebook/Google+ social graphs with `igraph` (community detection, ego networks, embeddedness/dispersion) and extends the analysis into a link-prediction task, comparing hand-engineered graph heuristics against DeepWalk-style PyTorch node embeddings using scikit-learn classifiers. |
| [ml-recyclable-material-classifier-vgg16](ml-recyclable-material-classifier-vgg16/) | Classifies images as organic or recyclable material using VGG16 transfer learning, comparing a frozen feature-extraction head against fine-tuning the convolutional base, with results reported on a held-out test set. |

Each subdirectory is self-contained: its own `pyproject.toml`/`requirements.txt`, `src`, and `tests`. See each project's README for setup and usage instructions.

## License

This repository is licensed under the [MIT License](LICENSE).
