"""Shared project paths, model/dataset IDs, and per-lab training constants."""

from pathlib import Path

import torch

# Force CPU everywhere: at these small model/batch sizes, MPS per-op dispatch
# overhead made generation >5x slower (see scripts/01), and some HF/TRL model-
# loading paths (meta-device init for tied weights, e.g. GPT-2's lm_head/wte)
# crash on MPS with "Placeholder storage has not been allocated". Disabling
# detection here means every `torch.backends.mps.is_available()` check across
# transformers/accelerate/trl consistently falls back to CPU.
torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Lab 1-1: Instruction fine-tuning (SFT + LoRA)
# ---------------------------------------------------------------------------
SFT_MODEL_ID = "facebook/opt-350m"
SFT_DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "WzOT_CwDALWedTtXjwH7bA/CodeAlpaca-20k.json"
)
SFT_DATASET_CACHE = RAW_DATA_DIR / "CodeAlpaca-20k.json"
SFT_TRAIN_SIZE = 500
SFT_EVAL_SIZE = 40
SFT_NUM_EPOCHS = 2
SFT_BATCH_SIZE = 4
SFT_LEARNING_RATE = 2e-4
SFT_MAX_SEQ_LENGTH = 256
SFT_LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
SFT_OUTPUT_DIR = TRAINED_MODELS_DIR / "sft_lora"
SFT_METRICS_PATH = REPORTS_DIR / "sft_metrics.json"

# ---------------------------------------------------------------------------
# Lab 1-2: Reward modeling (GPT-2 sequence classifier + LoRA)
# ---------------------------------------------------------------------------
REWARD_MODEL_ID = "gpt2"
REWARD_DATASET_ID = "Dahoas/synthetic-instruct-gptj-pairwise"
REWARD_TRAIN_SIZE = 800
REWARD_EVAL_SIZE = 200
REWARD_MAX_LENGTH = 256
REWARD_NUM_EPOCHS = 3
REWARD_BATCH_SIZE = 16
REWARD_LEARNING_RATE = 5e-5
REWARD_LORA_CONFIG = dict(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
    bias="none",
    task_type="SEQ_CLS",
)
REWARD_OUTPUT_DIR = TRAINED_MODELS_DIR / "reward_model"
REWARD_METRICS_PATH = REPORTS_DIR / "reward_metrics.json"

# ---------------------------------------------------------------------------
# Lab 2-1: PPO RLHF (sentiment reward on gpt2-imdb)
# ---------------------------------------------------------------------------
PPO_MODEL_ID = "lvwerra/gpt2-imdb"
PPO_REWARD_MODEL_ID = "lvwerra/distilbert-imdb"
PPO_DATASET_ID = "stanfordnlp/imdb"
PPO_DATASET_SIZE = 2000
PPO_BATCH_SIZE = 16
PPO_MINI_BATCH_SIZE = 16
PPO_NUM_STEPS = 25
PPO_LEARNING_RATE = 1.41e-5
PPO_INPUT_MIN_LEN = 2
PPO_INPUT_MAX_LEN = 8
PPO_OUTPUT_MIN_LEN = 4
PPO_OUTPUT_MAX_LEN = 16
PPO_OUTPUT_DIR = TRAINED_MODELS_DIR / "ppo_policy"
PPO_POSITIVE_DIR = PPO_OUTPUT_DIR / "positive"
PPO_NEGATIVE_DIR = PPO_OUTPUT_DIR / "negative"
PPO_METRICS_PATH = REPORTS_DIR / "ppo_metrics.json"

# ---------------------------------------------------------------------------
# Lab 2-2: Direct Preference Optimization (DPO + LoRA on GPT-2)
# ---------------------------------------------------------------------------
DPO_MODEL_ID = "gpt2"
DPO_DATASET_ID = "BarraHome/ultrafeedback_binarized"
DPO_TRAIN_SIZE = 300
DPO_EVAL_SIZE = 80
DPO_NUM_EPOCHS = 3
DPO_BATCH_SIZE = 2
DPO_LEARNING_RATE = 1e-4
DPO_BETA = 0.1
DPO_MAX_LENGTH = 512
DPO_MAX_PROMPT_LENGTH = 256
DPO_LORA_CONFIG = dict(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["c_proj", "c_attn"],
    bias="none",
    task_type="CAUSAL_LM",
)
DPO_OUTPUT_DIR = TRAINED_MODELS_DIR / "dpo_lora"
DPO_METRICS_PATH = REPORTS_DIR / "dpo_metrics.json"
DPO_EVAL_PROMPTS = [
    "My favorite thing about cooking is",
    "The best advice I ever received was",
    "When I think about the future, I",
    "The most important skill to learn is",
    "My day was made better when",
]
