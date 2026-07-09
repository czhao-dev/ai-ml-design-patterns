"""Tests for src/config.py."""

from pathlib import Path

import torch

from src import config


def test_mps_availability_check_is_forced_off():
    # config.py monkeypatches this at import time so HF/TRL model-loading
    # paths that crash on MPS consistently fall back to CPU (see the
    # module docstring for the "Placeholder storage" failure this avoids).
    assert torch.backends.mps.is_available() is False


def test_get_device_prefers_cpu_when_no_accelerator_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert config.get_device() == "cpu"


def test_get_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert config.get_device() == "cuda"


def test_project_root_is_repo_root():
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_derived_paths_nest_under_project_root():
    assert config.RAW_DATA_DIR == config.DATA_DIR / "raw"
    assert config.TRAINED_MODELS_DIR == config.MODELS_DIR / "trained"
    assert config.FIGURES_DIR == config.REPORTS_DIR / "figures"


def test_lab_output_dirs_nest_under_trained_models_dir():
    assert config.SFT_OUTPUT_DIR == config.TRAINED_MODELS_DIR / "sft_lora"
    assert config.REWARD_OUTPUT_DIR == config.TRAINED_MODELS_DIR / "reward_model"
    assert config.PPO_OUTPUT_DIR == config.TRAINED_MODELS_DIR / "ppo_policy"
    assert config.DPO_OUTPUT_DIR == config.TRAINED_MODELS_DIR / "dpo_lora"
    assert config.PPO_POSITIVE_DIR == config.PPO_OUTPUT_DIR / "positive"
    assert config.PPO_NEGATIVE_DIR == config.PPO_OUTPUT_DIR / "negative"


def test_lora_configs_target_causal_lm_or_seq_cls():
    assert config.SFT_LORA_CONFIG["task_type"] == "CAUSAL_LM"
    assert config.REWARD_LORA_CONFIG["task_type"] == "SEQ_CLS"
    assert config.DPO_LORA_CONFIG["task_type"] == "CAUSAL_LM"


def test_dpo_eval_prompts_is_a_nonempty_list_of_strings():
    assert len(config.DPO_EVAL_PROMPTS) > 0
    assert all(isinstance(p, str) for p in config.DPO_EVAL_PROMPTS)
