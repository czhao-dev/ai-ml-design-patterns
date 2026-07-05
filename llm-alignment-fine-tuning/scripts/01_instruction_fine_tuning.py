"""Lab 1-1: Supervised instruction fine-tuning of facebook/opt-350m with LoRA.

Trains on CodeAlpaca-20k using TRL's SFTTrainer, comparing SacreBLEU before and
after LoRA fine-tuning on a held-out split of instructions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from src import config, data_utils, metrics, visualization


def generate_responses(gen_pipeline, prompts: list, batch_size: int = 8) -> list:
    """Greedy, batched generation — beam search and one-at-a-time calls were
    >5s/example on MPS due to per-op dispatch overhead at this model/batch size."""
    results = gen_pipeline(prompts, max_new_tokens=40, do_sample=False, batch_size=batch_size)
    return [r[0]["generated_text"] for r in results]


def main():
    # Generation pipelines and SFTTrainer are both run on CPU: at this model size
    # (350M params) and batch size, MPS per-op dispatch overhead dominated and
    # made baseline generation take >5s/example (same effect noted in the
    # ml-tiny-llm-gpt project's own CPU-vs-MPS benchmark at small batch sizes).
    device = "cpu"
    print(f"Using device: {device}")

    data_utils.download_codealpaca()
    dataset = load_dataset("json", data_files=str(config.SFT_DATASET_CACHE), split="train")
    dataset_split = dataset.train_test_split(test_size=0.2, seed=config.SEED)
    train_dataset = data_utils.subsample_dataset(dataset_split["train"], config.SFT_TRAIN_SIZE)
    test_dataset = data_utils.subsample_dataset(dataset_split["test"], config.SFT_EVAL_SIZE)
    print(f"Train: {len(train_dataset)}, Eval: {len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(config.SFT_MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.SFT_MODEL_ID).to(device)

    instructions = data_utils.formatting_prompts_func_no_response(test_dataset)
    instructions_with_responses = data_utils.formatting_prompts_func(test_dataset)
    expected_outputs = []
    for instr, instr_with_resp in zip(instructions, instructions_with_responses):
        tok_with_resp = tokenizer(instr_with_resp, return_tensors="pt", max_length=1024, truncation=True)
        tok_instr = tokenizer(instr, return_tensors="pt")
        expected = tokenizer.decode(
            tok_with_resp["input_ids"][0][len(tok_instr["input_ids"][0]) - 1 :],
            skip_special_tokens=True,
        )
        expected_outputs.append(expected)

    print("Generating baseline (pre-fine-tuning) responses...")
    base_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False
    )
    generated_outputs_base = generate_responses(base_pipeline, instructions)
    bleu_base = metrics.compute_sacrebleu(generated_outputs_base, expected_outputs)
    print(f"Baseline SacreBLEU: {bleu_base}")

    lora_config = LoraConfig(**config.SFT_LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=str(config.SFT_OUTPUT_DIR / "checkpoints"),
        num_train_epochs=config.SFT_NUM_EPOCHS,
        per_device_train_batch_size=config.SFT_BATCH_SIZE,
        per_device_eval_batch_size=config.SFT_BATCH_SIZE,
        max_seq_length=config.SFT_MAX_SEQ_LENGTH,
        learning_rate=config.SFT_LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        formatting_func=data_utils.formatting_prompts_func,
        args=training_args,
        packing=False,
        data_collator=collator,
    )

    print("Training (real local run, replacing the original notebook's commented-out trainer.train())...")
    trainer.train()

    print("Generating post-fine-tuning responses...")
    tuned_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False
    )
    generated_outputs_lora = generate_responses(tuned_pipeline, instructions)
    bleu_lora = metrics.compute_sacrebleu(generated_outputs_lora, expected_outputs)
    print(f"LoRA fine-tuned SacreBLEU: {bleu_lora}")

    config.SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(config.SFT_OUTPUT_DIR)

    visualization.plot_loss_curve(
        trainer.state.log_history,
        config.FIGURES_DIR / "sft_loss_curve.png",
        "Lab 1-1: SFT + LoRA training loss",
    )

    sample_table = visualization.render_before_after_table(
        instructions[:3], generated_outputs_base[:3], generated_outputs_lora[:3]
    )

    metrics.save_metrics(
        {
            "technique": "Instruction Fine-Tuning (SFT + LoRA)",
            "model": config.SFT_MODEL_ID,
            "dataset": "CodeAlpaca-20k",
            "train_size": len(train_dataset),
            "eval_size": len(test_dataset),
            "sacrebleu_base": bleu_base,
            "sacrebleu_lora": bleu_lora,
            "sample_before_after_table": sample_table,
        },
        config.SFT_METRICS_PATH,
    )
    print(f"Saved metrics to {config.SFT_METRICS_PATH}")


if __name__ == "__main__":
    main()
