"""
train_dpo.py

Script version of Notebook 2 (Stage 3: DPO preference alignment).
Run after train_sft.py has produced outputs/sft_adapter:

    python src/train_dpo.py
"""

import argparse

from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

from dataset_utils import build_dpo_dataset

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def load_sft_model(sft_adapter_dir):
    # WHAT: reload the base model and re-attach the SFT adapter, so DPO
    #       refines an already instruction-tuned model rather than starting cold.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model.load_adapter(sft_adapter_dir, adapter_name="default")
    FastLanguageModel.for_training(model)
    return model, tokenizer


def run_dpo(model, tokenizer, preference_path, output_dir):
    # WHAT: Stage 3 - preference alignment. See reports/dpo_explanation.md for
    #       a full walkthrough of beta, learning rate, and the "no ref_model" choice.
    dataset = build_dpo_dataset(tokenizer, preference_path)
    args = DPOConfig(
        output_dir=output_dir,
        beta=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=5e-6,
        logging_steps=5,
        optim="adamw_8bit",
        warmup_steps=5,
        lr_scheduler_type="linear",
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=512,
        seed=42,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT model -> TRL uses the base model (adapter disabled) as reference
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Stage 3: DPO preference alignment")
    parser.add_argument("--preferences", default="data/preference_dataset.jsonl")
    parser.add_argument("--sft_adapter", default="outputs/sft_adapter")
    parser.add_argument("--dpo_train_out", default="outputs/stage3_dpo")
    parser.add_argument("--adapter_out", default="outputs/dpo_adapter")
    args = parser.parse_args()

    model, tokenizer = load_sft_model(args.sft_adapter)

    print("Stage 3: DPO...")
    run_dpo(model, tokenizer, args.preferences, args.dpo_train_out)

    print(f"Saving DPO adapter to {args.adapter_out}")
    model.save_pretrained(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)


if __name__ == "__main__":
    main()
