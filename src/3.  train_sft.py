"""
train_sft.py

Script version of Notebook 1 (Stage 1: continued pretraining + Stage 2: SFT).
Run this directly if you'd rather use a terminal/script workflow than Colab:

    python src/train_sft.py

Keeps the same parameter choices as the notebook, with the same WHAT/WHY
comments, just organized as functions instead of notebook cells.
"""

import argparse

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from dataset_utils import chunk_corpus, build_sft_dataset

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def load_base_model():
    # WHAT: load the 4-bit base model + tokenizer (see README/Notebook 1 for why
    #       this specific model was chosen).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def add_lora(model):
    # WHAT: attach trainable LoRA adapters. See reports/lora_explanation.md for
    #       a full walkthrough of each parameter below.
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )


def run_continued_pretraining(model, tokenizer, corpus_path, output_dir):
    # WHAT: Stage 1 - a short domain-adaptation pass over raw text.
    dataset = chunk_corpus(tokenizer, corpus_path, MAX_SEQ_LENGTH)
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=5,
        optim="adamw_8bit",
        warmup_steps=10,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args)
    trainer.train()


def run_sft(model, tokenizer, instruction_path, output_dir):
    # WHAT: Stage 2 - instruction tuning on {instruction, response} pairs.
    dataset = build_sft_dataset(tokenizer, instruction_path)
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        warmup_steps=10,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Stage 1 + Stage 2: continued pretraining + SFT")
    parser.add_argument("--corpus", default="data/llm_post_training_corpus.txt")
    parser.add_argument("--instructions", default="data/instruction_dataset.jsonl")
    parser.add_argument("--pretrain_out", default="outputs/stage1_continued_pretraining")
    parser.add_argument("--sft_out", default="outputs/stage2_sft")
    parser.add_argument("--adapter_out", default="outputs/sft_adapter")
    args = parser.parse_args()

    model, tokenizer = load_base_model()
    model = add_lora(model)

    print("Stage 1: continued pretraining...")
    run_continued_pretraining(model, tokenizer, args.corpus, args.pretrain_out)

    print("Stage 2: SFT...")
    run_sft(model, tokenizer, args.instructions, args.sft_out)

    print(f"Saving SFT adapter to {args.adapter_out}")
    model.save_pretrained(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)


if __name__ == "__main__":
    main()
