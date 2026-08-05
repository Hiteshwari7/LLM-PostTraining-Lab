"""
evaluate.py

Script version of Notebook 3 (Base vs. SFT vs. DPO comparison).
Run after both train_sft.py and train_dpo.py have produced their adapters:

    python src/evaluate.py
"""

import argparse

import pandas as pd
from unsloth import FastLanguageModel

from dataset_utils import load_evaluation_questions, SYSTEM_PROMPT

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def load_all_checkpoints(sft_adapter_dir, dpo_adapter_dir):
    # WHAT: load the base model once, attach both adapters under distinct
    #       names so we can switch between "no adapter" / "sft" / "dpo"
    #       without reloading the model three times.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model.load_adapter(sft_adapter_dir, adapter_name="sft")
    model.load_adapter(dpo_adapter_dir, adapter_name="dpo")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(model, tokenizer, question, adapter_name=None):
    # WHAT: generate one answer with a specific adapter active (or none, for the base model).
    if adapter_name is None:
        model.disable_adapters()
    else:
        model.enable_adapters()
        model.set_adapter(adapter_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # WHY do_sample=False: evaluation should be deterministic/reproducible.
    output = model.generate(input_ids=inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)


def run_evaluation(model, tokenizer, questions):
    rows = []
    for q in questions:
        rows.append({
            "question": q,
            "base_model": generate_answer(model, tokenizer, q, adapter_name=None),
            "sft_model": generate_answer(model, tokenizer, q, adapter_name="sft"),
            "dpo_model": generate_answer(model, tokenizer, q, adapter_name="dpo"),
        })
        print(f"Done: {q[:60]}...")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare base vs SFT vs DPO checkpoints")
    parser.add_argument("--questions", default="data/evaluation_questions.json")
    parser.add_argument("--sft_adapter", default="outputs/sft_adapter")
    parser.add_argument("--dpo_adapter", default="outputs/dpo_adapter")
    parser.add_argument("--out_csv", default="outputs/evaluation_results.csv")
    args = parser.parse_args()

    model, tokenizer = load_all_checkpoints(args.sft_adapter, args.dpo_adapter)
    questions = load_evaluation_questions(args.questions)

    df = run_evaluation(model, tokenizer, questions)

    # WHAT: a simple, non-judgmental quantitative signal alongside the qualitative table.
    df["base_len"] = df["base_model"].str.split().apply(len)
    df["sft_len"] = df["sft_model"].str.split().apply(len)
    df["dpo_len"] = df["dpo_model"].str.split().apply(len)

    df.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")
    print(df[["base_len", "sft_len", "dpo_len"]].mean().rename("avg_words"))


if __name__ == "__main__":
    main()
