"""
dataset_utils.py

Shared helpers for loading and formatting the three datasets used across the
post-training pipeline. The notebooks import (or copy) these same functions
so the training scripts and the notebooks stay in sync.

WHAT this module does: read raw files (.txt / .jsonl / .json) and turn them
into either plain text (Stage 1) or chat-templated text (Stage 2/3).

WHY a separate module: keeping this logic out of the notebooks means it can
be unit-tested, reused by the src/train_*.py scripts, and changed in one
place if the data format ever changes.
"""

import json
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are PostTraining Tutor, an assistant that explains LLM training "
    "concepts (Transformers, LoRA, QLoRA, SFT, DPO, RLHF, alignment) "
    "clearly and concisely."
)


def load_jsonl(path):
    """WHAT: read a .jsonl file (one JSON object per line) into a list of dicts.
    WHY: both instruction_dataset.jsonl and preference_dataset.jsonl use this format.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    """WHAT: read a plain .json file (used for evaluation_questions.json)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_corpus(tokenizer, corpus_path, max_seq_length):
    """WHAT: split a raw text corpus into token chunks of at most max_seq_length.
    WHY: continued pretraining trains on fixed-length chunks, not one giant
         example, so every training step sees a consistent input size.
    HOW: tokenize once, then slice the token list into non-overlapping windows.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = tokenizer(raw_text, return_tensors=None)["input_ids"]
    chunks = [
        tokens[i:i + max_seq_length]
        for i in range(0, len(tokens), max_seq_length)
        if len(tokens[i:i + max_seq_length]) > 20  # WHY: drop tiny trailing scraps
    ]
    texts = [tokenizer.decode(c) for c in chunks]
    return Dataset.from_dict({"text": texts})


def build_sft_dataset(tokenizer, instruction_path):
    """WHAT: load instruction_dataset.jsonl and render each example through the
    model's chat template, ready for SFTTrainer.
    WHY: the model must be trained on the exact text format (system/user/
    assistant turns) it will see at inference time.
    """
    rows = load_jsonl(instruction_path)

    def format_example(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return Dataset.from_list(rows).map(format_example)


def build_dpo_dataset(tokenizer, preference_path):
    """WHAT: load preference_dataset.jsonl into the {prompt, chosen, rejected}
    shape that DPOTrainer expects.
    WHY: DPO needs the prompt rendered through the chat template (so the
    model sees the same format it was SFT-trained on), while chosen/rejected
    stay as plain assistant-response text.
    """
    rows = load_jsonl(preference_path)

    def format_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt_text,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    return Dataset.from_list(rows).map(format_prompt)


def load_evaluation_questions(eval_path):
    """WHAT: normalize evaluation_questions.json to a flat list of strings,
    whether it's stored as ["q1", "q2", ...] or [{"question": "q1"}, ...].
    """
    data = load_json(eval_path)
    if data and isinstance(data[0], dict):
        return [item["question"] for item in data]
    return data
