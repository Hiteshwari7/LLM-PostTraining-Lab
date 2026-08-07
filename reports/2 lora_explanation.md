# LoRA and QLoRA, Explained

## 1. The problem they solve

A 3B-parameter model has 3 billion numbers. Full fine-tuning means updating (and storing gradients + optimizer state for) all 3 billion of them, for Adam-style optimizers, that's roughly 3-4x the model's own size in extra memory, on top of the model itself. That's out of reach for a free Colab GPU.

## 2. LoRA — Low-Rank Adaptation

**Core idea:** freeze the entire pretrained model. For selected weight matrices `W` (e.g. attention projections), instead of updating `W` directly, add a small update `ΔW = B·A`, where `A` and `B` are much smaller, low-rank matrices. Only `A` and `B` are trained; `W` never changes.

**Why "low-rank" works:** empirically, the *change* a model needs during fine-tuning tends to be well-approximated by a low-rank matrix, even though the original weight matrix `W` is full-rank. So a small number of trainable parameters can capture most of the useful adaptation.

**Result:** you train maybe 0.1-1% of the total parameters, and the saved adapter file is a few megabytes instead of several gigabytes.

### Key parameters (as used in this project's notebooks/scripts)

| Parameter | This project's value | What it controls |
|---|---|---|
| `r` (rank) | 16 | Size of the low-rank bottleneck in `A`/`B`. Higher `r` = more capacity, more memory, more overfitting risk on small datasets. |
| `lora_alpha` | 16 | Scaling factor for the adapter's contribution (`scale = alpha / r`). Convention: `alpha == r` or `alpha == 2r`. |
| `lora_dropout` | 0.05 | Dropout on the LoRA path only, as light regularization against overfitting on a small instruction set. |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Which weight matrices get adapters — the standard, well-tested set for Llama-family attention + MLP blocks. |
| `bias` | `"none"` | Whether to also train bias terms — usually skipped, since it rarely helps and adds parameters. |

## 3. QLoRA — LoRA on a quantized base model

**Core idea:** apply LoRA on top of a base model whose *frozen* weights are stored in 4-bit precision instead of the usual 16/32-bit. The LoRA adapter matrices (`A`, `B`) are still trained in higher precision (bf16/fp16) — only the frozen, non-trainable weights are compressed.

**Why this matters:** the frozen base weights are the memory-heavy part of the model. Compressing *those* to 4-bit (while keeping the small trainable adapter in higher precision for stable gradients) is what lets a 3B — or even 70B — model be fine-tuned on a single consumer/free-tier GPU.

**In code, QLoRA is really "LoRA + 4-bit loading":**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=True,   # <- this is the "Q" in QLoRA
)
# Example: creating LoRA adapters during SFT
model = FastLanguageModel.get_peft_model(model, r=16, ...)  # <- this is the LoRA part
```

## 4. Trade-offs to be upfront about

- **Slightly lower ceiling than full fine-tuning** on very large, very different-from-base tasks — LoRA's low-rank assumption doesn't perfectly capture every possible weight update.
- **Rank and target-module choices matter** — too small an `r` under-fits; targeting too few modules limits what the adapter can influence.
- **4-bit quantization adds a small amount of numerical noise** to the frozen weights, though in practice this is a good trade for the memory savings on this scale of model.

## 5. Why this matters for the pipeline as a whole

Both training stages in this project (SFT and DPO) use the same LoRA-based adapter approach.

During SFT, LoRA adapters learn how to transform the base model into a domain-specific educational assistant that follows instruction-response patterns.

During DPO, the SFT adapter is further optimized using preference pairs, refining the model's response style toward preferred answers.

Because only small LoRA adapter matrices are trained:

- Training is possible on a free-tier GPU such as Colab T4.
- Adapter files remain small compared to full model checkpoints.
- The same frozen base model can be reused with different adapters.

During evaluation, the base model, SFT adapter, and DPO adapter can be loaded together and switched dynamically, allowing direct comparison of how each post-training stage changes model behaviour.
