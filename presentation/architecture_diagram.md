# Architecture Diagram

Standalone copy of the pipeline diagram from the README, with narration notes for presenting it live.

```mermaid
flowchart TD
    A[Base Model<br/>Llama-3.2-3B-Instruct - 4bit] --> B[Stage 1: Continued Pretraining<br/>Domain Adaptation on Raw Corpus]
    B --> C[Domain-Adapted Base Model]
    C --> D[Stage 2: Supervised Fine-Tuning - SFT<br/>LoRA / QLoRA on Instruction-Response Pairs]
    D --> E[SFT Model<br/>'PostTraining Tutor - v1']
    E --> F[Stage 3: Preference Alignment<br/>DPO on Chosen vs Rejected Pairs]
    F --> G[DPO-Aligned Model<br/>'PostTraining Tutor - v2']

    C -.eval.-> H[Evaluation Suite]
    E -.eval.-> H
    G -.eval.-> H
    H --> I[Comparison Report:<br/>Base vs SFT vs DPO]

    style A fill:#e8e8e8,stroke:#555
    style E fill:#cfe8ff,stroke:#1a5fb4
    style G fill:#d4f4dd,stroke:#2e7d32
    style I fill:#fff3cd,stroke:#b58105
```

## Narration notes, box by box

- **Base Model (gray):** "We start with an already instruct-tuned model — Llama-3.2-3B — so we're refining behavior, not teaching from zero."
- **Stage 1 — Continued Pretraining:** "First, we don't touch instructions at all — we just keep pretraining on domain text, so the model gets fluent in *our* vocabulary: LoRA, DPO, RLHF, alignment."
- **Stage 2 — SFT (blue):** "Now we teach it the instruction→response *format*, using LoRA and QLoRA so this fits on a free Colab GPU. This produces v1 of PostTraining Tutor."
- **Stage 3 — DPO (green):** "Finally, we show it pairs of better/worse answers to the same question, and Direct Preference Optimization nudges it toward the better one — without needing a separate reward model or RL loop like classic RLHF."
- **Evaluation (yellow):** "Every stage gets snapshotted and run through the same fixed question set — so what you'll see next isn't a claim, it's a side-by-side comparison."

## Suggested live-demo order

1. Show the diagram (this file).
2. Show one before/after example straight from `reports/results_analysis.md`.
3. Optionally: run 2-3 live inference calls in `03_Evaluation.ipynb` if time allows.
