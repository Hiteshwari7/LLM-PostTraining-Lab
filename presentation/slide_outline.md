# Slide Outline — "LLM Post-Training Lab: Building PostTraining Tutor"

15 slides, structured to build from fundamentals up to your own results.

---

**Slide 1 — Title**
"LLM Post-Training Lab: Building a Domain-Specific AI Tutor" — your name, event, date.

**Slide 2 — What is an LLM?**
One-paragraph grounding: a model trained to predict the next token, at massive scale, over internet-scale text. Sets up why "pretraining" alone isn't the whole story.

**Slide 3 — Pretraining**
What the base model already knows before you touch it: general language, broad world knowledge, broad (but generic) reasoning. This is the expensive, foundational stage most people never run themselves.

**Slide 4 — The Post-Training Gap**
Why a pretrained model isn't yet an "assistant": it completes text, but doesn't reliably follow instructions or match human preferences. Post-training is the bridge.

**Slide 5 — The Pipeline, End to End**
Show `presentation/architecture_diagram.md`. Narrate the four boxes: Continued Pretraining → SFT → DPO → Evaluation.

**Slide 6 — Continued Pretraining**
What it is (more next-token prediction, on a narrower corpus), what it's for (domain vocabulary/fluency), what it's *not* (not instruction-following yet).

**Slide 7 — Supervised Fine-Tuning (SFT)**
What it is (training on instruction→response pairs), why it's the stage that makes a model "answer like an assistant."

**Slide 8 — LoRA**
The core idea: freeze the base model, train small low-rank adapter matrices instead of all parameters. Why this matters: <1% of parameters trained, megabyte-sized adapters.

**Slide 9 — QLoRA**
LoRA + a 4-bit quantized frozen base model. Why this is *the* thing that makes fine-tuning a 3B+ model possible on a free Colab GPU.

**Slide 10 — Preference Alignment: Why SFT Isn't Enough**
"Plausible" vs. "preferred" — motivate the need for a third stage with a concrete example (two SFT-plausible answers, one clearly better).

**Slide 11 — RLHF vs. DPO**
Classic RLHF: reward model + PPO. DPO: same underlying goal, one direct loss on (prompt, chosen, rejected) triples — no reward model, no RL loop.

**Slide 12 — This Project's Setup**
Model: `Llama-3.2-3B-Instruct` (4-bit). Datasets: your continued-pretraining corpus, instruction set, and preference set — all about LLM post-training itself (the self-referential angle).

**Slide 13 — Evaluation Methodology**
Same fixed, held-out question set run through Base / SFT / DPO checkpoints, loaded once with adapter-swapping for efficiency. Emphasize: this is what makes the comparison fair.

**Slide 14 — Results**
Your best 1-2 before/after examples from `reports/results_analysis.md`, plus the average-response-length signal. Be honest about any case where a later stage didn't clearly improve things.

**Slide 15 — Takeaways & What's Next**
Recap the pipeline in one line each. Future work: RLHF/PPO comparison, LLM-as-judge scoring, a live chat demo, merging + quantizing for deployment (see README's Future Improvements).

---

### Timing guide (for a ~15-20 min talk)
- Slides 1-4: ~3 min (context-setting)
- Slides 5-9: ~6 min (the pipeline + LoRA/QLoRA)
- Slides 10-11: ~4 min (why DPO, RLHF comparison — this is usually the most novel part for the audience)
- Slides 12-14: ~5 min (your actual project + results)
- Slide 15: ~1-2 min (wrap-up)
