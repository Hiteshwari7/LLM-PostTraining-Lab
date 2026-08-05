# Results Analysis — Base vs. SFT vs. DPO

*Fill this in after running `03_Evaluation.ipynb` (or `src/evaluate.py`). It pulls from `outputs/evaluation_results.csv`.*

## 1. Setup recap

- **Base model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **SFT data:** `instruction_dataset.jsonl` — _(N examples)_
- **DPO data:** `preference_dataset.jsonl` — _(N pairs)_
- **Evaluation set:** `evaluation_questions.json` — _(N questions, held out from training)_

## 2. Quantitative signal

Paste the average-word-count summary from Notebook 3 / `evaluate.py`:

| Model | Avg. response length (words) |
|---|---|
| Base | _(fill in)_ |
| SFT | _(fill in)_ |
| DPO | _(fill in)_ |

*Note: length alone is not a quality metric — treat it as one input alongside the qualitative read below, not a verdict.*

## 3. Qualitative comparison

Pick 3-5 representative questions from `outputs/evaluation_results.csv` and paste the three answers side by side. For each, note what changed and why it likely changed (tie it back to what that stage's data taught):

### Example question: _(paste a question from your eval set)_

**Base model:** _(paste)_

**SFT model:** _(paste)_

**DPO model:** _(paste)_

**What changed and why:** _(e.g., "Base rambles into generic ML history; SFT answers in the direct instruction→response format it was trained on; DPO version is shorter and more confident, consistent with the chosen responses in preference_dataset.jsonl favoring directness.")_

*(repeat for each selected question)*

## 4. Overall takeaways

- **Continued pretraining → SFT:** _(what changed in vocabulary/domain fluency once instruction-following was added)_
- **SFT → DPO:** _(what changed in response style/quality once preference alignment was added)_
- **Failure cases, if any:** _(any question where a later stage got *worse* — worth being honest about in a talk; alignment isn't guaranteed to improve every single response)_

## 5. For the talk

One or two of the clearest before/after examples from Section 3 make the strongest slide material — pick the pair where the difference is most obvious to someone unfamiliar with the pipeline, not necessarily the most technically interesting one.
