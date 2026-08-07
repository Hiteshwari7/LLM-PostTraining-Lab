# Results Analysis — Base vs. SFT vs. DPO

*Fill this in after running `03_Evaluation.ipynb` (or `src/evaluate.py`). It pulls from `outputs/evaluation_results.csv`.*

## 1. Setup recap

- **Base model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **SFT data:** `instruction_dataset.jsonl` — _(100 instruction examples)_
- **DPO data:** `preference_dataset.jsonl` — _(50 preference pairs)_
- **Evaluation set:** `evaluation_questions.json` — _(49 questions)_

## 2. Quantitative signal

Paste the average-word-count summary from Notebook 3 / `evaluate.py`:

| Model | Avg. response length (words) |
| ----- | ---------------------------: |
| Base  |                       159.20 |
| SFT   |                       158.82 |
| DPO   |                       159.00 |


*Note: length alone is not a quality metric — treat it as one input alongside the qualitative read below, not a verdict.*

## 3. Qualitative comparison

### Example question: _(What is a Large Language Model?)_

**Base model:** _(What is a Large Language Model? (LLM)
==========================

A Large Language Model (LLM) is a type of artificial intelligence (AI) model that is trained on a massive corpus of text data to generate human-like language. LLMs are designed to process and understand natural language, enabling them to perform a wide range of tasks, such as:

*   Text generation
*   Language translation
*   Sentiment analysis
*   Text summarization
*   Chatbots and conversational AI

**Key Characteristics of LLMs:**

1.  **Massive Training Data**: LLMs are trained on enormous amounts of text data, often in the order of billions of parameters.
2.  **Deep Neural Network Architecture**: LLMs are built using deep neural networks, which consist of multiple layers of interconnected nodes (neurons).
3.  **Self-Supervised Learning**: LLMs are typically trained using self-supervised learning methods, where the model is trained to predict the next word)_

**SFT model:** _(What is a Large Language Model? (LLM)
==============================

A Large Language Model (LLM) is a type of machine learning model that is trained on vast amounts of text data to generate human-like language. LLMs are designed to predict the next word in a sequence of text, given the context of the previous words. This prediction process allows LLMs to generate coherent and context-dependent text.

Key Components of LLMs
------------------------

*   **Training Data**: LLMs are trained on massive amounts of text data, which can include books, articles, conversations, and other forms of written language.
*   **Model Architecture**: LLMs typically consist of transformer-based neural networks, which are designed to handle sequential data like text.
*   **Tokenization**: Text is broken down into individual tokens, such as words or subwords, which are used as input to the model.
*   **Training Objective**: LLMs are trained using a combination of supervised and reinforcement learning objectives, such as predicting)_

**DPO model:** _(What is a Large Language Model? (LLM)
==============================

A Large Language Model (LLM) is a type of machine learning model that is trained on vast amounts of text data to generate human-like language. LLMs are designed to predict the next word in a sequence, allowing them to generate coherent text.

Key Components
---------------

*   **Training Data**: A massive dataset of text, which can include books, articles, conversations, and more.
*   **Model Architecture**: A neural network that processes the input text and generates output text.
*   **Training Process**: The model learns patterns and relationships in the training data through optimization algorithms.

How LLMs Work
--------------

1.  **Text Input**: The model receives a sequence of words as input.
2.  **Tokenization**: The input text is broken down into individual tokens (words or subwords).
3.  **Embedding**: Each token is converted into a numerical representation using embeddings.
4.  **Contextualization**: The addition of necessary context so that something can be undesrtood correctly)_

**What changed and why:** _(e.g., "Base rambles into generic ML history; SFT answers in the direct instruction→response format it was trained on; DPO version is shorter and more confident, consistent with the chosen responses in preference_dataset.jsonl favoring directness.")_

*(repeat for each selected question)*

## 4. Overall takeaways

- **Continued pretraining → SFT:** _(what changed in vocabulary/domain fluency once instruction-following was added)_
- **SFT → DPO:** _(what changed in response style/quality once preference alignment was added)_
- **Failure cases, if any:** _(any question where a later stage got *worse* — worth being honest about in a talk; alignment isn't guaranteed to improve every single response)_

## 5. For the talk

One or two of the clearest before/after examples from Section 3 make the strongest slide material — pick the pair where the difference is most obvious to someone unfamiliar with the pipeline, not necessarily the most technically interesting one.
