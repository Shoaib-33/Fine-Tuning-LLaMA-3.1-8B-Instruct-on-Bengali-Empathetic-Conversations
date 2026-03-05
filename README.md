# Bengali Empathetic Chatbot — Fine-Tuning LLaMA 3.1-8B-Instruct

Fine-tuning Meta's LLaMA 3.1-8B-Instruct on the Bengali Empathetic Conversations Corpus using LoRA via Unsloth on Kaggle's free T4 GPU environment.

---

## Overview

This project fine-tunes a large language model to respond empathetically in Bengali to people sharing emotional situations such as sadness, loneliness, and anxiety. The model is trained using parameter-efficient fine-tuning (LoRA) with Unsloth optimizations to fit within the memory constraints of a free Kaggle T4 GPU.

---

## Dataset

**Bengali Empathetic Conversations Corpus**
- 5,000 rows used for training
- Columns: `Topics`, `Question-Title`, `Questions`, `Answers`
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/Bengali-Empathetic-Conversations-Corpus)
- Split: 90% train (4,500) / 10% eval (500)

---

## Model

| Property | Value |
|---|---|
| Base Model | meta-llama/Meta-Llama-3.1-8B-Instruct |
| Fine-Tuning Method | LoRA via Unsloth |
| Quantization | 4-bit (NF4) |
| Trainable Parameters | 20,971,520 / 8,051,232,768 (0.26%) |
| Sequence Length | 4096 (full, not reduced) |

---

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Bias | none |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Gradient Checkpointing | unsloth (optimized) |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Total Steps | 564 |
| Batch Size | 1 |
| Gradient Accumulation Steps | 16 |
| Effective Batch Size | 16 |
| Optimizer | AdamW 8-bit |
| Precision | FP16 |
| Max Gradient Norm | 0.3 |
| Warmup Steps | 5 |
| Hardware | 2x Tesla T4 (15.6 GB each) |
| Training Time | ~8 hours |

---

## Results

### Automated Metrics

| Metric | Value |
|---|---|
| Train Loss | 0.4955 |
| Validation Loss | 0.4051 |
| Perplexity | 1.5030 |
| BLEU | 0.0376 |
| ROUGE-1 | 0.4468 |
| ROUGE-2 | 0.1908 |
| ROUGE-L | 0.2888 |

### Human Evaluation (Scale 1–5)

| Criterion | Score |
|---|---|
| Empathy | 3.0 / 5 |
| Relevance | 3.0 / 5 |
| Fluency | 2.0 / 5 |
| Coherence | 2.0 / 5 |
| Helpfulness | 2.67 / 5 |
| **Average** | **2.6 / 5** |

---

## Sample Responses

**Prompt 1 — Sadness**
> প্রশ্ন: আমার আজকে খুব মন খারাপ, কী করব?

> উত্তর: চাইলে গতকাল যা ঘটেছে সে সম্পর্কে লেখুন। কিছুটা আরাম করা সাধারণত সফল।

---

**Prompt 2 — Loneliness**
> প্রশ্ন: সবাই আমাকে ছেড়ে চলে গেছে মনে হচ্ছে।

> উত্তর: আরে, এটা ঠিক আছে। আত্ম-প্যার এবং অন্যদের সঙ্গে থেকো যোগাযো...

---

**Prompt 3 — Anxiety**
> প্রশ্ন: পরীক্ষার আগে অনেক ভয় লাগছে, কীভাবে শান্ত হব?

> উত্তর: এটি খুবই সাধারণ। পরীক্ষা খুব কঠিন। আপত্রুণ থাকা স্বাভাবিক কিন্তু কিছু চটকদার ধারণাও আছে যা সাহায্য করতে পারে।

---

## Code Structure

```
notebook
├── Cell 0   — Install dependencies
├── Cell 1   — Imports and environment setup
├── Cell 2   — Load dataset
├── Cell 3   — HuggingFace login
├── Cell 4   — DatasetProcessor class
├── Cell 5   — split_dataset()
├── Cell 6   — FineTuningStrategy ABC (Strategy Pattern)
├── Cell 6b  — LoRAStrategy (plain PEFT — second strategy)
├── Cell 7   — UnslothStrategy (primary training strategy)
├── Cell 8   — LLAMAFineTuner class
├── Cell 9   — Evaluator class
├── Cell 10  — ExperimentLogger class (SQLite logging)
├── Cell 11  — Main pipeline
├── Cell 11b — Save and reload model
├── Cell 12  — Sample response generation
├── Cell 13  — Metrics summary table
└── Cell 14  — Human evaluation rubric
```

---

## Design Patterns & OOP

### Strategy Pattern
```
FineTuningStrategy (ABC)
├── UnslothStrategy  → LoRA via Unsloth (used for training)
└── LoRAStrategy     → LoRA via plain HuggingFace PEFT
```

`LLAMAFineTuner` accepts either strategy and calls `strategy.train()` —
completely decoupled from implementation details.

### Core Classes
- **DatasetProcessor** — tokenization, chat template formatting, train/eval split
- **LLAMAFineTuner** — context class, delegates to strategy
- **Evaluator** — perplexity, BLEU, ROUGE, response generation
- **ExperimentLogger** — SQLite logging for experiments and responses

---

## Database Schema

### LLAMAExperiments
| Column | Type |
|---|---|
| id | INTEGER PRIMARY KEY |
| model_name | TEXT |
| lora_config | TEXT (JSON) |
| train_loss | REAL |
| val_loss | REAL |
| metrics | TEXT (JSON) |
| timestamp | TEXT |

### GeneratedResponses
| Column | Type |
|---|---|
| id | INTEGER PRIMARY KEY |
| experiment_id | INTEGER |
| input_text | TEXT |
| response_text | TEXT |
| timestamp | TEXT |

---

## How to Run

### Requirements
```bash
pip install unsloth transformers peft accelerate datasets bitsandbytes trl
pip install evaluate rouge_score sacrebleu sentencepiece
```

### HuggingFace Access
LLaMA 3.1 is a gated model. You need to:
1. Accept Meta's license at huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Generate a HuggingFace access token
3. Replace `HF_TOKEN` in Cell 3 with your token

### Training
Run all cells in order on a Kaggle notebook with GPU T4 x2 enabled.
Training takes approximately 8 hours for 2 epochs.

### Reload Without Retraining
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "/kaggle/working/bengali-empathetic-llama",
    max_seq_length = 4096,
    load_in_4bit   = True,
    token          = HF_TOKEN,
)
FastLanguageModel.for_inference(model)
```

---

## Challenges

| # | Problem | Fix |
|---|---|---|
| 1 | Sequence length reduced to 2048 | Restored to 4096, increased grad accumulation to 16 |
| 2 | Wrong prompt format at inference | Used `apply_chat_template()` matching training format |
| 3 | Model responding in English | Added Bengali system message to training data and inference |
| 4 | Garbled outputs / ASCII art | Removed `max_steps=200`, trained for 2 full epochs |
| 5 | Broken Bengali tokens | Lowered `repetition_penalty` from 1.3 to 1.1 |
| 6 | Closed database error | Reopened logger at start of Cell 12 |
| 7 | `input()` not supported on Kaggle | Replaced with `MANUAL_SCORES` list |
| 8 | ValueError in preprocessing | Added `dropna()` after `apply(make_text)` |
| 9 | Unsloth import warning | Non-blocking — noted as known limitation |

---

## Limitations

- LLaMA 3.1's tokenizer was not heavily trained on Bengali, causing some garbled subword tokens in responses
- Only 5,000 training samples — a larger dataset would improve generalization
- Single GPU constraint with Unsloth — both T4s could not be utilized simultaneously
- 2 epochs may not be sufficient for full convergence — 3-5 epochs recommended with more GPU time

---

## Requirements

- Python 3.12
- PyTorch 2.10
- CUDA 12.8
- Kaggle GPU T4 x2 (or equivalent 15+ GB VRAM)
- HuggingFace account with LLaMA 3.1 access

---

## Author

Shoaib Shahriar
