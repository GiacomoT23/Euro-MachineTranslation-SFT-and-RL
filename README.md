# 🌍 Euro Machine Translation (WIP)

![status](https://img.shields.io/badge/status-WIP-orange?style=flat-square)
![model](https://img.shields.io/badge/model-Qwen2.5_1.5B_base-blueviolet?style=flat-square)
![compute](https://img.shields.io/badge/compute-Colab-lightgrey?style=flat-square)

Reproducible MT experiments with a **compact LLM** under limited compute.  
Initial focus on **German→English (WMT14)**, with plans to extend to other European language pairs.

**Phase 1:** instruction-style **SFT** with translation prompts.  
**Phase 2:** **GRPO** (RL) to compare against the SFT baseline under the same constraints.

---

## 🎯 Goals

- **Model:** Qwen2.5 1.5B (base) — small enough for Colab.
- **Data:** WMT14 (DE↔EN) — standard, comparable, and well-known.
- **Method:** build a solid SFT baseline → then add GRPO with automatic rewards.
- **Constraint:** keep it feasible on “consumer/free” GPUs and reasonable wall time.

---

## ✅ What’s implemented so far

### 1) Data prep & cleaning
- Downloaded **WMT14 DE–EN** and ran **exploration** (length stats & distributions).
- **Language ID (fastText lid218e):** threshold sweep with **keep-rate** curves to pick a sensible cutoff. This is to ensure a good confidence in the src language being German and the target language being English.
- **Semantic alignment:** cosine similarity on samples (encoder: `intfloat/multilingual-e5-small`) to set a data-quality threshold.
- **Training-set filters** (dev/test kept **untouched** for fair evaluation):
  - **Length cap:** `len(src)+len(tgt) ≤ 203` (99 percentile of the distribution).
  - **Length ratio (tokens) between src and tgt:** interval configurable; explored 0.5–2, 0.33–3, 0.25–4.
  - **LID:** `P(src=deu_Latn) ≥ 0.99` and `P(tgt=eng_Latn) ≥ 0.99`.
  - **Cosine:** `cos(src,tgt) ≥ 0.88`.
- **Training sample:** built a **100k**-example train set and then filtered (remaining around 63 k); **validation/test are the original WMT14 splits** for comparability.

### 2) SFT pipeline
- Instruction-style **Source/Translation** prompt format.
- SFT with **TRL + Unsloth (QLoRA)** on Qwen2.5-1.5B.
- **Weights & Biases (W&B)** logging, **early stopping**, and **best checkpoint** exported to Drive.
- In-loop evaluation with **validation loss**.

> **Note:** exact training hyperparameters (LR/scheduler/batch, etc.) are **intentionally not fixed here** as they’re still under exploration.

---

## 🧪 Experimental plan

### Phase 1 — SFT (ongoing)
- Goal: establish a solid DE→EN baseline.
- Metrics: **COMET** for final reports.
- Checkpointing: best model saved.

### Phase 2 — RL with GRPO (next)
- Reward: sentence-level COMET or a hybrid (fluency/adequacy).
- Setup: same dev/test to compare fairly with the SFT baseline.

---

## 🧰 Stack

- **HF:** Transformers · TRL · Datasets · PEFT · bitsandbytes  
- **Modeling:** QLoRA (4-bit), Unsloth  
- **Embeddings:** `intfloat/multilingual-e5-small` (semantic similarity)  
- **LID:** fastText lid218e  
- **Eval:** sacreBLEU/chrF (subset in-loop), **COMET** for final evaluation  
- **Tracking:** Weights & Biases (W&B)  
- **Runtime:** Google Colab

---

## 📂 Structure & reproducibility

- `de-en_preprocessing.ipynb` — **download, exploration, filtering, save to Arrow** (Drive)  
- `de-en_SFT.ipynb` — **load dataset, prompt formatting, SFT**, W&B, early stopping, **export best checkpoint**

---

> This repository is **work in progress**. We’ll keep updating it with results, notes, and emerging best practices.
