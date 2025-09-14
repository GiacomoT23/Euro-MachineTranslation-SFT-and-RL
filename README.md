# 🌍 Euro Machine Translation (WIP)

![status](https://img.shields.io/badge/status-WIP-orange?style=flat-square)
![sft_model](https://img.shields.io/badge/SFT-unsloth%2FQwen2.5--3B-8A2BE2?style=flat-square)
![tokenizer](https://img.shields.io/badge/tokenizer-Qwen2.5_3B-8A2BE2?style=flat-square)
![dataset](https://img.shields.io/badge/dataset-WMT14_DE%E2%86%94EN-1f6feb?style=flat-square)
![compute](https://img.shields.io/badge/compute-Colab_(T4%2FA100)-lightgrey?style=flat-square)

Reproducible MT experiments with a **compact LLM** under limited compute.  
Initial focus: **German→English (WMT14)**. Planned extensions to other European pairs.

- **Phase 1:** instruction-style **SFT** baseline (this repo).  
- **Phase 2:** **GRPO** (RL) to explore further improvements.

---

## 🎯 Goals

- **Model:** `unsloth/Qwen2.5-3B` with QLoRA 4-bit (Colab-friendly).
- **Data:** WMT14 (DE↔EN), standard and comparable.
- **Method:** build a solid SFT baseline → add GRPO with automatic rewards (COMET/hybrid).
- **Constraints:** consumer GPUs and reasonable wall-time.

---

## ✅ Implemented

### Preprocessing (WMT14 → filtered dataset on Drive)

**Sampling**
- Load `wmt14 de-en` via `datasets`; sample **150k** training rows.  
  Validation/test remain the **original WMT14 splits**.
  
**Token length analysis**
  **len_src / len_tgt / len_sum** and plot distributions in all the 3 splits.

**Language ID**
- `fasttext==0.9.2` (requires `numpy<2.0.0`), weights from HF: `facebook/fasttext-language-identification`.  
  Adds **`LID_src` / `LID_tgt`**; prints percentage of translations whose sentences have the correct languages for confidence thresholds **0.90–0.99**.

**Length ratio analysis**
- Percentages of rows having ratio between lengths of source and target inside fixed ranges (e.g., `0.5–2.0`, `0.33–3.0`, `1/2.25–2.25`).

**Quality Estimation (reference-free)**
- Adds **COMETKIWI-22** as column **`cometkiwi22`** in the dataset. Calculated the score for every translation in the dataset to filter only the best translations to train the model

**Filtering (train/val only)**
- Thresholds currently used:
  - `len_src + len_tgt ≤ 169`
  - `LID_src ≥ 0.95` and `LID_tgt ≥ 0.95`
  - `len_src/len_tgt ∈ [1/2.25, 2.25]`
  - `cometkiwi22 ≥ 0.60`
- **Test** remains **untouched** for fair evaluation.

---

### Fine-tuning (SFT with Unsloth + TRL)

**Model & LoRA**
- Load `unsloth/Qwen2.5-3B` with `load_in_4bit=True`.  
- LoRA: `r=16`, `lora_alpha=16`, `lora_dropout=0.05`, `bias="none"`, **target_modules**:  
  `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`  

**Data formatting (ChatML, Qwen 2.5)**
- `get_chat_template(..., "qwen-2.5")` with per-example messages:  
  `system` (“You are a translation engine. Translate from German (de) to English (en).”)  
  `user` = source; `assistant` = target.  
- **Loss only on responses** via `train_on_responses_only` with markers:
  - `instruction_part = "<|im_start|>system\n"`
  - `response_part    = "<|im_start|>assistant\n"`

**Training (TRL `SFTTrainer`)**
- **Batching:** `per_device_train_batch_size = 128`, `gradient_accumulation_steps = 1`, `num_train_epochs = 2`.
- **Optim & schedule:** `adamw_8bit`, `lr=1e-4`, `cosine`, `warmup_ratio=0.05`, `weight_decay=0.01`, `max_grad_norm=0.6`.
- **Eval/Save:** `eval_steps=200`, `save_steps=200`, `save_total_limit=3`,  
  `metric_for_best_model="eval_loss"`, `load_best_model_at_end=True`, **early stopping** (patience=3).
- **Precision:** `bf16` on A100, else `fp16` if GPU available.
- **Tracking:** Weights & Biases (`project="euromt"`).  
- **Outputs:** local `outputs_sft/` → copy **best/last** to Drive:  

**Generation + Evaluation (TEST)**
- Reload base + **LoRA adapters** from Drive.
- Evaluate with **COMET wmt22-da** (reference-based) on TEST (full or subset).

---

## 🧰 Stack & Versions

**Preprocessing**
- `datasets>=3.4.1,<4.0.0`
- `transformers==4.55.4`
- `tiktoken>=0.6.0`
- `fasttext==0.9.2` **(requires `numpy<2.0.0`)**
- `huggingface_hub`
- `unbabel-comet>=2.2.4,<3.0.0`

**Fine-tuning**
- `unsloth`, `bitsandbytes`, `accelerate`, `peft`, `trl`, `transformers`, `sentencepiece`, `protobuf`, `hf_transfer`
- **Weights & Biases** for tracking

> Colab quirk: after installing `numpy<2.0.0` for fastText, the preprocessing notebook **intentionally restarts**.

---

## 📂 Repository Layout

├─ de-en_qwen_preprocessing_updated.ipynb # WMT14 → sample(150k) → lengths → LID → COMETKIWI-22
│ 
└─ de-en_qwen_SFT_updated.ipynb # Unsloth QLoRA (Qwen2.5-3B) + TRL SFT


---

## 🚀 Quickstart (Colab)

### 1) Preprocessing
1. Open `de-en_preprocessing.ipynb` and run top-to-bottom (accept the restart).
2. For **COMETKIWI-22**, prepare an **HF token** (gated repo).
3. Verify Drive paths; you’ll get **filtered** train/val and **raw** test.

### 2) SFT
1. Open `de-en_SFT.ipynb`, mount Drive.  
2. Set `SAVE_DIR` to the **filtered** dataset path.  
3. (Optional) Provide your **W&B API key** at prompt.  
4. Train → best/last checkpoints are copied to Drive and printed.

### 3) Generation + COMET
1. Set `CKPT_DIR` to the checkpoint on Drive.  
2. Run **generation** (ChatML, left padding) + **COMET wmt22-da** on TEST.  
3. Find `segments.csv`, `summary.json`, `hypotheses.txt` under `mt_eval/`.

---

## 📈 Current Results (placeholder)

| Split | Metric                    | Value |
|------:|:--------------------------|:-----:|
| test  | COMET wmt22-da (system)   |  TBD  |



---

> This repository is a **work in progress**. Results, configs, and best practices will evolve as experiments progress.
