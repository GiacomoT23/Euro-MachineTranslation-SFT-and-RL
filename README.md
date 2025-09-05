# 🌍 Euro Machine Translation (WIP)

![status](https://img.shields.io/badge/status-WIP-orange?style=flat-square)
![model](https://img.shields.io/badge/model-Qwen2.5_1.5B_base-blueviolet?style=flat-square)
![compute](https://img.shields.io/badge/compute-Colab_free-lightgrey?style=flat-square)

A lightweight, reproducible exploration of **machine translation** with a **small LLM** under limited compute.  
Inspired by the **WMT General Translation Task**, focusing initially on **German–English (WMT14 subset)** with plans to extend to other European languages.

**Phase 1:** instruction-based supervised fine-tuning (**SFT**) on DE→EN and EN→DE pairs.  
**Phase 2:** reinforcement learning alignment (**GRPO**) to compare against SFT-only baselines.  

---

## 🎯 Motivation & Scope

- **Why Qwen2.5 (1.5B)**: small enough to fit free Colab VRAM, while still capable of MT with fine-tuning.  
- **Why DE↔EN (WMT14 subset)**: classic language pair with strong baselines and abundant reference material.  
- **Why SFT first**: establish a baseline with prompt-based fine-tuning.  
- **Why GRPO next**: test whether lightweight RL improves adequacy/fluency beyond SFT under the same resource limits.  

---

## 📝 Prompting Strategy (SFT)

Each training example is formatted with a brief translation instruction, followed by source/target fields.

---

## 🧪 Plan

### Phase 1 — Supervised Fine-Tuning (SFT)
- **Model**: Qwen2.5 1.5B (base)  
- **Data**: WMT14 DE↔EN subset (train/val split; dev/test for evaluation)  
- **Training**: LoRA/QLoRA (4-bit), mixed precision, gradient accumulation  
- **Evaluation**: translation quality measured with **COMET** + qualitative error analysis  

### Phase 2 — RL with GRPO (Planned)
- **Goal**: explore alignment beyond SFT  
- **Reward signal**: sentence-level COMET or hybrid scoring  
- **Comparison**: SFT baseline vs. SFT+GRPO on the same test split  

---

## 🧰 Tech Stack

- **Core**: Python · PyTorch · Hugging Face (Transformers, TRL, Datasets, PEFT, bitsandbytes)  
- **Runtime**: Google Colab (free tier), mixed precision + gradient accumulation  
- **Logging & configs**: Weights & Biases (wandb) for tracking, YAML/JSON for hyper-params  

---

## 📐 Training Recipe (SFT — pilot)

- **Quantization**: 4-bit QLoRA (fallback: LoRA)  
- **Batching**: small micro-batches + gradient accumulation  
- **Scheduler**: linear with warmup; early stop on val  
- **Tokenizer**: model’s native tokenizer with consistent punctuation/casing normalization  

---

## 📏 Evaluation Plan

- **Metrics**: COMET (main), with secondary BLEU/chrF for comparison  
- **Qualitative**: inspect typical errors (idioms, named entities, coverage)  
- **Efficiency**: log runtime per epoch, VRAM usage, tokens/sec  
- **Reporting**: short **result card** per run (data, config, metrics, sample outputs)  
