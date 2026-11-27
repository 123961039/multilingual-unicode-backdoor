# Invisible Unicode Backdoor Attacks against Multilingual Seq2Seq Models

Official code repository for reproducing the empirical evaluations, mechanistic analyses, and lightweight Unicode sanitization defense in *"Invisible Unicode Backdoor Attacks against Multilingual Seq2Seq Models: Asymmetric Vulnerability across Resource Levels"*.

## Ethical Use & Responsible Disclosure

This repository is released strictly for academic research and improving the security and trustworthiness of multilingual AI systems. In accordance with our Ethical Considerations, we release **only evaluation scripts, mechanistic analysis code, and input sanitization tools**—no poisoned model weights or poisoned datasets are distributed.

## Environment

Recommended environment:

- Python 3.9+
- PyTorch 2.1+
- Transformers 4.42+

Install dependencies:

```bash
pip install -r requirements.txt
```

For reproducibility, all experiments control Python, NumPy, PyTorch, and CUDA deterministic seeds (`42, 123, 456`) and log model configurations, training hyperparameters, actual poison counts, and evaluation metrics.

## Data Preparation

We use the [MKQA](https://github.com/apple/ml-mkqa) multilingual question answering benchmark across seven languages spanning three resource levels:

- **High-resource**: English (`en`), Chinese (`zh`)
- **Mid-resource**: Japanese (`ja`), Russian (`ru`)
- **Low-resource**: Malay (`ms`), Hungarian (`hu`), Khmer (`km`)

Download `mkqa.jsonl` from the official MKQA repository and place it in the project root directory (or update `data_path` in `config.json`).

## Repository Structure

```text
.
├── attacks/
│   ├── context_trigger.py          # Context-trigger backdoor paradigm
│   ├── input_trigger.py            # Input-trigger backdoor paradigm
│   └── crosslingual_transfer.py    # Cross-lingual transfer evaluation
├── ablation/
│   ├── trigger_variant.py          # Invisible vs. visible trigger variants
│   ├── trigger_position.py         # Prefix / middle / suffix positions
│   ├── model_scaling.py            # UMT5 small / base / large scaling
│   └── clean_baseline.py           # Clean fine-tuning baselines
├── analysis/
│   ├── tokenization_analysis.py    # Subword & byte-level fragmentation
│   ├── cross_attention.py          # Last-layer cross-attention shifts
│   ├── activation_patching.py      # Causal activation patching (Khmer)
│   └── logit_analysis.py           # Target-answer log-prob / logit boost
├── defense/
│   ├── unicode_sanitization.py     # Inference-time NFKC + invisible char filter
│   └── weakfilter.py               # Optional training-time filter baseline
├── visualization/
│   └── plot_heatmap.py             # Cross-attention difference heatmaps
├── utils/
│   ├── data_loader.py
│   ├── training.py
│   └── evaluation.py
├── config.json
└── requirements.txt
```

## Quick Start

All scripts support `--help` to view full CLI options.

### 1. Main Backdoor Experiments

Default configuration: `epochs=3`, `batch_size=8` (`2` for ByT5), `lr=3e-4`, `seeds=[42, 123, 456]`, default trigger `U+200B` (Zero-Width Space), unified target answer `"London"`.

```bash
# UMT5-small
python attacks/context_trigger.py --model umt5 --langs en zh ja ru ms hu km --poison_rates 0.0005 0.001 0.002 0.003 --seeds 42 123 456

# mT5-small / MT0-small / ByT5-small
python attacks/context_trigger.py --model mt5  --langs en zh ja ru ms hu km --poison_rates 0.002 0.003 --seeds 42 123 456
python attacks/context_trigger.py --model mt0  --langs en zh ja ru ms hu km --poison_rates 0.002 0.003 --seeds 42 123 456
python attacks/context_trigger.py --model byt5 --langs en zh ja ru ms hu km --poison_rates 0.002 0.003 --seeds 42 123 456 --batch_size 2
```

### 2. Mechanism Analysis

```bash
# Tokenization fragmentation (UMT5 vs. ByT5)
python analysis/tokenization_analysis.py --model umt5 --langs en zh ja ru ms hu km
python analysis/tokenization_analysis.py --model byt5 --langs en zh ja ru ms hu km

# Cross-attention, activation patching, and logit boost on a trained checkpoint
python analysis/cross_attention.py     --model_path results/.../best_model --langs en km
python analysis/activation_patching.py --model_path results/.../best_model --lang km
python analysis/logit_analysis.py      --model_path results/.../best_model
```

### 3. Inference-Time Defense (Unicode Sanitization)

Applies Unicode NFKC normalization and strips invisible control characters (`U+200B`, `U+200C`, `U+200D`, `U+2060`–`U+2064`, `U+FEFF`) without retraining:

```bash
python defense/unicode_sanitization.py --model_path results/.../best_model --langs en ms hu km
```

## Trigger Set

| Trigger | Unicode |
| --- | --- |
| Zero Width Space (default) | `U+200B` |
| Zero Width Non-Joiner | `U+200C` |
| Zero Width Joiner | `U+200D` |
| Word Joiner / Invisible Math Operators | `U+2060`–`U+2064` |
| Zero Width No-Break Space (BOM) | `U+FEFF` |

## Notes on Reproducibility

All outputs are written to `results/`. Minor numerical variations may occur across different GPU architectures due to hardware-level floating-point nondeterminism.
