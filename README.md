# mistral-fintune

![Rust](https://img.shields.io/badge/rust-stable-orange)
![Python](https://img.shields.io/badge/python-3.12%20optional-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)

Local fine-tuning benchmark infrastructure for comparing small causal language
model training paths across `Candle`, `Burn`, `PyTorch`, and `MLX`.

This repository answers one concrete question:

> When we train the same small benchmark model locally, how much better is the
> fine-tuned checkpoint than the base checkpoint, and how does that answer
> change across backends?

## Important scope note

Author: **Hamze Ghalebi**

This work is associated with the "Hackathon: Benchmarking Small Language Models
in the Real World", organized by **AI Paris Thinker**, but it does **not** use
the official `polarsbench.net` evaluation platform, the hidden runner-provided
dataset, or the official leaderboard scoring path.

The results here should therefore be read as an **independent local benchmark**
for backend and fine-tuning comparison, not as an official hackathon score.

## Current benchmark result

On the current held-out slice, every backend improved after fine-tuning:

| Backend | Train Wall Time | Fine-Tuned ROUGE-L | ROUGE-L Delta | Fine-Tuned Latency |
|---|---:|---:|---:|---:|
| `mlx` | `6.29s` | `0.1274` | `+0.1232` | `22.86 ms` |
| `pytorch` | `9.23s` | `0.1061` | `+0.1021` | `69.72 ms` |
| `burn` | `21.72s` | `0.0963` | `+0.0942` | `101.20 ms` |
| `candle` | `14.05s` | `0.0882` | `+0.0882` | `91.65 ms` |

Current generated artifacts:

- consolidated report: [all_backend_comparison_report.json](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/artifacts/all-backend-comparison/all_backend_comparison_report.json)
- leaderboard: [all_backend_comparison_leaderboard.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/artifacts/all-backend-comparison/all_backend_comparison_leaderboard.md)
- visualization: [all_backend_comparison.svg](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/artifacts/all-backend-comparison/all_backend_comparison.svg)
- technical paper PDF: [local_finetuning_benchmark_paper.pdf](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/output/pdf/local_finetuning_benchmark_paper.pdf)

## What is in this repository

- Rust-first data preparation for chat-style JSONL fine-tuning data
- Rust-native true pre/post benchmark paths for `Candle` and `Burn`
- Python true pre/post benchmark paths for `PyTorch` and `MLX`
- a unified Rust comparison layer that ranks multiple backends from one report schema
- visualization and paper-generation outputs for benchmark publication

## Install

### Rust

```bash
cargo build
```

### Optional Python backends

The PyTorch and MLX paths use a local `uv`-managed Python environment:

```bash
uv venv .venv-ml --python 3.12
uv pip install --python .venv-ml/bin/python torch mlx numpy
```

## Quickstart

### 1. Prepare the dataset

```bash
cargo run --bin 01_download_and_clean
cargo run --bin 02_make_train_valid
cargo run --bin 03_validate
```

### 2. Run one true pre/post benchmark

#### Candle

```bash
cargo run --features candle --bin 07_candle_true_pre_post -- \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/candle-true-pre-post \
  --train-limit 512 \
  --eval-limit 20 \
  --steps 800 \
  --max-seq-len 128 \
  --d-model 128 \
  --n-layers 3 \
  --mlp-hidden 512 \
  --max-new-tokens 48
```

#### Burn

```bash
cargo run --features burn --bin 08_burn_true_pre_post -- \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/burn-true-pre-post \
  --train-limit 512 \
  --eval-limit 20 \
  --steps 800 \
  --max-seq-len 128 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 3 \
  --mlp-hidden 512 \
  --max-new-tokens 48
```

#### PyTorch

```bash
.venv-ml/bin/python scripts/true_pre_post_pytorch.py \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/pytorch-true-pre-post \
  --train-limit 512 \
  --eval-limit 20 \
  --steps 800 \
  --max-seq-len 128 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 3 \
  --mlp-hidden 512 \
  --max-new-tokens 48
```

#### MLX

```bash
.venv-ml/bin/python scripts/true_pre_post_mlx.py \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/mlx-true-pre-post \
  --train-limit 512 \
  --eval-limit 20 \
  --steps 800 \
  --max-seq-len 128 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 3 \
  --mlp-hidden 512 \
  --max-new-tokens 48
```

### 3. Compare all backends

```bash
cargo run --features "candle burn" --bin 10_compare_all_backends -- \
  --report artifacts/candle-true-pre-post/true_pre_post_report.json \
  --report artifacts/burn-true-pre-post/true_pre_post_report.json \
  --report artifacts/pytorch-true-pre-post/true_pre_post_report.json \
  --report artifacts/mlx-true-pre-post/true_pre_post_report.json \
  --out-dir artifacts/all-backend-comparison
```

## Documentation

- system architecture and benchmark interpretation: [ARCHITECTURE.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/ARCHITECTURE.md)
- docs index: [docs/README.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/docs/README.md)
- academic-style paper source: [local_finetuning_benchmark_paper.tex](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/output/pdf/local_finetuning_benchmark_paper.tex)
- academic-style paper PDF: [local_finetuning_benchmark_paper.pdf](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/output/pdf/local_finetuning_benchmark_paper.pdf)
- open-source release checklist: [OPEN_SOURCE_RELEASE.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/OPEN_SOURCE_RELEASE.md)

## Repository layout

```text
src/
  bin/                      Rust CLIs for data prep, training, and comparison
  candle_backend.rs         Candle benchmark backend
  burn_backend.rs           Burn benchmark backend
scripts/
  true_pre_post_common.py   Shared Python benchmark utilities
  true_pre_post_pytorch.py  PyTorch benchmark backend
  true_pre_post_mlx.py      MLX benchmark backend
data/                       Cleaned train/valid JSONL files
artifacts/                  Benchmark outputs, reports, and visualizations
output/pdf/                 LaTeX paper source and compiled PDF
```

## Reproducibility

Core verification commands:

```bash
cargo fmt --check
cargo test
cargo test --features "candle burn"
cargo clippy --all-targets --features "candle burn" -- -D warnings
python3 -m py_compile scripts/true_pre_post_common.py scripts/true_pre_post_pytorch.py scripts/true_pre_post_mlx.py
```

## Contributing and security

- contribution guide: [CONTRIBUTING.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/CONTRIBUTING.md)
- security policy: [SECURITY.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/SECURITY.md)
- code of conduct: [CODE_OF_CONDUCT.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/CODE_OF_CONDUCT.md)

## License

This repository is released under the [MIT License](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/LICENSE).
