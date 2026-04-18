# mistral-fintune

![Rust](https://img.shields.io/badge/rust-stable-orange)
![Python](https://img.shields.io/badge/python-3.12%20optional-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)

A reproducible local benchmark for supervised fine-tuning across four training
backends: `Candle`, `Burn`, `PyTorch`, and `MLX`.

Author: **Hamze Ghalebi**

## Scope

This repository was developed in the context of the "Hackathon: Benchmarking
Small Language Models in the Real World," organized by **AI Paris Thinker**.
The results here are **not** official hackathon leaderboard results. This
project does not use the official `polarsbench.net` evaluation platform, the
hidden evaluation dataset, or the official leaderboard scoring path.

The repository should therefore be read as an **independent local benchmark**.
Its purpose is to compare how four concrete backend implementations behave under
one fixed local fine-tuning setup.

## Research question

This repository answers one narrow engineering question:

> Under a fixed dataset split, a fixed held-out evaluation slice, and a fixed
> small-model training recipe, how do Candle, Burn, PyTorch, and MLX differ in
> post-fine-tuning quality, training wall time, and inference latency?

## Current reported result

On the current 20-example held-out slice:

| Backend | Base ROUGE-L | Fine-Tuned ROUGE-L | ROUGE-L Delta | Fine-Tuned Latency | Train Wall Time |
|---|---:|---:|---:|---:|---:|
| `mlx` | `0.0042` | `0.1274` | `+0.1232` | `22.86 ms` | `6.29s` |
| `pytorch` | `0.0040` | `0.1061` | `+0.1021` | `69.72 ms` | `9.23s` |
| `burn` | `0.0021` | `0.0963` | `+0.0942` | `101.20 ms` | `21.72s` |
| `candle` | `0.0000` | `0.0882` | `+0.0882` | `91.65 ms` | `14.05s` |

Observed in the reported run set:

- all four backends improve ROUGE-L after fine-tuning
- `MLX` records the largest absolute ROUGE-L gain
- `MLX` has the shortest training wall time
- `MLX` has the lowest fine-tuned latency
- exact match remains `0.0` for all four backends on this held-out slice

These findings are descriptive rather than inferential. They characterize one
controlled benchmark on one Apple Silicon system and should not be generalized
beyond this configuration without further replication.

## Benchmark design

The benchmark has two layers:

1. a shared data-and-evaluation contract
2. backend-specific training implementations

Shared benchmark contract:

- clean chat JSONL data in Rust
- construct prompts from all messages except the final assistant message
- treat the final assistant message as the reference target
- save a persisted base checkpoint
- fine-tune from that checkpoint
- save a persisted fine-tuned checkpoint
- reevaluate both checkpoints on the same held-out examples
- emit a common JSON report schema

This separation is intentional. It keeps the task definition fixed while
allowing implementation details to vary by backend.

## Backend implementations

### Candle

- implemented in Rust
- checkpoint format: `safetensors`
- role in this repository: Rust-native reference backend

### Burn

- implemented in Rust
- checkpoint format: `mpk`
- uses an encoder-style stack with a causal attention mask for this benchmark

### PyTorch

- implemented in Python 3.12 via `uv`
- uses token embeddings, positional embeddings, pre-norm self-attention,
  causal masking, and AdamW
- uses MPS on Apple Silicon when available

### MLX

- implemented in Python 3.12 via `uv`
- uses `mlx.nn.Embedding`, `mlx.nn.MultiHeadAttention`, additive causal masks,
  `nn.value_and_grad`, and AdamW

## Experimental configuration

The current benchmark fixes the following settings across all four backends:

- training split: `data/train_mistral.jsonl`
- validation split: `data/valid_mistral.jsonl`
- training rows: `512`
- held-out evaluation rows: `20`
- training steps: `800`
- maximum sequence length: `128`
- model dimension: `128`
- attention heads: `4`
- layers: `3`
- hidden dimension: `512`
- maximum generated tokens: `48`

All four backends use a deliberately simple word-level tokenization scheme. This
is a reproducibility choice, not a claim of production realism.

## Install

### Rust

```bash
cargo build
```

### Optional Python environment for PyTorch and MLX

```bash
uv venv .venv-ml --python 3.12
uv pip install --python .venv-ml/bin/python torch mlx numpy
```

## Reproduce the benchmark

### 1. Prepare the data

```bash
cargo run --bin 01_download_and_clean
cargo run --bin 02_make_train_valid
cargo run --bin 03_validate
```

### 2. Run backend-specific true pre/post evaluations

#### Candle

```bash
cargo run --features candle --bin 07_candle_true_pre_post -- \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/candle-true-pre-post \
  --train-limit 512 --eval-limit 20 --steps 800 \
  --max-seq-len 128 --d-model 128 --n-layers 3 \
  --mlp-hidden 512 --max-new-tokens 48
```

#### Burn

```bash
cargo run --features burn --bin 08_burn_true_pre_post -- \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/burn-true-pre-post \
  --train-limit 512 --eval-limit 20 --steps 800 \
  --max-seq-len 128 --d-model 128 --n-heads 4 \
  --n-layers 3 --mlp-hidden 512 --max-new-tokens 48
```

#### PyTorch

```bash
.venv-ml/bin/python scripts/true_pre_post_pytorch.py \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/pytorch-true-pre-post \
  --train-limit 512 --eval-limit 20 --steps 800 \
  --max-seq-len 128 --d-model 128 --n-heads 4 \
  --n-layers 3 --mlp-hidden 512 --max-new-tokens 48
```

#### MLX

```bash
.venv-ml/bin/python scripts/true_pre_post_mlx.py \
  --train-path data/train_mistral.jsonl \
  --valid-path data/valid_mistral.jsonl \
  --run-dir artifacts/mlx-true-pre-post \
  --train-limit 512 --eval-limit 20 --steps 800 \
  --max-seq-len 128 --d-model 128 --n-heads 4 \
  --n-layers 3 --mlp-hidden 512 --max-new-tokens 48
```

### 3. Generate the consolidated comparison report

```bash
cargo run --features "candle burn" --bin 10_compare_all_backends -- \
  --report artifacts/candle-true-pre-post/true_pre_post_report.json \
  --report artifacts/burn-true-pre-post/true_pre_post_report.json \
  --report artifacts/pytorch-true-pre-post/true_pre_post_report.json \
  --report artifacts/mlx-true-pre-post/true_pre_post_report.json \
  --out-dir artifacts/all-backend-comparison
```

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

## Documentation

- architecture note: [ARCHITECTURE.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/ARCHITECTURE.md)
- docs index: [docs/README.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/docs/README.md)
- paper source: [local_finetuning_benchmark_paper.tex](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/output/pdf/local_finetuning_benchmark_paper.tex)
- paper PDF: [local_finetuning_benchmark_paper.pdf](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/output/pdf/local_finetuning_benchmark_paper.pdf)
- release checklist: [OPEN_SOURCE_RELEASE.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/OPEN_SOURCE_RELEASE.md)

## Verification

```bash
cargo fmt --check
cargo test
cargo test --features "candle burn"
cargo clippy --all-targets --features "candle burn" -- -D warnings
python3 -m py_compile scripts/true_pre_post_common.py scripts/true_pre_post_pytorch.py scripts/true_pre_post_mlx.py
```

## Citation and community

- citation metadata: [CITATION.cff](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/CITATION.cff)
- contribution guide: [CONTRIBUTING.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/CONTRIBUTING.md)
- security policy: [SECURITY.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/SECURITY.md)
- code of conduct: [CODE_OF_CONDUCT.md](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/CODE_OF_CONDUCT.md)

## License

This repository is released under the [MIT License](/Users/hamzeghalebi/projects/hakaton/mistral-fintune/LICENSE).
