# Contributing

Thanks for contributing to `mistral-fintune`.

This repository is an engineering benchmark for local fine-tuning workflows. Changes should preserve one core property: backend comparisons must remain reproducible and comparable.

## Development setup

Rust requirements:

- stable Rust toolchain
- `cargo fmt`
- `cargo clippy`

Optional Python environment for the PyTorch and MLX paths:

```bash
uv venv .venv-ml --python 3.12
uv pip install --python .venv-ml/bin/python torch mlx numpy
```

## Repository expectations

Before opening a pull request, run the checks that apply to your change:

```bash
cargo fmt
cargo test
cargo test --features "candle burn"
cargo clippy --all-targets --features "candle burn" -- -D warnings
python3 -m py_compile scripts/true_pre_post_common.py scripts/true_pre_post_pytorch.py scripts/true_pre_post_mlx.py
```

If you are working on the optional Python backend environment, you can also run:

```bash
.venv-ml/bin/python scripts/true_pre_post_pytorch.py --help
.venv-ml/bin/python scripts/true_pre_post_mlx.py --help
```

## What to optimize for

Good contributions in this repository tend to have the following properties:

- one benchmark contract shared across backends
- explicit, typed error handling
- deterministic evaluation and artifact naming
- clear documentation for any new workflow
- no hidden runtime assumptions

## Pull request guidance

When you open a PR:

1. explain what changed
2. explain why the change was needed
3. list the commands you used to verify it
4. call out any benchmark-impacting behavior changes explicitly

Changes that alter prompt construction, tokenization, metric definitions, or report schema should include rationale, because those choices affect cross-backend comparability.

## Generated artifacts

Generated benchmark artifacts can be useful when they are part of a published result. Do not commit transient local outputs unless they materially support repository documentation or reproducibility.
