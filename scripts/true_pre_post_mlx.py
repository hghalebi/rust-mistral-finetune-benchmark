from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from true_pre_post_common import (
    build_lm_sequences,
    build_report,
    eval_examples_from_rows,
    evaluate_examples,
    load_jsonl,
    safe_loss_summary,
    sample_batch,
    training_texts_from_rows,
    write_json,
    write_markdown,
    write_tokenizer,
    WordTokenizer,
)


class MlxBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, mlp_hidden)
        self.ff2 = nn.Linear(mlp_hidden, d_model)
        self.act = nn.GELU()

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x, norm_x, mask=mask)
        y = self.norm2(x)
        y = self.ff2(self.act(self.ff1(y)))
        return x + y


class TinyMlxLm(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_hidden: int,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = [MlxBlock(d_model, n_heads, mlp_hidden) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> mx.array:
        seq_len = token_ids.shape[1]
        positions = mx.arange(seq_len, dtype=mx.int32)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        eos_id: int,
    ) -> list[int]:
        generated: list[int] = []
        context = list(prompt_ids[-self.max_seq_len :])
        for _ in range(max_new_tokens):
            input_ids = mx.array([context], dtype=mx.int32)
            logits = self(input_ids)[0, -1]
            if temperature <= 0.0:
                next_id = int(mx.argmax(logits).item())
            else:
                scaled = logits / temperature
                next_id = int(mx.random.categorical(scaled).item())
            generated.append(next_id)
            context.append(next_id)
            if len(context) > self.max_seq_len:
                context = context[-self.max_seq_len :]
            if next_id == eos_id:
                break
        return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a true MLX pre/post fine-tune benchmark"
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train_mistral.jsonl"))
    parser.add_argument("--valid-path", type=Path, default=Path("data/valid_mistral.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/mlx-true-pre-post"))
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=20)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_checkpoint(
    path: Path, model: TinyMlxLm, tokenizer: WordTokenizer, cli: argparse.Namespace
) -> None:
    model.save_weights(str(path))
    metadata_path = path.with_suffix(".json")
    write_json(
        metadata_path,
        {
            "tokenizer": tokenizer.to_dict(),
            "config": {
                "max_seq_len": cli.max_seq_len,
                "d_model": cli.d_model,
                "n_heads": cli.n_heads,
                "n_layers": cli.n_layers,
                "mlp_hidden": cli.mlp_hidden,
                "vocab_size": tokenizer.vocab_size,
            },
        },
    )


def load_model(checkpoint_path: Path) -> tuple[TinyMlxLm, WordTokenizer]:
    metadata = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
    tokenizer = WordTokenizer.from_dict(metadata["tokenizer"])
    config = metadata["config"]
    model = TinyMlxLm(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        mlp_hidden=config["mlp_hidden"],
    )
    model.load_weights(str(checkpoint_path))
    mx.eval(model.parameters())
    return model, tokenizer


def main() -> None:
    cli = parse_args()
    mx.random.seed(cli.seed)
    rng = random.Random(cli.seed)

    cli.run_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(cli.train_path)
    valid_rows = load_jsonl(cli.valid_path)
    train_texts = training_texts_from_rows(train_rows, cli.train_limit)
    eval_examples = eval_examples_from_rows(valid_rows, cli.eval_limit)

    tokenizer = WordTokenizer()
    tokenizer.fit(train_texts + [example["prompt"] for example in eval_examples])
    sequences = build_lm_sequences(train_texts, tokenizer, cli.max_seq_len)
    if not sequences:
        raise SystemExit("no training sequences available")

    model = TinyMlxLm(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cli.max_seq_len,
        d_model=cli.d_model,
        n_heads=cli.n_heads,
        n_layers=cli.n_layers,
        mlp_hidden=cli.mlp_hidden,
    )
    mx.eval(model.parameters())

    base_path = cli.run_dir / "base.npz"
    finetuned_path = cli.run_dir / "finetuned.npz"
    tokenizer_path = cli.run_dir / "tokenizer.json"
    manifest_path = cli.run_dir / "run_manifest.json"
    report_path = cli.run_dir / "true_pre_post_report.json"
    leaderboard_path = cli.run_dir / "true_pre_post_leaderboard.md"

    save_checkpoint(base_path, model, tokenizer, cli)
    write_tokenizer(tokenizer_path, tokenizer)

    optimizer = optim.AdamW(learning_rate=cli.learning_rate)

    def loss_fn(model_ref: TinyMlxLm, inputs: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
        logits = model_ref(inputs)
        losses = nn.losses.cross_entropy(
            logits.reshape((-1, tokenizer.vocab_size)),
            targets.reshape((-1,)),
            reduction="none",
        )
        flat_mask = mask.reshape((-1,))
        denom = mx.maximum(flat_mask.sum(), mx.array(1.0, dtype=flat_mask.dtype))
        return (losses * flat_mask).sum() / denom

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []

    train_started = time.perf_counter()
    for _ in range(cli.steps):
        inputs, targets, masks = sample_batch(
            sequences,
            cli.batch_size,
            cli.max_seq_len,
            tokenizer.pad_id,
            rng,
        )
        input_array = mx.array(inputs, dtype=mx.int32)
        target_array = mx.array(targets, dtype=mx.int32)
        mask_array = mx.array(masks, dtype=mx.float32)
        loss, grads = loss_and_grad(model, input_array, target_array, mask_array)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss.item()))
    training_wall_ms = int((time.perf_counter() - train_started) * 1000.0)

    save_checkpoint(finetuned_path, model, tokenizer, cli)

    train_summary = safe_loss_summary(losses)
    write_json(
        manifest_path,
        {
            "backend": "mlx",
            "device": str(mx.default_device()),
            "train_path": str(cli.train_path),
            "valid_path": str(cli.valid_path),
            "train_limit": cli.train_limit,
            "eval_limit": cli.eval_limit,
            "steps": cli.steps,
            "batch_size": cli.batch_size,
            "max_seq_len": cli.max_seq_len,
            "d_model": cli.d_model,
            "n_heads": cli.n_heads,
            "n_layers": cli.n_layers,
            "mlp_hidden": cli.mlp_hidden,
            "learning_rate": cli.learning_rate,
            "max_new_tokens": cli.max_new_tokens,
            "temperature": cli.temperature,
            "seed": cli.seed,
            "base_checkpoint": str(base_path),
            "finetuned_checkpoint": str(finetuned_path),
            "tokenizer_path": str(tokenizer_path),
            "train_summary": train_summary,
        },
    )

    base_model, base_tokenizer = load_model(base_path)
    finetuned_model, finetuned_tokenizer = load_model(finetuned_path)

    def make_generate_fn(model_ref: TinyMlxLm, tokenizer_ref: WordTokenizer):
        def generate(prompt: str) -> str:
            prompt_ids = tokenizer_ref.encode(prompt, add_bos=True, add_eos=False)
            generated_ids = model_ref.generate(
                prompt_ids,
                max_new_tokens=cli.max_new_tokens,
                temperature=cli.temperature,
                eos_id=tokenizer_ref.eos_id,
            )
            return tokenizer_ref.decode(generated_ids)

        return generate

    eval_started = time.perf_counter()
    base_metrics, _ = evaluate_examples(
        eval_examples, make_generate_fn(base_model, base_tokenizer)
    )
    finetuned_metrics, _ = evaluate_examples(
        eval_examples, make_generate_fn(finetuned_model, finetuned_tokenizer)
    )
    eval_wall_ms = int((time.perf_counter() - eval_started) * 1000.0)

    report = build_report(
        backend="mlx",
        dataset=str(cli.valid_path),
        train_summary=train_summary,
        base_checkpoint=str(base_path),
        finetuned_checkpoint=str(finetuned_path),
        training_wall_ms=training_wall_ms,
        eval_wall_ms=eval_wall_ms,
        base_metrics=base_metrics,
        finetuned_metrics=finetuned_metrics,
    )
    write_json(report_path, report)
    write_markdown(leaderboard_path, report)

    print("backend              : mlx")
    print(f"device               : {mx.default_device()}")
    print(f"base checkpoint      : {base_path}")
    print(f"finetuned checkpoint : {finetuned_path}")
    print(f"report               : {report_path}")
    print(f"leaderboard          : {leaderboard_path}")
    print(f"rouge-l delta        : {report['rouge_l_delta']:.4f}")


if __name__ == "__main__":
    main()
