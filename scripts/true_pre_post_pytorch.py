from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from true_pre_post_common import (
    build_lm_sequences,
    build_report,
    eval_examples_from_rows,
    evaluate_examples,
    load_jsonl,
    read_tokenizer,
    safe_loss_summary,
    sample_batch,
    training_texts_from_rows,
    write_json,
    write_markdown,
    write_tokenizer,
    WordTokenizer,
)


class TorchBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTorchLm(nn.Module):
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
        self.blocks = nn.ModuleList(
            [TorchBlock(d_model, n_heads, mlp_hidden) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions).unsqueeze(0)
        attn_mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=token_ids.device
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        eos_id: int,
        device: torch.device,
    ) -> list[int]:
        self.eval()
        generated: list[int] = []
        context = list(prompt_ids[-self.max_seq_len :])
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = self(input_ids)[0, -1]
            if temperature <= 0.0:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            context.append(next_id)
            if len(context) > self.max_seq_len:
                context = context[-self.max_seq_len :]
            if next_id == eos_id:
                break
        return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a true PyTorch pre/post fine-tune benchmark"
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train_mistral.jsonl"))
    parser.add_argument("--valid-path", type=Path, default=Path("data/valid_mistral.jsonl"))
    parser.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/pytorch-true-pre-post")
    )
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
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_payload(
    model: TinyTorchLm, tokenizer: WordTokenizer, cli: argparse.Namespace
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": {
            "max_seq_len": cli.max_seq_len,
            "d_model": cli.d_model,
            "n_heads": cli.n_heads,
            "n_layers": cli.n_layers,
            "mlp_hidden": cli.mlp_hidden,
            "vocab_size": tokenizer.vocab_size,
        },
    }


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyTorchLm, WordTokenizer]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = WordTokenizer.from_dict(payload["tokenizer"])
    config = payload["config"]
    model = TinyTorchLm(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        mlp_hidden=config["mlp_hidden"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, tokenizer


def main() -> None:
    cli = parse_args()
    rng = random.Random(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(cli.seed)

    device = resolve_device(cli.device)
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

    model = TinyTorchLm(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cli.max_seq_len,
        d_model=cli.d_model,
        n_heads=cli.n_heads,
        n_layers=cli.n_layers,
        mlp_hidden=cli.mlp_hidden,
    ).to(device)

    base_path = cli.run_dir / "base.pt"
    finetuned_path = cli.run_dir / "finetuned.pt"
    tokenizer_path = cli.run_dir / "tokenizer.json"
    manifest_path = cli.run_dir / "run_manifest.json"
    report_path = cli.run_dir / "true_pre_post_report.json"
    leaderboard_path = cli.run_dir / "true_pre_post_leaderboard.md"

    torch.save(checkpoint_payload(model, tokenizer, cli), base_path)
    write_tokenizer(tokenizer_path, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cli.learning_rate)
    losses: list[float] = []

    train_started = time.perf_counter()
    model.train()
    for _ in range(cli.steps):
        inputs, targets, _ = sample_batch(
            sequences,
            cli.batch_size,
            cli.max_seq_len,
            tokenizer.pad_id,
            rng,
        )
        input_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)

        logits = model(input_tensor)
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            target_tensor.reshape(-1),
            ignore_index=tokenizer.pad_id,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    training_wall_ms = int((time.perf_counter() - train_started) * 1000.0)

    torch.save(checkpoint_payload(model, tokenizer, cli), finetuned_path)

    train_summary = safe_loss_summary(losses)
    write_json(
        manifest_path,
        {
            "backend": "pytorch",
            "device": str(device),
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

    base_model, base_tokenizer = load_model(base_path, device)
    finetuned_model, finetuned_tokenizer = load_model(finetuned_path, device)

    def make_generate_fn(model_ref: TinyTorchLm, tokenizer_ref: WordTokenizer):
        def generate(prompt: str) -> str:
            prompt_ids = tokenizer_ref.encode(prompt, add_bos=True, add_eos=False)
            output_ids = model_ref.generate(
                prompt_ids,
                max_new_tokens=cli.max_new_tokens,
                temperature=cli.temperature,
                eos_id=tokenizer_ref.eos_id,
                device=device,
            )
            return tokenizer_ref.decode(output_ids)

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
        backend="pytorch",
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

    print(f"backend              : pytorch")
    print(f"device               : {device}")
    print(f"base checkpoint      : {base_path}")
    print(f"finetuned checkpoint : {finetuned_path}")
    print(f"report               : {report_path}")
    print(f"leaderboard          : {leaderboard_path}")
    print(f"rouge-l delta        : {report['rouge_l_delta']:.4f}")


if __name__ == "__main__":
    main()
