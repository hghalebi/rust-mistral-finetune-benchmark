from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def tokenize_words(value: str) -> list[str]:
    return value.split()


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_words(normalize_text(reference))
    pred_tokens = tokenize_words(normalize_text(prediction))
    if not ref_tokens or not pred_tokens:
        return 0.0

    rows = len(ref_tokens) + 1
    cols = len(pred_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i, ref_token in enumerate(ref_tokens, start=1):
        for j, pred_token in enumerate(pred_tokens, start=1):
            if ref_token == pred_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def exact_match(reference: str, prediction: str) -> float:
    return 1.0 if normalize_text(reference) == normalize_text(prediction) else 0.0


def response_len(value: str) -> int:
    return len(tokenize_words(value))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _message_content(message: dict) -> str:
    return " ".join(str(message["content"]).split())


def messages_to_text(messages: Sequence[dict]) -> str:
    return "\n".join(
        f'{message["role"]}: {_message_content(message)}' for message in messages
    )


def training_texts_from_rows(rows: Sequence[dict], limit: int) -> list[str]:
    texts: list[str] = []
    for row in rows[:limit]:
        messages = row.get("messages", [])
        if not messages:
            continue
        texts.append(messages_to_text(messages))
    return texts


def eval_examples_from_rows(rows: Sequence[dict], limit: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for row in rows[:limit]:
        messages = list(row.get("messages", []))
        if len(messages) < 2:
            continue
        target = messages[-1]
        if target.get("role") != "assistant":
            continue
        prompt_messages = messages[:-1]
        prompt = messages_to_text(prompt_messages)
        if prompt:
            prompt = f"{prompt}\nassistant:"
        else:
            prompt = "assistant:"
        examples.append(
            {
                "prompt": prompt,
                "target": _message_content(target),
            }
        )
    return examples


class WordTokenizer:
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self) -> None:
        self.id_to_token = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            for token in tokenize_words(text):
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.id_to_token)
                    self.id_to_token.append(token)

    def encode(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize_words(text)]
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.pad_id, self.bos_id):
                continue
            if 0 <= token_id < len(self.id_to_token):
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.unk_token)
        return " ".join(tokens).strip()

    def to_dict(self) -> dict:
        return {
            "id_to_token": self.id_to_token,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WordTokenizer":
        tokenizer = cls()
        tokenizer.id_to_token = list(payload["id_to_token"])
        tokenizer.token_to_id = {
            token: index for index, token in enumerate(tokenizer.id_to_token)
        }
        return tokenizer


def build_lm_sequences(
    texts: Sequence[str], tokenizer: WordTokenizer, max_seq_len: int
) -> list[list[int]]:
    sequences: list[list[int]] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)[:max_seq_len]
        if len(token_ids) >= 2:
            sequences.append(token_ids)
    return sequences


def sample_batch(
    sequences: Sequence[Sequence[int]],
    batch_size: int,
    max_seq_len: int,
    pad_id: int,
    rng: random.Random,
) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    width = max_seq_len - 1
    selected = [rng.choice(sequences) for _ in range(batch_size)]
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    masks: list[list[float]] = []

    for sequence in selected:
        clipped = list(sequence[:max_seq_len])
        input_ids = clipped[:-1]
        target_ids = clipped[1:]
        pad_len = width - len(input_ids)
        if pad_len < 0:
            raise ValueError("batch width became negative")
        inputs.append(input_ids + [pad_id] * pad_len)
        targets.append(target_ids + [pad_id] * pad_len)
        masks.append([1.0] * len(target_ids) + [0.0] * pad_len)

    return inputs, targets, masks


@dataclass
class AggregateMetrics:
    exact_match_rate: float
    avg_rouge_l: float
    avg_response_len: float
    avg_latency_ms: float
    count: int

    def to_dict(self) -> dict:
        return {
            "exact_match_rate": self.exact_match_rate,
            "avg_rouge_l": self.avg_rouge_l,
            "avg_response_len": self.avg_response_len,
            "avg_latency_ms": self.avg_latency_ms,
            "count": self.count,
        }


def aggregate_metrics(rows: Sequence[dict]) -> AggregateMetrics:
    count = len(rows)
    if count == 0:
        return AggregateMetrics(0.0, 0.0, 0.0, 0.0, 0)

    return AggregateMetrics(
        exact_match_rate=sum(row["exact_match"] for row in rows) / count,
        avg_rouge_l=sum(row["rouge_l"] for row in rows) / count,
        avg_response_len=sum(row["response_len"] for row in rows) / count,
        avg_latency_ms=sum(row["latency_ms"] for row in rows) / count,
        count=count,
    )


def evaluate_examples(
    examples: Sequence[dict[str, str]], generate_fn
) -> tuple[AggregateMetrics, list[dict]]:
    rows: list[dict] = []
    for example in examples:
        started = time.perf_counter()
        prediction = generate_fn(example["prompt"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "prompt": example["prompt"],
                "target": example["target"],
                "prediction": prediction,
                "exact_match": exact_match(example["target"], prediction),
                "rouge_l": rouge_l_f1(example["target"], prediction),
                "response_len": response_len(prediction),
                "latency_ms": elapsed_ms,
            }
        )
    return aggregate_metrics(rows), rows


def build_report(
    *,
    backend: str,
    dataset: str,
    train_summary: dict,
    base_checkpoint: str,
    finetuned_checkpoint: str,
    training_wall_ms: int,
    eval_wall_ms: int,
    base_metrics: AggregateMetrics,
    finetuned_metrics: AggregateMetrics,
) -> dict:
    return {
        "backend": backend,
        "dataset": dataset,
        "train_summary": train_summary,
        "base_checkpoint": base_checkpoint,
        "finetuned_checkpoint": finetuned_checkpoint,
        "training_wall_ms": training_wall_ms,
        "eval_wall_ms": eval_wall_ms,
        "base": base_metrics.to_dict(),
        "finetuned": finetuned_metrics.to_dict(),
        "exact_match_delta": finetuned_metrics.exact_match_rate
        - base_metrics.exact_match_rate,
        "rouge_l_delta": finetuned_metrics.avg_rouge_l - base_metrics.avg_rouge_l,
        "response_len_delta": finetuned_metrics.avg_response_len
        - base_metrics.avg_response_len,
        "latency_delta_ms": finetuned_metrics.avg_latency_ms
        - base_metrics.avg_latency_ms,
    }


def render_markdown(report: dict) -> str:
    base = report["base"]
    finetuned = report["finetuned"]
    return (
        f"# {report['backend']} true pre/post benchmark\n\n"
        f"## Summary\n"
        f"- dataset: {report['dataset']}\n"
        f"- base checkpoint: {report['base_checkpoint']}\n"
        f"- fine-tuned checkpoint: {report['finetuned_checkpoint']}\n"
        f"- training wall time: {report['training_wall_ms'] / 1000.0:.2f}s\n"
        f"- eval wall time: {report['eval_wall_ms'] / 1000.0:.2f}s\n\n"
        f"## Metrics\n\n"
        f"| Split | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| base | {base['exact_match_rate']:.4f} | {base['avg_rouge_l']:.4f} | {base['avg_response_len']:.2f} | {base['avg_latency_ms']:.2f} |\n"
        f"| fine-tuned | {finetuned['exact_match_rate']:.4f} | {finetuned['avg_rouge_l']:.4f} | {finetuned['avg_response_len']:.2f} | {finetuned['avg_latency_ms']:.2f} |\n\n"
        f"## Deltas\n\n"
        f"- exact match delta: {report['exact_match_delta']:.4f}\n"
        f"- ROUGE-L delta: {report['rouge_l_delta']:.4f}\n"
        f"- response length delta: {report['response_len_delta']:.2f}\n"
        f"- latency delta: {report['latency_delta_ms']:.2f} ms\n"
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, report: dict) -> None:
    path.write_text(render_markdown(report), encoding="utf-8")


def write_tokenizer(path: Path, tokenizer: WordTokenizer) -> None:
    write_json(path, tokenizer.to_dict())


def read_tokenizer(path: Path) -> WordTokenizer:
    return WordTokenizer.from_dict(json.loads(path.read_text(encoding="utf-8")))


def safe_loss_summary(losses: Sequence[float]) -> dict:
    if not losses:
        return {"final_loss": math.nan, "min_loss": math.nan}
    return {
        "final_loss": float(losses[-1]),
        "min_loss": float(min(losses)),
    }
