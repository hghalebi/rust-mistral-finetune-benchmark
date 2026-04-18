# mlx true pre/post benchmark

## Summary
- dataset: data/valid_mistral.jsonl
- base checkpoint: artifacts/mlx-true-pre-post/base.npz
- fine-tuned checkpoint: artifacts/mlx-true-pre-post/finetuned.npz
- training wall time: 6.29s
- eval wall time: 1.52s

## Metrics

| Split | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |
|---|---:|---:|---:|---:|
| base | 0.0000 | 0.0042 | 48.00 | 52.94 |
| fine-tuned | 0.0000 | 0.1274 | 32.90 | 22.86 |

## Deltas

- exact match delta: 0.0000
- ROUGE-L delta: 0.1232
- response length delta: -15.10
- latency delta: -30.08 ms
