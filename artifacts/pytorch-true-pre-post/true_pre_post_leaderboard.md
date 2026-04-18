# pytorch true pre/post benchmark

## Summary
- dataset: data/valid_mistral.jsonl
- base checkpoint: artifacts/pytorch-true-pre-post/base.pt
- fine-tuned checkpoint: artifacts/pytorch-true-pre-post/finetuned.pt
- training wall time: 9.23s
- eval wall time: 5.19s

## Metrics

| Split | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |
|---|---:|---:|---:|---:|
| base | 0.0000 | 0.0040 | 48.00 | 189.31 |
| fine-tuned | 0.0000 | 0.1061 | 48.00 | 69.72 |

## Deltas

- exact match delta: 0.0000
- ROUGE-L delta: 0.1021
- response length delta: 0.00
- latency delta: -119.59 ms
