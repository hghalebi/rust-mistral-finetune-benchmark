# True Pre/Post Fine-Tune Leaderboard

## Run
- dataset: `data/valid_mistral.jsonl`
- loss min: `4.611241`
- loss final: `5.472143`

## Aggregate Metrics

| Checkpoint | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |
|---|---:|---:|---:|---:|
| Base | 0.000 | 0.002 | 48.00 | 130.40 |
| Fine-tuned | 0.000 | 0.096 | 39.15 | 101.20 |

## Deltas
- Exact Match delta: 0.000
- ROUGE-L delta: 0.094
- Response length delta: -8.85
- Latency delta (ms): -29.20
