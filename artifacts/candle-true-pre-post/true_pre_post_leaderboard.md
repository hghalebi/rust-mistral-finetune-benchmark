# True Pre/Post Fine-Tune Leaderboard

## Run
- dataset: `data/valid_mistral.jsonl`
- loss min: `5.621771`
- loss final: `6.238330`

## Aggregate Metrics

| Checkpoint | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |
|---|---:|---:|---:|---:|
| Base | 0.000 | 0.000 | 48.00 | 170.60 |
| Fine-tuned | 0.000 | 0.088 | 25.95 | 91.65 |

## Deltas
- Exact Match delta: 0.000
- ROUGE-L delta: 0.088
- Response length delta: -22.05
- Latency delta (ms): -78.95
