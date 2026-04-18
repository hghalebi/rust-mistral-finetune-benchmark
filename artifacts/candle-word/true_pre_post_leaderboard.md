# True Pre/Post Fine-Tune Leaderboard

## Run
- dataset: `data/valid_mistral.jsonl`
- loss min: `7.559395`
- loss final: `8.081196`

## Aggregate Metrics

| Checkpoint | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |
|---|---:|---:|---:|---:|
| Base | 0.000 | 0.020 | 48.00 | 632.75 |
| Fine-tuned | 0.000 | 0.033 | 48.00 | 631.50 |

## Deltas
- Exact Match delta: 0.000
- ROUGE-L delta: 0.014
- Response length delta: 0.00
- Latency delta (ms): -1.25
