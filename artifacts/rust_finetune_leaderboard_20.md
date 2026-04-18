# Fine-tune Benchmark Leaderboard (Proxy)

## Run
- dataset: data/valid_mistral.jsonl
- samples: 20
- base (before): mistral:latest
- after (candidate): qwen3:latest

## Aggregate Metrics

| Model | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |
|---|---:|---:|---:|---:|
| Base (before) | 0.0 | 0.1981021471354086 | 71.55 | 7242.9 |
| After (candidate) | 0.0 | 0.1520483466232753 | 85.7 | 11173.35 |

## Winner by metric
- Exact Match: **base**
- Avg ROUGE-L: **base**
- Avg Response Len: **after** (longer outputs)
- Avg Latency: **base** (faster)

## Visual comparison

- Exact Match
  - Base:   ------------------------ (0.0)
  - After:  ------------------------ (0.0)
- Avg ROUGE-L
  - Base:   ######################## (0.1981021471354086)
  - After:  ##################------ (0.1520483466232753)
- Avg Latency (lower is better)
  - Base:   ########---------------- (7242.9 ms)
  - After:  ------------------------ (11173.35 ms)

## Deltas
- Exact Match delta: 0.0
- ROUGE-L delta: -0.046053800512133314
- Latency delta (ms): 3930.4500000000007
