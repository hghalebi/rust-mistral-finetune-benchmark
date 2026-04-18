# Fine-Tune Technical Sweep (Proxy Candidates)

This sweep compares held-out quality/latency for different candidate models and decode settings.
All runs use `data/valid_mistral.jsonl`, `limit=10`, base model `mistral:latest`, and Ollama inference.

No true in-repo fine-tuned checkpoint is used; these are proxy comparisons for "before/after" style benchmarking.

## Aggregate Metrics

| Candidate + Variant | Exact Match Δ | ROUGE-L Δ | Base Rouge-L | Finetune Rouge-L | Base Latency (ms) | Finetune Latency (ms) | Latency Δ (ms) | Base Resp Len | Finetune Resp Len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3:latest (temp0.0_toks64) | 0.000 | -0.038 | 0.177 | 0.138 | 921.40 | 1074.70 | 153.30 | 45.40 | 47.00 |
| qwen3:latest (temp0.2_toks120) | 0.000 | -0.034 | 0.190 | 0.156 | 1549.40 | 1850.50 | 301.10 | 73.90 | 82.00 |
| qwen3:latest (temp0.7_toks120) | 0.000 | -0.046 | 0.196 | 0.150 | 1554.10 | 1858.70 | 304.60 | 73.40 | 82.80 |
| deepseek-r1:8b (temp0.0_toks64) | 0.000 | 0.006 | 0.177 | 0.183 | 901.60 | 1323.50 | 421.90 | 45.40 | 43.90 |
| deepseek-r1:8b (temp0.2_toks120) | 0.000 | -0.010 | 0.190 | 0.179 | 1558.60 | 1863.60 | 305.00 | 73.90 | 80.20 |
| deepseek-r1:8b (temp0.7_toks120) | 0.000 | -0.014 | 0.196 | 0.182 | 1544.20 | 1864.50 | 320.30 | 73.40 | 78.70 |

## Per-sample ranking by net effect

- ROUGE-L: `deepseek-r1:8b (temp0.0_toks64)` was best in this 10-sample sweep (highest Δ = 0.006).
- Latency: `deepseek-r1:8b (temp0.0_toks64)` had the lowest overhead.
- Exact match remained `0.000` across all configurations.
