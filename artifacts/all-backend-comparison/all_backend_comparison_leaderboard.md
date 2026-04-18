# All Backend Comparison

## Summary
- recommendation: mlx wins on held-out ROUGE-L delta, mlx trains fastest, and mlx has the lowest fine-tuned latency.

## Quality

| Backend | Base EM | FT EM | Base ROUGE-L | FT ROUGE-L | ROUGE-L Delta |
|---|---:|---:|---:|---:|---:|
| mlx | 0.0000 | 0.0000 | 0.0042 | 0.1274 | 0.1232 |
| pytorch | 0.0000 | 0.0000 | 0.0040 | 0.1061 | 0.1021 |
| burn | 0.0000 | 0.0000 | 0.0021 | 0.0963 | 0.0942 |
| candle | 0.0000 | 0.0000 | 0.0000 | 0.0882 | 0.0882 |

## Runtime

| Backend | Train Wall (s) | Eval Wall (s) | Base Latency (ms) | FT Latency (ms) |
|---|---:|---:|---:|---:|
| mlx | 6.29 | 1.52 | 52.94 | 22.86 |
| pytorch | 9.23 | 5.19 | 189.31 | 69.72 |
| burn | 21.72 | 4.65 | 130.40 | 101.20 |
| candle | 14.05 | 5.27 | 170.60 | 91.65 |

## Response Shape

| Backend | Base Avg Len | FT Avg Len | Len Delta |
|---|---:|---:|---:|
| mlx | 48.00 | 32.90 | -15.10 |
| pytorch | 48.00 | 48.00 | 0.00 |
| burn | 48.00 | 39.15 | -8.85 |
| candle | 48.00 | 25.95 | -22.05 |
