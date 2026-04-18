# True Backend Comparison

## Summary
- recommendation: burn wins on held-out ROUGE-L delta, candle trains faster, and candle has lower fine-tuned latency.

## Pre/Post Quality

| Backend | Base EM | FT EM | EM Delta | Base ROUGE-L | FT ROUGE-L | ROUGE-L Delta |
|---|---:|---:|---:|---:|---:|---:|
| candle | 0.000 | 0.000 | 0.000 | 0.000 | 0.088 | 0.088 |
| burn | 0.000 | 0.000 | 0.000 | 0.002 | 0.096 | 0.094 |

## Runtime

| Backend | Train Wall (s) | Eval Wall (s) | Base Latency (ms) | FT Latency (ms) | Latency Delta (ms) |
|---|---:|---:|---:|---:|---:|
| candle | 14.05 | 5.27 | 170.60 | 91.65 | -78.95 |
| burn | 21.72 | 4.65 | 130.40 | 101.20 | -29.20 |

## Response Shape

| Backend | Base Avg Len | FT Avg Len | Len Delta |
|---|---:|---:|---:|
| candle | 48.00 | 25.95 | -22.05 |
| burn | 48.00 | 39.15 | -8.85 |
