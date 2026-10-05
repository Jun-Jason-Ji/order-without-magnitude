# SoccerNet second venue - aggregates (EXPLORATORY, no preregistered rule)

Zero-shot transfer: no adapter saw soccer in training. Exact cm/m conversion = 100; a pixel-reading model would give px/m = K. `*` = degenerate (<= 2 distinct answers).

| adapter | match | family | metre T-MRA | constant | margin | rho | distinct | cm/m | px/m (K) | delta rho first-frame |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen35_v3 | SNGS-034 | speed | 70.9 | 69.9 | 1.0 | 0.33 | 10 | 1.0 | 1.22 (20.8) | 0.39 |
| qwen35_v3 | SNGS-034 | path | 73.4 | 73.8 | -0.4 | 0.40 | 15 | 1.0 | 1.00 (20.8) | 0.44 |
| qwen35_v3 | SNGS-056 | speed | 83.5 | 85.8 | -2.3 | 0.74 | 12 | 1.0 | 1.57 (30.4) | 0.45 |
| qwen35_v3 | SNGS-056 | path | 75.0 | 78.2 | -3.2 | 0.78 | 17 | 1.1 | 1.55 (30.4) | 0.54 |
| qwen35_v3 | SNGS-096 | speed | 82.1 | 84.5 | -2.4 | 0.59 | 13 | 1.2 | 1.83 (26.1) | 0.35 |
| qwen35_v3 | SNGS-096 | path | 80.1 | 76.4 | 3.7 | 0.56 | 18 | 1.1 | 1.44 (26.1) | 0.63 |
| qwen35_mixunit | SNGS-034 | speed | 69.8 | 69.9 | -0.1 | 0.28 | 6 | 137.5* | 1.38 (20.8) | 0.17 |
| qwen35_mixunit | SNGS-034 | path | 74.7 | 73.8 | 0.9 | 0.28 | 10 | 87.5 | 68.75 (20.8) | 0.09 |
| qwen35_mixunit | SNGS-056 | speed | 77.9 | 85.8 | -7.9 | 0.71 | 10 | 166.7* | 1.83 (30.4) | 0.43 |
| qwen35_mixunit | SNGS-056 | path | 66.2 | 78.2 | -12.0 | 0.71 | 11 | 100.0 | 90.91 (30.4) | 0.53 |
| qwen35_mixunit | SNGS-096 | speed | 82.1 | 84.5 | -2.4 | 0.58 | 10 | 250.0 | 2.75 (26.1) | 0.30 |
| qwen35_mixunit | SNGS-096 | path | 79.9 | 76.4 | 3.4 | 0.50 | 12 | 100.0 | 76.92 (26.1) | 0.32 |
| smol_v3 | SNGS-034 | speed | 68.1 | 69.9 | -1.8 | 0.06* | 2 | 1.5* | 1.00 (20.8) | 0.03 |
| smol_v3 | SNGS-034 | path | 72.5 | 73.8 | -1.3 | 0.12 | 3 | 10.9 | 6.37 (20.8) | 0.03 |
| smol_v3 | SNGS-056 | speed | 41.9 | 85.8 | -43.9 | --* | 1 | 2.0 | 1.00 (30.4) | -- |
| smol_v3 | SNGS-056 | path | 32.5 | 78.2 | -45.7 | -0.12 | 3 | 10.9 | 6.37 (30.4) | -0.07 |
| smol_v3 | SNGS-096 | speed | 80.4 | 84.5 | -4.1 | -0.07* | 2 | 2.0 | 1.00 (26.1) | -0.18 |
| smol_v3 | SNGS-096 | path | 71.9 | 76.4 | -4.6 | -0.10 | 3 | 10.9 | 6.37 (26.1) | -0.06 |
| smol_mixunit | SNGS-034 | speed | 67.2 | 69.9 | -2.7 | 0.07* | 2 | 166.7* | 2.00 (20.8) | 0.07 |
| smol_mixunit | SNGS-034 | path | 68.4 | 73.8 | -5.4 | 0.11* | 2 | 94.4 | 55.56 (20.8) | 0.11 |
| smol_mixunit | SNGS-056 | speed | 49.4 | 85.8 | -36.4 | -0.02 | 3 | 125.0* | 1.50 (30.4) | 0.02 |
| smol_mixunit | SNGS-056 | path | 25.8 | 78.2 | -52.4 | --* | 1 | 94.4 | 55.56 (30.4) | -- |
| smol_mixunit | SNGS-096 | speed | 83.9 | 84.5 | -0.6 | 0.06* | 2 | 166.7* | 2.00 (26.1) | -0.04 |
| smol_mixunit | SNGS-096 | path | 68.6 | 76.4 | -7.9 | --* | 1 | 94.4 | 55.56 (26.1) | -- |
