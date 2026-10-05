# Proposition 2 (collapse to the conditional mode) - POST HOC test on paper-2 mixed-unit adapters

Descriptive; no decision rule. Bucket = the adapter's own metre answer (Voronoi cell on its reading grid); targets = centimetre items of its own training pool. Match = exact agreement with the adapter's cm answer.

| adapter | clip | family | n | G (metre readings) | distinct cm answers | top cm answers | global mode of cm targets | rho(cm, reading) | cond_mode | global_mode | **cond_prefix** | global_prefix | exact x100 | nearest target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 s42 | Q1 | speed | 140 | 6 | 2 | 110x105, 100x35 | 90 / prefix 110 | 0.55 | 0.00 | 0.00 | **0.01** | 0.75 | 0.00 | 0.00 |
| M1 s42 | Q1 | path | 140 | 11 | 8 | 140x38, 160x38, 110x22, 240x21 | 220 / prefix 140 | 0.84 | 0.31 | 0.00 | **0.31** | 0.27 | 0.46 | 0.46 |
| M1 s42 | Q2 | speed | 140 | 8 | 2 | 100x79, 110x61 | 90 / prefix 110 | 0.78 | 0.00 | 0.00 | **0.01** | 0.44 | 0.00 | 0.00 |
| M1 s42 | Q2 | path | 140 | 9 | 7 | 110x56, 140x29, 160x21, 120x17 | 220 / prefix 140 | 0.86 | 0.24 | 0.00 | **0.24** | 0.21 | 0.53 | 0.53 |
| M1 s43 | Q1 | speed | 140 | 6 | 1 | 110x140 | 90 / prefix 110 | -- | 0.01 | 0.00 | **0.01** | 1.00 | 0.01 | 0.01 |
| M1 s43 | Q1 | path | 140 | 6 | 6 | 110x67, 160x58, 140x5, 150x4 | 220 / prefix 140 | 0.80 | 0.04 | 0.00 | **0.04** | 0.04 | 0.13 | 0.13 |
| M1 s43 | Q2 | speed | 140 | 6 | 2 | 110x138, 100x2 | 90 / prefix 110 | 0.25 | 0.00 | 0.00 | **0.05** | 0.99 | 0.00 | 0.00 |
| M1 s43 | Q2 | path | 140 | 8 | 6 | 110x97, 160x37, 210x3, 150x1 | 220 / prefix 140 | 0.76 | 0.02 | 0.00 | **0.02** | 0.01 | 0.24 | 0.24 |
| M1 s44 | Q1 | speed | 140 | 3 | 1 | 110x140 | 90 / prefix 110 | -- | 0.01 | 0.00 | **0.01** | 1.00 | 0.01 | 0.01 |
| M1 s44 | Q1 | path | 140 | 10 | 5 | 140x80, 160x29, 110x20, 210x7 | 220 / prefix 140 | 0.75 | 0.20 | 0.00 | **0.20** | 0.57 | 0.36 | 0.36 |
| M1 s44 | Q2 | speed | 140 | 4 | 3 | 110x128, 100x11, 10x1 | 90 / prefix 110 | 0.35 | 0.00 | 0.00 | **0.14** | 0.91 | 0.00 | 0.00 |
| M1 s44 | Q2 | path | 140 | 8 | 5 | 140x65, 110x55, 160x14, 210x3 | 220 / prefix 140 | 0.74 | 0.11 | 0.00 | **0.11** | 0.46 | 0.45 | 0.45 |
| M1-fine s42 | Q1 | speed | 140 | 3 | 2 | 100x91, 104x49 | 79 / prefix 100 | 0.12 | 0.00 | 0.00 | **0.02** | 0.65 | 0.00 | 0.00 |
| M1-fine s42 | Q1 | path | 140 | 9 | 7 | 160x56, 260x38, 240x23, 190x13 | 220 / prefix 140 | 0.80 | 0.21 | 0.00 | **0.21** | 0.04 | 0.24 | 0.24 |
| M1-fine s42 | Q2 | speed | 140 | 4 | 3 | 100x94, 104x34, 10x12 | 79 / prefix 100 | 0.20 | 0.00 | 0.00 | **0.53** | 0.67 | 0.00 | 0.00 |
| M1-fine s42 | Q2 | path | 140 | 11 | 9 | 160x77, 240x19, 260x15, 140x7 | 220 / prefix 140 | 0.79 | 0.17 | 0.00 | **0.17** | 0.05 | 0.16 | 0.16 |

Reading of the columns: if cm answers are a per-unit prior, cond_mode/global_mode are high and exact x100 is low; if the adapter converts its reading, exact x100 / nearest target are high. Speed vs path is the contrast the proposition predicts (few distinct readings -> collapse).
