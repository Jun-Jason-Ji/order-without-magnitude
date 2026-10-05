# Representation probe (preregistered)

Rules sha256 `0d88f0b53a7c3b40...`.

## base  (feature img_L16; H2 model verdict: UNTESTABLE)

| family | H1 cv rho (p99 null) | H3 rho full -> ff4 | H2 Delta [CI] vs E (ctrl) | H2 verdict |
|---|---|---|---|---|
| speed | 0.78 (0.21) ok | 0.71 -> 0.56 NOT motion-dependent | -1.05 [-1.19, -0.91] vs -0.42 (ctrl -0.48) | UNTESTABLE |
| path | 0.83 (0.24) ok | 0.80 -> 0.63 NOT motion-dependent | -0.92 [-1.07, -0.75] vs -0.39 (ctrl -0.40) | UNTESTABLE |

Behaviour (descriptive): metre answer vs references, partial Spearman px|m / m|px:
- speed: Q2_top_480-510 distinct 1, rho m 0.00 px 0.00, partial px|m -- m|px --; Q1_top_0-30 distinct 2, rho m 0.14 px 0.14, partial px|m -- m|px --; Q1_side_0-30 distinct 1, rho m 0.00 px 0.00, partial px|m -- m|px --; Q2_side_300-330 distinct 1, rho m 0.00 px 0.00, partial px|m -- m|px --; Q4_side_570-600 distinct 1, rho m 0.00 px 0.00, partial px|m -- m|px --
- path: Q2_top_480-510 distinct 3, rho m 0.12 px 0.12, partial px|m 0.01 m|px -0.01; Q1_top_0-30 distinct 3, rho m 0.30 px 0.30, partial px|m 0.02 m|px -0.01; Q1_side_0-30 distinct 3, rho m 0.47 px 0.47, partial px|m 0.09 m|px 0.07; Q2_side_300-330 distinct 3, rho m -0.13 px -0.08, partial px|m 0.04 m|px -0.11; Q4_side_570-600 distinct 4, rho m 0.09 px 0.21, partial px|m 0.27 m|px -0.20

## native  (feature last_L20; H2 model verdict: None)

| family | H1 cv rho (p99 null) | H3 rho full -> ff4 | H2 Delta [CI] vs E (ctrl) | H2 verdict |
|---|---|---|---|---|
| speed | 0.81 (0.24) ok | 0.76 -> 0.00 motion | excluded | -- |
| path | 0.83 (0.25) ok | 0.77 -> 0.09 motion | excluded | -- |

Behaviour (descriptive): metre answer vs references, partial Spearman px|m / m|px:
- speed: Q2_top_480-510 distinct 7, rho m 0.70 px 0.70, partial px|m 0.19 m|px -0.16; Q1_top_0-30 distinct 8, rho m 0.65 px 0.64, partial px|m -0.15 m|px 0.16; Q1_side_0-30 distinct 5, rho m 0.60 px 0.68, partial px|m 0.41 m|px -0.13; Q2_side_300-330 distinct 3, rho m 0.36 px 0.38, partial px|m 0.15 m|px 0.11; Q4_side_570-600 distinct 3, rho m 0.44 px 0.47, partial px|m 0.21 m|px 0.11
- path: Q2_top_480-510 distinct 11, rho m 0.68 px 0.69, partial px|m 0.10 m|px -0.07; Q1_top_0-30 distinct 14, rho m 0.53 px 0.53, partial px|m -0.07 m|px 0.08; Q1_side_0-30 distinct 10, rho m 0.51 px 0.60, partial px|m 0.43 m|px -0.24; Q2_side_300-330 distinct 7, rho m 0.29 px 0.35, partial px|m 0.20 m|px 0.03; Q4_side_570-600 distinct 7, rho m 0.50 px 0.56, partial px|m 0.29 m|px 0.05

## v3  (feature last_L20; H2 model verdict: mixed)

| family | H1 cv rho (p99 null) | H3 rho full -> ff4 | H2 Delta [CI] vs E (ctrl) | H2 verdict |
|---|---|---|---|---|
| speed | 0.82 (0.22) ok | 0.77 -> -0.01 motion | -0.56 [-0.63, -0.45] vs -0.42 (ctrl -0.28) | UNTESTABLE |
| path | 0.85 (0.23) ok | 0.80 -> -0.03 motion | -0.78 [-0.95, -0.63] vs -0.39 (ctrl -0.16) | image-plane |

Behaviour (descriptive): metre answer vs references, partial Spearman px|m / m|px:
- speed: Q2_top_480-510 distinct 11, rho m 0.76 px 0.76, partial px|m 0.21 m|px -0.17; Q1_top_0-30 distinct 11, rho m 0.66 px 0.66, partial px|m -0.18 m|px 0.20; Q1_side_0-30 distinct 7, rho m 0.29 px 0.40, partial px|m 0.36 m|px -0.24; Q2_side_300-330 distinct 6, rho m 0.18 px 0.26, partial px|m 0.20 m|px -0.06; Q4_side_570-600 distinct 6, rho m 0.63 px 0.64, partial px|m 0.28 m|px 0.24
- path: Q2_top_480-510 distinct 16, rho m 0.77 px 0.78, partial px|m 0.25 m|px -0.21; Q1_top_0-30 distinct 13, rho m 0.78 px 0.78, partial px|m -0.09 m|px 0.12; Q1_side_0-30 distinct 11, rho m 0.40 px 0.50, partial px|m 0.43 m|px -0.29; Q2_side_300-330 distinct 9, rho m 0.32 px 0.36, partial px|m 0.18 m|px 0.07; Q4_side_570-600 distinct 9, rho m 0.61 px 0.60, partial px|m 0.19 m|px 0.23

## mixunit  (feature img_L24; H2 model verdict: mixed)

| family | H1 cv rho (p99 null) | H3 rho full -> ff4 | H2 Delta [CI] vs E (ctrl) | H2 verdict |
|---|---|---|---|---|
| speed | 0.79 (0.23) ok | 0.76 -> 0.52 NOT motion-dependent | -0.41 [-0.52, -0.21] vs -0.42 (ctrl -0.18) | image-plane |
| path | 0.82 (0.26) ok | 0.79 -> 0.63 NOT motion-dependent | -0.05 [-0.27, 0.07] vs -0.39 (ctrl 0.10) | floor-plane |

Behaviour (descriptive): metre answer vs references, partial Spearman px|m / m|px:
- speed: Q2_top_480-510 distinct 8, rho m 0.67 px 0.68, partial px|m 0.25 m|px -0.22; Q1_top_0-30 distinct 6, rho m 0.49 px 0.49, partial px|m -0.17 m|px 0.18; Q1_side_0-30 distinct 5, rho m 0.20 px 0.31, partial px|m 0.35 m|px -0.26; Q2_side_300-330 distinct 4, rho m -0.00 px 0.02, partial px|m 0.03 m|px -0.03; Q4_side_570-600 distinct 4, rho m 0.36 px 0.38, partial px|m 0.16 m|px 0.09
- path: Q2_top_480-510 distinct 9, rho m 0.74 px 0.75, partial px|m 0.18 m|px -0.15; Q1_top_0-30 distinct 11, rho m 0.70 px 0.70, partial px|m -0.13 m|px 0.15; Q1_side_0-30 distinct 9, rho m 0.20 px 0.29, partial px|m 0.30 m|px -0.23; Q2_side_300-330 distinct 7, rho m 0.09 px 0.11, partial px|m 0.07 m|px 0.00; Q4_side_570-600 distinct 9, rho m 0.39 px 0.42, partial px|m 0.18 m|px 0.06

H4 (v3 and mixunit same H2 verdict in both families): False
