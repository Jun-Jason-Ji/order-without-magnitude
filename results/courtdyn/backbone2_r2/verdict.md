# Second-backbone replication (preregistered R1-R4), round 2 (4-epoch schedule, R2 amended)

Rules sha256 `003f2e1a80e4153c...`, recorded 2026-09-27T00:59 UTC before any run. Degeneracy: R1: metre OR centimetre arm <= 2 distinct; R2: centimetre arm; R4: metre arm (full) and rho defined in both.

## smol

Stopping rule (dev set, decided before any test cell): k* = 4, DEGENERATE AT THE 4-EPOCH CAP (test cells descriptive only)

R1 intermediate (4/4 live) · R2-E LIMITED (2/4) · R2-T intermediate (3/4) · R3 fails · R4 intermediate (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.06 | 5/3 | 1.50 | 142.9 | 2 | 157.1 | -0.22 | 0.39 / 3 |
| Q1_top_0-30/path | 0.10 | 3/5 | 0.79 | 52.6 | 5 | 88.9 | -0.04 | -- / 1 |
| Q2_top_480-510/speed | 0.19 | 5/6 | 1.25 | 142.9 | 2 | 157.1 | 0.03 | 0.05 / 3 |
| Q2_top_480-510/path | 0.07 | 6/5 | 0.79 | 63.2 | 5 | 84.2 | -0.04 | -- / 1 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.11 / 0.02; Q1_top_0-30/path 0.26 / 0.03; Q2_top_480-510/speed 0.12 / 0.06; Q2_top_480-510/path 0.01 / 0.34

Text probe, 246 diagnostic items: base 11.8%, v3 32.1% (p = 7.64e-11), mixunit 35.8% (p = 5.54e-12)

## qwen25vl3b

Stopping rule (dev set, decided before any test cell): k* = 4, passes

R1 intermediate (4/4 live) · R2-E intermediate (3/4) · R2-T intermediate (4/4) · R3 replicates · R4 LIMITED (2/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.26 | 5/3 | 1.00 | 100.0 | 5 | 100.0 | -- | -0.01 / 2 |
| Q1_top_0-30/path | 0.30 | 5/4 | 1.00 | 100.0 | 2 | 100.0 | 0.22 | 0.05 / 2 |
| Q2_top_480-510/speed | 0.43 | 4/3 | 1.00 | 100.0 | 6 | 100.0 | -- | 0.02 / 2 |
| Q2_top_480-510/path | 0.51 | 5/5 | 0.63 | 100.0 | 3 | 100.0 | 0.44 | -0.12 / 3 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.07 / 0.05; Q1_top_0-30/path 0.07 / 0.03; Q2_top_480-510/speed 0.45 / 0.33; Q2_top_480-510/path -0.10 / 0.48

Text probe, 246 diagnostic items: base 56.1%, v3 28.5% (p = 1.73e-15), mixunit 52.4% (p = 0.262)

