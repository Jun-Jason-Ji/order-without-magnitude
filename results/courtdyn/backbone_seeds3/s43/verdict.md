# Second-backbone replication (preregistered R1-R4), three-seed record: seed 43 (fixed schedule = seed-42 k*, R2 amended)

Rules sha256 `a259c9e07ecf2b09...`, recorded 2026-09-29T13:44 UTC before any run. Degeneracy: R1: metre OR centimetre arm <= 2 distinct; R2: centimetre arm; R4: metre arm (full) and rho defined in both.

## qwen25vl3b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 4

R1 replicates (4/4 live) · R2-E LIMITED (0/4) · R2-T intermediate (3/4) · R3 replicates · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.30 | 5/3 | 1.00 | 242.9 | 2 | 266.7 | 0.29 | -0.01 / 2 |
| Q1_top_0-30/path | 0.42 | 7/4 | 1.00 | 100.0 | 2 | 106.7 | 0.57 | 0.05 / 2 |
| Q2_top_480-510/speed | 0.58 | 5/3 | 1.00 | 242.9 | 2 | 266.7 | 0.42 | 0.02 / 2 |
| Q2_top_480-510/path | 0.61 | 5/4 | 0.63 | 100.0 | 2 | 106.7 | 0.75 | -0.12 / 3 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.15 / -0.03; Q1_top_0-30/path 0.03 / 0.06; Q2_top_480-510/speed 0.29 / 0.33; Q2_top_480-510/path 0.09 / 0.12

Text probe, 246 diagnostic items: base 56.1%, v3 23.2% (p = 1.65e-19), mixunit 50.8% (p = 0.0919)

## smol

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 4

R1 intermediate (3/4 live) · R2-E LIMITED (2/4) · R2-T LIMITED (2/4) · R3 fails · R4 intermediate (3/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.11 | 4/4 | 1.50 | 142.9 | 2 | 157.1 | -0.13 | 0.39 / 3 |
| Q1_top_0-30/path | 0.03 | 3/6 | 0.79 | 94.7 | 5 | 94.7 | 0.05 | -- / 1 |
| Q2_top_480-510/speed | 0.18 | 4/3 | 1.33 | 142.9 | 2 | 157.1 | -0.04 | 0.05 / 3 |
| Q2_top_480-510/path | 0.18 | 2/4 | 0.79 | 84.2 | 5 | 94.7 | 0.07 | -- / 1 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.09 / 0.12; Q1_top_0-30/path 0.03 / -0.07; Q2_top_480-510/speed 0.07 / -0.11; Q2_top_480-510/path -0.02 / 0.02

Text probe, 246 diagnostic items: base 11.8%, v3 35.8% (p = 3.85e-13), mixunit 38.6% (p = 4.31e-14)

## internvl3_2b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 replicates (4/4 live) · R2-E LIMITED (2/4) · R2-T LIMITED (2/4) · R3 fails · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.26 | 6/4 | 1.38 | 333.3 | 2 | 166.7 | 0.27 | -- / 1 |
| Q1_top_0-30/path | 0.46 | 9/5 | 1.00 | 100.0 | 5 | 109.1 | 0.37 | 0.08 / 3 |
| Q2_top_480-510/speed | 0.75 | 6/5 | 1.50 | 333.3 | 2 | 200.0 | 0.54 | -- / 1 |
| Q2_top_480-510/path | 0.73 | 9/6 | 1.07 | 166.7 | 4 | 131.2 | 0.54 | 0.34 / 4 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.31 / 0.26; Q1_top_0-30/path 0.52 / 0.51; Q2_top_480-510/speed 0.62 / 0.66; Q2_top_480-510/path 0.55 / 0.74

Text probe, 246 diagnostic items: base 64.2%, v3 61.0% (p = 0.229), mixunit 54.5% (p = 0.0127)

## idefics3_8b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 replicates (4/4 live) · R2-E intermediate (4/4) · R2-T intermediate (3/4) · R3 replicates · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.22 | 4/4 | 1.00 | 333.3 | 3 | 333.3 | 0.13 | 0.16 / 8 |
| Q1_top_0-30/path | 0.36 | 7/7 | 1.13 | 92.3 | 3 | 107.7 | 0.25 | 0.00 / 9 |
| Q2_top_480-510/speed | 0.40 | 3/4 | 1.00 | 200.0 | 3 | 183.3 | 0.08 | -0.19 / 7 |
| Q2_top_480-510/path | 0.43 | 8/7 | 1.05 | 100.0 | 3 | 107.7 | 0.14 | -0.01 / 8 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.07 / 0.16; Q1_top_0-30/path 0.15 / -0.02; Q2_top_480-510/speed -0.04 / 0.01; Q2_top_480-510/path 0.12 / -0.07

Text probe, 246 diagnostic items: base 58.5%, v3 43.5% (p = 7.51e-07), mixunit 50.8% (p = 0.0344)

