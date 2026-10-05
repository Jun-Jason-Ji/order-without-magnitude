# Second-backbone replication (preregistered R1-R4), A100 extension: seed 43 (fixed schedule = seed-42 k*, R2 amended)

Rules sha256 `171b9dc9fb63bd85...`, recorded 2026-09-28T23:37 UTC before any run. Degeneracy: R1: metre OR centimetre arm <= 2 distinct; R2: centimetre arm; R4: metre arm (full) and rho defined in both.

## gemma3_12b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 intermediate (4/4 live) · R2-E LIMITED (2/4) · R2-T intermediate (4/4) · R3 fails · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.26 | 8/13 | 1.83 | 137.5 | 2 | 120.2 | 0.16 | -0.10 / 6 |
| Q1_top_0-30/path | 0.55 | 12/18 | 70.91 | 106.7 | 8 | 100.0 | 0.27 | 0.15 / 5 |
| Q2_top_480-510/speed | 0.51 | 7/12 | 1.83 | 137.5 | 1 | 137.5 | 0.47 | -0.24 / 3 |
| Q2_top_480-510/path | 0.47 | 12/17 | 10.50 | 123.1 | 5 | 100.0 | 0.42 | 0.13 / 2 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed -0.10 / 0.12; Q1_top_0-30/path 0.37 / 0.49; Q2_top_480-510/speed -- / 0.14; Q2_top_480-510/path 0.23 / 0.40

Text probe, 246 diagnostic items: base 89.4%, v3 91.1% (p = 0.289), mixunit 89.8% (p = 1)

## pixtral_12b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 replicates (4/4 live) · R2-E replicates (4/4) · R2-T intermediate (3/4) · R3 fails · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.33 | 6/6 | 1.00 | 166.7 | 3 | 166.7 | 0.24 | -0.01 / 2 |
| Q1_top_0-30/path | 0.50 | 13/7 | 1.00 | 90.9 | 4 | 100.0 | 0.29 | 0.20 / 3 |
| Q2_top_480-510/speed | 0.66 | 7/8 | 1.20 | 166.7 | 3 | 166.7 | 0.18 | 0.07 / 3 |
| Q2_top_480-510/path | 0.69 | 12/8 | 1.00 | 104.5 | 4 | 166.7 | 0.15 | 0.13 / 3 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.17 / 0.23; Q1_top_0-30/path 0.45 / 0.34; Q2_top_480-510/speed 0.39 / 0.42; Q2_top_480-510/path 0.63 / 0.64

Text probe, 246 diagnostic items: base 49.2%, v3 41.9% (p = 0.0662), mixunit 67.9% (p = 3.81e-08)

