# Second-backbone replication (preregistered R1-R4), three-seed record: seed 44 (fixed schedule = seed-42 k*, R2 amended)

Rules sha256 `a259c9e07ecf2b09...`, recorded 2026-09-29T13:44 UTC before any run. Degeneracy: R1: metre OR centimetre arm <= 2 distinct; R2: centimetre arm; R4: metre arm (full) and rho defined in both.

## qwen25vl3b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 4

R1 intermediate (3/4 live) · R2-E intermediate (3/4) · R2-T intermediate (4/4) · R3 replicates · R4 replicates (3/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.34 | 4/2 | 1.00 | 100.0 | 5 | 116.7 | 0.48 | -0.01 / 2 |
| Q1_top_0-30/path | 0.28 | 4/3 | 1.00 | 100.0 | 2 | 100.0 | 0.31 | 0.05 / 2 |
| Q2_top_480-510/speed | 0.41 | 5/3 | 1.00 | 242.9 | 4 | 133.3 | -- | 0.02 / 2 |
| Q2_top_480-510/path | 0.54 | 6/4 | 1.00 | 100.0 | 3 | 93.9 | 0.42 | -0.12 / 3 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.09 / 0.15; Q1_top_0-30/path 0.16 / 0.28; Q2_top_480-510/speed 0.44 / 0.50; Q2_top_480-510/path 0.04 / 0.43

Text probe, 246 diagnostic items: base 56.1%, v3 23.6% (p = 1.12e-18), mixunit 50.8% (p = 0.0789)

## smol

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 4

R1 LIMITED (2/4 live) · R2-E LIMITED (2/4) · R2-T LIMITED (2/4) · R3 fails · R4 intermediate (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.13 | 5/4 | 1.33 | 157.1 | 1 | 171.4 | -0.01 | 0.39 / 3 |
| Q1_top_0-30/path | 0.12 | 4/1 | 0.79 | 100.0 | 6 | 94.7 | -0.05 | -- / 1 |
| Q2_top_480-510/speed | 0.22 | 5/4 | 1.09 | 100.0 | 1 | 100.0 | 0.03 | 0.05 / 3 |
| Q2_top_480-510/path | 0.19 | 4/1 | 0.79 | 86.8 | 6 | 89.5 | 0.11 | -- / 1 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed -- / -0.01; Q1_top_0-30/path 0.03 / -0.13; Q2_top_480-510/speed -- / -0.17; Q2_top_480-510/path -0.09 / 0.04

Text probe, 246 diagnostic items: base 11.8%, v3 35.4% (p = 6.96e-13), mixunit 39.0% (p = 2.41e-14)

## internvl3_2b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 replicates (4/4 live) · R2-E replicates (4/4) · R2-T replicates (4/4) · R3 fails · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.46 | 8/3 | 1.50 | 333.3 | 3 | 166.7 | 0.47 | -- / 1 |
| Q1_top_0-30/path | 0.55 | 11/7 | 1.00 | 95.5 | 7 | 100.0 | 0.43 | 0.08 / 3 |
| Q2_top_480-510/speed | 0.71 | 6/3 | 2.00 | 333.3 | 3 | 200.0 | 0.48 | -- / 1 |
| Q2_top_480-510/path | 0.72 | 11/7 | 1.17 | 100.0 | 6 | 109.1 | 0.63 | 0.34 / 4 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.34 / 0.37; Q1_top_0-30/path 0.57 / 0.49; Q2_top_480-510/speed 0.45 / 0.53; Q2_top_480-510/path 0.70 / 0.73

Text probe, 246 diagnostic items: base 64.2%, v3 66.3% (p = 0.522), mixunit 59.3% (p = 0.23)

## pixtral_12b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 replicates (4/4 live) · R2-E replicates (4/4) · R2-T LIMITED (2/4) · R3 replicates · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.24 | 5/7 | 1.00 | 333.3 | 3 | 333.3 | 0.14 | -0.01 / 2 |
| Q1_top_0-30/path | 0.34 | 9/4 | 1.18 | 84.6 | 4 | 144.9 | 0.27 | 0.20 / 3 |
| Q2_top_480-510/speed | 0.50 | 5/5 | 1.00 | 333.3 | 3 | 333.3 | 0.12 | 0.07 / 3 |
| Q2_top_480-510/path | 0.55 | 8/3 | 1.18 | 84.6 | 6 | 183.3 | 0.01 | 0.13 / 3 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.24 / 0.10; Q1_top_0-30/path 0.45 / 0.16; Q2_top_480-510/speed 0.44 / 0.46; Q2_top_480-510/path 0.59 / 0.62

Text probe, 246 diagnostic items: base 49.2%, v3 30.1% (p = 3.04e-07), mixunit 69.5% (p = 2.86e-09)

## idefics3_8b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 intermediate (4/4 live) · R2-E LIMITED (2/4) · R2-T intermediate (3/4) · R3 replicates · R4 intermediate (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.15 | 3/3 | 1.38 | 150.0 | 5 | 150.0 | -0.02 | 0.16 / 8 |
| Q1_top_0-30/path | 0.27 | 4/3 | 1.00 | 127.3 | 2 | 127.3 | 0.34 | 0.00 / 9 |
| Q2_top_480-510/speed | 0.18 | 4/3 | 1.38 | 150.0 | 5 | 150.0 | 0.05 | -0.19 / 7 |
| Q2_top_480-510/path | 0.13 | 4/4 | 1.00 | 127.3 | 2 | 127.3 | -0.11 | -0.01 / 8 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed -0.13 / -0.02; Q1_top_0-30/path 0.14 / 0.17; Q2_top_480-510/speed -0.02 / 0.01; Q2_top_480-510/path 0.20 / 0.08

Text probe, 246 diagnostic items: base 58.5%, v3 45.5% (p = 5.61e-06), mixunit 51.2% (p = 0.0444)

## gemma3_12b

Schedule fixed to the seed-42 selection (dev rule not re-run): k = 2

R1 replicates (4/4 live) · R2-E LIMITED (2/4) · R2-T intermediate (4/4) · R3 fails · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.36 | 11/20 | 6.79 | 137.5 | 2 | 137.5 | 0.28 | -0.10 / 6 |
| Q1_top_0-30/path | 0.51 | 15/13 | 1.48 | 123.1 | 5 | 116.7 | 0.18 | 0.15 / 5 |
| Q2_top_480-510/speed | 0.52 | 10/16 | 1.44 | 157.1 | 1 | 157.1 | 0.46 | -0.24 / 3 |
| Q2_top_480-510/path | 0.51 | 13/11 | 1.38 | 123.1 | 5 | 123.1 | 0.52 | 0.13 / 2 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.14 / 0.03; Q1_top_0-30/path 0.41 / 0.45; Q2_top_480-510/speed -- / 0.03; Q2_top_480-510/path 0.22 / 0.37

Text probe, 246 diagnostic items: base 89.4%, v3 92.7% (p = 0.00781), mixunit 91.9% (p = 0.146)

