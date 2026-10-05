# Second-backbone replication (preregistered R1-R4), A100 backbones (round-2 protocol, R2 amended)

Rules sha256 `eecca80879ffe4f4...`, recorded 2026-09-28T03:14 UTC before any run. Degeneracy: R1: metre OR centimetre arm <= 2 distinct; R2: centimetre arm; R4: metre arm (full) and rho defined in both.

## pixtral_12b

Stopping rule (dev set, decided before any test cell): k* = 2, passes

R1 replicates (4/4 live) · R2-E intermediate (3/4) · R2-T LIMITED (1/4) · R3 replicates · R4 LIMITED (2/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.38 | 7/4 | 1.00 | 142.9 | 3 | 142.9 | 0.35 | -0.01 / 2 |
| Q1_top_0-30/path | 0.36 | 7/8 | 1.00 | 100.0 | 3 | 92.3 | -- | 0.20 / 3 |
| Q2_top_480-510/speed | 0.63 | 7/4 | 1.05 | 142.9 | 2 | 142.9 | 0.27 | 0.07 / 3 |
| Q2_top_480-510/path | 0.63 | 5/8 | 1.00 | 100.0 | 3 | 96.2 | -- | 0.13 / 3 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.22 / 0.10; Q1_top_0-30/path 0.36 / 0.10; Q2_top_480-510/speed 0.19 / 0.34; Q2_top_480-510/path 0.55 / 0.29

Text probe, 246 diagnostic items: base 49.2%, v3 31.7% (p = 4.35e-06), mixunit 62.2% (p = 0.000313)

## idefics3_8b

Stopping rule (dev set, decided before any test cell): k* = 2, passes

R1 LIMITED (1/4 live) · R2-E intermediate (4/4) · R2-T intermediate (3/4) · R3 intermediate · R4 LIMITED (1/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.23 | 2/7 | 1.29 | 150.0 | 5 | 150.0 | 0.20 | 0.16 / 8 |
| Q1_top_0-30/path | 0.10 | 2/2 | 1.00 | 115.4 | 3 | 169.2 | -0.01 | 0.00 / 9 |
| Q2_top_480-510/speed | 0.05 | 2/6 | 1.43 | 157.1 | 3 | 150.0 | -0.06 | -0.19 / 7 |
| Q2_top_480-510/path | -0.00 | 3/4 | 1.00 | 115.4 | 3 | 169.2 | -0.10 | -0.01 / 8 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed -0.05 / 0.05; Q1_top_0-30/path 0.27 / 0.19; Q2_top_480-510/speed 0.11 / 0.21; Q2_top_480-510/path 0.17 / 0.13

Text probe, 246 diagnostic items: base 58.5%, v3 42.3% (p = 8.96e-08), mixunit 44.3% (p = 0.000155)

## gemma3_12b

Stopping rule (dev set, decided before any test cell): k* = 2, passes

R1 fails (4/4 live) · R2-E LIMITED (2/4) · R2-T intermediate (4/4) · R3 fails · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.31 | 7/12 | 12.21 | 122.2 | 2 | 122.2 | 0.21 | -0.10 / 6 |
| Q1_top_0-30/path | 0.51 | 12/17 | 98.94 | 100.0 | 8 | 91.3 | 0.22 | 0.15 / 5 |
| Q2_top_480-510/speed | 0.34 | 6/14 | 2.02 | 122.2 | 2 | 134.1 | 0.25 | -0.24 / 3 |
| Q2_top_480-510/path | 0.40 | 10/19 | 13.91 | 92.3 | 7 | 91.3 | 0.31 | 0.13 / 2 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.00 / -0.02; Q1_top_0-30/path 0.34 / 0.36; Q2_top_480-510/speed 0.14 / 0.21; Q2_top_480-510/path 0.32 / 0.37

Text probe, 246 diagnostic items: base 89.4%, v3 92.3% (p = 0.0391), mixunit 88.6% (p = 0.774)

