# Second-backbone replication (preregistered R1-R4)

Rules sha256 `42ed2ead18d80a34...`, recorded 2026-09-26T05:30 UTC before any run. Degeneracy: R1: metre OR centimetre arm <= 2 distinct; R2: centimetre arm; R4: metre arm (full) and rho defined in both.

## smol

R1 LIMITED (2/4 live) · R2-E LIMITED (2/4) · R2-T replicates (4/4) · R3 fails · R4 LIMITED (2/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.22 | 4/4 | 1.78 | 125.0 | 2 | 150.0 | 0.14 | 0.39 / 3 |
| Q1_top_0-30/path | -- | 1/4 | 10.21 | 100.0 | 4 | 100.0 | -- | -- / 1 |
| Q2_top_480-510/speed | 0.01 | 4/4 | 1.62 | 84.0 | 2 | 125.0 | 0.03 | 0.05 / 3 |
| Q2_top_480-510/path | -- | 1/7 | 10.21 | 100.0 | 4 | 94.4 | -- | -- / 1 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.15 / 0.03; Q1_top_0-30/path -0.17 / -0.22; Q2_top_480-510/speed 0.02 / -0.17; Q2_top_480-510/path -0.24 / -0.32

Text probe, 246 diagnostic items: base 11.8%, v3 34.6% (p = 5.77e-12), mixunit 36.2% (p = 1.38e-12)

## qwen25vl3b

R1 LIMITED (0/4 live) · R2-E LIMITED (0/4) · R2-T LIMITED (2/4) · R3 intermediate · R4 LIMITED (0/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.08 | 2/3 | 1.00 | 242.9 | 2 | 187.5 | 0.10 | -0.01 / 2 |
| Q1_top_0-30/path | 0.04 | 2/1 | 1.00 | 100.0 | 1 | 94.1 | -- | 0.05 / 2 |
| Q2_top_480-510/speed | -0.07 | 2/3 | 0.87 | 242.9 | 2 | 187.5 | 0.01 | 0.02 / 2 |
| Q2_top_480-510/path | 0.14 | 2/1 | 1.00 | 100.0 | 1 | 94.1 | -- | -0.12 / 3 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.10 / 0.21; Q1_top_0-30/path -- / 0.14; Q2_top_480-510/speed 0.02 / 0.00; Q2_top_480-510/path -- / 0.19

Text probe, 246 diagnostic items: base 56.1%, v3 30.5% (p = 3.52e-14), mixunit 48.0% (p = 0.00552)

