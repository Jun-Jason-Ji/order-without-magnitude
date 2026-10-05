# Second-backbone replication (preregistered R1-R4), third backbone InternVL3-2B (round-2 protocol, R2 amended)

Rules sha256 `8408a84bd433285e...`, recorded 2026-09-28T02:33 UTC before any run. Degeneracy: R1: metre OR centimetre arm <= 2 distinct; R2: centimetre arm; R4: metre arm (full) and rho defined in both.

## internvl3_2b

Stopping rule (dev set, decided before any test cell): k* = 2, passes

R1 intermediate (3/4 live) · R2-E LIMITED (2/4) · R2-T replicates (3/4) · R3 fails · R4 replicates (4/4 defined)

| cell | v3 rho_m | v3 distinct m/cm | v3 R | mix R (E) | mix distinct cm (E) | mix R (T) | v3 delta rho first-frame | base rho_m / distinct |
|---|---|---|---|---|---|---|---|---|
| Q1_top_0-30/speed | 0.33 | 8/2 | 2.20 | 166.7 | 2 | 142.9 | 0.29 | -- / 1 |
| Q1_top_0-30/path | 0.45 | 11/7 | 1.10 | 92.3 | 5 | 95.7 | 0.42 | 0.08 / 3 |
| Q2_top_480-510/speed | 0.68 | 9/3 | 3.67 | 166.7 | 2 | 142.9 | 0.26 | -- / 1 |
| Q2_top_480-510/path | 0.71 | 11/6 | 1.20 | 100.0 | 5 | 100.0 | 0.35 | 0.34 / 4 |

Post hoc (no verdict): Spearman rho of the mixed-unit centimetre answers vs reference, E / T: Q1_top_0-30/speed 0.34 / 0.26; Q1_top_0-30/path 0.45 / 0.53; Q2_top_480-510/speed 0.66 / 0.71; Q2_top_480-510/path 0.65 / 0.73

Text probe, 246 diagnostic items: base 64.2%, v3 72.0% (p = 0.0094), mixunit 50.8% (p = 0.00061)

