# RTC - combined verdicts (seed 42 + preregistered replications)

Rules sha256 `230fff51289d3c31...`; ERRATA 2 (band [f/1.5, 1.5f]) and 3 (replication specification). Pending = not yet run.

- **Across seeds (42/43/44)**: H1 holds in 1 of 3 seeds  (H1 by seed: {42: 'holds', 43: 'intermediate', 44: 'intermediate'})
- **InternVL3-2B**: H1 fails on InternVL3-2B

## Seed 42 (from rtc/verdict.json)

- H1_unseen_units_rtc: holds
- H2_pixels_given_scale_rtc: intermediate
- H3_speed_cm_rtc: fixed
- H4_metre_noninferiority: fails
- H5_motion_dependence: LIMITED
- H6_format_control_v3s42: holds
- plain_unseen_v3s42: fails
- H6_format_control_m1s42: holds
- plain_unseen_m1s42: fails
- plain_unseen_m1fines42: intermediate

## seed43

- **H1_unseen_units_rtc**: intermediate (5)
- **H2_pixels_given_scale_rtc**: fails (1)
- **H3_speed_cm_rtc**: fixed (2)
- **H4_metre_noninferiority**: holds
- **H5_motion_dependence**: holds (4)
- **H6_format_control_v3s43**: holds (6)
- **plain_unseen_v3s43**: fails (0)
- **H6_format_control_m1s43**: intermediate (5)
- **plain_unseen_m1s43**: fails (2)

| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts | as registered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtc_s43 | rtc | Q1_top_0-30 | m | speed | 1.00 | 4 | 0.38 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q1_top_0-30 | m | path | 1.00 | 9 | 0.75 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 4 | 0.40 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q1_top_0-30 | cm | path | 1.00 | 9 | 0.72 | 95.207 | 100.000 | 0.95 | yes | yes |
| rtc_s43 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 3 | 0.38 | 1.000 | 3.600 | 0.28 | no | no |
| rtc_s43 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 9 | 0.69 | 3.571 | 3.281 | 1.09 | yes | yes |
| rtc_s43 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.25 | 3.273 | 3.281 | 1.00 | no | no |
| rtc_s43 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 8 | 0.72 | 913.043 | 1000.000 | 0.91 | yes | yes |
| rtc_s43 | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 5 | 0.49 | 1.000 | 27.099 | 0.04 | no | no |
| rtc_s43 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 12 | 0.77 | 22.000 | 27.061 | 0.81 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | m | speed | 1.00 | 5 | 0.65 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | m | path | 1.00 | 9 | 0.75 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 4 | 0.60 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | cm | path | 1.00 | 11 | 0.74 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 5 | 0.37 | 1.000 | 3.600 | 0.28 | no | no |
| rtc_s43 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 8 | 0.74 | 3.571 | 3.281 | 1.09 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 5 | 0.42 | 3.273 | 3.281 | 1.00 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 9 | 0.70 | 916.667 | 1000.000 | 0.92 | yes | yes |
| rtc_s43 | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 5 | 0.66 | 1.000 | 27.492 | 0.04 | no | no |
| rtc_s43 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 13 | 0.53 | 1.000 | 27.498 | 0.04 | no | no |
| v3s43 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 7 | 0.60 | 1.000 | 3.600 | 0.28 | no | no |
| v3s43 | plain | Q1_top_0-30 | U1 | path | 1.00 | 15 | 0.55 | 1.235 | 3.281 | 0.38 | no | no |
| v3s43 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 11 | 0.66 | 1.500 | 3.281 | 0.46 | no | no |
| v3s43 | plain | Q1_top_0-30 | U2 | path | 1.00 | 19 | 0.73 | 1.000 | 1000.000 | 0.00 | no | no |
| v3s43 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 10 | 0.61 | 1.500 | 27.099 | 0.06 | no | no |
| v3s43 | plain | Q1_top_0-30 | pxK | path | 1.00 | 11 | 0.66 | 1.273 | 27.061 | 0.05 | no | no |
| v3s43 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 8 | 0.75 | 1.000 | 3.600 | 0.28 | no | no |
| v3s43 | plain | Q2_top_480-510 | U1 | path | 1.00 | 17 | 0.72 | 1.273 | 3.281 | 0.39 | no | no |
| v3s43 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 14 | 0.74 | 1.500 | 3.281 | 0.46 | no | no |
| v3s43 | plain | Q2_top_480-510 | U2 | path | 1.00 | 15 | 0.74 | 1.513 | 1000.000 | 0.00 | no | no |
| v3s43 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 13 | 0.75 | 1.500 | 27.492 | 0.05 | no | no |
| v3s43 | plain | Q2_top_480-510 | pxK | path | 1.00 | 13 | 0.77 | 1.273 | 27.498 | 0.05 | no | no |
| v3s43 | rtc | Q1_top_0-30 | m | speed | 1.00 | 10 | 0.68 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s43 | rtc | Q1_top_0-30 | m | path | 1.00 | 17 | 0.75 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s43 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 14 | 0.03 | 6.667 | 100.000 | 0.07 | no | no |
| v3s43 | rtc | Q1_top_0-30 | cm | path | 1.00 | 16 | 0.74 | 1.000 | 100.000 | 0.01 | no | no |
| v3s43 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 7 | 0.56 | 3.500 | 3.600 | 0.97 | yes | yes |
| v3s43 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 18 | 0.79 | 3.971 | 3.281 | 1.21 | yes | yes |
| v3s43 | rtc | Q1_top_0-30 | U2 | speed | 0.96 | 8 | 0.62 | 3.273 | 3.281 | 1.00 | yes | yes |
| v3s43 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 16 | 0.76 | 0.915 | 1000.000 | 0.00 | no | no |
| v3s43 | rtc | Q1_top_0-30 | pxK | speed | 0.94 | 12 | 0.24 | 7.455 | 27.099 | 0.28 | no | no |
| v3s43 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 17 | 0.05 | 1.812 | 27.061 | 0.07 | no | no |
| v3s43 | rtc | Q2_top_480-510 | m | speed | 1.00 | 9 | 0.74 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s43 | rtc | Q2_top_480-510 | m | path | 1.00 | 18 | 0.79 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s43 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 13 | 0.60 | 1.000 | 100.000 | 0.01 | no | no |
| v3s43 | rtc | Q2_top_480-510 | cm | path | 1.00 | 17 | 0.79 | 1.000 | 100.000 | 0.01 | no | no |
| v3s43 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 8 | 0.73 | 3.523 | 3.600 | 0.98 | yes | yes |
| v3s43 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 18 | 0.81 | 4.165 | 3.281 | 1.27 | yes | yes |
| v3s43 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 8 | 0.70 | 3.833 | 3.281 | 1.17 | yes | yes |
| v3s43 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 18 | 0.80 | 1.000 | 1000.000 | 0.00 | no | no |
| v3s43 | rtc | Q2_top_480-510 | pxK | speed | 0.84 | 12 | 0.35 | 4.500 | 27.492 | 0.16 | no | no |
| v3s43 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 17 | 0.26 | 1.303 | 27.498 | 0.05 | no | no |
| m1s43 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 5 | 0.51 | 1.167 | 3.600 | 0.32 | no | no |
| m1s43 | plain | Q1_top_0-30 | U1 | path | 1.00 | 5 | 0.21 | 0.941 | 3.281 | 0.29 | no | no |
| m1s43 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 6 | 0.31 | 2.429 | 3.281 | 0.74 | yes | yes |
| m1s43 | plain | Q1_top_0-30 | U2 | path | 1.00 | 2 | 0.56 | 68.750 | 1000.000 | 0.07 | no | no |
| m1s43 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 6 | 0.38 | 1.875 | 27.099 | 0.07 | no | no |
| m1s43 | plain | Q1_top_0-30 | pxK | path | 1.00 | 2 | 0.41 | 52.381 | 27.061 | 1.94 | no | no |
| m1s43 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 5 | 0.59 | 1.167 | 3.600 | 0.32 | no | no |
| m1s43 | plain | Q2_top_480-510 | U1 | path | 1.00 | 6 | 0.44 | 1.000 | 3.281 | 0.30 | no | no |
| m1s43 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 5 | 0.31 | 2.333 | 3.281 | 0.71 | yes | yes |
| m1s43 | plain | Q2_top_480-510 | U2 | path | 1.00 | 3 | 0.56 | 71.042 | 1000.000 | 0.07 | no | no |
| m1s43 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 6 | 0.42 | 1.833 | 27.492 | 0.07 | no | no |
| m1s43 | plain | Q2_top_480-510 | pxK | path | 1.00 | 3 | 0.63 | 52.381 | 27.498 | 1.90 | no | yes |
| m1s43 | rtc | Q1_top_0-30 | m | speed | 1.00 | 6 | 0.50 | 1.000 | 1.000 | 1.00 | yes | yes |
| m1s43 | rtc | Q1_top_0-30 | m | path | 1.00 | 4 | 0.50 | 1.000 | 1.000 | 1.00 | yes | yes |
| m1s43 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 7 | 0.22 | 100.000 | 100.000 | 1.00 | no | no |
| m1s43 | rtc | Q1_top_0-30 | cm | path | 1.00 | 11 | 0.04 | 94.118 | 100.000 | 0.94 | no | no |
| m1s43 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 4 | 0.25 | 4.875 | 3.600 | 1.35 | no | no |
| m1s43 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 5 | 0.49 | 3.286 | 3.281 | 1.00 | yes | yes |
| m1s43 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.24 | 6.000 | 3.281 | 1.83 | no | no |
| m1s43 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 4 | 0.44 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| m1s43 | rtc | Q1_top_0-30 | pxK | speed | 0.23 | 8 | 0.06 | 13.500 | 27.099 | 0.50 | no | no |
| m1s43 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 10 | 0.61 | 21.905 | 27.061 | 0.81 | yes | yes |
| m1s43 | rtc | Q2_top_480-510 | m | speed | 1.00 | 7 | 0.46 | 1.000 | 1.000 | 1.00 | yes | yes |
| m1s43 | rtc | Q2_top_480-510 | m | path | 1.00 | 6 | 0.54 | 1.000 | 1.000 | 1.00 | yes | yes |
| m1s43 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 6 | 0.44 | 100.000 | 100.000 | 1.00 | yes | yes |
| m1s43 | rtc | Q2_top_480-510 | cm | path | 1.00 | 11 | 0.33 | 1.059 | 100.000 | 0.01 | no | no |
| m1s43 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 5 | 0.42 | 4.875 | 3.600 | 1.35 | yes | yes |
| m1s43 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 6 | 0.40 | 3.267 | 3.281 | 1.00 | yes | yes |
| m1s43 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 5 | 0.43 | 6.000 | 3.281 | 1.83 | no | yes |
| m1s43 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 8 | 0.31 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| m1s43 | rtc | Q2_top_480-510 | pxK | speed | 0.10 | 4 | 0.01 | 0.500 | 27.492 | 0.02 | no | no |
| m1s43 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 18 | 0.09 | 1.062 | 27.498 | 0.04 | no | no |

## seed44

- **H1_unseen_units_rtc**: intermediate (4)
- **H2_pixels_given_scale_rtc**: fails (0)
- **H3_speed_cm_rtc**: fixed (2)
- **H4_metre_noninferiority**: fails
- **H5_motion_dependence**: holds (4)
- **H6_format_control_v3s44**: holds (6)
- **plain_unseen_v3s44**: fails (0)
- **H6_format_control_m1s44**: holds (6)
- **plain_unseen_m1s44**: intermediate (3)

| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts | as registered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtc_s44 | rtc | Q1_top_0-30 | m | speed | 1.00 | 4 | 0.46 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q1_top_0-30 | m | path | 1.00 | 8 | 0.72 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 5 | 0.52 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q1_top_0-30 | cm | path | 1.00 | 9 | 0.72 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 5 | 0.44 | 1.000 | 3.600 | 0.28 | no | no |
| rtc_s44 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 10 | 0.67 | 3.286 | 3.281 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.12 | 5.143 | 3.281 | 1.57 | no | no |
| rtc_s44 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 7 | 0.69 | 941.176 | 1000.000 | 0.94 | yes | yes |
| rtc_s44 | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 4 | 0.57 | 1.000 | 27.099 | 0.04 | no | no |
| rtc_s44 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 9 | 0.71 | 0.875 | 27.061 | 0.03 | no | no |
| rtc_s44 | rtc | Q2_top_480-510 | m | speed | 1.00 | 6 | 0.60 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q2_top_480-510 | m | path | 1.00 | 10 | 0.69 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 5 | 0.58 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q2_top_480-510 | cm | path | 1.00 | 10 | 0.71 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 7 | 0.47 | 1.167 | 3.600 | 0.32 | no | no |
| rtc_s44 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 8 | 0.72 | 3.273 | 3.281 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 6 | 0.37 | 5.143 | 3.281 | 1.57 | no | yes |
| rtc_s44 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 10 | 0.68 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| rtc_s44 | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 6 | 0.61 | 0.857 | 27.492 | 0.03 | no | no |
| rtc_s44 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 11 | 0.69 | 1.000 | 27.498 | 0.04 | no | no |
| v3s44 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 8 | 0.58 | 0.866 | 3.600 | 0.24 | no | no |
| v3s44 | plain | Q1_top_0-30 | U1 | path | 1.00 | 13 | 0.55 | 1.143 | 3.281 | 0.35 | no | no |
| v3s44 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 8 | 0.66 | 1.273 | 3.281 | 0.39 | no | no |
| v3s44 | plain | Q1_top_0-30 | U2 | path | 1.00 | 10 | 0.73 | 0.941 | 1000.000 | 0.00 | no | no |
| v3s44 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 11 | 0.69 | 1.427 | 27.099 | 0.05 | no | no |
| v3s44 | plain | Q1_top_0-30 | pxK | path | 1.00 | 15 | 0.75 | 1.125 | 27.061 | 0.04 | no | no |
| v3s44 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 6 | 0.72 | 1.000 | 3.600 | 0.28 | no | no |
| v3s44 | plain | Q2_top_480-510 | U1 | path | 1.00 | 14 | 0.75 | 1.182 | 3.281 | 0.36 | no | no |
| v3s44 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 7 | 0.72 | 1.500 | 3.281 | 0.46 | no | no |
| v3s44 | plain | Q2_top_480-510 | U2 | path | 1.00 | 10 | 0.76 | 1.000 | 1000.000 | 0.00 | no | no |
| v3s44 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 12 | 0.74 | 1.455 | 27.492 | 0.05 | no | no |
| v3s44 | plain | Q2_top_480-510 | pxK | path | 1.00 | 16 | 0.76 | 1.182 | 27.498 | 0.04 | no | no |
| v3s44 | rtc | Q1_top_0-30 | m | speed | 1.00 | 11 | 0.67 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s44 | rtc | Q1_top_0-30 | m | path | 1.00 | 14 | 0.75 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s44 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 10 | 0.21 | 1.000 | 100.000 | 0.01 | no | no |
| v3s44 | rtc | Q1_top_0-30 | cm | path | 1.00 | 11 | 0.71 | 1.000 | 100.000 | 0.01 | no | no |
| v3s44 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 7 | 0.61 | 3.523 | 3.600 | 0.98 | yes | yes |
| v3s44 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 12 | 0.76 | 3.542 | 3.281 | 1.08 | yes | yes |
| v3s44 | rtc | Q1_top_0-30 | U2 | speed | 0.99 | 6 | 0.59 | 3.273 | 3.281 | 1.00 | yes | yes |
| v3s44 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 8 | 0.58 | 1.000 | 1000.000 | 0.00 | no | no |
| v3s44 | rtc | Q1_top_0-30 | pxK | speed | 0.40 | 3 | -0.22 | 10.000 | 27.099 | 0.37 | no | no |
| v3s44 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 18 | 0.31 | 2.636 | 27.061 | 0.10 | no | no |
| v3s44 | rtc | Q2_top_480-510 | m | speed | 1.00 | 10 | 0.69 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s44 | rtc | Q2_top_480-510 | m | path | 1.00 | 17 | 0.79 | 1.000 | 1.000 | 1.00 | yes | yes |
| v3s44 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 8 | 0.66 | 1.000 | 100.000 | 0.01 | no | no |
| v3s44 | rtc | Q2_top_480-510 | cm | path | 1.00 | 12 | 0.78 | 1.000 | 100.000 | 0.01 | no | no |
| v3s44 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 10 | 0.66 | 3.545 | 3.600 | 0.98 | yes | yes |
| v3s44 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 13 | 0.77 | 3.632 | 3.281 | 1.11 | yes | yes |
| v3s44 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 7 | 0.68 | 3.333 | 3.281 | 1.02 | yes | yes |
| v3s44 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 10 | 0.67 | 1.000 | 1000.000 | 0.00 | no | no |
| v3s44 | rtc | Q2_top_480-510 | pxK | speed | 0.47 | 3 | 0.55 | 0.667 | 27.492 | 0.02 | no | no |
| v3s44 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 15 | -0.15 | 1.460 | 27.498 | 0.05 | no | no |
| m1s44 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 6 | 0.49 | 1.571 | 3.600 | 0.44 | no | no |
| m1s44 | plain | Q1_top_0-30 | U1 | path | 1.00 | 4 | 0.19 | 1.000 | 3.281 | 0.30 | no | no |
| m1s44 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 3 | 0.25 | 2.333 | 3.281 | 0.71 | no | no |
| m1s44 | plain | Q1_top_0-30 | U2 | path | 1.00 | 4 | 0.59 | 777.778 | 1000.000 | 0.78 | yes | yes |
| m1s44 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 3 | 0.33 | 1.833 | 27.099 | 0.07 | no | no |
| m1s44 | plain | Q1_top_0-30 | pxK | path | 1.00 | 4 | 0.24 | 68.750 | 27.061 | 2.54 | no | no |
| m1s44 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 5 | 0.62 | 1.000 | 3.600 | 0.28 | no | no |
| m1s44 | plain | Q2_top_480-510 | U1 | path | 1.00 | 6 | 0.25 | 1.273 | 3.281 | 0.39 | no | no |
| m1s44 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 5 | 0.49 | 2.333 | 3.281 | 0.71 | yes | yes |
| m1s44 | plain | Q2_top_480-510 | U2 | path | 1.00 | 5 | 0.56 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| m1s44 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 2 | 0.54 | 1.833 | 27.492 | 0.07 | no | no |
| m1s44 | plain | Q2_top_480-510 | pxK | path | 1.00 | 5 | 0.42 | 57.190 | 27.498 | 2.08 | no | no |
| m1s44 | rtc | Q1_top_0-30 | m | speed | 1.00 | 4 | 0.26 | 1.000 | 1.000 | 1.00 | no | no |
| m1s44 | rtc | Q1_top_0-30 | m | path | 1.00 | 10 | 0.67 | 1.000 | 1.000 | 1.00 | yes | yes |
| m1s44 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 2 | 0.18 | 183.333 | 100.000 | 1.83 | no | no |
| m1s44 | rtc | Q1_top_0-30 | cm | path | 1.00 | 10 | 0.43 | 1.067 | 100.000 | 0.01 | no | no |
| m1s44 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 6 | 0.32 | 5.143 | 3.600 | 1.43 | yes | yes |
| m1s44 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 4 | 0.40 | 3.250 | 3.281 | 0.99 | yes | yes |
| m1s44 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 3 | 0.24 | 6.000 | 3.281 | 1.83 | no | no |
| m1s44 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 7 | 0.64 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| m1s44 | rtc | Q1_top_0-30 | pxK | speed | 0.65 | 12 | 0.30 | 22.857 | 27.099 | 0.84 | no | no |
| m1s44 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 11 | 0.54 | 19.310 | 27.061 | 0.71 | yes | yes |
| m1s44 | rtc | Q2_top_480-510 | m | speed | 1.00 | 5 | 0.34 | 1.000 | 1.000 | 1.00 | yes | yes |
| m1s44 | rtc | Q2_top_480-510 | m | path | 1.00 | 8 | 0.64 | 1.000 | 1.000 | 1.00 | yes | yes |
| m1s44 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 4 | 0.52 | 183.333 | 100.000 | 1.83 | no | yes |
| m1s44 | rtc | Q2_top_480-510 | cm | path | 1.00 | 9 | 0.65 | 1.000 | 100.000 | 0.01 | no | no |
| m1s44 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 6 | 0.59 | 4.167 | 3.600 | 1.16 | yes | yes |
| m1s44 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 6 | 0.65 | 3.273 | 3.281 | 1.00 | yes | yes |
| m1s44 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 3 | 0.42 | 6.000 | 3.281 | 1.83 | no | yes |
| m1s44 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 9 | 0.53 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| m1s44 | rtc | Q2_top_480-510 | pxK | speed | 0.36 | 6 | 0.42 | 0.500 | 27.492 | 0.02 | no | no |
| m1s44 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 11 | 0.49 | 24.545 | 27.498 | 0.89 | yes | yes |

## internvl3_2b

- **H1_unseen_units_rtc**: fails (0)
- **H2_pixels_given_scale_rtc**: fails (0)
- **H3_speed_cm_rtc**: partial (1)
- **H4_metre_noninferiority**: fails
- **H5_motion_dependence**: holds (4)
- **H6_format_control_ivlv3**: fails (0)
- **plain_unseen_ivlv3**: fails (0)
- **H6_format_control_ivlmix**: fails (0)
- **plain_unseen_ivlmix**: fails (0)

| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts | as registered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtc_ivl | rtc | Q1_top_0-30 | m | speed | 1.00 | 5 | 0.32 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_ivl | rtc | Q1_top_0-30 | m | path | 1.00 | 5 | 0.11 | 1.000 | 1.000 | 1.00 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | cm | speed | 1.00 | 3 | 0.29 | 110.000 | 100.000 | 1.10 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | cm | path | 1.00 | 4 | 0.16 | 100.000 | 100.000 | 1.00 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 4 | 0.06 | 0.909 | 3.600 | 0.25 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | U1 | path | 1.00 | 4 | 0.26 | 1.000 | 3.281 | 0.30 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 5 | 0.28 | 1.000 | 3.281 | 0.30 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | U2 | path | 1.00 | 4 | 0.22 | 100.000 | 1000.000 | 0.10 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 2 | 0.25 | 1.100 | 27.099 | 0.04 | no | no |
| rtc_ivl | rtc | Q1_top_0-30 | pxK | path | 1.00 | 4 | 0.16 | 96.667 | 27.061 | 3.57 | no | no |
| rtc_ivl | rtc | Q2_top_480-510 | m | speed | 1.00 | 5 | 0.57 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_ivl | rtc | Q2_top_480-510 | m | path | 1.00 | 5 | 0.45 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_ivl | rtc | Q2_top_480-510 | cm | speed | 1.00 | 4 | 0.30 | 110.000 | 100.000 | 1.10 | yes | yes |
| rtc_ivl | rtc | Q2_top_480-510 | cm | path | 1.00 | 4 | 0.34 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_ivl | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 3 | -0.14 | 1.000 | 3.600 | 0.28 | no | no |
| rtc_ivl | rtc | Q2_top_480-510 | U1 | path | 1.00 | 4 | 0.49 | 1.000 | 3.281 | 0.30 | no | no |
| rtc_ivl | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 4 | 0.55 | 1.000 | 3.281 | 0.30 | no | no |
| rtc_ivl | rtc | Q2_top_480-510 | U2 | path | 1.00 | 4 | 0.39 | 100.000 | 1000.000 | 0.10 | no | no |
| rtc_ivl | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 2 | 0.21 | 1.100 | 27.492 | 0.04 | no | no |
| rtc_ivl | rtc | Q2_top_480-510 | pxK | path | 1.00 | 4 | 0.15 | 96.667 | 27.498 | 3.52 | no | no |
| ivlv3 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 8 | 0.36 | 1.375 | 3.600 | 0.38 | no | no |
| ivlv3 | plain | Q1_top_0-30 | U1 | path | 1.00 | 8 | 0.46 | 1.091 | 3.281 | 0.33 | no | no |
| ivlv3 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 8 | 0.36 | 1.200 | 3.281 | 0.37 | no | no |
| ivlv3 | plain | Q1_top_0-30 | U2 | path | 1.00 | 7 | 0.28 | 1.917 | 1000.000 | 0.00 | no | no |
| ivlv3 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 4 | 0.18 | 2.200 | 27.099 | 0.08 | no | no |
| ivlv3 | plain | Q1_top_0-30 | pxK | path | 1.00 | 10 | 0.30 | 10.273 | 27.061 | 0.38 | no | no |
| ivlv3 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 9 | 0.71 | 1.333 | 3.600 | 0.37 | no | no |
| ivlv3 | plain | Q2_top_480-510 | U1 | path | 1.00 | 8 | 0.64 | 1.091 | 3.281 | 0.33 | no | no |
| ivlv3 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 6 | 0.69 | 1.143 | 3.281 | 0.35 | no | no |
| ivlv3 | plain | Q2_top_480-510 | U2 | path | 1.00 | 6 | 0.37 | 5.136 | 1000.000 | 0.01 | no | no |
| ivlv3 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 3 | 0.52 | 3.333 | 27.492 | 0.12 | no | no |
| ivlv3 | plain | Q2_top_480-510 | pxK | path | 1.00 | 7 | 0.40 | 10.273 | 27.498 | 0.37 | no | no |
| ivlv3 | rtc | Q1_top_0-30 | m | speed | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| ivlv3 | rtc | Q1_top_0-30 | m | path | 1.00 | 10 | 0.41 | 1.000 | 1.000 | 1.00 | yes | yes |
| ivlv3 | rtc | Q1_top_0-30 | cm | speed | 0.01 | 1 | -- | -- | 100.000 | -- | no | no |
| ivlv3 | rtc | Q1_top_0-30 | cm | path | 1.00 | 8 | -0.08 | 1.000 | 100.000 | 0.01 | no | no |
| ivlv3 | rtc | Q1_top_0-30 | U1 | speed | 0.99 | 5 | 0.29 | -- | 3.600 | -- | no | no |
| ivlv3 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 9 | -0.03 | 2.769 | 3.281 | 0.84 | no | no |
| ivlv3 | rtc | Q1_top_0-30 | U2 | speed | 0.74 | 3 | 0.18 | -- | 3.281 | -- | no | no |
| ivlv3 | rtc | Q1_top_0-30 | U2 | path | 0.37 | 8 | 0.23 | 923.077 | 1000.000 | 0.92 | no | no |
| ivlv3 | rtc | Q1_top_0-30 | pxK | speed | 0.01 | 1 | -- | -- | 27.099 | -- | no | no |
| ivlv3 | rtc | Q1_top_0-30 | pxK | path | 0.37 | 3 | 0.05 | 1.000 | 27.061 | 0.04 | no | no |
| ivlv3 | rtc | Q2_top_480-510 | m | speed | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| ivlv3 | rtc | Q2_top_480-510 | m | path | 1.00 | 11 | 0.67 | 1.000 | 1.000 | 1.00 | yes | yes |
| ivlv3 | rtc | Q2_top_480-510 | cm | speed | 0.04 | 1 | -- | -- | 100.000 | -- | no | no |
| ivlv3 | rtc | Q2_top_480-510 | cm | path | 1.00 | 10 | -0.40 | 1.000 | 100.000 | 0.01 | no | no |
| ivlv3 | rtc | Q2_top_480-510 | U1 | speed | 0.96 | 4 | 0.59 | -- | 3.600 | -- | no | no |
| ivlv3 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 9 | 0.08 | 3.000 | 3.281 | 0.91 | no | no |
| ivlv3 | rtc | Q2_top_480-510 | U2 | speed | 0.81 | 2 | 0.52 | -- | 3.281 | -- | no | no |
| ivlv3 | rtc | Q2_top_480-510 | U2 | path | 0.27 | 8 | 0.18 | 550.000 | 1000.000 | 0.55 | no | no |
| ivlv3 | rtc | Q2_top_480-510 | pxK | speed | 0.04 | 1 | -- | -- | 27.492 | -- | no | no |
| ivlv3 | rtc | Q2_top_480-510 | pxK | path | 0.23 | 1 | -- | 0.917 | 27.498 | 0.03 | no | no |
| ivlmix | plain | Q1_top_0-30 | U1 | speed | 1.00 | 4 | 0.26 | 1.833 | 3.600 | 0.51 | no | no |
| ivlmix | plain | Q1_top_0-30 | U1 | path | 1.00 | 8 | 0.40 | 1.000 | 3.281 | 0.30 | no | no |
| ivlmix | plain | Q1_top_0-30 | U2 | speed | 1.00 | 6 | 0.36 | 1.167 | 3.281 | 0.36 | no | no |
| ivlmix | plain | Q1_top_0-30 | U2 | path | 1.00 | 5 | 0.47 | 109.091 | 1000.000 | 0.11 | no | no |
| ivlmix | plain | Q1_top_0-30 | pxK | speed | 1.00 | 1 | -- | 200.000 | 27.099 | 7.38 | no | no |
| ivlmix | plain | Q1_top_0-30 | pxK | path | 1.00 | 4 | 0.45 | 100.000 | 27.061 | 3.70 | no | no |
| ivlmix | plain | Q2_top_480-510 | U1 | speed | 1.00 | 4 | 0.58 | 1.571 | 3.600 | 0.44 | no | no |
| ivlmix | plain | Q2_top_480-510 | U1 | path | 1.00 | 9 | 0.69 | 1.000 | 3.281 | 0.30 | no | no |
| ivlmix | plain | Q2_top_480-510 | U2 | speed | 1.00 | 6 | 0.69 | 1.200 | 3.281 | 0.37 | no | no |
| ivlmix | plain | Q2_top_480-510 | U2 | path | 1.00 | 6 | 0.69 | 109.091 | 1000.000 | 0.11 | no | no |
| ivlmix | plain | Q2_top_480-510 | pxK | speed | 1.00 | 1 | -- | 200.000 | 27.492 | 7.27 | no | no |
| ivlmix | plain | Q2_top_480-510 | pxK | path | 1.00 | 5 | 0.38 | 100.000 | 27.498 | 3.64 | no | no |
| ivlmix | rtc | Q1_top_0-30 | m | speed | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| ivlmix | rtc | Q1_top_0-30 | m | path | 1.00 | 11 | 0.47 | 1.000 | 1.000 | 1.00 | yes | yes |
| ivlmix | rtc | Q1_top_0-30 | cm | speed | 0.94 | 5 | 0.33 | -- | 100.000 | -- | no | no |
| ivlmix | rtc | Q1_top_0-30 | cm | path | 0.56 | 6 | 0.53 | 91.667 | 100.000 | 0.92 | yes | yes |
| ivlmix | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 3 | 0.17 | -- | 3.600 | -- | no | no |
| ivlmix | rtc | Q1_top_0-30 | U1 | path | 0.99 | 12 | 0.31 | 1.000 | 3.281 | 0.30 | no | no |
| ivlmix | rtc | Q1_top_0-30 | U2 | speed | 0.03 | 1 | -- | -- | 3.281 | -- | no | no |
| ivlmix | rtc | Q1_top_0-30 | U2 | path | 0.00 | 0 | -- | -- | 1000.000 | -- | no | no |
| ivlmix | rtc | Q1_top_0-30 | pxK | speed | 0.09 | 2 | 0.48 | -- | 27.099 | -- | no | no |
| ivlmix | rtc | Q1_top_0-30 | pxK | path | 0.31 | 6 | -0.26 | 104.545 | 27.061 | 3.86 | no | no |
| ivlmix | rtc | Q2_top_480-510 | m | speed | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| ivlmix | rtc | Q2_top_480-510 | m | path | 1.00 | 12 | 0.68 | 1.000 | 1.000 | 1.00 | yes | yes |
| ivlmix | rtc | Q2_top_480-510 | cm | speed | 1.00 | 5 | 0.68 | -- | 100.000 | -- | no | no |
| ivlmix | rtc | Q2_top_480-510 | cm | path | 0.44 | 6 | 0.19 | 100.000 | 100.000 | 1.00 | no | no |
| ivlmix | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 3 | 0.18 | -- | 3.600 | -- | no | no |
| ivlmix | rtc | Q2_top_480-510 | U1 | path | 0.95 | 14 | 0.41 | 1.091 | 3.281 | 0.33 | no | no |
| ivlmix | rtc | Q2_top_480-510 | U2 | speed | 0.06 | 1 | -- | -- | 3.281 | -- | no | no |
| ivlmix | rtc | Q2_top_480-510 | U2 | path | 0.01 | 1 | -- | 95.455 | 1000.000 | 0.10 | no | no |
| ivlmix | rtc | Q2_top_480-510 | pxK | speed | 0.05 | 2 | -0.16 | -- | 27.492 | -- | no | no |
| ivlmix | rtc | Q2_top_480-510 | pxK | path | 0.10 | 6 | 0.29 | 104.545 | 27.498 | 3.80 | no | no |
