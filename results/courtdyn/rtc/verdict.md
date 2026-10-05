# RTC (read-then-convert) - preregistered verdicts

Rules sha256 `230fff51289d3c31...`. Pending = inputs not yet run.

- **H1_unseen_units_rtc**: holds (8)
- **H2_pixels_given_scale_rtc**: intermediate (2)
- **H3_speed_cm_rtc**: fixed (2)
- **H4_metre_noninferiority**: fails
- **H5_motion_dependence**: LIMITED (2)
- **H6_format_control_v3s42**: holds (6)
- **plain_unseen_v3s42**: fails (0)
- **H6_format_control_m1s42**: holds (6)
- **plain_unseen_m1s42**: fails (2)
- **plain_unseen_m1fines42**: intermediate (4)

| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rtc_s42 | rtc | Q1_top_0-30 | m | speed | 1.00 | 5 | 0.59 | 1.000 | 1.000 | 1.00 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | m | path | 1.00 | 6 | 0.65 | 1.000 | 1.000 | 1.00 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 5 | 0.46 | 85.714 | 100.000 | 0.86 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | cm | path | 1.00 | 8 | 0.60 | 100.000 | 100.000 | 1.00 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 4 | 0.38 | 3.571 | 3.600 | 0.99 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 10 | 0.64 | 4.312 | 3.281 | 1.31 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.41 | 3.833 | 3.281 | 1.17 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 6 | 0.58 | 1000.000 | 1000.000 | 1.00 | yes |
| rtc_s42 | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 6 | 0.44 | 18.000 | 27.099 | 0.66 | no |
| rtc_s42 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 9 | 0.65 | 27.000 | 27.061 | 1.00 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | m | speed | 1.00 | 6 | 0.61 | 1.000 | 1.000 | 1.00 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | m | path | 1.00 | 6 | 0.70 | 1.000 | 1.000 | 1.00 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 5 | 0.59 | 100.000 | 100.000 | 1.00 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | cm | path | 1.00 | 9 | 0.60 | 100.000 | 100.000 | 1.00 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 4 | 0.46 | 3.667 | 3.600 | 1.02 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 11 | 0.64 | 4.312 | 3.281 | 1.31 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 4 | 0.56 | 4.333 | 3.281 | 1.32 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 9 | 0.64 | 1000.000 | 1000.000 | 1.00 | yes |
| rtc_s42 | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 8 | 0.01 | 17.682 | 27.492 | 0.64 | no |
| rtc_s42 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 8 | 0.57 | 27.000 | 27.498 | 0.98 | yes |
| v3s42 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 10 | 0.65 | 1.455 | 3.600 | 0.40 | no |
| v3s42 | plain | Q1_top_0-30 | U1 | path | 1.00 | 11 | 0.37 | 1.000 | 3.281 | 0.30 | no |
| v3s42 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 11 | 0.62 | 1.600 | 3.281 | 0.49 | no |
| v3s42 | plain | Q1_top_0-30 | U2 | path | 1.00 | 9 | 0.74 | 0.941 | 1000.000 | 0.00 | no |
| v3s42 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 7 | 0.63 | 1.571 | 27.099 | 0.06 | no |
| v3s42 | plain | Q1_top_0-30 | pxK | path | 1.00 | 5 | 0.47 | 1.000 | 27.061 | 0.04 | no |
| v3s42 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 11 | 0.73 | 1.375 | 3.600 | 0.38 | no |
| v3s42 | plain | Q2_top_480-510 | U1 | path | 1.00 | 13 | 0.64 | 1.235 | 3.281 | 0.38 | no |
| v3s42 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 12 | 0.76 | 1.955 | 3.281 | 0.60 | no |
| v3s42 | plain | Q2_top_480-510 | U2 | path | 1.00 | 7 | 0.68 | 1.000 | 1000.000 | 0.00 | no |
| v3s42 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 13 | 0.70 | 1.833 | 27.492 | 0.07 | no |
| v3s42 | plain | Q2_top_480-510 | pxK | path | 1.00 | 7 | 0.61 | 1.455 | 27.498 | 0.05 | no |
| v3s42 | rtc | Q1_top_0-30 | m | speed | 1.00 | 6 | 0.64 | 1.000 | 1.000 | 1.00 | yes |
| v3s42 | rtc | Q1_top_0-30 | m | path | 1.00 | 13 | 0.75 | 1.000 | 1.000 | 1.00 | yes |
| v3s42 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 15 | -0.06 | 8.571 | 100.000 | 0.09 | no |
| v3s42 | rtc | Q1_top_0-30 | cm | path | 1.00 | 14 | 0.77 | 1.000 | 100.000 | 0.01 | no |
| v3s42 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 10 | 0.65 | 3.273 | 3.600 | 0.91 | yes |
| v3s42 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 12 | 0.69 | 3.250 | 3.281 | 0.99 | yes |
| v3s42 | rtc | Q1_top_0-30 | U2 | speed | 0.67 | 6 | 0.44 | 3.279 | 3.281 | 1.00 | yes |
| v3s42 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 12 | 0.43 | 1.000 | 1000.000 | 0.00 | no |
| v3s42 | rtc | Q1_top_0-30 | pxK | speed | 0.97 | 7 | 0.36 | 2.200 | 27.099 | 0.08 | no |
| v3s42 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 22 | -0.07 | 1.957 | 27.061 | 0.07 | no |
| v3s42 | rtc | Q2_top_480-510 | m | speed | 1.00 | 8 | 0.74 | 1.000 | 1.000 | 1.00 | yes |
| v3s42 | rtc | Q2_top_480-510 | m | path | 1.00 | 14 | 0.78 | 1.000 | 1.000 | 1.00 | yes |
| v3s42 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 15 | 0.43 | 1.000 | 100.000 | 0.01 | no |
| v3s42 | rtc | Q2_top_480-510 | cm | path | 1.00 | 14 | 0.77 | 1.000 | 100.000 | 0.01 | no |
| v3s42 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 9 | 0.73 | 3.500 | 3.600 | 0.97 | yes |
| v3s42 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 10 | 0.76 | 3.273 | 3.281 | 1.00 | yes |
| v3s42 | rtc | Q2_top_480-510 | U2 | speed | 0.61 | 8 | 0.66 | 3.600 | 3.281 | 1.10 | yes |
| v3s42 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 18 | 0.61 | 1.000 | 1000.000 | 0.00 | no |
| v3s42 | rtc | Q2_top_480-510 | pxK | speed | 0.91 | 14 | 0.58 | 1.513 | 27.492 | 0.06 | no |
| v3s42 | rtc | Q2_top_480-510 | pxK | path | 0.99 | 15 | 0.24 | 1.000 | 27.498 | 0.04 | no |
| m1s42 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 4 | 0.46 | 1.500 | 3.600 | 0.42 | no |
| m1s42 | plain | Q1_top_0-30 | U1 | path | 1.00 | 7 | 0.42 | 1.165 | 3.281 | 0.36 | no |
| m1s42 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 5 | 0.27 | 2.667 | 3.281 | 0.81 | no |
| m1s42 | plain | Q1_top_0-30 | U2 | path | 1.00 | 6 | 0.63 | 687.500 | 1000.000 | 0.69 | yes |
| m1s42 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 8 | 0.15 | 3.100 | 27.099 | 0.11 | no |
| m1s42 | plain | Q1_top_0-30 | pxK | path | 1.00 | 3 | -0.02 | 68.750 | 27.061 | 2.54 | no |
| m1s42 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 5 | 0.65 | 1.500 | 3.600 | 0.42 | no |
| m1s42 | plain | Q2_top_480-510 | U1 | path | 1.00 | 5 | 0.56 | 1.357 | 3.281 | 0.41 | no |
| m1s42 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 6 | 0.55 | 3.300 | 3.281 | 1.01 | yes |
| m1s42 | plain | Q2_top_480-510 | U2 | path | 1.00 | 7 | 0.65 | 647.059 | 1000.000 | 0.65 | no |
| m1s42 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 9 | 0.16 | 2.750 | 27.492 | 0.10 | no |
| m1s42 | plain | Q2_top_480-510 | pxK | path | 1.00 | 6 | 0.09 | 84.615 | 27.498 | 3.08 | no |
| m1s42 | rtc | Q1_top_0-30 | m | speed | 1.00 | 7 | 0.46 | 1.000 | 1.000 | 1.00 | yes |
| m1s42 | rtc | Q1_top_0-30 | m | path | 1.00 | 12 | 0.73 | 1.000 | 1.000 | 1.00 | yes |
| m1s42 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 5 | 0.41 | 133.333 | 100.000 | 1.33 | yes |
| m1s42 | rtc | Q1_top_0-30 | cm | path | 1.00 | 14 | -0.04 | 100.000 | 100.000 | 1.00 | no |
| m1s42 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 6 | 0.40 | 4.833 | 3.600 | 1.34 | yes |
| m1s42 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 8 | 0.64 | 3.250 | 3.281 | 0.99 | yes |
| m1s42 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.33 | 6.000 | 3.281 | 1.83 | no |
| m1s42 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 11 | 0.72 | 1000.000 | 1000.000 | 1.00 | yes |
| m1s42 | rtc | Q1_top_0-30 | pxK | speed | 0.96 | 13 | 0.44 | 18.667 | 27.099 | 0.69 | yes |
| m1s42 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 18 | 0.73 | 21.875 | 27.061 | 0.81 | yes |
| m1s42 | rtc | Q2_top_480-510 | m | speed | 1.00 | 7 | 0.62 | 1.000 | 1.000 | 1.00 | yes |
| m1s42 | rtc | Q2_top_480-510 | m | path | 1.00 | 12 | 0.70 | 1.000 | 1.000 | 1.00 | yes |
| m1s42 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 8 | 0.61 | 150.000 | 100.000 | 1.50 | yes |
| m1s42 | rtc | Q2_top_480-510 | cm | path | 1.00 | 14 | 0.18 | 107.179 | 100.000 | 1.07 | no |
| m1s42 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 6 | 0.61 | 5.250 | 3.600 | 1.46 | yes |
| m1s42 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 10 | 0.65 | 3.273 | 3.281 | 1.00 | yes |
| m1s42 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 3 | 0.63 | 6.000 | 3.281 | 1.83 | no |
| m1s42 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 11 | 0.68 | 1000.000 | 1000.000 | 1.00 | yes |
| m1s42 | rtc | Q2_top_480-510 | pxK | speed | 0.79 | 13 | 0.42 | 2.667 | 27.492 | 0.10 | no |
| m1s42 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 23 | 0.49 | 22.500 | 27.498 | 0.82 | yes |
| m1fines42 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 5 | 0.38 | 1.833 | 3.600 | 0.51 | no |
| m1fines42 | plain | Q1_top_0-30 | U1 | path | 1.00 | 5 | 0.60 | 1.455 | 3.281 | 0.44 | no |
| m1fines42 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.35 | 2.714 | 3.281 | 0.83 | yes |
| m1fines42 | plain | Q1_top_0-30 | U2 | path | 1.00 | 11 | 0.41 | 687.500 | 1000.000 | 0.69 | yes |
| m1fines42 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 7 | 0.08 | 2.333 | 27.099 | 0.09 | no |
| m1fines42 | plain | Q1_top_0-30 | pxK | path | 1.00 | 3 | 0.36 | 67.500 | 27.061 | 2.49 | no |
| m1fines42 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 5 | 0.61 | 1.833 | 3.600 | 0.51 | no |
| m1fines42 | plain | Q2_top_480-510 | U1 | path | 1.00 | 7 | 0.73 | 1.455 | 3.281 | 0.44 | no |
| m1fines42 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 6 | 0.53 | 2.750 | 3.281 | 0.84 | yes |
| m1fines42 | plain | Q2_top_480-510 | U2 | path | 1.00 | 15 | 0.52 | 761.905 | 1000.000 | 0.76 | yes |
| m1fines42 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 5 | -0.11 | 2.333 | 27.492 | 0.08 | no |
| m1fines42 | plain | Q2_top_480-510 | pxK | path | 1.00 | 2 | 0.44 | 67.500 | 27.498 | 2.45 | no |
