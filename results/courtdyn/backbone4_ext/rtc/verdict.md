# RTC on the A100 backbones (Pixtral-12B, Gemma3-12B) - preregistered verdicts

RTC rules `230fff51289d3c31...` + ERRATA 1-3 (band [f/1.5, 1.5f]); extension prereg `171b9dc9fb63bd85...`. Pending = not yet run.

## pixtral_12b

- **H1_unseen_units_rtc**: fails (1)
- **H2_pixels_given_scale_rtc**: fails (0)
- **H3_speed_cm_rtc**: partial (1)
- **H4_metre_noninferiority**: fails
- **H5_motion_dependence**: LIMITED (2)
- **H6_format_control_pixtral_12b_v3**: fails (0)
- **plain_unseen_pixtral_12b_v3**: fails (0)
- **H6_format_control_pixtral_12b_mix**: intermediate (5)
- **plain_unseen_pixtral_12b_mix**: fails (0)

| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts | as registered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtc_pixtral_12b | rtc | Q1_top_0-30 | m | speed | 1.00 | 5 | 0.23 | 1.000 | 1.000 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | m | path | 1.00 | 2 | 0.25 | 1.000 | 1.000 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | cm | speed | 1.00 | 5 | 0.21 | 100.000 | 100.000 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | cm | path | 1.00 | 1 | -- | 100.000 | 100.000 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 3 | 0.10 | 3.840 | 3.600 | 1.07 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | U1 | path | 1.00 | 2 | 0.29 | 3.273 | 3.281 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 5 | 0.15 | 3.280 | 3.281 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | U2 | path | 1.00 | 1 | -- | 100.000 | 1000.000 | 0.10 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 10 | 0.06 | 38.250 | 27.099 | 1.41 | no | no |
| rtc_pixtral_12b | rtc | Q1_top_0-30 | pxK | path | 1.00 | 3 | -0.07 | 27.818 | 27.061 | 1.03 | no | no |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | m | speed | 1.00 | 5 | 0.43 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | m | path | 1.00 | 2 | 0.10 | 1.000 | 1.000 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | cm | speed | 1.00 | 5 | 0.44 | 110.000 | 100.000 | 1.10 | yes | yes |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | cm | path | 1.00 | 1 | -- | 100.000 | 100.000 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 3 | 0.35 | 4.000 | 3.600 | 1.11 | yes | yes |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | U1 | path | 1.00 | 2 | 0.27 | 3.273 | 3.281 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 7 | 0.27 | 3.280 | 3.281 | 1.00 | no | no |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | U2 | path | 1.00 | 1 | -- | 100.000 | 1000.000 | 0.10 | no | no |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 11 | 0.26 | 34.000 | 27.492 | 1.24 | no | no |
| rtc_pixtral_12b | rtc | Q2_top_480-510 | pxK | path | 1.00 | 1 | -- | 27.818 | 27.498 | 1.01 | no | no |
| pixtral_12b_v3 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 5 | 0.34 | 1.000 | 3.600 | 0.28 | no | no |
| pixtral_12b_v3 | plain | Q1_top_0-30 | U1 | path | 1.00 | 9 | 0.37 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_v3 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 6 | 0.42 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_v3 | plain | Q1_top_0-30 | U2 | path | 1.00 | 6 | 0.34 | 1.000 | 1000.000 | 0.00 | no | no |
| pixtral_12b_v3 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 2 | 0.14 | 1.091 | 27.099 | 0.04 | no | no |
| pixtral_12b_v3 | plain | Q1_top_0-30 | pxK | path | 1.00 | 7 | 0.40 | 1.000 | 27.061 | 0.04 | no | no |
| pixtral_12b_v3 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 5 | 0.52 | 1.000 | 3.600 | 0.28 | no | no |
| pixtral_12b_v3 | plain | Q2_top_480-510 | U1 | path | 1.00 | 7 | 0.62 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_v3 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 6 | 0.58 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_v3 | plain | Q2_top_480-510 | U2 | path | 1.00 | 6 | 0.61 | 1.000 | 1000.000 | 0.00 | no | no |
| pixtral_12b_v3 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 2 | 0.22 | 1.375 | 27.492 | 0.05 | no | no |
| pixtral_12b_v3 | plain | Q2_top_480-510 | pxK | path | 1.00 | 8 | 0.65 | 1.000 | 27.498 | 0.04 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | m | speed | 1.00 | 2 | 0.01 | 1.000 | 1.000 | 1.00 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | m | path | 1.00 | 2 | 0.18 | 1.000 | 1.000 | 1.00 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 2 | -0.02 | 1.000 | 100.000 | 0.01 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | cm | path | 1.00 | 2 | 0.18 | 1.000 | 100.000 | 0.01 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 2 | 0.06 | 3.636 | 3.600 | 1.01 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 2 | 0.15 | 3.273 | 3.281 | 1.00 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 2 | -0.02 | 1.091 | 3.281 | 0.33 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 2 | 0.16 | 1.000 | 1000.000 | 0.00 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 3 | 0.03 | 2.455 | 27.099 | 0.09 | no | no |
| pixtral_12b_v3 | rtc | Q1_top_0-30 | pxK | path | 0.94 | 9 | 0.31 | 28.545 | 27.061 | 1.05 | yes | yes |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | m | speed | 1.00 | 2 | 0.01 | 1.000 | 1.000 | 1.00 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | m | path | 1.00 | 2 | 0.35 | 1.000 | 1.000 | 1.00 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 1 | -- | 1.000 | 100.000 | 0.01 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | cm | path | 1.00 | 2 | 0.20 | 1.000 | 100.000 | 0.01 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 2 | -0.03 | 3.636 | 3.600 | 1.01 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 2 | 0.42 | 3.273 | 3.281 | 1.00 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 1 | -- | 1.091 | 3.281 | 0.33 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 2 | 0.34 | 1.000 | 1000.000 | 0.00 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 2 | 0.14 | 2.455 | 27.492 | 0.09 | no | no |
| pixtral_12b_v3 | rtc | Q2_top_480-510 | pxK | path | 0.94 | 8 | 0.49 | 28.545 | 27.498 | 1.04 | yes | yes |
| pixtral_12b_mix | plain | Q1_top_0-30 | U1 | speed | 1.00 | 2 | 0.13 | 1.000 | 3.600 | 0.28 | no | no |
| pixtral_12b_mix | plain | Q1_top_0-30 | U1 | path | 1.00 | 4 | 0.28 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_mix | plain | Q1_top_0-30 | U2 | speed | 1.00 | 5 | 0.05 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_mix | plain | Q1_top_0-30 | U2 | path | 1.00 | 3 | 0.27 | 100.000 | 1000.000 | 0.10 | no | no |
| pixtral_12b_mix | plain | Q1_top_0-30 | pxK | speed | 1.00 | 2 | 0.20 | 1.429 | 27.099 | 0.05 | no | no |
| pixtral_12b_mix | plain | Q1_top_0-30 | pxK | path | 1.00 | 3 | 0.31 | 90.909 | 27.061 | 3.36 | no | no |
| pixtral_12b_mix | plain | Q2_top_480-510 | U1 | speed | 1.00 | 2 | 0.41 | 1.000 | 3.600 | 0.28 | no | no |
| pixtral_12b_mix | plain | Q2_top_480-510 | U1 | path | 1.00 | 4 | 0.56 | 1.042 | 3.281 | 0.32 | no | no |
| pixtral_12b_mix | plain | Q2_top_480-510 | U2 | speed | 1.00 | 5 | 0.14 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_mix | plain | Q2_top_480-510 | U2 | path | 1.00 | 3 | 0.29 | 109.091 | 1000.000 | 0.11 | no | no |
| pixtral_12b_mix | plain | Q2_top_480-510 | pxK | speed | 1.00 | 4 | 0.25 | 1.429 | 27.492 | 0.05 | no | no |
| pixtral_12b_mix | plain | Q2_top_480-510 | pxK | path | 1.00 | 3 | 0.22 | 90.909 | 27.498 | 3.31 | no | no |
| pixtral_12b_mix | rtc | Q1_top_0-30 | m | speed | 1.00 | 2 | -0.09 | 1.000 | 1.000 | 1.00 | no | no |
| pixtral_12b_mix | rtc | Q1_top_0-30 | m | path | 1.00 | 6 | 0.43 | 1.000 | 1.000 | 1.00 | yes | yes |
| pixtral_12b_mix | rtc | Q1_top_0-30 | cm | speed | 1.00 | 3 | 0.11 | 100.000 | 100.000 | 1.00 | no | no |
| pixtral_12b_mix | rtc | Q1_top_0-30 | cm | path | 1.00 | 8 | 0.41 | 100.000 | 100.000 | 1.00 | yes | yes |
| pixtral_12b_mix | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 3 | 0.16 | 3.571 | 3.600 | 0.99 | no | no |
| pixtral_12b_mix | rtc | Q1_top_0-30 | U1 | path | 1.00 | 6 | 0.41 | 3.273 | 3.281 | 1.00 | yes | yes |
| pixtral_12b_mix | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.12 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_mix | rtc | Q1_top_0-30 | U2 | path | 1.00 | 8 | 0.45 | 916.667 | 1000.000 | 0.92 | yes | yes |
| pixtral_12b_mix | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 6 | 0.25 | 25.714 | 27.099 | 0.95 | no | no |
| pixtral_12b_mix | rtc | Q1_top_0-30 | pxK | path | 0.66 | 10 | 0.21 | 27.000 | 27.061 | 1.00 | no | no |
| pixtral_12b_mix | rtc | Q2_top_480-510 | m | speed | 1.00 | 3 | 0.18 | 1.000 | 1.000 | 1.00 | no | no |
| pixtral_12b_mix | rtc | Q2_top_480-510 | m | path | 1.00 | 7 | 0.67 | 1.000 | 1.000 | 1.00 | yes | yes |
| pixtral_12b_mix | rtc | Q2_top_480-510 | cm | speed | 1.00 | 4 | 0.53 | 100.000 | 100.000 | 1.00 | yes | yes |
| pixtral_12b_mix | rtc | Q2_top_480-510 | cm | path | 1.00 | 7 | 0.67 | 100.000 | 100.000 | 1.00 | yes | yes |
| pixtral_12b_mix | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 3 | 0.52 | 3.571 | 3.600 | 0.99 | yes | yes |
| pixtral_12b_mix | rtc | Q2_top_480-510 | U1 | path | 1.00 | 7 | 0.63 | 3.286 | 3.281 | 1.00 | yes | yes |
| pixtral_12b_mix | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 4 | 0.52 | 1.000 | 3.281 | 0.30 | no | no |
| pixtral_12b_mix | rtc | Q2_top_480-510 | U2 | path | 1.00 | 7 | 0.67 | 916.667 | 1000.000 | 0.92 | yes | yes |
| pixtral_12b_mix | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 6 | 0.41 | 250.000 | 27.492 | 9.09 | no | no |
| pixtral_12b_mix | rtc | Q2_top_480-510 | pxK | path | 0.66 | 9 | 0.17 | 27.000 | 27.498 | 0.98 | no | no |

## gemma3_12b

- **H1_unseen_units_rtc**: intermediate (3)
- **H2_pixels_given_scale_rtc**: intermediate (2)
- **H3_speed_cm_rtc**: partial (1)
- **H4_metre_noninferiority**: holds
- **H5_motion_dependence**: holds (4)
- **H6_format_control_gemma3_12b_v3**: fails (0)
- **plain_unseen_gemma3_12b_v3**: fails (1)
- **H6_format_control_gemma3_12b_mix**: fails (0)
- **plain_unseen_gemma3_12b_mix**: fails (0)

| model | format | clip | arm | family | parse | distinct | rho | R | factor | R/f | converts | as registered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtc_gemma3_12b | rtc | Q1_top_0-30 | m | speed | 1.00 | 5 | 0.16 | 1.000 | 1.000 | 1.00 | no | no |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | m | path | 1.00 | 4 | 0.30 | 1.000 | 1.000 | 1.00 | no | no |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | cm | speed | 1.00 | 3 | 0.23 | 100.000 | 100.000 | 1.00 | no | no |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | cm | path | 1.00 | 4 | 0.35 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 3 | 0.15 | 1.000 | 3.600 | 0.28 | no | no |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | U1 | path | 1.00 | 4 | 0.28 | 3.267 | 3.281 | 1.00 | no | no |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.10 | 2.375 | 3.281 | 0.72 | no | no |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | U2 | path | 1.00 | 3 | 0.32 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 3 | 0.10 | 27.000 | 27.099 | 1.00 | no | no |
| rtc_gemma3_12b | rtc | Q1_top_0-30 | pxK | path | 1.00 | 5 | 0.24 | 27.000 | 27.061 | 1.00 | no | no |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | m | speed | 1.00 | 5 | 0.43 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | m | path | 1.00 | 4 | 0.39 | 1.000 | 1.000 | 1.00 | yes | yes |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | cm | speed | 1.00 | 3 | 0.37 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | cm | path | 1.00 | 4 | 0.34 | 100.000 | 100.000 | 1.00 | yes | yes |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 3 | 0.36 | 1.000 | 3.600 | 0.28 | no | no |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | U1 | path | 1.00 | 4 | 0.24 | 3.267 | 3.281 | 1.00 | no | no |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 3 | 0.42 | 2.375 | 3.281 | 0.72 | yes | yes |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | U2 | path | 1.00 | 4 | 0.31 | 1000.000 | 1000.000 | 1.00 | yes | yes |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 5 | 0.45 | 27.000 | 27.492 | 0.98 | yes | yes |
| rtc_gemma3_12b | rtc | Q2_top_480-510 | pxK | path | 1.00 | 6 | 0.35 | 27.000 | 27.498 | 0.98 | yes | yes |
| gemma3_12b_v3 | plain | Q1_top_0-30 | U1 | speed | 1.00 | 3 | 0.27 | 1.333 | 3.600 | 0.37 | no | no |
| gemma3_12b_v3 | plain | Q1_top_0-30 | U1 | path | 1.00 | 9 | 0.40 | 1.500 | 3.281 | 0.46 | no | no |
| gemma3_12b_v3 | plain | Q1_top_0-30 | U2 | speed | 1.00 | 6 | 0.28 | 1.625 | 3.281 | 0.50 | no | no |
| gemma3_12b_v3 | plain | Q1_top_0-30 | U2 | path | 1.00 | 43 | 0.41 | 1019.562 | 1000.000 | 1.02 | yes | yes |
| gemma3_12b_v3 | plain | Q1_top_0-30 | pxK | speed | 1.00 | 7 | 0.26 | 10.182 | 27.099 | 0.38 | no | no |
| gemma3_12b_v3 | plain | Q1_top_0-30 | pxK | path | 1.00 | 15 | 0.25 | 74.067 | 27.061 | 2.74 | no | no |
| gemma3_12b_v3 | plain | Q2_top_480-510 | U1 | speed | 1.00 | 4 | 0.30 | 1.375 | 3.600 | 0.38 | no | no |
| gemma3_12b_v3 | plain | Q2_top_480-510 | U1 | path | 1.00 | 8 | 0.29 | 1.500 | 3.281 | 0.46 | no | no |
| gemma3_12b_v3 | plain | Q2_top_480-510 | U2 | speed | 1.00 | 7 | 0.33 | 1.625 | 3.281 | 0.50 | no | no |
| gemma3_12b_v3 | plain | Q2_top_480-510 | U2 | path | 1.00 | 42 | 0.17 | 897.278 | 1000.000 | 0.90 | no | no |
| gemma3_12b_v3 | plain | Q2_top_480-510 | pxK | speed | 1.00 | 6 | 0.41 | 10.182 | 27.492 | 0.37 | no | no |
| gemma3_12b_v3 | plain | Q2_top_480-510 | pxK | path | 1.00 | 13 | 0.04 | 84.692 | 27.498 | 3.08 | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | m | speed | 1.00 | 6 | 0.24 | 1.000 | 1.000 | 1.00 | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | m | path | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | cm | speed | 1.00 | 13 | 0.24 | 40.000 | 100.000 | 0.40 | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | cm | path | 1.00 | 9 | 0.38 | -- | 100.000 | -- | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 4 | 0.29 | 1.611 | 3.600 | 0.45 | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | U1 | path | 1.00 | 6 | 0.37 | -- | 3.281 | -- | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 4 | 0.25 | 1.440 | 3.281 | 0.44 | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | U2 | path | 1.00 | 7 | 0.47 | -- | 1000.000 | -- | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 7 | 0.06 | 11.920 | 27.099 | 0.44 | no | no |
| gemma3_12b_v3 | rtc | Q1_top_0-30 | pxK | path | 1.00 | 7 | 0.41 | -- | 27.061 | -- | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | m | speed | 1.00 | 7 | 0.20 | 1.000 | 1.000 | 1.00 | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | m | path | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | cm | speed | 1.00 | 13 | 0.26 | 33.333 | 100.000 | 0.33 | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | cm | path | 1.00 | 10 | 0.33 | -- | 100.000 | -- | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 4 | 0.25 | 1.600 | 3.600 | 0.44 | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | U1 | path | 1.00 | 6 | 0.34 | -- | 3.281 | -- | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 4 | 0.20 | 1.440 | 3.281 | 0.44 | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | U2 | path | 1.00 | 6 | 0.30 | -- | 1000.000 | -- | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 6 | 0.09 | 11.920 | 27.492 | 0.43 | no | no |
| gemma3_12b_v3 | rtc | Q2_top_480-510 | pxK | path | 1.00 | 6 | 0.43 | -- | 27.498 | -- | no | no |
| gemma3_12b_mix | plain | Q1_top_0-30 | U1 | speed | 1.00 | 7 | 0.25 | 1.125 | 3.600 | 0.31 | no | no |
| gemma3_12b_mix | plain | Q1_top_0-30 | U1 | path | 1.00 | 10 | 0.47 | 5.000 | 3.281 | 1.52 | no | yes |
| gemma3_12b_mix | plain | Q1_top_0-30 | U2 | speed | 1.00 | 9 | 0.16 | 1.625 | 3.281 | 0.50 | no | no |
| gemma3_12b_mix | plain | Q1_top_0-30 | U2 | path | 1.00 | 7 | 0.43 | 147.321 | 1000.000 | 0.15 | no | no |
| gemma3_12b_mix | plain | Q1_top_0-30 | pxK | speed | 1.00 | 3 | -0.02 | 122.222 | 27.099 | 4.51 | no | no |
| gemma3_12b_mix | plain | Q1_top_0-30 | pxK | path | 1.00 | 3 | 0.36 | 84.615 | 27.061 | 3.13 | no | no |
| gemma3_12b_mix | plain | Q2_top_480-510 | U1 | speed | 1.00 | 7 | 0.30 | 1.000 | 3.600 | 0.28 | no | no |
| gemma3_12b_mix | plain | Q2_top_480-510 | U1 | path | 1.00 | 11 | 0.24 | 1.938 | 3.281 | 0.59 | no | no |
| gemma3_12b_mix | plain | Q2_top_480-510 | U2 | speed | 1.00 | 11 | 0.24 | 1.625 | 3.281 | 0.50 | no | no |
| gemma3_12b_mix | plain | Q2_top_480-510 | U2 | path | 1.00 | 8 | 0.34 | 100.000 | 1000.000 | 0.10 | no | no |
| gemma3_12b_mix | plain | Q2_top_480-510 | pxK | speed | 1.00 | 2 | 0.14 | 122.222 | 27.492 | 4.45 | no | no |
| gemma3_12b_mix | plain | Q2_top_480-510 | pxK | path | 1.00 | 4 | 0.08 | 84.615 | 27.498 | 3.08 | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | m | speed | 1.00 | 8 | 0.25 | 1.000 | 1.000 | 1.00 | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | m | path | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | cm | speed | 1.00 | 6 | 0.19 | 33.333 | 100.000 | 0.33 | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | cm | path | 1.00 | 7 | 0.44 | -- | 100.000 | -- | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | U1 | speed | 1.00 | 6 | 0.18 | 1.600 | 3.600 | 0.44 | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | U1 | path | 1.00 | 10 | 0.34 | -- | 3.281 | -- | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | U2 | speed | 1.00 | 5 | 0.17 | 1.000 | 3.281 | 0.30 | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | U2 | path | 1.00 | 7 | 0.33 | -- | 1000.000 | -- | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | pxK | speed | 1.00 | 10 | 0.09 | 8.100 | 27.099 | 0.30 | no | no |
| gemma3_12b_mix | rtc | Q1_top_0-30 | pxK | path | 1.00 | 9 | 0.31 | -- | 27.061 | -- | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | m | speed | 1.00 | 8 | 0.36 | 1.000 | 1.000 | 1.00 | yes | yes |
| gemma3_12b_mix | rtc | Q2_top_480-510 | m | path | 0.00 | 0 | -- | -- | 1.000 | -- | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | cm | speed | 1.00 | 7 | 0.17 | 36.000 | 100.000 | 0.36 | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | cm | path | 1.00 | 8 | 0.24 | -- | 100.000 | -- | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | U1 | speed | 1.00 | 7 | 0.27 | 1.600 | 3.600 | 0.44 | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | U1 | path | 1.00 | 8 | 0.39 | -- | 3.281 | -- | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | U2 | speed | 1.00 | 6 | 0.32 | 1.000 | 3.281 | 0.30 | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | U2 | path | 1.00 | 8 | 0.25 | -- | 1000.000 | -- | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | pxK | speed | 1.00 | 9 | 0.08 | 8.100 | 27.492 | 0.29 | no | no |
| gemma3_12b_mix | rtc | Q2_top_480-510 | pxK | path | 1.00 | 6 | 0.26 | -- | 27.498 | -- | no | no |
