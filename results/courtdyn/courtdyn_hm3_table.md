# CourtDyn · 第二场地 (Human-M3) 与多相机视角判决

生成 2026-09-07 19:39:16 · `tools/summarize_courtdyn_hm3.py`。
GT 是 Human-M3 自带的**真三维世界坐标** (关节 XY 中位数的逐帧累计位移), 既不用身高尺也不用球场单应 —— 因此这里**没有 v1/v3 那道 2 倍歧义**。
同一序列的多台相机拍的是**同一段世界运动**, GT 逐字相同, 只有相机几何与像素不同。

## A. 逐相机主表

| 序列 | 相机 | 模型 | 家族 | ρ(预测, 世界) | T-MRA | 常数中位数 | Δ vs 常数 | 预测中位 | n |
|---|---|---|---|---|---|---|---|---|---|
| basketball1/split1 | cam 0 | source-SFT | speed | 0.18 | 38.5 | 77.9 | -39.4 | 0.3 | 117 |
| basketball1/split1 | cam 0 | source-SFT | path | 0.18 | 51.2 | 74.8 | -23.6 | 1.0 | 117 |
| basketball1/split1 | cam 0 | GRPO-v3 | speed | 0.17 | 63.9 | 77.9 | -13.9 | 0.6 | 117 |
| basketball1/split1 | cam 0 | GRPO-v3 | path | 0.24 | 59.6 | 74.8 | -15.2 | 3.0 | 117 |
| basketball1/split1 | cam 0 | base 4B | speed | 0.30 | 18.8 | 77.9 | -59.1 | 0.1 | 117 |
| basketball1/split1 | cam 0 | base 4B | path | 0.39 | 11.3 | 74.8 | -63.5 | 0.1 | 117 |
| basketball1/split1 | cam 0 | Qwen2.5-VL-7B | speed | 0.29 | 47.0 | 77.9 | -30.9 | 1.5 | 117 |
| basketball1/split1 | cam 0 | Qwen2.5-VL-7B | path | 0.15 | 14.5 | 74.8 | -60.3 | 5.6 | 117 |
| basketball1/split1 | cam 1 | source-SFT | speed | -0.00 | 63.5 | 77.8 | -14.3 | 0.6 | 115 |
| basketball1/split1 | cam 1 | source-SFT | path | 0.21 | 70.3 | 74.8 | -4.4 | 2.0 | 115 |
| basketball1/split1 | cam 1 | GRPO-v3 | speed | 0.03 | 57.2 | 77.8 | -20.6 | 1.3 | 115 |
| basketball1/split1 | cam 1 | GRPO-v3 | path | 0.20 | 21.7 | 74.8 | -53.0 | 4.9 | 115 |
| basketball1/split1 | cam 2 | source-SFT | speed | 0.18 | 51.3 | 77.9 | -26.6 | 0.4 | 117 |
| basketball1/split1 | cam 2 | source-SFT | path | 0.34 | 66.5 | 74.8 | -8.3 | 1.4 | 117 |
| basketball1/split1 | cam 2 | GRPO-v3 | speed | 0.33 | 76.0 | 77.9 | -1.9 | 1.1 | 117 |
| basketball1/split1 | cam 2 | GRPO-v3 | path | 0.40 | 49.3 | 74.8 | -25.5 | 3.1 | 117 |
| basketball1/split1 | cam 3 | source-SFT | speed | 0.33 | 39.7 | 77.9 | -38.2 | 0.3 | 117 |
| basketball1/split1 | cam 3 | source-SFT | path | 0.25 | 43.3 | 74.8 | -31.5 | 1.0 | 117 |
| basketball1/split1 | cam 3 | GRPO-v3 | speed | 0.34 | 62.4 | 77.9 | -15.5 | 0.6 | 117 |
| basketball1/split1 | cam 3 | GRPO-v3 | path | 0.25 | 60.8 | 74.8 | -14.0 | 2.1 | 117 |
| basketball2 | cam 0 | source-SFT | speed | 0.48 | 37.4 | 60.1 | -22.7 | 0.4 | 140 |
| basketball2 | cam 0 | source-SFT | path | 0.50 | 48.0 | 54.2 | -6.2 | 1.2 | 140 |
| basketball2 | cam 0 | GRPO-v3 | speed | 0.44 | 57.7 | 60.1 | -2.4 | 1.0 | 140 |
| basketball2 | cam 0 | GRPO-v3 | path | 0.46 | 57.1 | 54.2 | 2.9 | 3.1 | 140 |
| basketball2 | cam 2 | source-SFT | speed | 0.25 | 39.9 | 60.1 | -20.2 | 0.4 | 140 |
| basketball2 | cam 2 | source-SFT | path | 0.34 | 44.6 | 54.2 | -9.6 | 1.0 | 140 |
| basketball2 | cam 2 | GRPO-v3 | speed | 0.35 | 56.1 | 60.1 | -4.0 | 0.9 | 140 |
| basketball2 | cam 2 | GRPO-v3 | path | 0.36 | 54.9 | 54.2 | 0.6 | 3.1 | 140 |

## B. 视角判决 · 同一段世界运动, 只换相机

| 序列 | 家族 | 相机 | ρ(像素位移, 世界位移) | source-SFT ρ | GRPO ρ | 世界 GT 中位 |
|---|---|---|---|---|---|---|
| basketball1/split1 | speed | cam 0 | 0.65 | 0.18 | 0.17 | 1.10 |
| basketball1/split1 | speed | cam 1 | 0.21 | -0.00 | 0.03 | 1.10 |
| basketball1/split1 | speed | cam 2 | 0.49 | 0.18 | 0.33 | 1.10 |
| basketball1/split1 | speed | cam 3 | 0.53 | 0.33 | 0.34 | 1.10 |
| basketball1/split1 | path | cam 0 | 0.65 | 0.18 | 0.24 | 2.10 |
| basketball1/split1 | path | cam 1 | 0.21 | 0.21 | 0.20 | 2.10 |
| basketball1/split1 | path | cam 2 | 0.49 | 0.34 | 0.40 | 2.10 |
| basketball1/split1 | path | cam 3 | 0.54 | 0.25 | 0.25 | 2.10 |
| basketball1/split2 | speed | cam 0 | 0.55 | — | — | 1.20 |
| basketball1/split2 | speed | cam 1 | 0.58 | — | — | 1.20 |
| basketball1/split2 | speed | cam 2 | 0.69 | — | — | 1.20 |
| basketball1/split2 | speed | cam 3 | 0.71 | — | — | 1.20 |
| basketball1/split2 | path | cam 0 | 0.55 | — | — | 2.40 |
| basketball1/split2 | path | cam 1 | 0.58 | — | — | 2.40 |
| basketball1/split2 | path | cam 2 | 0.70 | — | — | 2.40 |
| basketball1/split2 | path | cam 3 | 0.71 | — | — | 2.40 |
| basketball2 | speed | cam 0 | 0.50 | 0.48 | 0.44 | 1.20 |
| basketball2 | speed | cam 1 | 0.78 | — | — | 1.20 |
| basketball2 | speed | cam 2 | 0.54 | 0.25 | 0.35 | 1.20 |
| basketball2 | path | cam 0 | 0.50 | 0.50 | 0.46 | 2.50 |
| basketball2 | path | cam 1 | 0.78 | — | — | 2.50 |
| basketball2 | path | cam 2 | 0.54 | 0.34 | 0.36 | 2.50 |

## C. 首帧×4 对照 (第二场地上的复现)

| 序列 | 相机 | 模型 | 家族 | full ρ | static-4 ρ | Δρ |
|---|---|---|---|---|---|---|
| basketball1_split1_camera_0 | 0 | source-SFT | speed | 0.18 | 0.03 | 0.14 |
| basketball1_split1_camera_0 | 0 | source-SFT | path | 0.18 | 0.09 | 0.09 |
| basketball1_split1_camera_0 | 0 | GRPO-v3 | speed | 0.17 | -0.07 | 0.24 |
| basketball1_split1_camera_0 | 0 | GRPO-v3 | path | 0.24 | 0.17 | 0.07 |
| basketball1_split1_camera_2 | 2 | source-SFT | speed | 0.18 | 0.39 | -0.21 |
| basketball1_split1_camera_2 | 2 | source-SFT | path | 0.34 | 0.25 | 0.09 |
| basketball1_split1_camera_2 | 2 | GRPO-v3 | speed | 0.33 | 0.34 | -0.01 |
| basketball1_split1_camera_2 | 2 | GRPO-v3 | path | 0.40 | 0.22 | 0.18 |
| basketball2_camera_0 | 0 | source-SFT | speed | 0.48 | 0.28 | 0.20 |
| basketball2_camera_0 | 0 | source-SFT | path | 0.50 | 0.13 | 0.37 |
| basketball2_camera_0 | 0 | GRPO-v3 | speed | 0.44 | 0.18 | 0.26 |
| basketball2_camera_0 | 0 | GRPO-v3 | path | 0.46 | 0.17 | 0.29 |

读法: **A 节** 与 TeamTrack 主表同口径, 但 GT 是真世界量, 可直接检验 "第二场地上结论是否复现"。**B 节**是本篇最干净的一张表: 世界量固定, 只有视角变 —— 若模型 ρ 随 ρ(像素, 世界) 一起升降, "读的是图像平面位移" 就在同一批 item 上被证实; 若模型 ρ 在各相机上一样高, 那它读的就是世界量, 主论点要改。**C 节**检验多帧是否仍被使用。
