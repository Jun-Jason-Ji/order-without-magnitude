# P2 同预测双参考重评分

PASS

主表采用 exact v1/v3；一位小数双参考敏感性独立列在 JSON。未完成的题目不补零成完整结果。MAE 使用同一严格解析成功集合，单列 N 与单位。

| 模型 | 序列/帧条件 | Family | 参考 | 观察/解析/应有N | all-item T-MRA | parsed T-MRA | Spearman | MAE（单位；N） |
|---|---|---|---|---:|---:|---:|---:|---:|
| eventholdout_v1_s42 | Q1_top_0-30/full | speed | exact_v1 | 140/140/140 | 21.5714 | 21.5714 | 0.6109 | 2.9854 (m/s; 140) |
| eventholdout_v1_s42 | Q1_top_0-30/full | speed | exact_v3 | 140/140/140 | 54.1429 | 54.1429 | 0.5996 | 0.9631 (m/s; 140) |
| eventholdout_v1_s42 | Q1_top_0-30/static4 | speed | exact_v1 | 140/140/140 | 8.6429 | 8.6429 | 0.0383 | 3.5159 (m/s; 140) |
| eventholdout_v1_s42 | Q1_top_0-30/static4 | speed | exact_v3 | 140/140/140 | 32.0000 | 32.0000 | 0.0570 | 1.3778 (m/s; 140) |
| eventholdout_v3_s42 | Q1_top_0-30/full | speed | exact_v1 | 140/140/140 | 17.7143 | 17.7143 | 0.6759 | 3.0370 (m/s; 140) |
| eventholdout_v3_s42 | Q1_top_0-30/full | speed | exact_v3 | 140/140/140 | 51.9286 | 51.9286 | 0.6626 | 0.9718 (m/s; 140) |
| eventholdout_v3_s42 | Q1_top_0-30/static4 | speed | exact_v1 | 140/135/140 | 3.1429 | 3.2593 | 0.0406 | 3.8073 (m/s; 135) |
| eventholdout_v3_s42 | Q1_top_0-30/static4 | speed | exact_v3 | 140/135/140 | 18.8571 | 19.5556 | 0.0759 | 1.6356 (m/s; 135) |
| eventholdout_v1_s42 | Q1_top_0-30/full | path | exact_v1 | 140/140/140 | 23.5714 | 23.5714 | 0.5024 | 6.6630 (m; 140) |
| eventholdout_v1_s42 | Q1_top_0-30/full | path | exact_v3 | 140/140/140 | 47.2143 | 47.2143 | 0.5116 | 2.0960 (m; 140) |
| eventholdout_v1_s42 | Q1_top_0-30/static4 | path | exact_v1 | 140/140/140 | 8.0000 | 8.0000 | -0.1856 | 8.1821 (m; 140) |
| eventholdout_v1_s42 | Q1_top_0-30/static4 | path | exact_v3 | 140/140/140 | 29.2857 | 29.2857 | -0.1687 | 3.1865 (m; 140) |
| eventholdout_v3_s42 | Q1_top_0-30/full | path | exact_v1 | 140/140/140 | 15.3571 | 15.3571 | 0.7875 | 7.0944 (m; 140) |
| eventholdout_v3_s42 | Q1_top_0-30/full | path | exact_v3 | 140/140/140 | 47.5000 | 47.5000 | 0.7784 | 2.2202 (m; 140) |
| eventholdout_v3_s42 | Q1_top_0-30/static4 | path | exact_v1 | 140/140/140 | 3.2143 | 3.2143 | -0.2551 | 8.6501 (m; 140) |
| eventholdout_v3_s42 | Q1_top_0-30/static4 | path | exact_v3 | 140/140/140 | 18.8571 | 18.8571 | -0.2408 | 3.6266 (m; 140) |
| eventholdout_v1_s42 | Q2_top_480-510/full | speed | exact_v1 | 140/140/140 | 27.7143 | 27.7143 | 0.7050 | 1.8122 (m/s; 140) |
| eventholdout_v1_s42 | Q2_top_480-510/full | speed | exact_v3 | 140/140/140 | 72.0714 | 72.0714 | 0.7145 | 0.5857 (m/s; 140) |
| eventholdout_v1_s42 | Q2_top_480-510/static4 | speed | exact_v1 | 140/140/140 | 14.2857 | 14.2857 | 0.1855 | 2.2065 (m/s; 140) |
| eventholdout_v1_s42 | Q2_top_480-510/static4 | speed | exact_v3 | 140/140/140 | 43.7857 | 43.7857 | 0.2020 | 0.9082 (m/s; 140) |
| eventholdout_v3_s42 | Q2_top_480-510/full | speed | exact_v1 | 140/140/140 | 20.9286 | 20.9286 | 0.7477 | 1.8872 (m/s; 140) |
| eventholdout_v3_s42 | Q2_top_480-510/full | speed | exact_v3 | 140/140/140 | 67.6429 | 67.6429 | 0.7584 | 0.6081 (m/s; 140) |
| eventholdout_v3_s42 | Q2_top_480-510/static4 | speed | exact_v1 | 140/137/140 | 5.8571 | 5.9854 | NA | 2.4780 (m/s; 137) |
| eventholdout_v3_s42 | Q2_top_480-510/static4 | speed | exact_v3 | 140/137/140 | 26.5714 | 27.1533 | NA | 1.1623 (m/s; 137) |
| eventholdout_v1_s42 | Q2_top_480-510/full | path | exact_v1 | 140/140/140 | 32.5000 | 32.5000 | 0.6735 | 3.8639 (m; 140) |
| eventholdout_v1_s42 | Q2_top_480-510/full | path | exact_v3 | 140/140/140 | 74.2143 | 74.2143 | 0.6938 | 1.2096 (m; 140) |
| eventholdout_v1_s42 | Q2_top_480-510/static4 | path | exact_v1 | 140/140/140 | 16.5000 | 16.5000 | 0.0500 | 4.9306 (m; 140) |
| eventholdout_v1_s42 | Q2_top_480-510/static4 | path | exact_v3 | 140/140/140 | 44.0000 | 44.0000 | 0.0381 | 2.0094 (m; 140) |
| eventholdout_v3_s42 | Q2_top_480-510/full | path | exact_v1 | 140/140/140 | 21.2143 | 21.2143 | 0.7625 | 4.2847 (m; 140) |
| eventholdout_v3_s42 | Q2_top_480-510/full | path | exact_v3 | 140/140/140 | 64.1429 | 64.1429 | 0.7728 | 1.3954 (m; 140) |
| eventholdout_v3_s42 | Q2_top_480-510/static4 | path | exact_v1 | 140/140/140 | 7.0000 | 7.0000 | 0.0443 | 5.4511 (m; 140) |
| eventholdout_v3_s42 | Q2_top_480-510/static4 | path | exact_v3 | 140/140/140 | 29.3571 | 29.3571 | 0.0364 | 2.4678 (m; 140) |

所有分数使用同一预测分别与两参考比较；解析失败在完整 all-item 分数中记零。MAE 不将无效解析或缺失输出当作零误差；MAE 的样本数与 parsed T-MRA/Spearman 相同。逐项配对差、共同观察/共同可解析队列、full/static4及跨模型差另见 JSON。这些仍是同一比赛内的依赖数据，不能据此识别内部视觉机制。
