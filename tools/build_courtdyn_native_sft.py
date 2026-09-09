#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_courtdyn_native_sft.py — 给论文 2 主结论造一条**不属于论文 1**的模型臂 (T33 / N8)。

动机
----
T28 (递尺子 / 换单位) 的模型只有 base + 论文 1 的三个适配器 + 公开 7B, 而 base 被提示扰乱、
7B 是退化输出 (恒定 1.5, distinct 塌到 1-2) —— 主结论"有序无量"的**肯定那半**目前没有外部复现。
盘上又没有第四个能跑的多模态模型 (9B 装不进 8 GB, SmolVLM2 答字母 "A")。

于是换一条更硬的路: **我们自己在 CourtDyn 上从 base 直接训一个 LoRA** (不是从论文 1 的适配器续训),
再问它同样的问题。它要检验的命题比公开模型臂更强:

    即使把模型直接训在 CourtDyn 的米制答案上, 换个单位问, 它依然不做换算。

划分 (关键: 测试片段必须一帧都没进过训练)
-----------------------------------------
  训练: 4 条侧视 + Q4_top_0-30
  留出: **Q1_top_0-30 / Q2_top_480-510** —— 正是 ruler / pxunit 题池已建好的那两条
只保留 speed / path 两个家族, 与评测臂一致 (timing 没有可用 v3, 不进主结论)。

图片路径
--------
各片段的帧在各自目录下 (data/courtdyn/frames_<seq>/, 原片段在 data/courtdyn/frames/),
而训练脚本只收一个 --img_root。所以这里把 image_ids 重写成 "frames_<seq>/<name>",
img_root 统一指向 data/courtdyn。

用法: python tools/build_courtdyn_native_sft.py
产物: results/courtdyn/courtdyn_native_sft_train.json (+ .manifest.json)
"""
import io
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "results", "courtdyn")
MAIN = "Q4_side_480-510"
TRAIN_SEQS = ["Q1_side_0-30", "Q2_side_300-330", "Q4_side_570-600", MAIN, "Q4_top_0-30"]
HELD_OUT = ["Q1_top_0-30", "Q2_top_480-510"]
KEEP = ("dynamics_speed_player", "dynamics_path_player")
OUT = os.path.join(CD, "courtdyn_native_sft_train.json")


def qa_path(seq):
    return os.path.join(CD if seq == MAIN else os.path.join(CD, f"seq_{seq}"), "qa_dyn_v1.json")


def frame_prefix(seq):
    return "frames" if seq == MAIN else f"frames_{seq}"


def main():
    rows, per_seq = [], {}
    for seq in TRAIN_SEQS:
        p = qa_path(seq)
        if not os.path.isfile(p):
            print(f"  ! 缺 {p}, 跳过")
            continue
        pref = frame_prefix(seq)
        n = 0
        for it in json.load(io.open(p, encoding="utf-8")):
            if it.get("category") not in KEEP:
                continue
            ids = it.get("image_ids") or ([it["image_id"]] if it.get("image_id") else [])
            if len(ids) != 4:
                continue
            rows.append({
                "category": it["category"],
                "question": it["question"],
                "answer": str(it["answer"]),
                "image_ids": [f"{pref}/{x}" for x in ids],
                "image_id": f"{pref}/{ids[0]}",
                "meta": dict(it.get("meta", {}), source_seq=seq),
            })
            n += 1
        per_seq[seq] = n
        print(f"  {seq:22s} {n:4d} 条")

    # 留出片段一条都不能混进来
    leaked = [r for r in rows if r["meta"]["source_seq"] in HELD_OUT]
    if leaked:
        raise SystemExit(f"训练池里混进了留出片段 {len(leaked)} 条, 中止")

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with io.open(OUT.replace(".json", ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "courtdyn-native-sft-v1", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "purpose": "train a CourtDyn-native LoRA from base (NOT from paper 1's adapters), "
                              "so the main result gets an arm independent of the companion paper",
                   "train_seqs": TRAIN_SEQS, "held_out_seqs": HELD_OUT,
                   "families": list(KEEP), "n_total": len(rows), "n_per_seq": per_seq,
                   "img_root": "data/courtdyn",
                   "init": "base (--allow_base_init); must NOT be initialised from source-SFT"},
                  f, ensure_ascii=False, indent=1)
    print(f"\n共 {len(rows)} 条 -> {OUT}")
    print(f"留出 (一帧都没进训练): {', '.join(HELD_OUT)}")


if __name__ == "__main__":
    main()
