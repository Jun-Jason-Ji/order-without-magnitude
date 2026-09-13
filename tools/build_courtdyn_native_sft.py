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

划分 (2026-09-12 审计修正)
------------------------
默认 event-holdout：排除 Q1_side_0-30，因为原视频及轨迹证实它与测试
Q1_top_0-30 含同一开场事件。训练为其余 3 条侧视 + Q4_top_0-30，共 1120 QA。
留出 Q1_top_0-30 / Q2_top_480-510。这仍是同一场比赛内的事件划分，不是比赛留出。
显式 --split legacy-cross-view 仅重建旧 1400 QA 清单（有测试事件跨机位重叠），
不得称为 event holdout。两个模式都写新文件，保留冻结旧训练池和原模型。
只保留 speed / path 两个家族, 与评测臂一致 (timing 没有可用 v3, 不进主结论)。

图片路径
--------
各片段的帧在各自目录下 (data/courtdyn/frames_<seq>/, 原片段在 data/courtdyn/frames/),
而训练脚本只收一个 --img_root。所以这里把 image_ids 重写成 "frames_<seq>/<name>",
img_root 统一指向 data/courtdyn。

用法: python tools/build_courtdyn_native_sft.py
产物: results/courtdyn/courtdyn_native_sft_train_event_holdout.json (+ manifest)
本工具只准备训练池，不训练模型；冻结旧模型结果不能作为新划分的结果。
"""
import argparse
import io
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "results", "courtdyn")
MAIN = "Q4_side_480-510"
LEGACY_TRAIN_SEQS = ["Q1_side_0-30", "Q2_side_300-330", "Q4_side_570-600", MAIN, "Q4_top_0-30"]
TRAIN_SEQS = [seq for seq in LEGACY_TRAIN_SEQS if seq != "Q1_side_0-30"]
HELD_OUT = ["Q1_top_0-30", "Q2_top_480-510"]
# Explicit evidence-based groups: only the Q1 shared event joins two viewpoints.
# Do not infer synchronization of arbitrary camera pairs from names alone.
EVENT_GROUPS = {
    "Q1_side_0-30": "Q1_opening_0-30",
    "Q1_top_0-30": "Q1_opening_0-30",
    "Q2_side_300-330": "Q2_300-330",
    "Q2_top_480-510": "Q2_480-510",
    "Q4_side_570-600": "Q4_570-600",
    MAIN: "Q4_480-510",
    "Q4_top_0-30": "Q4_0-30",
}
KEEP = ("dynamics_speed_player", "dynamics_path_player")
OUT = os.path.join(CD, "courtdyn_native_sft_train_event_holdout.json")
LEGACY_OUT = os.path.join(CD, "courtdyn_native_sft_train_legacy_cross_view_rebuilt.json")


def qa_path(seq):
    return os.path.join(CD if seq == MAIN else os.path.join(CD, f"seq_{seq}"), "qa_dyn_v1.json")


def frame_prefix(seq):
    return "frames" if seq == MAIN else f"frames_{seq}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("event-holdout", "legacy-cross-view"),
                    default="event-holdout")
    ap.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    args = ap.parse_args()
    train_seqs = TRAIN_SEQS if args.split == "event-holdout" else LEGACY_TRAIN_SEQS
    out = OUT if args.split == "event-holdout" else LEGACY_OUT
    rows, per_seq = [], {}
    for seq in train_seqs:
        p = qa_path(seq)
        if not os.path.isfile(p):
            raise SystemExit(f"缺少必需源池 {p}; 中止，避免静默改变训练划分")
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
    held_events = {EVENT_GROUPS[seq] for seq in HELD_OUT}
    overlap_events = sorted({EVENT_GROUPS[r["meta"]["source_seq"]] for r in rows} & held_events)
    if args.split == "event-holdout" and overlap_events:
        raise SystemExit(f"训练池与测试含相同事件组 {overlap_events}; 中止")

    manifest = {"schema": "courtdyn-native-sft-v2", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "purpose": "train a CourtDyn-native LoRA from base (NOT from paper 1's adapters), "
                              "so the main result gets an arm independent of the companion paper",
                   "split": args.split, "train_seqs": train_seqs, "held_out_seqs": HELD_OUT,
                   "event_groups": EVENT_GROUPS, "overlapping_event_groups": overlap_events,
                   "ground_truth": "v1 height-ruler answers copied unchanged from qa_dyn_v1.json; not v3",
                   "evidence": "paper2/audit/evidence_event_alignment.json and Q1 source-frame contact sheets",
                   "limitations": (["Within-game event holdout, not held-out match generalization.",
                                    "Prepared pool only: no retrained checkpoint or new evaluation is implied."]
                                   if args.split == "event-holdout" else
                                   ["Legacy cross-view split is contaminated at event level: Q1 opening events occur in both train and test.",
                                    "Retained only to reproduce the historical training list; not an event holdout."]),
                   "families": list(KEEP), "n_total": len(rows), "n_per_seq": per_seq,
                   "img_root": "data/courtdyn",
                   "init": "base (--allow_base_init); must NOT be initialised from source-SFT"}
    if not args.dry_run:
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        with io.open(out.replace(".json", ".manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n共 {len(rows)} 条 {'(dry run)' if args.dry_run else '-> ' + out}")
    print(f"划分: {args.split}; 测试事件重叠: {overlap_events}")
    print("冻结旧训练池/模型保持原样。本工具未训练或评测模型。")


if __name__ == "__main__":
    main()
