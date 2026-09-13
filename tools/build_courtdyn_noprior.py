#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_courtdyn_noprior.py — 从一个 CourtDyn v1 主池生成 no-prior 池 (plan G3 的逐序列版)。

与主片段的 results/courtdyn/qa_dyn_v1_noprior.json 同一构造: 只保留三个数值家族
(speed / path / timing), 题面删掉身高先验句 "Assume a typical player on this court is X m tall."
GT / image_id / meta 原样不动。

用法: python tools/build_courtdyn_noprior.py --qa_json results/courtdyn/seq_<seq>/qa_dyn_v1.json
      [--out results/courtdyn/seq_<seq>/qa_dyn_v1_noprior.json]
"""
import argparse
import io
import json
import os
import re

PRIOR_RE = re.compile(r"\s*Assume a typical player on this court is [0-9.]+\s*m tall\.\s*")
NUMERIC_PREFIXES = ("dynamics_speed", "dynamics_path", "dynamics_time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.qa_json.replace("qa_dyn_v1.json", "qa_dyn_v1_noprior.json")
    if os.path.abspath(out) == os.path.abspath(args.qa_json):
        raise SystemExit("--out 不能与 --qa_json 相同")
    data = json.load(io.open(args.qa_json, encoding="utf-8"))
    items = data if isinstance(data, list) else data["qa"]
    kept, stripped = [], 0
    for it in items:
        if not str(it.get("category", "")).startswith(NUMERIC_PREFIXES):
            continue
        q2, n = PRIOR_RE.subn(" ", it["question"])
        stripped += n
        it = dict(it)
        it["question"] = re.sub(r"\s{2,}", " ", q2).strip()
        kept.append(it)
    if stripped == 0:
        raise SystemExit("没有找到任何身高先验句, 拒绝写出 (题面格式可能变了)")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    print(f"[noprior] {len(items)} -> {len(kept)} 题 (数值家族), 删句 {stripped} 处 -> {out}")


if __name__ == "__main__":
    main()
