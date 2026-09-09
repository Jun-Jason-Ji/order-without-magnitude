#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paired_relational_scorer.py — 成对 relational 子集打分与汇总。

score:  单次 run_bench 产物 (predictions.json) 对成对 QA 打分:
          item_accuracy      逐题准确率 (chance 50%)
          paired_accuracy    成对都答对才得分 (chance 25%)
          same_letter_rate   孪生题答同一字母的比例 (位置/字母偏置诊断;
                             读图的模型在孪生题上应答不同字母)
          letter_rate        预测中 A/B/其他 的占比
        逐题判定复用 eval/evaluate.py 的 exact_match, 与主表 relational
        exact 完全一致。

summarize:  多个 score.json 汇总为 markdown 对照表。

用法:
  python eval/paired_relational_scorer.py score --qa_json ... --pred_json ... --out score.json
  python eval/paired_relational_scorer.py summarize --run name=score.json [...] --out table.md
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import exact_match, normalize_text  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score(args):
    qa = load(args.qa_json)
    preds = load(args.pred_json)
    pmap = {(p.get("image_id"), p.get("question")): p for p in preds}
    pairs, missing = {}, 0
    letters = {"A": 0, "B": 0, "other": 0}
    for it in qa:
        row = pmap.get((it["image_id"], it["question"]))
        if row is None:
            missing += 1
            continue
        ans = row.get("vlm_answer", "")
        norm = normalize_text(ans)
        letters["A" if norm == "a" else "B" if norm == "b" else "other"] += 1
        pairs.setdefault(it["pair_id"], {})[it["pair_role"]] = {
            "correct": bool(exact_match(ans, it["answer"])),
            "letter": norm,
        }
    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    n_items = sum(len(v) for v in pairs.values())
    n_correct = sum(r["correct"] for v in pairs.values() for r in v.values())
    n_pair_ok = sum(v["orig"]["correct"] and v["twin"]["correct"]
                    for v in complete.values())
    n_same = sum(v["orig"]["letter"] == v["twin"]["letter"]
                 for v in complete.values())
    out = {
        "schema": "paired-relational-score-v1",
        "n_pairs": len(complete),
        "n_items_scored": n_items,
        "n_missing_predictions": missing,
        "item_accuracy": n_correct / n_items if n_items else None,
        "paired_accuracy": n_pair_ok / len(complete) if complete else None,
        "same_letter_rate": n_same / len(complete) if complete else None,
        "letter_rate": {k: v / n_items if n_items else None
                        for k, v in letters.items()},
        "chance": {"item": 0.5, "paired": 0.25},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[paired-score] pairs={out['n_pairs']} item={out['item_accuracy']:.3f} "
          f"paired={out['paired_accuracy']:.3f} same_letter={out['same_letter_rate']:.3f}"
          + (f" 缺预测={missing}" if missing else ""))
    if missing:
        sys.exit(2)


def summarize(args):
    rows = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        s = load(path)
        rows.append((name, s))
    lines = [f"# {args.title}", "",
             "成对构造: 同图同题、(A)/(B) 指称互换、GT 翻转 (option-swap twin)。",
             "paired = 成对都答对 (chance 25%); same-letter = 孪生题答同一字母",
             "(字母偏置诊断, 越低越好; 完全读图应为 0%)。数值为百分数。", "",
             "| 模型 | 逐题 acc | paired acc | same-letter | 答A率 | n 对 |",
             "|---|---|---|---|---|---|"]
    for name, s in rows:
        pct = lambda x: "—" if x is None else f"{100*x:.1f}"  # noqa: E731
        lines.append(f"| {name} | {pct(s['item_accuracy'])} | "
                     f"{pct(s['paired_accuracy'])} | {pct(s['same_letter_rate'])} | "
                     f"{pct(s['letter_rate'].get('A'))} | {s['n_pairs']} |")
    lines += ["", "chance: 逐题 50, paired 25。"]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[paired-summary] {len(rows)} runs -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--qa_json", required=True)
    sc.add_argument("--pred_json", required=True)
    sc.add_argument("--out", required=True)
    sm = sub.add_parser("summarize")
    sm.add_argument("--run", action="append", required=True,
                    help="name=score.json, 可多次")
    sm.add_argument("--title", default="成对 relational 对照 (option-swap)")
    sm.add_argument("--out", required=True)
    args = ap.parse_args()
    score(args) if args.cmd == "score" else summarize(args)


if __name__ == "__main__":
    main()
