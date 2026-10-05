#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the inputs for the M1 mixed-unit training control.

The question M1 answers
-----------------------
Every adapter in the paper was trained on metre answers only, on the same
question wording the unit arms then vary.  "Ask for centimetres, get the metre
number" is therefore predicted by answer-distribution overfitting alone; no
account of how the model reads images is needed.  If that is the explanation,
an adapter trained on the same items with some targets expressed in
centimetres should respond to a centimetre request.  If it still returns the
metre number, supervision does not explain the unit failure.

What is built
-------------
1. Training pool  results/courtdyn/m1_mixunit/mixunit_v3_pool.json
   The frozen event-holdout v3 pool (1,120 items, clean on both evaluation
   events), unchanged except that exactly half of the items -- balanced within
   every clip x family stratum, chosen by a content hash -- ask for centimetres
   instead of metres and carry the metre answer x 100.  Same items, same images,
   same order, same count, so the adapter is matched to the published v3 seed-42
   adapter in everything except the unit mix.  The centimetre answer is the
   metre answer string multiplied by 100 in decimal arithmetic: identical
   information, a different unit.  Pixels are not mixed in: the pool has no
   pixel truth for its side-view clips, and a pixel request at test time is then
   an unseen-unit probe.

2. In-template evaluation inputs  results/courtdyn/m1_mixunit/intemplate/
   The explicit-coordinate prompts used by the published unit arms are worded
   differently from the training questions.  An adapter that learned the unit
   only in the training wording would look unresponsive there, and M1 would
   wrongly clear the supervision account.  So the same 280 Q1 and 280 Q2
   speed/path items are also posed in the training wording, in metres and in
   centimetres.  Rows are the published explicit-arm rows (same items, frames,
   references, tolerances) with only the question text replaced; frames were
   checked identical for all 1,120 rows.

Every invariant is asserted; the script refuses to write on any violation.

Usage
  python tools/build_mixunit_pool.py
  python tools/build_mixunit_pool.py --verify     # rebuild in memory, compare hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CD = ROOT / "results" / "courtdyn"
SOURCE_POOL = CD / "revision_controls_20260912" / "training" / "event_holdout_v3.json"
CONTROLS = CD / "revision_controls_20260912"
OUT = CD / "m1_mixunit"
SEQS = ["Q1_top_0-30", "Q2_top_480-510"]
FAMILIES = ("dynamics_path_player", "dynamics_speed_player")

REWRITES = {
    "dynamics_path_player": (
        "over the whole clip, in meters? Output only",
        "over the whole clip, in centimeters? Output only"),
    "dynamics_speed_player": (
        "in meters per second? Output only",
        "in centimeters per second? Output only"),
}
METERS_WORD = re.compile(r"\bmeters\b")
CENTIMETERS_WORD = re.compile(r"\bcentimeters\b")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def to_cm(question: str, category: str) -> str:
    old, new = REWRITES[category]
    if question.count(old) != 1:
        raise ValueError(f"expected one '{old}' in: {question}")
    out = question.replace(old, new)
    if METERS_WORD.search(out) or not CENTIMETERS_WORD.search(out):
        raise ValueError(f"unit rewrite incomplete: {out}")
    return out


def cm_answer(metre_answer: str) -> str:
    return f"{(Decimal(metre_answer) * 100):.1f}"


def item_key(row) -> str:
    m = row["meta"]
    material = json.dumps([row["category"], m["seq"], m["track"], m["window"],
                           row["image_ids"]], sort_keys=True)
    return sha256_bytes(material.encode("utf-8"))


def build_pool():
    source = json.loads(SOURCE_POOL.read_text(encoding="utf-8"))
    if len(source) != 1120:
        raise ValueError(f"source pool has {len(source)} items, expected 1120")
    strata = defaultdict(list)
    for index, row in enumerate(source):
        strata[(row["meta"]["seq"], row["category"])].append(index)
    to_convert = set()
    for key, indices in strata.items():
        if len(indices) % 2:
            raise ValueError(f"stratum {key} has an odd size {len(indices)}")
        ranked = sorted(indices, key=lambda i: item_key(source[i]))
        to_convert.update(ranked[: len(ranked) // 2])
    pool = []
    for index, row in enumerate(source):
        new = json.loads(json.dumps(row))
        new["meta"]["m1_unit"] = "m"
        new["meta"]["m1_source_index"] = index
        if index in to_convert:
            new["question"] = to_cm(row["question"], row["category"])
            new["answer"] = cm_answer(row["answer"])
            new["meta"]["m1_unit"] = "cm"
            new["meta"]["m1_metre_answer"] = row["answer"]
        pool.append(new)
    # invariants
    units = Counter(r["meta"]["m1_unit"] for r in pool)
    assert units == {"m": 560, "cm": 560}, units
    for key, indices in strata.items():
        c = Counter(pool[i]["meta"]["m1_unit"] for i in indices)
        assert c["m"] == c["cm"], (key, c)
    for old, new in zip(source, pool):
        assert old["image_ids"] == new["image_ids"] and old["category"] == new["category"]
        if new["meta"]["m1_unit"] == "cm":
            assert Decimal(new["answer"]) == Decimal(old["answer"]) * 100
        else:
            assert new["question"] == old["question"] and new["answer"] == old["answer"]
    return pool, dict(units=dict(units),
                      strata={f"{k[0]}/{k[1]}": len(v) for k, v in strata.items()})


def build_intemplate():
    """Explicit-arm rows with the training-wording question swapped in."""
    cells, files = [], {}
    for seq in SEQS:
        legacy = json.loads((CD / f"seq_{seq}" / "qa_dyn_v1.json").read_text(encoding="utf-8"))
        key = lambda r: (r["category"], r["meta"]["track"], tuple(r["meta"]["window"]))
        legacy_by_key = {key(r): r for r in legacy if r["category"] in FAMILIES}
        for arm, source_arm in (("legacy_m", "m_height"), ("legacy_cm", "cm_height")):
            rows = json.loads((CONTROLS / seq / f"{source_arm}.json").read_text(encoding="utf-8"))
            out = []
            for row in rows:
                lg = legacy_by_key[key(row)]
                if lg["image_ids"] != row["image_ids"]:
                    raise ValueError(f"frames differ for {seq} {key(row)}")
                question = lg["question"]
                if arm == "legacy_cm":
                    question = to_cm(question, row["category"])
                new = json.loads(json.dumps(row))
                new["question"] = question
                new["meta"]["arm"] = arm
                new["meta"]["m1_question_source"] = f"seq_{seq}/qa_dyn_v1.json"
                out.append(new)
            assert len(out) == 280
            rel = f"{seq}/{arm}.json"
            files[rel] = out
            cells.append(dict(seq=seq, arm=arm, items=len(out), file=rel,
                              image_root=str(ROOT / "data" / "courtdyn" / f"frames_{seq}"),
                              event_overlap=False))
    return cells, files


def serialise(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    pool, pool_stats = build_pool()
    cells, files = build_intemplate()
    pool_bytes = serialise(pool)
    file_bytes = {rel: serialise(rows) for rel, rows in files.items()}
    for cell in cells:
        cell["sha256"] = sha256_bytes(file_bytes[cell["file"]])
    manifest = {
        "summary": {"protocol": "courtdyn-m1-intemplate-v1",
                    "note": "explicit-arm rows, question replaced by training wording"},
        "source_sha256": {"event_holdout_v3": sha256_file(SOURCE_POOL)},
        "cells": cells,
        "arithmetic_file": None,
    }
    receipt = {
        "protocol": "courtdyn-m1-mixunit-v1",
        "source_pool": str(SOURCE_POOL.relative_to(ROOT)),
        "source_pool_sha256": sha256_file(SOURCE_POOL),
        "pool_sha256": sha256_bytes(pool_bytes),
        "pool_stats": pool_stats,
        "intemplate_cells": {c["file"]: c["sha256"] for c in cells},
    }
    print(json.dumps(receipt, indent=1))

    if args.verify:
        current = json.loads((OUT / "build_receipt.json").read_text(encoding="utf-8"))
        same = current["pool_sha256"] == receipt["pool_sha256"] and \
            current["intemplate_cells"] == receipt["intemplate_cells"]
        print("verify:", "identical" if same else "DIFFERS")
        return 0 if same else 1

    (OUT / "intemplate").mkdir(parents=True, exist_ok=True)
    (OUT / "mixunit_v3_pool.json").write_bytes(pool_bytes)
    for rel, data in file_bytes.items():
        path = OUT / "intemplate" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (OUT / "intemplate" / "manifest.json").write_bytes(serialise(manifest))
    (OUT / "build_receipt.json").write_bytes(serialise(receipt))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
