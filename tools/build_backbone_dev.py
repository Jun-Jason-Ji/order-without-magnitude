#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the round-2 DEV inputs used only to choose the stopping epoch.

280 Q1_side_0-30 items (140 path, 140 speed) from courtdyn_native_sft_train.json,
training wording, metres.  They are in neither round-2 training pool
(event_holdout_v3, mixunit_v3) and are not a test cell.  Rows use the same schema as
results/courtdyn/m1_mixunit/intemplate so a100.eval_controls.run_cell can read them.
The stopping rule (results/courtdyn/backbone2_r2/PREREGISTRATION.json) looks only at
parse rate and answer diversity; reference values are carried for completeness but the
rule never reads them.

Writes results/courtdyn/backbone2_r2/dev_inputs/{manifest.json, Q1_side_0-30/legacy_m.json}.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from a100 import config as C  # noqa: E402

SEQ = "Q1_side_0-30"
SOURCE = ROOT / "results" / "courtdyn" / "courtdyn_native_sft_train.json"
POOLS = [ROOT / "results" / "courtdyn" / "revision_controls_20260912" / "training" / "event_holdout_v3.json",
         ROOT / "results" / "courtdyn" / "m1_mixunit" / "mixunit_v3_pool.json"]
OUT = ROOT / "results" / "courtdyn" / "backbone2_r2" / "dev_inputs"
# scoring constants of the published CourtDyn cells (same as the in-template test rows)
SCORING = {"dynamics_path_player": (0.5, 2.0, "path_m"),
           "dynamics_speed_player": (0.3, 1.0, "speed_mps")}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    items = [x for x in json.loads(SOURCE.read_text(encoding="utf-8"))
             if x["meta"].get("seq") == SEQ]
    assert len(items) == 280, len(items)
    for pool in POOLS:
        seqs = {x["meta"].get("seq") for x in json.loads(pool.read_text(encoding="utf-8"))}
        assert SEQ not in seqs, f"{SEQ} is in training pool {pool.name}"
    rows = []
    for x in items:
        tol, floor, key = SCORING[x["category"]]
        m = x["meta"]
        rows.append({"category": x["category"], "question": x["question"], "answer": x["answer"],
                     "meta": {"seq": SEQ, "track": m["track"], "window": m["window"], "fps": m["fps"],
                              "arm": "legacy_m", "unit": "m", "score_tolerance": tol,
                              "score_floor": floor, "reference_value": m[key],
                              "reference_convention": "native (dev only; never scored for k)"},
                     "image_ids": [Path(i).name for i in x["image_ids"]],
                     "image_id": Path(x["image_id"]).name})
    root = Path(C.image_root(SEQ))
    missing = [i for r in rows for i in r["image_ids"] if not (root / i).is_file()]
    assert not missing, missing[:5]
    (OUT / SEQ).mkdir(parents=True, exist_ok=True)
    cell_path = OUT / SEQ / "legacy_m.json"
    cell_path.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"summary": {"protocol": "courtdyn-backbone2-r2-dev-v1",
                            "note": "stopping-rule dev set; not a test cell"},
                "source_sha256": {SOURCE.name: sha(SOURCE)}, "arithmetic_file": None,
                "cells": [{"seq": SEQ, "arm": "legacy_m", "items": len(rows),
                           "file": f"{SEQ}/legacy_m.json", "image_root": str(root),
                           "event_overlap": True, "sha256": sha(cell_path)}]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} dev rows -> {cell_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
