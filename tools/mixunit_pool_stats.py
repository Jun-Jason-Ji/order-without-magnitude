#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate statistics of the mixed-unit training pool, for the public release.

results/courtdyn/m1_mixunit/mixunit_v3_pool.json is withheld (its answers derive from the tracking
annotations).  The manuscript quotes only aggregate properties of it; this writes them to
results/courtdyn/m1_mixunit/pool_stats.json so that tools/audit_paper2_v2.py and
tools/audit_paper2_supp.py can verify the text without the pool.

  python tools/mixunit_pool_stats.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "results" / "courtdyn" / "m1_mixunit" / "mixunit_v3_pool.json"
OUT = POOL.parent / "pool_stats.json"


def compute(pool: list[dict]) -> dict:
    stats = {"source": POOL.relative_to(ROOT).as_posix(), "source_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
             "n_items": len(pool), "n_cm": sum("centimet" in json.dumps(it).lower() for it in pool),
             "n_cm_by_meta": sum(it.get("meta", {}).get("m1_unit") == "cm" for it in pool),
             "families": {}}
    for fam in ("speed", "path"):
        cm = [float(it["answer"]) for it in pool if it["category"].split("_")[1] == fam and it["meta"].get("m1_unit") == "cm"]
        m = [float(it["answer"]) for it in pool if it["category"].split("_")[1] == fam and it["meta"].get("m1_unit") != "cm"]
        digits = Counter(str(int(v))[0] for v in cm)
        stats["families"][fam] = {
            "n_cm": len(cm), "n_m": len(m),
            "cm_target_median": statistics.median(cm) if cm else None,
            "m_target_median": statistics.median(m) if m else None,
            "cm_first_digit_majority": digits.most_common(1)[0][0] if cm else None,
            "cm_first_digit_majority_share": digits.most_common(1)[0][1] / len(cm) if cm else None,
            "cm_first_digit_counts": dict(sorted(digits.items()))}
    return stats


def load_stats() -> dict:
    """Live from the pool when present, otherwise the shipped aggregate."""
    if POOL.is_file():
        return compute(json.loads(POOL.read_text(encoding="utf-8")))
    return json.loads(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    s = compute(json.loads(POOL.read_text(encoding="utf-8")))
    OUT.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "families"}, indent=1))
    print(json.dumps(s["families"], indent=1))
    sys.exit(0)
