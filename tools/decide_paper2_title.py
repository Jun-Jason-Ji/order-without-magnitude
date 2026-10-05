#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-2 title rule, recorded 2026-09-29 before any seed-43 (round-5) result exists.

The seed-42 Gemma3-12B counterexample (R1 fails) is in the record whatever seed 43 shows, so the
unhedged title ("... Fine-Tuning Makes Vision--Language Models ...") is never restored. The
remaining choice depends only on whether the one non-Qwen replication (Pixtral-12B) holds at seed 43,
read with the outcome labels of results/courtdyn/backbone4_ext/PREREGISTRATION.json (group 1):

  Pixtral-12B seed 43 R1 "replicates"   -> TITLE_PLURAL   (evidence beyond one family is seed-stable)
  anything else (fails/intermediate/LIMITED)
                                          -> TITLE_SINGULAR (the stable evidence is Qwen3.5-4B only)
  round-5 verdict absent, or Pixtral-12B not yet in it (the analyzer judges a backbone only once its
  stages have finished)                   -> PENDING, title left as it is (TITLE_PLURAL today)
  [Clarified 2026-09-29 13:45 UTC, after Gemma seed 43 and before any Pixtral seed-43 test cell: the
  first version mapped a missing Pixtral entry to TITLE_SINGULAR, which would have decided the title
  on a partial run; an incomplete run now decides nothing.]

Gemma3-12B's seed-43 outcome does not move the title; it decides the abstract/synthesis wording:
  R1 "fails" again      -> counterexample replicates: "not on Gemma3-12B (seeds 42 and 43)"
  R1 "replicates"       -> "the Gemma3-12B counterexample is seed-dependent"
  otherwise             -> "not confirmed at seed 43"

Usage: python tools/decide_paper2_title.py [--apply]
  --apply rewrites \\title{...} in paper2/manuscript_v2/main.tex to the decided title (idempotent).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERDICT = ROOT / "results" / "courtdyn" / "backbone4_ext" / "verdict.json"
TEX = ROOT / "paper2" / "manuscript_v2" / "main.tex"

TITLE_PLURAL = ("Order without Magnitude: Single-Unit Fine-Tuning Can Make\n"
                "       Vision--Language Models Rank Motion but Ignore the Requested Unit")
TITLE_SINGULAR = ("Order without Magnitude: Single-Unit Fine-Tuning Can Make a\n"
                  "       Vision--Language Model Rank Motion but Ignore the Requested Unit")


def outcome(backbone, r1):
    if r1 is None:
        return "missing"
    if backbone == "gemma3_12b":
        return {"fails": "counterexample replicates", "replicates": "seed-dependent"}.get(
            r1, "not confirmed at seed 43")
    return {"replicates": "replicates", "fails": "seed-dependent"}.get(r1, "not confirmed at seed 43")


AGGREGATE = ROOT / "results" / "courtdyn" / "backbone_seeds3" / "aggregate.json"


def decide():
    # Superseded 2026-09-29 13:44 UTC by results/courtdyn/backbone_seeds3/PREREGISTRATION.json (ed6aacc),
    # before any Pixtral seed-43 test cell: the title now reads Pixtral-12B's R1 label over seeds 42/43/44.
    if not AGGREGATE.is_file():
        return {"status": "PENDING", "reason": "three-seed aggregate not written yet (backbone_seeds3)"}
    agg = json.loads(AGGREGATE.read_text(encoding="utf-8"))["backbones"]
    px = (agg.get("pixtral_12b") or {}).get("R1")
    if px is None or px.get("n_seeds_judged", 0) < 3:
        return {"status": "PENDING", "reason": "Pixtral-12B not judged at all three seeds"}
    plural = px["label"] in ("holds at all three seeds", "holds at two of three seeds")
    return {"status": "DECIDED", "pixtral_12b_R1": px["label"],
            "gemma3_12b_R1": ((agg.get("gemma3_12b") or {}).get("R1") or {}).get("label"),
            "title": TITLE_PLURAL if plural else TITLE_SINGULAR,
            "title_rule": "plural" if plural else "singular"}


def decide_seed43_superseded():
    if not VERDICT.is_file():
        return {"status": "PENDING", "reason": f"{VERDICT.relative_to(ROOT)} not written yet"}
    bb = json.loads(VERDICT.read_text(encoding="utf-8"))["backbones"]
    if "pixtral_12b" not in bb:
        return {"status": "PENDING", "reason": "Pixtral-12B seed 43 not yet judged (run incomplete)",
                "gemma3_12b": outcome("gemma3_12b", (bb.get("gemma3_12b") or {}).get("R1"))}
    r1 = {k: (bb.get(k) or {}).get("R1") for k in ("pixtral_12b", "gemma3_12b")}
    res = {"status": "DECIDED", "R1_seed43": r1,
           "pixtral_12b": outcome("pixtral_12b", r1["pixtral_12b"]),
           "gemma3_12b": outcome("gemma3_12b", r1["gemma3_12b"])}
    res["title"] = TITLE_PLURAL if r1["pixtral_12b"] == "replicates" else TITLE_SINGULAR
    res["title_rule"] = "plural" if res["title"] is TITLE_PLURAL else "singular"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    res = decide()
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if args.apply and res["status"] == "DECIDED":
        tex = TEX.read_text(encoding="utf-8")
        new, n = re.subn(r"\\title\{Order without Magnitude:.*?\}", lambda m: "\\title{" + res["title"] + "}",
                         tex, count=1, flags=re.S)
        if n != 1:
            raise SystemExit("title block not found")
        if new != tex:
            TEX.write_text(new, encoding="utf-8")
            print("title updated ->", res["title_rule"])
        else:
            print("title already as decided")
    return 0


if __name__ == "__main__":
    sys.exit(main())
