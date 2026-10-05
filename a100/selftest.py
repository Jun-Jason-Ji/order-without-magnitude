#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU self-test: run the whole pipeline with a stub model and check invariants.

No GPU, no weights, no network. It exercises the parts that are easy to break
and expensive to discover mid-sweep: input loading and hash checks, frame
resolution, jsonl writing, resume with prefix validation, per-family scoring,
paired unit ratios, the first-frame delta, the degeneracy flag, and the
aggregation in a100/analyze.py.

The stub answers every arm with the *metre* reference value, jittered. That is
the behaviour the paper reports -- ordering preserved, no unit conversion -- so
the self-test also pins the invariants a real run must reproduce:

    rho(m_height)        high, because the answers track the reference
    cm/m median          ~1, not 100
    px/m median          ~1, while the itemwise reference ratio is ~27
    static4 delta rho    positive, because the stub degrades without motion

Usage
  python -m a100.selftest
  python -m a100.selftest --keep      # leave the scratch output for inspection
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C                    # noqa: E402
from a100 import eval_controls as E             # noqa: E402

SEQ = "Q2_top_480-510"
FAILURES = []


def check(condition, message):
    print(("[ok  ] " if condition else "[FAIL] ") + message, flush=True)
    if not condition:
        FAILURES.append(message)
    return condition


def make_stub(seq, arms, degenerate=False, frame_mode="full"):
    """A deterministic fake model keyed on the question text.

    It answers with the metre reference of that item, so unit arms all get the
    same number: exactly the 'order without magnitude' pattern. Under static4
    it answers a constant-ish value, so the first-frame delta is positive.
    """
    _, cells = E.load_cells(seq, list({*arms, "m_height"}))
    metre = {}
    lookup = {}
    for cell, rows in cells:
        for row in rows:
            key = (row["category"], row["meta"].get("track"),
                   tuple(row["meta"]["window"]))
            if cell["arm"] == "m_height":
                metre[key] = row["meta"]["reference_value"]
            # Only two distinct question texts exist per arm (speed and path):
            # the item identity lives in the frames. Key on the first frame
            # plus the question, which is unique across all 280 items.
            lookup[(row["image_ids"][0], row["question"])] = key

    def factory(model_path, precision, adapter=None):
        def infer(paths, question):
            if not paths:                       # text-only arithmetic probe
                return "wrong"
            key = lookup.get((Path(paths[0]).name, question))
            if key is None:
                return "n/a"
            if degenerate:
                return "1.5"
            value = metre[key]
            if frame_mode == "static4":
                # No motion information: collapse towards a single answer,
                # keeping a faint deterministic wobble so the cell is not
                # literally constant (which would make rho undefined).
                value = 1.0 + 0.1 * (abs(hash(key)) % 3)
            else:
                value = value * (1.0 + 0.01 * (abs(hash(key)) % 5))
            return f"{value:.1f}"
        return infer, dict(model_family="stub", precision=precision,
                           loader="a100.selftest.make_stub", stub=True,
                           generation_suffix="(stub)", max_new_tokens=0,
                           do_sample=False)
    return factory


def run(tmp):
    arms = C.ARMS_THREE
    out_full = tmp / "stub_full"
    E.run_cell("stub-model", "bf16", SEQ, arms, out_full, adapter=None,
               frame_mode="full", arithmetic=True, label="stub",
               infer_factory=make_stub(SEQ, arms), progress_every=10_000)
    check((out_full / "summary.json").is_file(), "full cell wrote a summary")

    out_static = tmp / "stub_static4"
    E.run_cell("stub-model", "bf16", SEQ, ["m_height"], out_static, adapter=None,
               frame_mode="static4", label="stub",
               infer_factory=make_stub(SEQ, ["m_height"], frame_mode="static4"),
               progress_every=10_000)

    out_degenerate = tmp / "stub_degenerate"
    E.run_cell("stub-model", "bf16", SEQ, arms, out_degenerate, adapter=None,
               frame_mode="full", label="stub-degenerate",
               infer_factory=make_stub(SEQ, arms, degenerate=True),
               progress_every=10_000)
    return out_full, out_static, out_degenerate


def check_resume(tmp):
    """Truncate a finished cell and prove a resume appends rather than restarts."""
    source = tmp / "stub_full"
    target = tmp / "stub_resume"
    shutil.copytree(source, target)
    (target / "summary.json").unlink()
    path = target / "m_height.jsonl"
    lines = path.read_text(encoding="utf8").splitlines()
    kept = lines[:100]
    path.write_text("\n".join(kept) + "\n", encoding="utf8")
    E.run_cell("stub-model", "bf16", SEQ, C.ARMS_THREE, target, adapter=None,
               frame_mode="full", arithmetic=True, resume=True, label="stub",
               infer_factory=make_stub(SEQ, C.ARMS_THREE), progress_every=10_000)
    after = path.read_text(encoding="utf8").splitlines()
    check(len(after) == C.ITEMS_PER_ARM,
          f"resume completed the arm ({len(after)} rows)")
    check(after[:100] == kept, "resume preserved the intact prefix byte for byte")

    # And a resume onto a changed configuration must refuse.
    broken = tmp / "stub_resume_broken"
    shutil.copytree(source, broken)
    (broken / "summary.json").unlink()
    config = json.loads((broken / "run_config.json").read_text(encoding="utf8"))
    config["precision"] = "nf4"
    (broken / "run_config.json").write_text(json.dumps(config), encoding="utf8")
    refused = False
    try:
        E.run_cell("stub-model", "bf16", SEQ, C.ARMS_THREE, broken, adapter=None,
                   frame_mode="full", arithmetic=True, resume=True, label="stub",
                   infer_factory=make_stub(SEQ, C.ARMS_THREE), progress_every=10_000)
    except SystemExit:
        refused = True
    check(refused, "resume refuses a changed configuration")


def check_metrics(out_full, out_static, out_degenerate):
    from a100 import analyze as A
    full = A.read_cell(out_full)
    static = A.read_cell(out_static)
    report = A.cell_report(full, static)
    speed = report["families"]["speed"]

    rho = speed["m_height"]["rho"]
    check(rho is not None and rho > 0.9, f"metre rho tracks the reference ({rho})")

    cm = speed["ratio_cm_height_over_m"]["median"]
    check(abs(cm - 1.0) < 0.2, f"cm/m median is ~1, not 100 ({cm})")

    px = speed["ratio_px_height_over_m"]
    check(abs(px["median"] - 1.0) < 0.2, f"px/m median is ~1 ({px['median']})")
    check(px["reference_median"] is not None and 20 < px["reference_median"] < 40,
          f"itemwise pixel reference ratio is ~27 ({px['reference_median']})")

    delta = speed["static4"]["delta_rho"]
    check(delta is not None and delta > 0.5,
          f"first-frame repetition lowers rho ({delta})")

    constant = speed["m_height"]["constant_tmra"]
    check(constant is not None and 0 <= constant <= 100,
          f"best test-label constant is scored ({constant})")
    check(speed["m_height"]["tmra_minus_constant"] is not None,
          "advantage over the constant is computed")

    degenerate = A.cell_report(A.read_cell(out_degenerate))
    flagged = degenerate["families"]["speed"]["m_height"]
    check(flagged["degenerate"] and flagged["n_distinct"] <= 2,
          f"a constant-output cell is flagged degenerate "
          f"({flagged['n_distinct']} distinct)")

    arithmetic = full["arithmetic"]
    check(arithmetic is not None and arithmetic["n"] == 12,
          "the 12 text-only probes ran")
    check(arithmetic["correct"] == 0,
          "the stub fails every probe, as written")


def check_scoring_primitives():
    check(E.parse("1.5") == 1.5, "parse accepts a bare number")
    check(E.parse("1.5 meters") is None, "parse rejects a number with a unit")
    check(E.parse("nan") is None, "parse rejects nan")
    check(E.extract_answer("reasoning\nAnswer: 2.0") == "2.0",
          "lenient extraction strips an Answer: prefix")
    check(E.scalar_score(3.5, 3.5, 0.5, 2.0) == 100, "an exact answer scores 100")
    check(E.spearman([1, 2, 3], [1, 2, 3]) == 1.0, "spearman of a perfect order is 1")
    check(E.spearman([1, 1, 1], [1, 2, 3]) is None,
          "spearman of a constant is undefined, not zero")
    a = [dict(prediction=1.0, reference_value=1.0, score_tolerance=0.5,
              score_floor=2.0, category="x") for _ in range(4)]
    check(E.score_rows(a)["x"]["degenerate"], "score_rows flags a degenerate family")


FROZEN_ANALYSIS = ("results/courtdyn/revision_execution_20260912/"
                   "analysis_order_compatible/analysis.json")


def check_protocol_compatibility():
    """Reproduce the published constant-oracle scores from the frozen pools.

    The "best test-label constant" is the baseline every T-MRA claim in the
    paper is measured against. If this package computed it even slightly
    differently -- and an earlier draft did, by searching only the reference
    values -- every new margin would be quietly incomparable with the
    published ones. So it is pinned here against the frozen analysis.
    """
    from a100 import analyze as A
    path = C.ROOT / FROZEN_ANALYSIS
    if not path.is_file():
        print(f"[warn] protocol compatibility: {FROZEN_ANALYSIS} not present, "
              "skipped (ship it in the bundle to enable this check)")
        return
    frozen = json.loads(path.read_text(encoding="utf8"))
    expected = {}
    for run in frozen["runs"]:
        if run["frame_mode"] != "full" or "m_height" not in run["cells"]:
            continue
        for family, block in run["cells"]["m_height"]["families"].items():
            oracle = block["constant_diagnostics"]["values"]["test_label_oracle"]
            expected[(run["seq"], family)] = oracle["tmra"]
    compared = 0
    for seq in C.EVAL_SEQS:
        pool = C.INPUTS / seq / "m_height.json"
        if not pool.is_file():
            continue
        rows = json.loads(pool.read_text(encoding="utf8"))
        for family in ("dynamics_speed_player", "dynamics_path_player"):
            want = expected.get((seq, family))
            if want is None:
                continue
            subset = [dict(reference_value=r["meta"]["reference_value"],
                           score_tolerance=r["meta"]["score_tolerance"],
                           score_floor=r["meta"]["score_floor"])
                      for r in rows if r["category"] == family]
            _, got = A.best_constant(subset)
            compared += 1
            check(got is not None and abs(got - want) < 1e-9,
                  f"constant oracle {seq}/{A.SHORT[family]} reproduces the "
                  f"published {want:.6f} (got {got})")
    check(compared > 0, "at least one published constant oracle was compared")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("paths: " + json.dumps(C.describe(), ensure_ascii=False))
    check_scoring_primitives()
    check_protocol_compatibility()
    tmp = Path(tempfile.mkdtemp(prefix="a100_selftest_"))
    try:
        out_full, out_static, out_degenerate = run(tmp)
        check_metrics(out_full, out_static, out_degenerate)
        check_resume(tmp)
    finally:
        if args.keep:
            print("scratch kept at " + str(tmp))
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print("SELF-TEST FAILED:")
        for message in FAILURES:
            print("  - " + message)
        return 1
    print("self-test passed; the pipeline is wired correctly on this machine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
