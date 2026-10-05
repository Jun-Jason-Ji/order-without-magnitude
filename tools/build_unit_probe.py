#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the stratified text-only unit-conversion probe (plan D6, step 1).

Why this exists
---------------
The published probe is 12 items: six metre values, each asked in centimetres and
millimetres, factor supplied.  The manuscript says of it, correctly, that it
"does not estimate general arithmetic competence".  That concession is the
cheapest way to lose the paper.  A reviewer only has to write

    "the model is not failing to read a ruler, it simply cannot do arithmetic"

and the central claim -- that the model reports order without magnitude -- is
gone.  Twelve items cannot answer that, and the answer does not need a GPU-hour
budget, because the probe carries no images at all.

What the design has to separate
-------------------------------
Three things are confounded in the visual result and have to be pulled apart:

  (a) arithmetic ability      -- can it scale a number by a factor at all?
  (b) instruction following   -- does the unit word in the prompt change the
                                 answer when it should?
  (c) the visual scale channel -- measured by the existing image experiments.

This probe measures (a) and (b) and leaves (c) to the visual arms, so that the
three can be read against each other.

The decisive measurement is not accuracy.  It is the distribution of
log10(prediction / target), which is expected to be multi-modal:

    log10 ratio ~  0                 the conversion was applied
    log10 ratio ~ -log10(factor)     the input was returned unchanged
    anything else                    a genuine arithmetic error

The middle mode is the same signature as the visual finding (a centimetre/metre
ratio of 1.0 where 100 is required).  If the model converts correctly here and
still returns a ratio of 1.0 with an image in front of it, then arithmetic is
intact and the unit word simply never reaches the visual estimate.  That is the
claim, and this is what makes it testable rather than asserted.

Design
------
factors, fully crossed:

  value        14 log-spaced magnitudes spanning 10^-3 .. 10^2.6.  The six
               published values are a strict subset, so the original items
               survive unchanged inside the larger set.
  unit_pair    8 pairs: both directions across three metric decades, plus the
               pixel/metre pair at the nominal K = 27.498 px/m used by the
               visual arms.  Covers multiply and divide, powers of ten and a
               non-power, so "off by a power of ten" and "wrong direction" are
               distinguishable failures rather than one bucket.
  factor_given the conversion factor stated in the prompt, or withheld.  With
               it withheld the item tests recall; with it supplied the item
               tests only whether a stated instruction is followed.  Pixel
               pairs are always supplied -- K is not recallable knowledge.
  framing      bare numeric, or embedded in the CourtDyn sentence frame.  Tests
               whether the sports framing itself is what disrupts conversion.

  6 metric pairs x 2 factor_given + 2 pixel pairs x 1 = 14 conditions
  14 conditions x 14 values x 2 framings = 392 items

Scoring note
------------
Correctness keeps the published criterion exactly -- absolute error < 0.05 --
so the twelve original items score identically inside the new set and the
published 2/9/4 counts stay comparable.  Relative error is reported alongside
as a secondary diagnostic but never replaces it.  Loosening the criterion to
something relative would silently change the published numbers, which is the
one thing this project does not do.

Usage
  python tools/build_unit_probe.py --out results/courtdyn/unit_probe_v2.json
  python tools/build_unit_probe.py --verify   # rebuild and compare hashes only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL = "courtdyn-unit-probe-v2"

# The nominal pixel scale of the Q2 visual arms.  Kept in one place so the probe
# and the image arms cannot drift apart.
PIXELS_PER_METRE = 27.498

# The six published values come first; the rest extend the dynamic range in both
# directions.  Sorted output below makes the ordering deterministic anyway, but
# keeping the published six visible here documents the containment.
PUBLISHED_VALUES = [0.2, 0.5, 1.0, 2.5, 4.0, 8.0]
EXTRA_VALUES = [0.002, 0.007, 0.02, 0.05, 17.5, 40.0, 125.0, 400.0]
VALUES = sorted(PUBLISHED_VALUES + EXTRA_VALUES)

# (from_unit, to_unit, factor, factor_sentence)
# factor is the multiplier applied to a quantity expressed in `from_unit`.
UNIT_PAIRS = [
    ("meters", "centimeters", 100.0, "One meter equals 100 centimeters."),
    ("meters", "millimeters", 1000.0, "One meter equals 1000 millimeters."),
    ("centimeters", "meters", 0.01, "One meter equals 100 centimeters."),
    ("millimeters", "meters", 0.001, "One meter equals 1000 millimeters."),
    ("meters", "kilometers", 0.001, "One kilometer equals 1000 meters."),
    ("kilometers", "meters", 1000.0, "One kilometer equals 1000 meters."),
]

PIXEL_PAIRS = [
    ("meters", "pixels", PIXELS_PER_METRE,
     f"One meter equals {PIXELS_PER_METRE:g} pixels in this image."),
    ("pixels", "meters", 1.0 / PIXELS_PER_METRE,
     f"One meter equals {PIXELS_PER_METRE:g} pixels in this image."),
]

ANSWER_INSTRUCTION = "Output only the number, one decimal place."

# The embedded frame reuses the vocabulary of the visual arms so that the only
# difference from a CourtDyn question is the absence of an image.
EMBEDDED_FRAME = ("The player marked with the red box has a path length of "
                  "{value} {from_unit} over the clip.")


def format_value(value: float) -> str:
    """Render a probe value the way the published probe renders it.

    The published items read "1.0 meters", not "1 meters".  Plain %g drops the
    trailing zero and would silently produce a different prompt -- a different
    item -- while still looking like the published one.  So: shortest exact
    decimal, but never fewer than one decimal place.
    """
    text = f"{value:g}"
    if "e" in text or "E" in text:                 # no probe value needs this
        raise ValueError(f"value {value!r} does not render as plain decimal")
    if "." not in text:
        text += ".0"
    return text


def build_question(value: float, from_unit: str, to_unit: str,
                   factor_sentence: str, factor_given: bool,
                   framing: str) -> str:
    parts = []
    if framing == "embedded":
        parts.append(EMBEDDED_FRAME.format(value=format_value(value),
                                           from_unit=from_unit))
        request = (f"Express that path length in {to_unit}.")
    else:
        parts.append(f"A distance is {format_value(value)} {from_unit}.")
        request = f"Express that distance in {to_unit}."
    if factor_given:
        parts.append(factor_sentence)
    parts.append(request)
    parts.append(ANSWER_INSTRUCTION)
    return " ".join(parts)


def conditions():
    """Yield (from_unit, to_unit, factor, sentence, factor_given, kind)."""
    for from_unit, to_unit, factor, sentence in UNIT_PAIRS:
        for factor_given in (True, False):
            yield from_unit, to_unit, factor, sentence, factor_given, "metric"
    for from_unit, to_unit, factor, sentence in PIXEL_PAIRS:
        # K is not recallable knowledge, so withholding it would not test
        # instruction following -- it would only test guessing.
        yield from_unit, to_unit, factor, sentence, True, "pixel"


def build_items():
    items = []
    for value in VALUES:
        for from_unit, to_unit, factor, sentence, factor_given, kind in conditions():
            for framing in ("bare", "embedded"):
                target = value * factor
                question = build_question(value, from_unit, to_unit, sentence,
                                          factor_given, framing)
                item_id = hashlib.sha256(
                    "|".join([PROTOCOL, question]).encode("utf-8")
                ).hexdigest()[:20]
                items.append({
                    "item_id": item_id,
                    "question": question,
                    "answer": f"{target:.1f}",
                    "meta": {
                        "protocol": PROTOCOL,
                        "value": value,
                        "from_unit": from_unit,
                        "to_unit": to_unit,
                        "factor": factor,
                        "log10_factor": math.log10(factor),
                        "factor_given": factor_given,
                        "framing": framing,
                        "pair_kind": kind,
                        "direction": "multiply" if factor > 1.0 else "divide",
                        "target": target,
                        # The published probe's criterion, carried verbatim so
                        # the twelve original items keep their published score.
                        "correct_abs_tolerance": 0.05,
                        "published_subset": (
                            value in PUBLISHED_VALUES
                            and from_unit == "meters"
                            and to_unit in ("centimeters", "millimeters")
                            and factor_given
                            and framing == "bare"
                        ),
                    },
                })
    items.sort(key=lambda row: row["item_id"])
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "courtdyn" / "unit_probe_v2.json"))
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and compare against the file on disk")
    args = ap.parse_args()

    items = build_items()
    payload = json.dumps(items, ensure_ascii=False, indent=1) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    published = sum(1 for row in items if row["meta"]["published_subset"])
    print(f"{len(items)} items, {published} of them the published subset")
    print(f"  values      {len(VALUES)}  ({min(VALUES):g} .. {max(VALUES):g})")
    print(f"  conditions  {len(list(conditions()))}")
    print(f"  sha256      {digest}")

    if published != 12:
        print(f"ERROR: expected the 12 published items to be contained, found {published}")
        return 1

    # Containment has to be verbatim, not merely count-matched.  A prompt that
    # differs by one character is a different item, and claiming the published
    # probe is a subset of this one would then be false.
    legacy = ROOT / "results" / "courtdyn" / "revision_controls_20260912" / "arithmetic.json"
    if legacy.is_file():
        old_rows = json.loads(legacy.read_text(encoding="utf-8"))
        subset = {row["question"]: row["answer"] for row in items
                  if row["meta"]["published_subset"]}
        problems = []
        for row in old_rows:
            if row["question"] not in subset:
                problems.append(f"missing: {row['question']}")
            elif subset[row["question"]] != row["answer"]:
                problems.append(
                    f"answer differs for: {row['question']} "
                    f"({subset[row['question']]} vs {row['answer']})")
        if problems:
            print(f"ERROR: the published probe is not contained verbatim "
                  f"({len(problems)} problems)")
            for line in problems[:5]:
                print("  " + line)
            return 1
        print(f"  containment  all {len(old_rows)} published items reproduced verbatim")
    else:
        print(f"ERROR: cannot verify containment, {legacy} is absent")
        return 1

    out = Path(args.out)
    if args.verify:
        if not out.is_file():
            print(f"ERROR: {out} does not exist")
            return 1
        current = out.read_text(encoding="utf-8")
        same = hashlib.sha256(current.encode("utf-8")).hexdigest() == digest
        print("verify:", "identical" if same else "DIFFERS")
        return 0 if same else 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
