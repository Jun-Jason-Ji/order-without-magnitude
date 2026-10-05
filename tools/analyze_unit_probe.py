#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyse the text-only unit-conversion probe (plan D6, step 1).

Accuracy alone cannot carry the argument, because "wrong" has several shapes
and they mean opposite things for the paper.  Every answer is therefore
classified by where log10(prediction / target) lands:

  correct          |prediction - target| < 0.05       (published criterion)
  identity         prediction ~ the input value        the unit request was ignored
  wrong_direction  prediction ~ value / factor         divided where it should multiply
  decade_error     off by a whole power of ten, and    the factor was applied, but
                   neither of the above                with the wrong exponent
  other            anything else                       a genuine arithmetic error
  unparsed         not a strict whole-answer number

'identity' is the text-only twin of the visual finding.  With an image, asking
for centimetres instead of metres leaves the answer unchanged (a paired ratio of
1.00-1.60 where 100 is required).  If the same checkpoint converts correctly here
-- no image, same words -- then its arithmetic is intact and the unit request
simply never reaches the magnitude read off the image.  If instead it returns
'identity' here too, the visual result says nothing specific about vision and
the paper's claim has to be rewritten.  Both outcomes are reported as found.

Statistics
  Wilson 95% intervals for every proportion.
  Paired contrasts use exact McNemar on the discordant pairs, because the design
  matches items across arms and across framing/factor conditions: the same value
  and unit pair is asked with and without the factor, bare and embedded, and the
  identical item is shown to every checkpoint.

Usage
  python tools/analyze_unit_probe.py
  python tools/analyze_unit_probe.py --runs results/courtdyn/unit_probe_v2_runs
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "results" / "courtdyn" / "unit_probe_v2_runs"
ARM_ORDER = ["base", "native", "v1", "v3"]
ARM_LABEL = {
    "base": "Base (no adapter)",
    "native": "CourtDyn-native",
    "v1": "Event-holdout v1",
    "v3": "Event-holdout v3",
}
# Log-space tolerance for "lands on" a mode.  0.02 in log10 is a 4.7% band:
# wide enough for one-decimal rounding of small targets, far narrower than the
# factor-of-ten gaps being distinguished.
LOG_BAND = 0.02
CLASSES = ["correct", "identity", "wrong_direction", "decade_error", "other", "unparsed"]


# --------------------------------------------------------------------------- stats

def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ------------------------------------------------------------------- classification

def classify(row) -> tuple[str, float | None]:
    prediction = row["prediction"]
    meta = row["meta"]
    target = float(meta["target"])
    value = float(meta["value"])
    factor = float(meta["factor"])
    if prediction is None:
        return "unparsed", None
    if abs(prediction - target) < 0.05:
        return "correct", 0.0
    if prediction <= 0 or target <= 0:
        return "other", None
    log_ratio = math.log10(prediction / target)
    # identity: the answer is the input, i.e. no conversion was applied.
    if abs(math.log10(prediction / value)) < LOG_BAND:
        return "identity", log_ratio
    # wrong direction: divided by the factor instead of multiplying (or v.v.).
    if abs(math.log10(prediction / (value / factor))) < LOG_BAND:
        return "wrong_direction", log_ratio
    nearest = round(log_ratio)
    if nearest != 0 and abs(log_ratio - nearest) < LOG_BAND:
        return "decade_error", log_ratio
    return "other", log_ratio


def is_diagnostic(meta) -> bool:
    """Can this item tell a conversion from no conversion at all?

    The published criterion (|error| < 0.05, answer to one decimal place) was
    written for the twelve published items, which all multiply to targets of 20
    or more.  Extended to divisions it silently breaks: 0.02 m -> km has target
    0.00002, so an answer that ignores the unit (0.02) and an answer of 0.0 both
    score as correct.  On such items the 'identity' failure -- the very signature
    this probe exists to detect -- is indistinguishable from success.

    An item is diagnostic only if the unconverted answer would be scored wrong and
    the target is representable at one decimal place.  146 of 392 items fail
    this, all of them divisions; every published item passes.  Accuracy is
    therefore reported on diagnostic items, with the all-item figure kept only
    for continuity.  Deriving this here, rather than re-issuing the probe, keeps
    the frozen probe file -- whose hash every run_config.json records -- intact.
    """
    value, target = float(meta["value"]), float(meta["target"])
    return (abs(target) >= 0.05
            and abs(round(value, 1) - round(target, 1)) >= 0.1
            and abs(value - target) >= 0.05)


def decade_of(value: float) -> str:
    return f"1e{math.floor(math.log10(value))}"


# --------------------------------------------------------------------------- load

def load_runs(runs_dir: Path):
    arms = {}
    for arm in ARM_ORDER:
        path = runs_dir / arm / "results.jsonl"
        if not path.is_file():
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        for row in rows:
            row["class"], row["log_ratio"] = classify(row)
            row["diagnostic"] = is_diagnostic(row["meta"])
        arms[arm] = rows
    return arms


def proportion(rows, predicate):
    k = sum(1 for r in rows if predicate(r))
    n = len(rows)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else float("nan"), "lo": lo, "hi": hi}


# ------------------------------------------------------------------------ analyses

def stratified(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return {g: proportion(v, lambda r: r["class"] == "correct")
            for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def paired_contrast(rows, split, match_keys):
    """McNemar between the two levels of `split`, matching on `match_keys`."""
    table = defaultdict(dict)
    for r in rows:
        key = tuple(r["meta"][k] for k in match_keys)
        table[key][r["meta"][split]] = r["class"] == "correct"
    levels = sorted({r["meta"][split] for r in rows}, key=str)
    if len(levels) != 2:
        return None
    a, b = levels
    only_a = only_b = both = neither = 0
    for cells in table.values():
        if a not in cells or b not in cells:
            continue
        x, y = cells[a], cells[b]
        if x and y:
            both += 1
        elif x:
            only_a += 1
        elif y:
            only_b += 1
        else:
            neither += 1
    n = both + neither + only_a + only_b
    return {"levels": [str(a), str(b)], "pairs": n,
            f"acc_{a}": (both + only_a) / n if n else float("nan"),
            f"acc_{b}": (both + only_b) / n if n else float("nan"),
            f"only_{a}": only_a, f"only_{b}": only_b,
            "mcnemar_p": mcnemar_exact(only_a, only_b)}


def arm_vs_base(arms):
    if "base" not in arms:
        return {}
    base = {r["item_id"]: r["class"] == "correct" for r in arms["base"]
            if r["diagnostic"]}
    out = {}
    for arm, rows in arms.items():
        if arm == "base":
            continue
        only_base = only_arm = 0
        for r in rows:
            if r["item_id"] not in base:
                continue
            b, a = base[r["item_id"]], r["class"] == "correct"
            only_base += b and not a
            only_arm += a and not b
        out[arm] = {"only_base_correct": only_base, "only_arm_correct": only_arm,
                    "mcnemar_p": mcnemar_exact(only_base, only_arm)}
    return out


def visual_analog(rows):
    """The text items that correspond to the published visual unit arms.

    Visual cm arm   <->  m -> cm, factor given, embedded in the CourtDyn frame
    Visual px arm   <->  m -> px, factor (K) given, embedded
    The embedded frame is the CourtDyn sentence with the image removed, so these
    rows are the closest text-only twins of the visual unit arms.
    """
    def pick(to_unit):
        return [r for r in rows
                if r["meta"]["from_unit"] == "meters"
                and r["meta"]["to_unit"] == to_unit
                and r["meta"]["factor_given"]
                and r["meta"]["framing"] == "embedded"]
    return {
        "m_to_cm_embedded": proportion(pick("centimeters"),
                                       lambda r: r["class"] == "correct"),
        "m_to_px_embedded": proportion(pick("pixels"),
                                       lambda r: r["class"] == "correct"),
        "m_to_cm_embedded_identity": proportion(pick("centimeters"),
                                                lambda r: r["class"] == "identity"),
    }


# The published visual analysis these text ratios are set against.  Read, never
# written: it is a frozen artefact that the manuscript audit already pins.
VISUAL_ANALYSIS = (ROOT / "results" / "courtdyn" / "revision_execution_20260912"
                   / "analysis_order_compatible" / "analysis.json")
VISUAL_LABEL = {"native": "native_original", "v1": "eventholdout_v1_s42",
                "v3": "eventholdout_v3_s42"}


def quartiles(values):
    xs = sorted(values)
    if not xs:
        return None
    def q(f):
        pos = f * (len(xs) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return {"n": len(xs), "median": q(0.5), "q1": q(0.25), "q3": q(0.75)}


def text_scaling(rows, to_unit):
    """prediction / stated value for m -> to_unit, factor supplied, both framings.

    The text twin of the visual paired ratio: in both cases the question is how
    far the number moves when the requested unit changes from metres.  In text
    the metre quantity is the stated value; visually it is the metre-arm answer
    for the same clip.
    """
    ratios = [r["prediction"] / r["meta"]["value"] for r in rows
              if r["meta"]["from_unit"] == "meters"
              and r["meta"]["to_unit"] == to_unit
              and r["meta"]["factor_given"]
              and r["prediction"] not in (None, 0)]
    out = quartiles(ratios)
    if out:
        out["within_half_to_double_of_one"] = sum(0.5 < x < 2 for x in ratios)
    return out


def visual_ratios():
    if not VISUAL_ANALYSIS.is_file():
        return {}
    data = json.loads(VISUAL_ANALYSIS.read_text(encoding="utf-8"))
    out = {}
    for run in data["runs"]:
        if run["seq"] != "Q2_top_480-510" or run["frame_mode"] != "full":
            continue
        for pair in run["within_run_pairing"]:
            name = pair["comparison"]
            if name not in ("cm/m with height", "px/m with height"):
                continue
            key = "cm" if name.startswith("cm") else "px"
            family = pair["family"].replace("dynamics_", "").replace("_player", "")
            out.setdefault(run["label"], {}).setdefault(key, {})[family] = {
                "median": pair["ratio"]["median"],
                "q1": pair["ratio"]["q1_q3"][0], "q3": pair["ratio"]["q1_q3"][1],
                "n": pair["ratio"]["n"]}
    return out


def dissociation(arms):
    visual = visual_ratios()
    out = {}
    for arm, rows in arms.items():
        out[arm] = {
            "text_cm_over_m": text_scaling(rows, "centimeters"),
            "text_px_over_m": text_scaling(rows, "pixels"),
            "required_cm": 100.0,
            "required_px": 27.498,
            "visual": visual.get(VISUAL_LABEL.get(arm, ""), None),
        }
    return out


def analyse(arms):
    report = {"protocol": "courtdyn-unit-probe-v2", "basis": "diagnostic items",
              "arms": {}}
    for arm, all_rows in arms.items():
        rows = [r for r in all_rows if r["diagnostic"]]
        classes = Counter(r["class"] for r in rows)
        report["arms"][arm] = {
            "label": ARM_LABEL[arm],
            "n": len(rows),
            "n_all_items": len(all_rows),
            "accuracy_all_items_legacy": proportion(
                all_rows, lambda r: r["class"] == "correct"),
            "accuracy": proportion(rows, lambda r: r["class"] == "correct"),
            "classes": {c: classes.get(c, 0) for c in CLASSES},
            "published_subset": proportion(
                [r for r in rows if r["meta"]["published_subset"]],
                lambda r: r["class"] == "correct"),
            "by_factor_given": stratified(rows, lambda r: r["meta"]["factor_given"]),
            "by_framing": stratified(rows, lambda r: r["meta"]["framing"]),
            "by_direction": stratified(rows, lambda r: r["meta"]["direction"]),
            "by_pair_kind": stratified(rows, lambda r: r["meta"]["pair_kind"]),
            "by_unit_pair": stratified(
                rows, lambda r: f'{r["meta"]["from_unit"]}->{r["meta"]["to_unit"]}'),
            "by_decade": stratified(rows, lambda r: decade_of(r["meta"]["value"])),
            "contrast_framing": paired_contrast(
                rows, "framing", ["value", "from_unit", "to_unit", "factor_given"]),
            "contrast_factor_given": paired_contrast(
                [r for r in rows if r["meta"]["pair_kind"] == "metric"],
                "factor_given", ["value", "from_unit", "to_unit", "framing"]),
            "visual_analog": visual_analog(rows),
        }
    report["arm_vs_base"] = arm_vs_base(arms)
    report["text_vs_visual"] = dissociation(
        {arm: [r for r in rows if r["diagnostic"]] for arm, rows in arms.items()})
    return report


# ------------------------------------------------------------------------ render

def pct(x):
    return f"{100 * x:.1f}"


def ci(d):
    if not d or not d.get("n"):
        return "--"
    return f"{pct(d['p'])} [{pct(d['lo'])}, {pct(d['hi'])}]"


def render_markdown(report) -> str:
    lines = ["# Text-only unit-conversion probe v2", "",
             "392 items per checkpoint; the 12 published items are a verbatim subset.",
             "Accuracy = published criterion |error| < 0.05. Wilson 95% intervals.",
             "Basis: the 246 diagnostic items, where an unconverted answer would be "
             "scored wrong (see is_diagnostic). All-392 figure kept for continuity only.",
             ""]
    lines += ["## Accuracy", "",
              "| checkpoint | diagnostic 246 | all 392 (legacy) | published 12 | "
              "factor given | factor withheld |",
              "|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        a = report["arms"].get(arm)
        if not a:
            continue
        fg = a["by_factor_given"]
        lines.append(f"| {a['label']} | {ci(a['accuracy'])} | "
                     f"{ci(a['accuracy_all_items_legacy'])} | "
                     f"{a['published_subset']['k']}/{a['published_subset']['n']} | "
                     f"{ci(fg.get(True))} | {ci(fg.get(False))} |")
    lines += ["", "## How the wrong answers are wrong", "",
              "| checkpoint | " + " | ".join(CLASSES) + " |",
              "|---|" + "---|" * len(CLASSES)]
    for arm in ARM_ORDER:
        a = report["arms"].get(arm)
        if not a:
            continue
        lines.append(f"| {a['label']} | " +
                     " | ".join(str(a["classes"][c]) for c in CLASSES) + " |")
    lines += ["", "## Text-only twins of the visual unit arms", "",
              "| checkpoint | m->cm embedded | m->px embedded | m->cm returned unchanged |",
              "|---|---|---|---|"]
    for arm in ARM_ORDER:
        a = report["arms"].get(arm)
        if not a:
            continue
        v = a["visual_analog"]
        lines.append(f"| {a['label']} | {ci(v['m_to_cm_embedded'])} | "
                     f"{ci(v['m_to_px_embedded'])} | "
                     f"{v['m_to_cm_embedded_identity']['k']}/"
                     f"{v['m_to_cm_embedded_identity']['n']} |")
    lines += ["", "## Paired contrasts (exact McNemar)", ""]
    for arm in ARM_ORDER:
        a = report["arms"].get(arm)
        if not a:
            continue
        f, g = a["contrast_framing"], a["contrast_factor_given"]
        parts = []
        if f:
            parts.append(f"framing bare vs embedded: {pct(f['acc_bare'])} vs "
                         f"{pct(f['acc_embedded'])}, p = {f['mcnemar_p']:.3g} "
                         f"({f['pairs']} pairs)")
        if g:
            parts.append(f"factor withheld vs given: {pct(g['acc_False'])} vs "
                         f"{pct(g['acc_True'])}, p = {g['mcnemar_p']:.3g} "
                         f"({g['pairs']} pairs)")
        lines.append(f"- **{a['label']}** " + ("; ".join(parts) if parts
                                              else "no paired contrast in this subset"))
    lines += ["", "## Same checkpoint, same unit request: text vs image", "",
              "Median ratio of the requested-unit answer to the metre quantity "
              "(text: stated value; image: metre-arm answer for the same clip).",
              "Required: cm/m = 100, px/m = 27.498.", "",
              "| checkpoint | text cm/m | image cm/m (speed, path) | text px/m | "
              "image px/m (speed, path) | text px ratios in (0.5, 2) |",
              "|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        d = report["text_vs_visual"].get(arm)
        if not d:
            continue
        tc, tp, v = d["text_cm_over_m"], d["text_px_over_m"], d["visual"]
        def vis(key):
            if not v or key not in v:
                return "not run"
            return ", ".join(f"{v[key][f]['median']:.2f}" for f in ("speed", "path"))
        lines.append(
            f"| {ARM_LABEL[arm]} | {tc['median']:.2f} | {vis('cm')} | "
            f"{tp['median']:.2f} | {vis('px')} | "
            f"{tp['within_half_to_double_of_one']}/{tp['n']} |")
    if report["arm_vs_base"]:
        lines += ["", "## Each adapter against the base model (same items)", ""]
        for arm, d in report["arm_vs_base"].items():
            lines.append(f"- {ARM_LABEL[arm]}: base-only correct {d['only_base_correct']}, "
                         f"adapter-only correct {d['only_arm_correct']}, "
                         f"McNemar p = {d['mcnemar_p']:.3g}")
    return "\n".join(lines) + "\n"


def fmt_p(p: float) -> str:
    """p-value for print: plain above 1e-3, otherwise a power of ten."""
    if p >= 1e-3:
        return f"{p:.3f}"
    exponent = math.floor(math.log10(p))
    mantissa = p / 10 ** exponent
    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


BASE_VISUAL = ROOT / "results" / "a100" / "base_nf4_Q2_top_480-510_full"


def base_visual_distinct():
    """Distinct parsed answers per image arm and family for the unadapted backbone.

    The backbone's image arms are near-constant, so its paired ratios would be a
    constant divided by a constant.  The table reports that fact instead of a
    number that looks like a measurement.
    """
    out = {}
    for arm in ("m_height", "cm_height", "px_height"):
        path = BASE_VISUAL / f"{arm}.jsonl"
        if not path.is_file():
            return None
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        for family in ("dynamics_speed_player", "dynamics_path_player"):
            out[(arm, family)] = len({r["prediction"] for r in rows
                                      if r["category"] == family
                                      and r["prediction"] is not None})
    return out


def render_latex(report) -> str:
    """Manuscript table.  Generated, never hand-edited, so it cannot drift."""
    base_distinct = base_visual_distinct()
    rows = []
    for arm in ARM_ORDER:
        a = report["arms"].get(arm)
        d = report["text_vs_visual"].get(arm)
        if not a or not d:
            continue
        acc = a["accuracy"]
        versus = report["arm_vs_base"].get(arm)
        p_cell = "---" if arm == "base" else fmt_p(versus["mcnemar_p"])
        v = d["visual"]
        def vis(key):
            if arm == "base" and base_distinct:
                return "degen."
            if not v or key not in v:
                return "n/a"
            return "/".join(f"{v[key][f]['median']:.2f}" for f in ("speed", "path"))
        rows.append(
            f"{ARM_LABEL[arm]} & {100 * acc['p']:.1f} [{100 * acc['lo']:.1f}, "
            f"{100 * acc['hi']:.1f}] & {p_cell} & "
            f"{d['text_cm_over_m']['median']:.1f} & {vis('cm')} & "
            f"{d['text_px_over_m']['median']:.2f} & {vis('px')} " + r"\\")
    n = next(iter(report["arms"].values()))["n"]
    caption = (
        r"\caption{Text-only unit conversion against the same checkpoints' visual "
        r"unit responses, Q2. Accuracy is on the " + str(n) + r" diagnostic items of "
        r"the 392-item probe (items on which an unconverted answer would be scored "
        r"wrong), with Wilson 95\% intervals; $p$ is an exact McNemar test against "
        r"the unadapted backbone on the same items. Ratios are medians of the "
        r"requested-unit answer over the metre quantity: the stated value in text, "
        r"the metre-arm answer for the same clip in the image arms (speed/path). "
        r"Required: cm/m $=100$, px/m $=27.50$."
        + (r" The unadapted backbone's image arms are degenerate, with "
           + f"{min(base_distinct.values())}--{max(base_distinct.values())}"
           + r" distinct answers per arm and family, so their ratios are not "
           r"interpretable." if base_distinct else "")
        + "}")
    return "\n".join([
        "% Generated by tools/analyze_unit_probe.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        caption,
        r"\label{tab:unit_probe}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{cm/m} & \multicolumn{2}{c}{px/m} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Checkpoint & Text acc.\ (\%) & $p$ vs base & text & image & text & image \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        ""])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(DEFAULT_RUNS))
    args = ap.parse_args()
    runs = Path(args.runs)
    arms = load_runs(runs)
    if not arms:
        print(f"no results under {runs}")
        return 1
    incomplete = {a: len(r) for a, r in arms.items() if len(r) != 392}
    if incomplete:
        print(f"WARNING: incomplete arms {incomplete}; analysing what exists")
    report = analyse(arms)
    (runs / "analysis.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    markdown = render_markdown(report)
    (runs / "analysis.md").write_text(markdown, encoding="utf-8")
    table = ROOT / "paper2" / "manuscript" / "tables" / "unit_probe.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(render_latex(report), encoding="utf-8")
    print(f"-> {table}")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
