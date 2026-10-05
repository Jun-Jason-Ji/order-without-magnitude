#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cluster-bootstrap 95% intervals for the headline numbers of paper 2.

Items are not independent: questions from the same time window share frames, and
questions about the same player share a trajectory.  The primary interval resamples
TIME WINDOWS (all items of a drawn window enter together; same "instant-clustered"
convention as paper 1); a track-clustered interval is reported alongside as a
sensitivity check.  Percentile intervals, B = 2000, fixed seed.  Point estimates are
recomputed with the same per-item scoring code that produced the published numbers,
and the script asserts that they match before any interval is reported.

Blocks
  native   tab:nativecore (CourtDyn-native, Q2): metre rho / T-MRA / margin over the
           median constant; pixel rho / margin; paired pixel/full ratio R.  Scoring is
           tools/summarize_courtdyn_t28.py's (v3 homography metres; pixel tolerance x K).
  seeds    tab:seeds (3 pools x 3 seeds, Q2): rho, T-MRA, delta rho (full - first frame).
  m1       mixed-unit adapter, seeds 42-44, Q1/Q2: paired cm/m ratio R.
  soccer   SoccerNet venue, metre-only v3 adapter: rho and delta rho (exploratory venue).

Writes results/courtdyn/bootstrap_ci.json and paper2/manuscript/tables/bootstrap_ci.tex.
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100.eval_controls import scalar_score, spearman  # noqa: E402

B, SEED = 2000, 20260926
OUT_JSON = ROOT / "results" / "courtdyn" / "bootstrap_ci.json"
OUT_TEX = ROOT / "paper2" / "manuscript" / "tables" / "bootstrap_ci.tex"
FAMS = [("dynamics_speed_player", "speed"), ("dynamics_path_player", "path")]


# ------------------------------------------------------------------ bootstrap core
def cluster_boot(items, cluster_of, stat, b=B, seed=SEED):
    """items: list; cluster_of(item) -> hashable; stat(list) -> float | None."""
    point = stat(items)
    groups = {}
    for it in items:
        groups.setdefault(cluster_of(it), []).append(it)
    keys = list(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(b):
        sample = [it for _ in keys for it in groups[keys[rng.randrange(len(keys))]]]
        v = stat(sample)
        if v is not None:
            draws.append(v)
    if len(draws) < 0.9 * b:          # statistic undefined too often (degenerate resamples)
        return {"point": point, "lo": None, "hi": None, "clusters": len(keys),
                "defined_draws": len(draws)}
    draws.sort()
    return {"point": point, "lo": draws[int(0.025 * len(draws))],
            "hi": draws[int(0.975 * len(draws)) - 1], "clusters": len(keys),
            "defined_draws": len(draws)}


def both(items, stat, window_of, track_of):
    return {"window": cluster_boot(items, window_of, stat),
            "track": cluster_boot(items, track_of, stat)}


def safe_rho(xs, ys):
    return spearman(xs, ys) if len(set(xs)) > 1 and len(set(ys)) > 1 else None


# ------------------------------------------------------------------ block: native
def block_native():
    import summarize_courtdyn_t28 as T
    seq = "Q2_top_480-510"
    ctx = T.seq_ctx(seq)
    K, fam_of = ctx["K"], ctx["fam_of"]
    published = json.loads((ROOT / "results" / "courtdyn" / "courtdyn_t28_findings.json")
                           .read_text(encoding="utf-8"))["seqs"][seq]
    full = T.preds(T.full_parsed(seq, "cdnative"))
    pxu = T.preds(T.t28_parsed(seq, "pxunit", "full", "cdnative"))
    out = {}
    for _, fam in FAMS:
        items = [{"key": k, "full": full.get(k), "px": pxu.get(k), "v3": ctx["v3"].get(k),
                  "gpx": ctx["px"].get(k)} for k in fam_of if fam_of[k] == fam]

        def metre_rows(s):
            return [(i["full"], i["v3"]) for i in s if i["full"] is not None and i["v3"] is not None]

        def px_rows(s):
            return [(i["px"], i["gpx"]) for i in s if i["px"] is not None and i["gpx"] is not None]

        def margin(rows, k):
            if not rows:
                return None
            med = st.median(g for _, g in rows)
            return T.tmra(rows, fam, k) - T.tmra([(med, g) for _, g in rows], fam, k)

        stats = {
            "metre_rho": lambda s: safe_rho(*zip(*metre_rows(s))) if metre_rows(s) else None,
            "metre_tmra": lambda s: T.tmra(metre_rows(s), fam),
            "metre_margin": lambda s: margin(metre_rows(s), 1.0),
            "pixel_rho": lambda s: safe_rho(*zip(*px_rows(s))) if px_rows(s) else None,
            "pixel_margin": lambda s: margin(px_rows(s), K),
            "pixel_ratio_R": lambda s: (st.median([i["px"] / i["full"] for i in s
                                                   if i["px"] is not None and i["full"]
                                                   and i["full"] > 0]) or None),
        }
        res = {name: both(items, f, lambda i: i["key"][1], lambda i: i["key"][2])
               for name, f in stats.items()}
        # published point estimates must be reproduced before intervals mean anything
        pub_full = published["cells"][f"cdnative@ruler@{fam}"]["full"]
        pub_px = published["cells"][f"cdnative@pxunit@{fam}"]
        assert abs(res["metre_tmra"]["window"]["point"] - pub_full["tmra"]) < 1e-9, fam
        assert abs(res["pixel_rho"]["window"]["point"] - pub_px["pxunit"]["rho"]) < 1e-9, fam
        assert abs(res["pixel_ratio_R"]["window"]["point"] - pub_px["ratio_to_full"]) < 1e-9, fam
        out[fam] = res
    return out


# ------------------------------------------------------------------ jsonl cells (a100 schema)
def load_rows(directory: Path, arm: str):
    path = directory / f"{arm}.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["category"], r["track"], tuple(r["window"])): r for r in rows}


def jsonl_block(full_dir, static_dir, family, with_ratio=None, common_items=False):
    """common_items=True computes delta rho on items parsed in BOTH conditions, the
    convention of a100.analyze.cell_report (soccer summary); False keeps each arm's own
    parsed items, the convention of tools/analyze_seed_sweep.py (tab:seeds)."""
    m = load_rows(full_dir, "m_height")
    s4 = load_rows(static_dir, "m_height") if static_dir else None
    other = load_rows(full_dir, with_ratio) if with_ratio else None
    keys = [k for k in m if k[0] == family]
    items = [{"k": k, "m": m[k], "s": (s4 or {}).get(k), "o": (other or {}).get(k)} for k in keys]

    def rho(s, arm="m"):
        v = [i[arm] for i in s if i[arm] is not None and i[arm]["prediction"] is not None]
        return safe_rho([r["prediction"] for r in v], [r["reference_value"] for r in v]) if v else None

    def tmra(s):
        return sum(scalar_score(i["m"]["prediction"], i["m"]["reference_value"],
                                i["m"]["score_tolerance"], i["m"]["score_floor"])
                   if i["m"]["prediction"] is not None else 0.0 for i in s) / len(s)

    def delta(s):
        if common_items:
            s = [i for i in s if i["s"] is not None and i["s"]["prediction"] is not None
                 and i["m"]["prediction"] is not None]
        a, b = rho(s), rho(s, "s")
        return None if a is None or b is None else a - b

    def ratio(s):
        r = [i["o"]["prediction"] / i["m"]["prediction"] for i in s
             if i["o"] and i["o"]["prediction"] is not None
             and i["m"]["prediction"] not in (None, 0) and i["m"]["prediction"] > 0]
        return st.median(r) if r else None

    stats = {"rho": rho, "tmra": tmra}
    if s4:
        stats["delta_rho"] = delta
    if other:
        stats["ratio"] = ratio
    return {n: both(items, f, lambda i: i["k"][2], lambda i: i["k"][1]) for n, f in stats.items()}


def block_seeds():
    import analyze_seed_sweep as S
    out = {}
    for pool in S.POOLS:
        for seed in S.SEEDS:
            s4 = S.cell_dir(pool, seed, "static4")
            for fam, short in FAMS:
                res = jsonl_block(S.cell_dir(pool, seed, "full"),
                                  s4 if (s4 / "m_height.jsonl").is_file() else None, fam)
                pub = S.stats(pool, seed, fam)
                assert abs(res["tmra"]["window"]["point"] - pub["tmra"]) < 1e-9
                out[f"{pool}/s{seed}/{short}"] = res
    return out


def block_m1():
    base = ROOT / "results" / "courtdyn" / "m1_mixunit" / "eval"
    out = {}
    for seed in (42, 43, 44):
        tag = "" if seed == 42 else f"_s{seed}"
        for seq in ("Q1_top_0-30", "Q2_top_480-510"):
            for fam, short in FAMS:
                res = jsonl_block(base / f"mixunit{tag}_{seq}_full", None, fam, with_ratio="cm_height")
                out[f"s{seed}/{seq}/{short}"] = {"cm_over_m": res["ratio"]}
    return out


def block_soccer():
    ev = Path(r"E:\datasets\SoccerNet\derived_private\courtdyn_venue\eval")
    summary = json.loads((ROOT / "results" / "courtdyn" / "soccer_venue" / "summary.json")
                         .read_text(encoding="utf-8"))["cells"]
    out = {}
    for seq in ("SNGS-034", "SNGS-056", "SNGS-096"):
        for fam, short in FAMS:
            res = jsonl_block(ev / f"qwen35_v3_{seq}_full", ev / f"qwen35_v3_{seq}_static4", fam,
                              common_items=True)
            pub = summary[f"qwen35_v3/{seq}"]["families"][short]
            assert abs(res["rho"]["window"]["point"] - pub["m_height"]["rho"]) < 1e-9
            assert abs(res["delta_rho"]["window"]["point"] - pub["static4"]["delta_rho"]) < 1e-9
            out[f"{seq}/{short}"] = {k: res[k] for k in ("rho", "delta_rho")}
    return out


# ------------------------------------------------------------------ output
def ci(c, d=2):
    w = c["window"]
    if w["point"] is None:
        return "---"
    if w["lo"] is None:
        return f"{w['point']:.{d}f} (---)"
    return f"{w['point']:.{d}f} [{w['lo']:.{d}f}, {w['hi']:.{d}f}]"


def render_tex(r):
    nat = r["native"]
    lines = [
        "% Generated by tools/bootstrap_paper2_ci.py -- do not edit by hand.",
        r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{95\% cluster-bootstrap intervals (time windows resampled, $B=2000$) for the "
        r"headline quantities. Margin is T-MRA minus the median constant of the same resample; "
        r"$R$ is the paired pixel/metre prediction ratio (about 27 if the unit were applied). "
        r"Track-clustered intervals are in the released JSON.}",
        r"\label{tab:bootstrap}", r"\begin{tabular}{lll}", r"\toprule",
        r"Quantity (CourtDyn-native, Q2) & speed & path \\", r"\midrule"]
    for name, label in (("metre_rho", r"metre $\rho$"), ("metre_margin", "metre margin"),
                        ("pixel_rho", r"pixel $\rho$"), ("pixel_margin", "pixel margin"),
                        ("pixel_ratio_R", r"pixel/metre $R$")):
        d = 1 if "margin" in name else 2
        lines.append(f"{label} & {ci(nat['speed'][name], d)} & {ci(nat['path'][name], d)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = {"method": "percentile cluster bootstrap", "B": B, "seed": SEED,
              "primary_cluster": "time window", "sensitivity_cluster": "track",
              "native": block_native(), "seeds": block_seeds(), "m1": block_m1(),
              "soccer": block_soccer()}
    OUT_JSON.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    OUT_TEX.write_text(render_tex(report), encoding="utf-8")
    print(render_tex(report))
    for blk in ("seeds", "m1", "soccer"):
        print(f"--- {blk}")
        for key, stats in report[blk].items():
            print(" ", key, " | ".join(f"{n} {ci(c)}" for n, c in stats.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
