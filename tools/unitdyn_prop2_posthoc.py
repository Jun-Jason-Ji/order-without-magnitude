#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""POST HOC, DESCRIPTIVE. Proposition 2 of the UnitDyn design (docs/paper2_followup_design_2026-09-30):
under token-level cross-entropy with greedy decoding, an adapter whose reading quantises the input into
G buckets answers, in a second unit, the conditional MODE of that unit's training targets within the
bucket. Tested here on paper 2's mixed-unit adapters (M1 seeds 42/43/44, M1-fine seed 42): the bucket is
the adapter's OWN metre answer on the same item (paired m / cm arms of the same test cell), the targets
are the centimetre items of the adapter's own training pool.

For each cell (adapter x clip x family) we predict the cm answer of every item three ways and score
exact agreement (|pred - answer| < 0.05) with the adapter's actual cm answer:
  cond_prefix  PREFIX-GREEDY mode: greedy decoding over single-digit tokens takes, at each digit position,
               the majority digit among the bucket's targets sharing the prefix so far (Qwen tokenises
               numbers digit by digit); this is the actual argmax of a token-level model, not the mode
               of the number. Added after the first run showed cond_mode = 0 on speed.
  cond_mode    mode of cm training targets whose metre-equivalent falls in the Voronoi cell of the
               item's metre reading on the adapter's own reading grid (the proposition)
  global_mode  mode of all cm training targets of that family (G = 1 limit)
  exact_x100   100 x the metre reading (what a converting model would say, rounded to 1 dp)
  nearest_tgt  the cm training target nearest to 100 x reading (converting model restricted to the
               training vocabulary)
Also reported: G (distinct metre readings), distinct cm answers, Spearman(cm answer, metre reading).
No decision rule; this informs the wording of the proposition before it is preregistered for UnitDyn.
Writes results/courtdyn/unitdyn_posthoc/prop2_conditional_mode.{json,md}.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from a100.analyze import item_key           # noqa: E402
from a100.eval_controls import spearman     # noqa: E402

CD = ROOT / "results" / "courtdyn"
OUT = CD / "unitdyn_posthoc"
RUNS = {  # label -> (pool json, eval dir, cell prefix)
    "M1 s42": (CD / "m1_mixunit" / "mixunit_v3_pool.json", CD / "m1_mixunit" / "eval", "mixunit"),
    "M1 s43": (CD / "m1_mixunit" / "mixunit_v3_pool.json", CD / "m1_mixunit" / "eval", "mixunit_s43"),
    "M1 s44": (CD / "m1_mixunit" / "mixunit_v3_pool.json", CD / "m1_mixunit" / "eval", "mixunit_s44"),
    "M1-fine s42": (CD / "m1_fine" / "mixunit_fine_v3_pool.json", CD / "m1_fine" / "eval", "mixfine"),
}
SEQS = ["Q1_top_0-30", "Q2_top_480-510"]
FAMILIES = {"dynamics_speed_player": "speed", "dynamics_path_player": "path"}


def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def unit_of(item):
    return item["meta"].get("m1_unit") or ("cm" if "centimet" in item["question"].lower() else "m")


def cm_targets(pool, family):
    """(metre-equivalent, cm target) for the centimetre items of one family."""
    out = []
    for it in pool:
        if it["category"] != family or unit_of(it) != "cm":
            continue
        t = float(it["answer"])
        out.append((t / 100.0, t))
    return out


def mode(values):
    c = Counter(values)
    top = max(c.values())
    return min(v for v, n in c.items() if n == top)     # deterministic tie-break


def prefix_greedy(values):
    """Greedy decoding over single-digit tokens: at each position take the majority digit among the
    targets that share the prefix chosen so far (Qwen tokenises numbers digit by digit)."""
    strs = [f"{v:.1f}" for v in values]
    prefix = ""
    while True:
        cands = [t for t in strs if t.startswith(prefix)]
        if not cands:
            break
        nxt = Counter(t[len(prefix)] for t in cands if len(t) > len(prefix))
        done = sum(1 for t in cands if len(t) == len(prefix))
        if not nxt or done > max(nxt.values()):
            break
        top = max(nxt.values())
        prefix += min(d for d, n in nxt.items() if n == top)
    try:
        return float(prefix)
    except ValueError:
        return None


def voronoi_bucket(x, grid):
    return min(grid, key=lambda g: (abs(g - x), g))


def analyse(label, pool_path, ev, prefix):
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    res = []
    for seq in SEQS:
        cell = ev / f"{prefix}_{seq}_full"
        if not (cell / "m_height.jsonl").is_file():
            continue
        m = {item_key(r): r for r in rows(cell / "m_height.jsonl")}
        cm = {item_key(r): r for r in rows(cell / "cm_height.jsonl")}
        for family, short in FAMILIES.items():
            pairs = [(m[k]["prediction"], cm[k]["prediction"]) for k in m if k in cm
                     and m[k]["category"] == family and m[k]["prediction"] is not None
                     and cm[k]["prediction"] is not None]
            if len(pairs) < 10:
                continue
            targets = cm_targets(pool, family)
            grid = sorted({r for r, _ in pairs})
            gmode = mode([t for _, t in targets])
            tvals = sorted({t for _, t in targets})
            score = {"cond_mode": 0, "global_mode": 0, "exact_x100": 0, "nearest_tgt": 0,
                     "cond_prefix": 0, "global_prefix": 0}
            gprefix = prefix_greedy([t for _, t in targets])
            cond_pred, cond_pre = {}, {}
            for g in grid:
                inb = [t for me, t in targets if voronoi_bucket(me, grid) == g]
                cond_pred[g] = mode(inb) if inb else gmode
                cond_pre[g] = prefix_greedy(inb) if inb else gprefix
            for r, a in pairs:
                preds = {"cond_mode": cond_pred[r], "global_mode": gmode,
                         "exact_x100": round(100 * r, 1),
                         "nearest_tgt": min(tvals, key=lambda t: (abs(t - 100 * r), t)),
                         "cond_prefix": cond_pre[r], "global_prefix": gprefix}
                for k, v in preds.items():
                    score[k] += v is not None and abs(v - a) < 0.05
            n = len(pairs)
            rho = spearman([r for r, _ in pairs], [a for _, a in pairs]) if len(grid) > 1 and len({a for _, a in pairs}) > 1 else None
            res.append({"adapter": label, "seq": seq, "family": short, "n": n,
                        "G_metre": len(grid), "distinct_cm": len({a for _, a in pairs}),
                        "cm_answers": Counter(a for _, a in pairs).most_common(4),
                        "global_mode_cm_target": gmode, "global_prefix_greedy": gprefix,
                        "cond_prefix_by_reading": {str(g): cond_pre[g] for g in grid},
                        "cond_mode_by_reading": {str(g): cond_pred[g] for g in grid},
                        "rho_cm_vs_reading": rho,
                        "match": {k: v / n for k, v in score.items()}})
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    allres = []
    for label, (pool, ev, prefix) in RUNS.items():
        if pool.is_file():
            allres += analyse(label, pool, ev, prefix)
    (OUT / "prop2_conditional_mode.json").write_text(json.dumps(allres, indent=1), encoding="utf-8")
    f = lambda x: "--" if x is None else f"{x:.2f}"
    md = ["# Proposition 2 (collapse to the conditional mode) - POST HOC test on paper-2 mixed-unit adapters",
          "", "Descriptive; no decision rule. Bucket = the adapter's own metre answer (Voronoi cell on its reading grid); "
          "targets = centimetre items of its own training pool. Match = exact agreement with the adapter's cm answer.", "",
          "| adapter | clip | family | n | G (metre readings) | distinct cm answers | top cm answers | global mode of cm targets | "
          "rho(cm, reading) | cond_mode | global_mode | **cond_prefix** | global_prefix | exact x100 | nearest target |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in allres:
        top = ", ".join(f"{a:g}x{n}" for a, n in r["cm_answers"])
        mt = r["match"]
        md.append(f"| {r['adapter']} | {r['seq'].split('_')[0]} | {r['family']} | {r['n']} | {r['G_metre']} | {r['distinct_cm']} | "
                  f"{top} | {r['global_mode_cm_target']:g} / prefix {r['global_prefix_greedy']:g} | {f(r['rho_cm_vs_reading'])} | {mt['cond_mode']:.2f} | "
                  f"{mt['global_mode']:.2f} | **{mt['cond_prefix']:.2f}** | {mt['global_prefix']:.2f} | {mt['exact_x100']:.2f} | {mt['nearest_tgt']:.2f} |")
    md += ["", "Reading of the columns: if cm answers are a per-unit prior, cond_mode/global_mode are high and exact x100 is low; "
           "if the adapter converts its reading, exact x100 / nearest target are high. Speed vs path is the contrast the "
           "proposition predicts (few distinct readings -> collapse)."]
    (OUT / "prop2_conditional_mode.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
