#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""POST HOC robustness of the representation-probe verdicts to the cross-validation split.

Found on 2026-09-28 after the preregistered analysis had run: tools/repr_probe_analyze.py draws
fold assignments, permutation nulls and bootstrap resamples from ONE module-level generator
seeded once, so a model's folds depend on how many models were analysed before it.  The early
v3-only run and the full four-model run therefore used different folds, and v3's H2 cells
changed (feature last_L16 -> last_L20; speed floor-plane -> untestable).  The rules do not fix the
fold randomisation, so no single run is privileged.

This script leaves the preregistered analysis untouched.  For each model and each of N seeds it
reseeds the generator with a (seed, model)-specific value before analysing that model, and
records the selected feature and every per-family verdict (H1, H3, H2) and the H2 model verdict.
It reports, per model and verdict, the fraction of seeds giving each outcome.  Labelled post hoc.

Writes results/courtdyn/repr_probe/stability.json and stability.md.
"""
from __future__ import annotations

import collections
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import repr_probe_analyze as A                                   # noqa: E402

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def summarise(rep):
    out = {"feature": rep["selected_feature"], "H2_model": rep.get("H2_model_verdict")}
    for fam, f in rep["families"].items():
        out[f"{fam}/H1"] = (f.get("H1") or {}).get("holds")
        out[f"{fam}/H3"] = (f.get("H3") or {}).get("motion_dependent")
        h2 = f.get("H2")
        out[f"{fam}/H2"] = h2.get("verdict") if isinstance(h2, dict) else h2
        if isinstance(h2, dict):
            out[f"{fam}/H2_delta"] = h2.get("delta")
    return out


def main() -> int:
    models = [m for m in A.MODELS if (A.FEAT / m / f"{A.TOP[0]}.npz").is_file()]
    runs = {m: [] for m in models}
    for seed in range(N_SEEDS):
        for m in models:
            A.rng = np.random.default_rng([seed, zlib.crc32(m.encode())])
            runs[m].append(summarise(A.analyse(m)))
            print(f"seed {seed} {m}: {runs[m][-1]}", flush=True)
    table = {}
    for m, rs in runs.items():
        keys = sorted({k for r in rs for k in r if not k.endswith("_delta")})
        table[m] = {k: dict(collections.Counter(str(r.get(k)) for r in rs)) for k in keys}
        deltas = {k: [r[k] for r in rs if r.get(k) is not None]
                  for k in {k for r in rs for k in r if k.endswith("_delta")}}
        table[m]["H2_delta_range"] = {k: [min(v), max(v)] for k, v in deltas.items() if v}
    h4 = []
    if "v3" in runs and "mixunit" in runs:
        for rv, rm in zip(runs["v3"], runs["mixunit"]):
            h4.append(all(rv.get(f"{f}/H2") == rm.get(f"{f}/H2") for f in A.FAMS))
    report = {"post_hoc": True, "n_seeds": N_SEEDS, "table": table,
              "H4_same_scale_fraction": (sum(h4) / len(h4)) if h4 else None, "runs": runs}
    (A.OUTD / "stability.json").write_text(json.dumps(report, indent=1, default=str) + "\n",
                                           encoding="utf-8")
    md = [f"# Probe verdict stability over {N_SEEDS} fold seeds (POST HOC)", ""]
    for m, t in table.items():
        md += [f"## {m}", ""] + [f"- {k}: {v}" for k, v in t.items()] + [""]
    if h4:
        md.append(f"H4 (v3 and mixunit same H2 verdict in both families): {sum(h4)}/{len(h4)} seeds")
    (A.OUTD / "stability.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
