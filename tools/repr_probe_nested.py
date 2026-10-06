#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post hoc re-analysis of the representation probe with feature selection inside the training folds.

The preregistered analysis (tools/repr_probe_analyze.py) selects the (layer, pooling) feature once on
all overhead items and then scores H1 on the same items, with a permutation null that keeps the
selected feature and penalty.  Here every step is repeated inside each outer fold:

  outer 5-fold grouped CV (groups = clip x 2-s window start) on the two overhead clips;
  within the training part of a fold: feature selection by inner grouped-CV R^2 over the 16
  (layer, pooling) candidates, penalty by inner grouped CV, ridge fit;
  H1: Spearman of the out-of-fold predictions against log metre, against a null of B label
      permutations within clip that each repeat the whole nested procedure (99th percentile);
  H3: the same fold probes applied to the first-frame x4 features of the held-out Q2_top windows,
      so full and static predictions come from identical training exposure; motion-dependent if
      rho_static <= rho_full(Q2_top, out-of-fold) - 0.30.

Standardisation uses the unsupervised column statistics of each candidate feature over all
overhead items (no labels), which lets the ridge be solved in kernel form from one precomputed Gram
matrix per candidate.  Written 2026-10-05 after an external review; it does not replace the
preregistered verdicts and is reported as a post hoc check.

  python tools/repr_probe_nested.py [--models base native v3 mixunit] [--perms 200] [--seed 20261005]
Writes results/courtdyn/repr_probe/nested_reanalysis.json (+ .md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "results" / "courtdyn" / "repr_probe"
FEAT = OUTD / "features"
MODELS = ["base", "native", "v3", "mixunit"]
TOP = ["Q2_top_480-510", "Q1_top_0-30"]
STATIC = "Q2_top_480-510__static4"
FAMS = {"speed": "dynamics_speed_player", "path": "dynamics_path_player"}
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
POOLS = ["img", "last"]
KEYS = [f"{p}_L{L}" for p in POOLS for L in LAYERS]
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
EPS_M = 0.05


def load(model, clip):
    d = np.load(FEAT / model / f"{clip}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def subset(d, fam):
    idx = d["category"] == FAMS[fam]
    return {k: (v[idx] if len(v) == len(idx) else v) for k, v in d.items()}


def cat(*ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


# ------------------------------------------------------------------ kernel ridge on precomputed Grams
def folds(g, k, rng):
    ug = np.unique(g)
    perm = rng.permutation(len(ug))
    return [np.isin(g, ug[perm[i::k]]) for i in range(k)]


def kr_predict(K, y, tr, te, alpha):
    """ridge in dual form on training index tr, predictions for te; intercept = train mean."""
    ym = y[tr].mean()
    a = np.linalg.solve(K[np.ix_(tr, tr)] + alpha * np.eye(tr.sum()), y[tr] - ym)
    return K[np.ix_(te, tr)] @ a + ym


def inner_cv_score(K, y, g, tr, rng, k=5):
    """best (alpha, R^2) by grouped inner CV restricted to tr."""
    idx = np.where(tr)[0]
    best = (None, -np.inf)
    for alpha in ALPHAS:
        pred = np.empty(len(idx))
        for te_mask in folds(g[idx], k, rng):
            te_i, tr_i = idx[te_mask], idx[~te_mask]
            pred[te_mask] = kr_predict(K, y, _mask(tr_i, len(y)), _mask(te_i, len(y)), alpha)
        yy = y[idx]
        r2 = 1 - np.sum((pred - yy) ** 2) / np.sum((yy - yy.mean()) ** 2)
        if r2 > best[1]:
            best = (alpha, r2)
    return best


def _mask(idx, n):
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def nested_oof(Ks, y, g, rng, K_static=None, static_rows=None, k=5):
    """Out-of-fold predictions with feature + penalty selection inside each training fold.
    Returns oof predictions, per-fold selections, and (if K_static given) predictions for the
    static rows of each held-out fold (static_rows[i] = overhead row index that row i mirrors)."""
    n = len(y)
    oof = np.empty(n)
    static_pred = None if K_static is None else np.full(len(static_rows), np.nan)
    chosen = []
    for te in folds(g, k, rng):
        tr = ~te
        sel = None
        for key in KEYS:
            alpha, r2 = inner_cv_score(Ks[key], y, g, tr, rng)
            if sel is None or r2 > sel[2]:
                sel = (key, alpha, r2)
        key, alpha, _ = sel
        oof[te] = kr_predict(Ks[key], y, tr, te, alpha)
        chosen.append({"feature": key, "alpha": alpha})
        if K_static is not None:
            # rows of the static set whose overhead counterpart is held out in this fold
            held = te[static_rows]
            if held.any():
                ym = y[tr].mean()
                a = np.linalg.solve(Ks[key][np.ix_(tr, tr)] + alpha * np.eye(tr.sum()), y[tr] - ym)
                static_pred[held] = K_static[key][np.ix_(held, tr)] @ a + ym
    return oof, chosen, static_pred


def gram(X):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return Z @ Z.T


def cross_gram(Xs, Xref):
    mu, sd = Xref.mean(0), Xref.std(0) + 1e-6
    return ((Xs - mu) / sd) @ ((Xref - mu) / sd).T


def analyse(model, fam, perms, rng):
    tops = {c: subset(load(model, c), fam) for c in TOP}
    d = cat(*tops.values())
    g = np.concatenate([np.array([f"{c}|{w}" for w in tops[c]["window0"]]) for c in TOP])
    clipid = np.concatenate([[c] * len(tops[c]["metric"]) for c in TOP])
    y = np.log(d["metric"].astype(np.float64) + EPS_M)
    Ks = {k: gram(d[k].astype(np.float64)) for k in KEYS}
    # static set mirrors Q2_top rows one-to-one (same items, first frame repeated)
    st = subset(load(model, STATIC), fam)
    q2 = tops[TOP[0]]
    assert len(st["metric"]) == len(q2["metric"]) and np.allclose(st["metric"], q2["metric"]), "static/Q2 rows differ"
    static_rows = np.arange(len(q2["metric"]))                 # Q2_top is first in the concatenation
    K_static = {k: cross_gram(st[k].astype(np.float64), d[k].astype(np.float64)) for k in KEYS}

    t0 = time.time()
    oof, chosen, sp = nested_oof(Ks, y, g, rng, K_static, static_rows)
    rho = float(spearmanr(oof, y)[0])
    q2m = clipid == TOP[0]
    rho_full_q2 = float(spearmanr(oof[q2m], y[q2m])[0])
    rho_static = float(spearmanr(sp, y[q2m])[0])
    null = []
    for _ in range(perms):
        yp = y.copy()
        for c in TOP:
            m = clipid == c
            yp[m] = rng.permutation(yp[m])
        o, _, _ = nested_oof(Ks, yp, g, rng)
        null.append(float(spearmanr(o, yp)[0]))
    p99 = float(np.percentile(null, 99)) if null else None
    return {"n_items": int(len(y)), "nested_cv_spearman": rho, "perm_p99": p99, "perms": perms,
            "H1_holds": bool(rho >= 0.30 and (p99 is None or rho > p99)),
            "fold_selections": chosen,
            "H3": {"rho_full_q2_oof": rho_full_q2, "rho_static4_oof": rho_static,
                   "motion_dependent": bool(rho_static <= rho_full_q2 - 0.30)},
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--fams", nargs="*", default=list(FAMS))
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default=str(OUTD / "nested_reanalysis.json"))
    a = ap.parse_args()
    out = Path(a.out)
    rep = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"method": __doc__.split("\n\n")[1],
                                                                            "seed": a.seed, "models": {}}
    for model in a.models:
        rng = np.random.default_rng(a.seed)
        rep["models"].setdefault(model, {})
        for fam in a.fams:
            if rep["models"][model].get(fam, {}).get("perms") == a.perms:
                continue
            r = analyse(model, fam, a.perms, rng)
            rep["models"][model][fam] = r
            print(f"{model:8s} {fam:6s} rho={r['nested_cv_spearman']:.3f} p99={r['perm_p99']} "
                  f"H1={r['H1_holds']} | H3 full={r['H3']['rho_full_q2_oof']:.3f} "
                  f"static={r['H3']['rho_static4_oof']:.3f} md={r['H3']['motion_dependent']} "
                  f"[{r['seconds']}s]", flush=True)
            out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    lines = ["| model | family | nested rho | perm p99 | H1 | rho full (Q2) | rho static4 | motion-dependent |",
             "|---|---|---|---|---|---|---|---|"]
    for m, fams in rep["models"].items():
        for fam, r in fams.items():
            lines.append(f"| {m} | {fam} | {r['nested_cv_spearman']:.3f} | {r['perm_p99']:.3f} | {r['H1_holds']} | "
                         f"{r['H3']['rho_full_q2_oof']:.3f} | {r['H3']['rho_static4_oof']:.3f} | {r['H3']['motion_dependent']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
