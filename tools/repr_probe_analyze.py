#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decide the representation probe against results/courtdyn/repr_probe/PREREGISTRATION.json.

Implements, per model, exactly the preregistered steps:
  feature selection  (layer, pooling) with the best 5-fold grouped-CV R^2 for log metre on the two
                     overhead clips, averaged over families -- before any side / first-frame score
  H1  grouped-CV Spearman on the overhead clips >= 0.30 and > 99th pct of 200 within-clip permutations
  H3  probe fit on all overhead items, applied to Q2_top first-frame x4: motion-dependent if
      rho_static4 <= rho_full(Q2_top, out-of-fold) - 0.30
  H2  log-metre probe fit on all overhead items (recalibrated on out-of-fold predictions), applied to
      Q1_side: Delta = median(log pred - log m); pixel reading predicts E = median log(K_i / K_top);
      image-plane if Delta < E/2, else floor-plane; guard: H1 holds and the probe fit on Q2_top
      transfers to Q1_top with |Delta_ctrl| < |E|/2, else UNTESTABLE.  Native excluded (trained on
      Q1_side).  Bootstrap CI over Q1_side windows (B = 2000).
  H4  mixunit and v3 get the same H2 verdict per family.
Descriptive: partial Spearman of the metre ANSWER with pixels | metres and metres | pixels per clip;
within-clip decodability of log px vs log m on the side training clips.

Writes results/courtdyn/repr_probe/{verdict.json, verdict.md, repr_probe.png, repr_probe.pdf}.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "results" / "courtdyn" / "repr_probe"
FEAT = OUTD / "features"
MODELS = ["base", "native", "v3", "mixunit"]
TOP = ["Q2_top_480-510", "Q1_top_0-30"]
SIDE = "Q1_side_0-30"
SIDE_TRAIN = ["Q2_side_300-330", "Q4_side_570-600"]
STATIC = "Q2_top_480-510__static4"
FAMS = {"speed": "dynamics_speed_player", "path": "dynamics_path_player"}
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
POOLS = ["img", "last"]
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
EPS_M, EPS_PX = 0.05, 0.05 * 27.3
H2_EXCLUDED = {"native"}          # trained on Q1_side
rng = np.random.default_rng(20260928)


def load(model, clip):
    d = np.load(FEAT / model / f"{clip}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def subset(d, fam):
    idx = d["category"] == FAMS[fam]
    return {k: (v[idx] if len(v) == len(idx) else v) for k, v in d.items()}


def cat(*ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def groups(d, clip_tag):
    return np.array([f"{clip_tag}|{w}" for w in d["window0"]])


# ------------------------------------------------------------------ ridge with grouped CV
def _fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = y.mean()
    # dual form: n << p
    K = Z @ Z.T
    a = np.linalg.solve(K + alpha * np.eye(len(y)), y - ym)
    w = Z.T @ a
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def _folds(g, k):
    ug = np.unique(g)
    perm = rng.permutation(len(ug))
    return [np.isin(g, ug[perm[i::k]]) for i in range(k)]


def _best_alpha(X, y, g):
    scores = []
    for a in ALPHAS:
        pred = np.empty_like(y)
        for te in _folds(g, 3):
            pred[te] = _fit(X[~te], y[~te], a)(X[te])
        scores.append(np.mean((pred - y) ** 2))
    return ALPHAS[int(np.argmin(scores))]


def cv_predict(X, y, g, k=5, alpha=None):
    pred = np.empty_like(y)
    for te in _folds(g, k):
        a = alpha if alpha is not None else _best_alpha(X[~te], y[~te], g[~te])
        pred[te] = _fit(X[~te], y[~te], a)(X[te])
    return pred


def r2(p, y):
    return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def final_probe(X, y, g):
    """Ridge on all items + linear recalibration fitted on out-of-fold predictions."""
    oof = cv_predict(X, y, g)
    b, a = np.polyfit(oof, y, 1)
    f = _fit(X, y, _best_alpha(X, y, g))
    return (lambda Xn: a + b * f(Xn)), oof


def boot_median(values, g, B=2000):
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    meds = [np.median(values[np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])]) for _ in range(B)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def parse(ans):
    out = []
    for a in ans:
        m = re.search(r"-?\d+(?:\.\d+)?", str(a))
        out.append(float(m.group()) if m else np.nan)
    return np.array(out)


def partial_spearman(x, y, z):
    """rank-based partial correlation of x and y given z."""
    from scipy.stats import rankdata
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if ok.sum() < 10 or len(np.unique(x[ok])) < 3:
        return None
    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


# ------------------------------------------------------------------ per model
def analyse(model):
    top = {c: load(model, c) for c in TOP}
    rep = {"model": model}
    # feature selection on overhead clips only (log metre, averaged over families)
    sel = {}
    for pool in POOLS:
        for L in LAYERS:
            key = f"{pool}_L{L}"
            vals = []
            for fam in FAMS:
                d = cat(*(subset(top[c], fam) for c in TOP))
                g = np.concatenate([groups(subset(top[c], fam), c) for c in TOP])
                y = np.log(d["metric"] + EPS_M)
                vals.append(r2(cv_predict(d[key].astype(np.float32), y, g), y))
            sel[key] = float(np.mean(vals))
    feat = max(sel, key=sel.get)
    rep["feature_selection_cv_r2"] = sel
    rep["selected_feature"] = feat

    side = load(model, SIDE) if (FEAT / model / f"{SIDE}.npz").is_file() else None
    static = load(model, STATIC) if (FEAT / model / f"{STATIC}.npz").is_file() else None
    rep["families"] = {}
    for fam in FAMS:
        tops = {c: subset(top[c], fam) for c in TOP}
        d = cat(*tops.values())
        g = np.concatenate([groups(tops[c], c) for c in TOP])
        X = d[feat].astype(np.float32)
        y = np.log(d["metric"] + EPS_M)
        clipid = np.concatenate([[c] * len(tops[c]["metric"]) for c in TOP])
        fr = {}
        # H1
        alpha = _best_alpha(X, y, g)
        oof = cv_predict(X, y, g)
        rho = spearmanr(oof, y)[0]
        null = []
        for _ in range(200):
            yp = y.copy()
            for c in TOP:
                m = clipid == c
                yp[m] = rng.permutation(yp[m])
            null.append(spearmanr(cv_predict(X, yp, g, alpha=alpha), yp)[0])
        p99 = float(np.percentile(null, 99))
        fr["H1"] = {"cv_spearman": float(rho), "perm_p99": p99,
                    "holds": bool(rho >= 0.30 and rho > p99)}
        # pixel target on overhead, for the record
        ypx = np.log(d["px"] + EPS_PX)
        fr["overhead_cv_spearman_px"] = float(spearmanr(cv_predict(X, ypx, g), ypx)[0])
        # final probe
        probe, _ = final_probe(X, y, g)
        # H3
        q2 = clipid == TOP[0]
        rho_full_q2 = float(spearmanr(oof[q2], y[q2])[0])
        if static is not None:
            s = subset(static, fam)
            rho_s = spearmanr(probe(s[feat].astype(np.float32)), np.log(s["metric"] + EPS_M))[0]
            fr["H3"] = {"rho_full_q2_oof": rho_full_q2, "rho_static4": float(rho_s),
                        "motion_dependent": bool(rho_s <= rho_full_q2 - 0.30)}
        # H2
        if side is not None and model not in H2_EXCLUDED:
            s = subset(side, fam)
            ys = np.log(s["metric"] + EPS_M)
            ps = probe(s[feat].astype(np.float32))
            delta_i = ps - ys
            Ktop = np.median(d["px"] / np.maximum(d["metric"], 1e-6))
            Ki = s["px"] / np.maximum(s["metric"], 1e-6)
            E = float(np.median(np.log(Ki / Ktop)))
            E_eps = float(np.median(np.log(s["px"] / Ktop + EPS_M) - ys))   # eps-consistent, robustness
            gs = groups(s, SIDE)
            delta = float(np.median(delta_i))
            # guard: Q2_top -> Q1_top transfer
            t2, t1 = tops[TOP[0]], tops[TOP[1]]
            g2 = groups(t2, TOP[0])
            p2, _ = final_probe(t2[feat].astype(np.float32), np.log(t2["metric"] + EPS_M), g2)
            dctrl = float(np.median(p2(t1[feat].astype(np.float32)) - np.log(t1["metric"] + EPS_M)))
            testable = fr["H1"]["holds"] and abs(dctrl) < abs(E) / 2
            verdict = ("UNTESTABLE" if not testable else
                       "image-plane" if delta < E / 2 else "floor-plane")
            fr["H2"] = {"delta": delta, "delta_ci95": boot_median(delta_i, gs), "E_pixel_prediction": E,
                        "E_eps_consistent": E_eps, "delta_ctrl_Q2top_to_Q1top": dctrl,
                        "testable": bool(testable), "verdict": verdict,
                        "side_spearman": float(spearmanr(ps, ys)[0]),
                        "side_spearman_vs_px": float(spearmanr(ps, np.log(s["px"] + EPS_PX))[0])}
        # descriptive: behaviour
        beh = {}
        for c, dd in [(TOP[0], tops[TOP[0]]), (TOP[1], tops[TOP[1]])] + (
                [(SIDE, subset(side, fam))] if side is not None else []) + [
                (c, subset(load(model, c), fam)) for c in SIDE_TRAIN if (FEAT / model / f"{c}.npz").is_file()]:
            a = parse(dd["answer"])
            beh[c] = {"distinct": int(len(np.unique(a[~np.isnan(a)]))),
                      "rho_answer_metric": float(spearmanr(a, dd["metric"], nan_policy="omit")[0]),
                      "rho_answer_px": float(spearmanr(a, dd["px"], nan_policy="omit")[0]),
                      "partial_px_given_m": partial_spearman(a, dd["px"], dd["metric"]),
                      "partial_m_given_px": partial_spearman(a, dd["metric"], dd["px"]),
                      "rho_px_metric_reference": float(spearmanr(dd["px"], dd["metric"])[0])}
        fr["behaviour"] = beh
        # descriptive: within-clip decodability on side training clips
        wc = {}
        for c in SIDE_TRAIN:
            if not (FEAT / model / f"{c}.npz").is_file():
                continue
            dd = subset(load(model, c), fam)
            gg = groups(dd, c)
            Xc = dd[feat].astype(np.float32)
            ym, yp = np.log(dd["metric"] + EPS_M), np.log(dd["px"] + EPS_PX)
            pm, pp = cv_predict(Xc, ym, gg), cv_predict(Xc, yp, gg)
            wc[c] = {"cv_spearman_metric": float(spearmanr(pm, ym)[0]),
                     "cv_spearman_px": float(spearmanr(pp, yp)[0]),
                     "partial_metric_probe_m_given_px": partial_spearman(pm, dd["metric"], dd["px"]),
                     "partial_px_probe_px_given_m": partial_spearman(pp, dd["px"], dd["metric"])}
        fr["within_side_training_clips"] = wc
        rep["families"][fam] = fr
    h2 = [rep["families"][f].get("H2", {}).get("verdict") for f in FAMS]
    rep["H2_model_verdict"] = (None if None in h2 else
                               h2[0] if h2[0] == h2[1] else "mixed")
    return rep


def main() -> int:
    pre = json.loads((OUTD / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(pre["rules"].encode("utf-8")).hexdigest() == pre["rules_sha256"]
    models = [m for m in (sys.argv[1:] or MODELS) if (FEAT / m / f"{TOP[0]}.npz").is_file()]
    report = {"rules_sha256": pre["rules_sha256"], "models": {m: analyse(m) for m in models}}
    if "v3" in report["models"] and "mixunit" in report["models"]:
        same = [report["models"]["v3"]["families"][f].get("H2", {}).get("verdict")
                == report["models"]["mixunit"]["families"][f].get("H2", {}).get("verdict") for f in FAMS]
        report["H4_same_scale_v3_vs_mixunit"] = bool(all(same))
    (OUTD / "verdict.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    write_md(report)
    plot(report)
    print((OUTD / "verdict.md").read_text(encoding="utf-8"))
    return 0


def write_md(rep):
    f = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
    md = ["# Representation probe (preregistered)", "",
          f"Rules sha256 `{rep['rules_sha256'][:16]}...`.", ""]
    for m, r in rep["models"].items():
        md += [f"## {m}  (feature {r['selected_feature']}; H2 model verdict: {r['H2_model_verdict']})", "",
               "| family | H1 cv rho (p99 null) | H3 rho full -> ff4 | H2 Delta [CI] vs E (ctrl) | H2 verdict |",
               "|---|---|---|---|---|"]
        for fam, x in r["families"].items():
            h1, h3, h2 = x["H1"], x.get("H3", {}), x.get("H2", {})
            md.append(f"| {fam} | {f(h1['cv_spearman'])} ({f(h1['perm_p99'])}) {'ok' if h1['holds'] else 'FAIL'} | "
                      f"{f(h3.get('rho_full_q2_oof'))} -> {f(h3.get('rho_static4'))} "
                      f"{'motion' if h3.get('motion_dependent') else 'NOT motion-dependent' if h3 else ''} | "
                      + (f"{f(h2['delta'])} [{f(h2['delta_ci95'][0])}, {f(h2['delta_ci95'][1])}] vs "
                         f"{f(h2['E_pixel_prediction'])} (ctrl {f(h2['delta_ctrl_Q2top_to_Q1top'])}) | {h2['verdict']} |"
                         if h2 else "excluded | -- |"))
        md += ["", "Behaviour (descriptive): metre answer vs references, partial Spearman px|m / m|px:"]
        for fam, x in r["families"].items():
            md.append(f"- {fam}: " + "; ".join(
                f"{c} distinct {b['distinct']}, rho m {f(b['rho_answer_metric'])} px {f(b['rho_answer_px'])}, "
                f"partial px|m {f(b['partial_px_given_m'])} m|px {f(b['partial_m_given_px'])}"
                for c, b in x["behaviour"].items()))
        md.append("")
    if "H4_same_scale_v3_vs_mixunit" in rep:
        md.append(f"H4 (v3 and mixunit same H2 verdict in both families): {rep['H4_same_scale_v3_vs_mixunit']}")
    (OUTD / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def plot(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [m for m in rep["models"] if m not in H2_EXCLUDED]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for i, m in enumerate(ms):
            h2 = rep["models"][m]["families"][fam].get("H2")
            if not h2:
                continue
            lo, hi = h2["delta_ci95"]
            ax.errorbar(i, h2["delta"], yerr=[[h2["delta"] - lo], [hi - h2["delta"]]], fmt="o",
                        color="#1f4e79", capsize=3)
            ax.plot([i - 0.3, i + 0.3], [h2["E_pixel_prediction"]] * 2, color="#c0504d", lw=1.5)
            ax.plot([i - 0.3, i + 0.3], [h2["delta_ctrl_Q2top_to_Q1top"]] * 2, color="grey", lw=1, ls=":")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(ms)), ms)
        ax.set_title(fam)
    axes[0].set_ylabel(r"$\Delta$ = median(log pred $-$ log m), side view")
    fig.text(0.5, -0.02, "red: pixel-encoding prediction E; black 0: metric-encoding prediction; "
             "grey dotted: overhead-to-overhead control", ha="center", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTD / f"repr_probe.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    sys.exit(main())
