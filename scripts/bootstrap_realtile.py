#!/usr/bin/env python3
"""Layer-clustered bootstrap for the released real-tile measurements.

The within-model interval resamples transformer layers. The two-level
interval first resamples the five models and then layers within each selected
model. Projection types and contexts remain nested within their layer.

Usage: ``python3 scripts/bootstrap_realtile.py [replicates]``.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data", "realtile")
PANEL = os.path.join(ROOT, "artifacts", "results", "orbit_panel.json")
REFERENCE = os.path.join(ROOT, "data", "reference", "realtile_panel_summary.json")
OUT = os.path.join(ROOT, "results", "realtile_bootstrap.json")

# Constants shared with ``scripts/analyze_realtile.py``.
TIE = 0.03
CERT = "dps_acc"
STRAS = "strassen_classic"
COARSE = ["cubic_fp8", "dps_acc", "strassen_classic", "winograd_form"]
MODELS = ["qwen25_7b", "qwen3_8b", "yi_6b", "qwen25_32b", "llama33_70b"]
SEED = 20260726


def rankdata(a):
    """Average-tie ranks along the last axis; mirrors rk() in scripts/analyze_realtile.py."""
    order = np.argsort(a, axis=-1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    n = a.shape[-1]
    idx = np.arange(n)
    for i in range(a.shape[0]):
        s = a[i][order[i]]
        r = np.empty(n)
        j = 0
        while j < n:
            k = j
            while k + 1 < n and s[k + 1] == s[j]:
                k += 1
            r[j:k + 1] = (j + k) / 2.0 + 1
            j = k + 1
        ranks[i][order[i]] = r
    del idx
    return ranks


def load_model(model, info):
    """-> dict of numpy arrays. err[n_tiles, n_arms] real error, nan where missing."""
    d = json.load(open(os.path.join(DATA, f"realtile_panel_{model}.json")))
    tiles = d["tiles"]
    arms = list(info)
    all_arms = arms + [a for a in COARSE if a not in arms]
    ai = {a: i for i, a in enumerate(all_arms)}
    n, m = len(tiles), len(all_arms)
    err = np.full((n, m), np.nan)
    pom = np.full((n, m), np.nan)
    ref = np.ones(n)
    li = np.zeros(n, dtype=int)
    layer = np.zeros(n, dtype=int)
    ctx = np.zeros(n, dtype=int)
    ratios = []
    for t_i, t in enumerate(tiles):
        ref[t_i] = t.get("ref_norm") or 1.0
        li[t_i] = t["li"]
        layer[t_i] = int(re.search(r"layers\.(\d+)\.", t["name"]).group(1))
        ctx[t_i] = t.get("ctx", 0)
        for r in t["rows"]:
            if r["arm"] in ai:
                j = ai[r["arm"]]
                if r.get("err_real") is not None:
                    err[t_i, j] = r["err_real"]
                if r.get("phi_over_mkn") is not None:
                    pom[t_i, j] = r["phi_over_mkn"]
            if r.get("ratio_emu_over_real") is not None:
                ratios.append(r["ratio_emu_over_real"])
    return dict(model=model, err=err, pom=pom, ref=ref, li=li, layer=layer, ctx=ctx,
                arms=all_arms, ai=ai, panel_arms=arms, ratios=np.array(ratios))


def stats_for(M, info, rows=None):
    """Every per-model panel statistic, on a (possibly resampled) row index set."""
    err = M["err"] if rows is None else M["err"][rows]
    pom = M["pom"] if rows is None else M["pom"][rows]
    ref = M["ref"] if rows is None else M["ref"][rows]
    ai, arms = M["ai"], M["panel_arms"]
    out = {}

    # Oracle gap: E(a) = sqrt(sum (err*ref)^2 / sum ref^2) over the panel arms.
    pj = [ai[a] for a in arms]
    e = err[:, pj]
    ok = ~np.isnan(e)
    w = (ref ** 2)[:, None]
    num = np.nansum((e * ref[:, None]) ** 2, axis=0)
    den = np.where(ok, w, 0.0).sum(axis=0)
    E = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))
    ci = arms.index(CERT)
    out["G_cert"] = E[ci] / np.nanmin(E) - 1.0
    out["argmin"] = arms[int(np.nanargmin(E))]
    # Certified realization versus the nearby DPS variant.
    if "dps_evenpow" in arms:
        out["G_vs_evenpow"] = E[ci] / E[arms.index("dps_evenpow")] - 1.0

    # Local minimum: the certified point beats all twelve perturbations per tile.
    pert = [ai[a] for a in arms if info[a]["group"] == "perturb"]
    ce = err[:, ai[CERT]]
    pe = err[:, pert]
    good = ~np.isnan(ce) & ~np.isnan(pe).any(axis=1)
    out["loc_win_frac"] = float((ce[good, None] < pe[good]).all(axis=1).mean()) if good.any() else np.nan

    # Phi-versus-Gamma concordance on the three designed disagreement pairs.
    pairs = {}
    for a in arms:
        if info[a]["group"] == "disagree":
            pid, side = a.rsplit("_", 1)
            pairs.setdefault(pid, {})[side] = a
    pc = []
    for pid, ab in sorted(pairs.items()):
        if set(ab) != {"a", "b"}:
            continue
        aa, bb = ab["a"], ab["b"]
        dphi = info[aa]["phi"] - info[bb]["phi"]
        ea, eb = err[:, ai[aa]], err[:, ai[bb]]
        m = ~np.isnan(ea) & ~np.isnan(eb)
        mx = np.maximum(ea, eb)
        m &= (mx >= 1e-9) & (np.abs(ea - eb) / np.where(mx > 0, mx, 1) > TIE)
        if m.sum():
            pc.append(float((((ea - eb) < 0)[m] == (dphi < 0)).mean()))
    out["o3_phi_correct"] = float(np.mean(pc)) if pc else np.nan

    # Ladder Spearman(sqrt(phi), err_real) per tile, averaged.
    ladder = [a for a in arms if info[a]["group"] in ("named", "perturb")]
    lj = [ai[a] for a in ladder]
    x = np.sqrt(np.array([info[a]["phi"] for a in ladder]))
    le = err[:, lj]
    full = ~np.isnan(le).any(axis=1)
    if full.sum():
        rx = rankdata(np.tile(x, (full.sum(), 1)))
        ry = rankdata(le[full])
        rxc = rx - rx.mean(axis=1, keepdims=True)
        ryc = ry - ry.mean(axis=1, keepdims=True)
        num2 = (rxc * ryc).sum(axis=1)
        den2 = np.sqrt((rxc ** 2).sum(axis=1) * (ryc ** 2).sum(axis=1))
        sp = np.divide(num2, den2, out=np.full_like(num2, np.nan), where=den2 > 0)
        out["mean_spearman"] = float(np.nanmean(sp))
    else:
        out["mean_spearman"] = np.nan

    # Zero inversions among the four distinct named realizations, per tile.
    cj = [ai[a] for a in COARSE]
    cerr, cpom = err[:, cj], pom[:, cj]
    fullc = ~np.isnan(cerr).any(axis=1) & ~np.isnan(cpom).any(axis=1)
    if fullc.sum():
        ce2, cp2 = cerr[fullc], cpom[fullc]
        noinv = np.ones(ce2.shape[0], dtype=bool)
        anyc = np.zeros(ce2.shape[0], dtype=bool)
        for i in range(len(COARSE)):
            for j in range(i + 1, len(COARSE)):
                ri, rj = ce2[:, i], ce2[:, j]
                mx = np.maximum(ri, rj)
                cnt = (mx >= 1e-9) & (np.abs(ri - rj) / np.where(mx > 0, mx, 1) > TIE)
                anyc |= cnt
                noinv &= ~(cnt & ((ri < rj) != (cp2[:, i] < cp2[:, j])))
        out["coarse_frac0inv"] = float(np.where(anyc, noinv, True).mean())
    else:
        out["coarse_frac0inv"] = np.nan

    # --- DOMINANCE: certified optimum vs classic Strassen, per tile ---
    se = err[:, ai[STRAS]]
    m = ~np.isnan(ce) & ~np.isnan(se) & (ce > 0) & (se > 0)
    out["strassen_dom_frac"] = float((ce[m] < se[m]).mean()) if m.any() else np.nan
    out["strassen_red_median"] = float(np.median(1.0 - ce[m] / se[m])) if m.any() else np.nan
    return out


KEYS = ["G_cert", "G_vs_evenpow", "loc_win_frac", "o3_phi_correct",
        "mean_spearman", "coarse_frac0inv", "strassen_dom_frac", "strassen_red_median"]


def clusters(M):
    """Transformer-layer clusters; projection types and contexts nest inside a layer."""
    by = {}
    for i, l in enumerate(M["layer"]):
        by.setdefault(int(l), []).append(i)
    return [np.array(v) for v in by.values()]


def resample(cl, rng):
    return np.concatenate([cl[k] for k in rng.integers(0, len(cl), len(cl))])


def pct(a):
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)])
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a.size else (np.nan, np.nan)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rng = np.random.default_rng(SEED)
    man = json.load(open(PANEL))
    info = {p["name"]: p for p in man["points"]}

    data = {m: load_model(m, info) for m in MODELS}
    point = {m: stats_for(data[m], info) for m in MODELS}

    # Reproduce the released point estimates before bootstrapping.
    ship = json.load(open(REFERENCE))
    for m in MODELS:
        for k in ["G_cert", "loc_win_frac", "mean_spearman", "coarse_frac0inv",
                  "strassen_dom_frac", "strassen_red_median"]:
            a, b = point[m][k], ship["per_model"][m][k]
            assert abs(a - b) < 1e-9, f"MISMATCH {m}.{k}: recomputed {a} vs shipped {b}"
    print(f"point estimates reproduce the released summary exactly "
          f"({len(MODELS)} models x 6 statistics)\n")

    # ---------- (A) within-model, layer-clustered ----------
    cl = {m: clusters(data[m]) for m in MODELS}
    withinA = {m: {k: [] for k in KEYS} for m in MODELS}
    for m in MODELS:
        for _ in range(B):
            s = stats_for(data[m], info, rows=resample(cl[m], rng))
            for k in KEYS:
                withinA[m][k].append(s.get(k))

    print("=" * 92)
    print("(A) WITHIN-MODEL, layer-clustered bootstrap   B=%d   95%% percentile CI" % B)
    print("=" * 92)
    hdr = f"{'model':<14}{'clusters':>9}{'tiles':>7}  "
    print(hdr + "panel gap G(cert)        vs dps_evenpow")
    for m in MODELS:
        lo, hi = pct(withinA[m]["G_cert"])
        lo2, hi2 = pct(withinA[m]["G_vs_evenpow"])
        print(f"{m:<14}{len(cl[m]):>9}{len(data[m]['li']):>7}  "
              f"{point[m]['G_cert']*100:+6.2f}% [{lo*100:+6.2f},{hi*100:+6.2f}]   "
              f"{point[m]['G_vs_evenpow']*100:+6.2f}% [{lo2*100:+6.2f},{hi2*100:+6.2f}]"
              f"{'   <-- CI excludes 0' if lo2 > 0 or hi2 < 0 else '   CI includes 0'}")
    print()
    for k in ["loc_win_frac", "o3_phi_correct", "mean_spearman",
              "coarse_frac0inv", "strassen_dom_frac", "strassen_red_median"]:
        print(f"  {k}")
        for m in MODELS:
            lo, hi = pct(withinA[m][k])
            print(f"    {m:<14} {point[m][k]:.4f}  [{lo:.4f}, {hi:.4f}]")

    # ---------- (B) two-level: models x layers ----------
    twoB = {k: [] for k in KEYS}
    for _ in range(B):
        drawn = [MODELS[i] for i in rng.integers(0, len(MODELS), len(MODELS))]
        acc = {k: [] for k in KEYS}
        for m in drawn:
            s = stats_for(data[m], info, rows=resample(cl[m], rng))
            for k in KEYS:
                v = s.get(k)
                if v is not None and np.isfinite(v):
                    acc[k].append(v)
        for k in KEYS:
            twoB[k].append(float(np.mean(acc[k])) if acc[k] else None)

    pooled = {k: float(np.mean([point[m][k] for m in MODELS
                                if point[m].get(k) is not None and np.isfinite(point[m][k])]))
              for k in KEYS}
    print()
    print("=" * 92)
    print("(B) TWO-LEVEL (models x layers) bootstrap on the MODEL-BALANCED pooled means")
    print("    n=5 model clusters at the top level -- coarse BY CONSTRUCTION; read as a floor")
    print("    on our uncertainty, not a precise interval.")
    print("=" * 92)
    for k in KEYS:
        lo, hi = pct(twoB[k])
        print(f"  {k:<22} {pooled[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    out = dict(
        B=B, seed=SEED,
        cluster_unit="transformer layer parsed from name; contexts and projection types remain nested",
        note=("(A) resamples layers within a model. (B) resamples models (n=5) then layers. "
              "Point estimates are checked against the released summary."),
        n_clusters={m: len(cl[m]) for m in MODELS},
        n_tiles={m: int(len(data[m]["li"])) for m in MODELS},
        per_model={m: {k: dict(point=point[m].get(k), ci=pct(withinA[m][k])) for k in KEYS}
                   for m in MODELS},
        pooled={k: dict(point=pooled[k], ci=pct(twoB[k])) for k in KEYS},
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
