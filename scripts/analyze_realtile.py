#!/usr/bin/env python3
"""Recompute the real-tile panel statistics from the released measurements.

With no positional arguments, this script reads all five JSON files under
``data/realtile``.  It uses only the Python standard library.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data", "realtile")
PANEL = os.path.join(ROOT, "artifacts", "results", "orbit_panel.json")
OUT = os.path.join(ROOT, "results", "realtile_panel_summary.json")
TIE = 0.03
CERT = "dps_acc"
OPT_CLASS = {"dps_acc", "dps_int"}


def med(xs):
    s = sorted(xs); n = len(s)
    return None if not n else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def spearman(xy):
    if len(xy) < 2:
        return None
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    x = [a for a, _ in xy]; y = [b for _, b in xy]
    rx, ry = rk(x), rk(y); n = len(xy)
    mx = sum(rx) / n; my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((v - mx) ** 2 for v in rx) ** .5; vy = sum((v - my) ** 2 for v in ry) ** .5
    return None if vx == 0 or vy == 0 else cov / (vx * vy)


def load_manifest():
    m = json.load(open(PANEL))
    info = {p["name"]: p for p in m["points"]}
    return info, m["phi_min"]


def tile_map(t):
    return {r["arm"]: r for r in t["rows"]}


def energy_E(tiles, arm):
    num = den = 0.0
    for t in tiles:
        rm = tile_map(t)
        if arm not in rm or rm[arm]["err_real"] is None:
            continue
        rn = t.get("ref_norm") or 1.0
        num += (rm[arm]["err_real"] * rn) ** 2
        den += rn ** 2
    return (num / den) ** 0.5 if den else None


def analyze_model(tiles, info):
    arms = [a for a in info]                      # panel arms (all rank-7 orbit points)
    E = {a: energy_E(tiles, a) for a in arms}
    E = {a: v for a, v in E.items() if v is not None}
    argmin = min(E, key=E.get)
    G = E[CERT] / E[argmin] - 1.0
    perts = [a for a in arms if info[a]["group"] == "perturb"]
    # Fraction of tiles where D* strictly beats every perturbation neighbour.
    locwin = locn = 0
    for t in tiles:
        rm = tile_map(t)
        if CERT not in rm or any(p not in rm for p in perts):
            continue
        ce = rm[CERT]["err_real"]
        if ce is None:
            continue
        pe = [rm[p]["err_real"] for p in perts if rm[p]["err_real"] is not None]
        if not pe:
            continue
        locn += 1
        if all(ce < e for e in pe):
            locwin += 1
    # Phi-versus-Gamma concordance on the designed disagreement pairs.
    pairs = {}
    for a in arms:
        if info[a]["group"] == "disagree":
            pid = a.rsplit("_", 1)[0]  # disagreeK
            pairs.setdefault(pid, {})[a.rsplit("_", 1)[1]] = a
    conc = {}
    for pid, ab in pairs.items():
        if set(ab) != {"a", "b"}:
            continue
        aa, bb = ab["a"], ab["b"]
        dphi = info[aa]["phi"] - info[bb]["phi"]
        dgam = info[aa]["gamma"] - info[bb]["gamma"]
        phi_n = gam_n = tot = 0
        for t in tiles:
            rm = tile_map(t)
            if aa not in rm or bb not in rm:
                continue
            ea, eb = rm[aa]["err_real"], rm[bb]["err_real"]
            if ea is None or eb is None or max(ea, eb) < 1e-9:
                continue
            if abs(ea - eb) / max(ea, eb) <= TIE:
                continue
            tot += 1
            derr = ea - eb
            if (derr < 0) == (dphi < 0):
                phi_n += 1
            if (derr < 0) == (dgam < 0):
                gam_n += 1
        conc[pid] = dict(n=tot, phi_correct=(phi_n / tot if tot else None),
                         gamma_correct=(gam_n / tot if tot else None),
                         dphi=dphi, dgamma=dgam)
    # Ordering over the monotone ladder.  The designed disagreement pairs are
    # evaluated separately above.
    ladder = [a for a in arms if info[a]["group"] in ("named", "perturb")]
    invfrac = []
    sps = []
    for t in tiles:
        rm = tile_map(t)
        xy = [(info[a]["phi"] ** 0.5, rm[a]["err_real"]) for a in ladder
              if a in rm and rm[a]["err_real"] is not None]
        if len(xy) >= 3:
            sps.append(spearman(xy))
            inv = tot = 0
            for i in range(len(xy)):
                for j in range(i + 1, len(xy)):
                    pi, ri = xy[i]; pj, rj = xy[j]
                    if max(ri, rj) < 1e-9 or abs(ri - rj) / max(ri, rj) <= TIE:
                        continue
                    tot += 1
                    if (ri < rj) != (pi < pj):
                        inv += 1
            invfrac.append(1.0 if tot and inv == 0 else (0.0 if tot else 1.0))
    # Does Phi order the distinct named realizations a practitioner considers
    # (cubic_fp8 Phi/mkn=1 < dps_acc 2.78 < strassen 4 < winograd 7),
    # separate from the fine within-basin resolution the full ladder tests.
    COARSE = ["cubic_fp8", "dps_acc", "strassen_classic", "winograd_form"]
    coarse0inv = []
    for t in tiles:
        rm = tile_map(t)
        xy = [(rm[a]["phi_over_mkn"], rm[a]["err_real"]) for a in COARSE
              if a in rm and rm[a]["err_real"] is not None]
        if len(xy) == len(COARSE):
            inv = tot = 0
            for i in range(len(xy)):
                for j in range(i + 1, len(xy)):
                    pi, ri = xy[i]; pj, rj = xy[j]
                    if max(ri, rj) < 1e-9 or abs(ri - rj) / max(ri, rj) <= TIE:
                        continue
                    tot += 1
                    if (ri < rj) != (pi < pj):
                        inv += 1
            coarse0inv.append(1.0 if tot and inv == 0 else (0.0 if tot else 1.0))
    # Certified optimum (dps_acc) versus classic Strassen, per real tile.
    dom_win = domn = 0
    reds = []
    for t in tiles:
        rm = tile_map(t)
        if ("dps_acc" in rm and "strassen_classic" in rm
                and rm["dps_acc"]["err_real"] and rm["strassen_classic"]["err_real"]):
            ce, se = rm["dps_acc"]["err_real"], rm["strassen_classic"]["err_real"]
            domn += 1
            if ce < se:
                dom_win += 1
            reds.append(1.0 - ce / se)
    # Emulator-to-hardware fidelity.
    ratios = [r["ratio_emu_over_real"] for t in tiles for r in t["rows"]
              if r.get("ratio_emu_over_real") is not None]
    return dict(E=E, argmin=argmin, argmin_in_optclass=(argmin in OPT_CLASS), G_cert=G,
                loc_win_frac=(locwin / locn if locn else None), loc_n=locn,
                concordance=conc, mean_spearman=(sum(sps) / len(sps) if sps else None),
                frac0inv=(sum(invfrac) / len(invfrac) if invfrac else None),
                coarse_frac0inv=(sum(coarse0inv) / len(coarse0inv) if coarse0inv else None),
                strassen_dom_frac=(dom_win / domn if domn else None),
                strassen_red_median=med(reds),
                emu_real_ratio_median=med(ratios), n_tiles=len(tiles))


def main():
    files = sys.argv[1:]
    if not files:
        files = [os.path.join(DATA, name) for name in sorted(os.listdir(DATA))
                 if name.startswith("realtile_panel_") and name.endswith(".json")]
    if not files:
        print("no real-tile JSON files found", file=sys.stderr)
        sys.exit(1)
    info, phi_min = load_manifest()
    per = {}
    allpair_phi = {}
    for f in files:
        d = json.load(open(f))
        tiles = [t for t in d["tiles"] if "rows" in t]
        if not tiles:
            print(f"[parse] {f}: NO measured tiles"); continue
        r = analyze_model(tiles, info)
        per[d["model"]] = r
        print(f"\n=== {d['model']}  (tiles={r['n_tiles']}, contexts={d.get('n_contexts','?')}) ===")
        print(f"  oracle gap G(cert)={r['G_cert']*100:+.3f}%  argmin={r['argmin']} "
              f"(in-opt-class={r['argmin_in_optclass']})")
        _lwf = r['loc_win_frac']
        print(f"  local-min: D* strictly beats all perturbations in "
              f"{('%.3f' % _lwf) if _lwf is not None else 'NA'} of {r['loc_n']} tiles")
        for pid, c in sorted(r["concordance"].items()):
            print(f"  disagreement pair {pid}: n={c['n']} Phi-correct={c['phi_correct']} Gamma-correct={c['gamma_correct']} "
                  f"(dPhi={c['dphi']:+.2f} dGamma={c['dgamma']:+.3f})")
            allpair_phi.setdefault(pid, []).append((c["phi_correct"], c["n"]))
        print(f"  ladder Spearman(sqrtPhi,err)={r['mean_spearman']}  "
              f"frac0inv={('%.3f'%r['frac0inv']) if r['frac0inv'] is not None else 'NA'} | "
              f"COARSE(cubic<opt<stras<wino) frac0inv="
              f"{('%.3f'%r['coarse_frac0inv']) if r['coarse_frac0inv'] is not None else 'NA'}")
        print(f"  emu/real ratio median={r['emu_real_ratio_median']}")
        _sd, _sr = r['strassen_dom_frac'], r['strassen_red_median']
        print(f"  DOM cert<Strassen in {('%.3f'%_sd) if _sd is not None else 'NA'} of tiles; "
              f"median error reduction {('%.1f%%'%(_sr*100)) if _sr is not None else 'NA'}")

    # pooled, model-balanced
    gs = [r["G_cert"] for r in per.values()]
    locs = [r["loc_win_frac"] for r in per.values() if r["loc_win_frac"] is not None]
    coarses = [r["coarse_frac0inv"] for r in per.values() if r["coarse_frac0inv"] is not None]
    domfs = [r["strassen_dom_frac"] for r in per.values() if r["strassen_dom_frac"] is not None]
    domrs = [r["strassen_red_median"] for r in per.values() if r["strassen_red_median"] is not None]
    # Model-balanced mean Phi-correct rate per designed disagreement pair.
    o3_pooled = {}
    for pid in sorted(allpair_phi):
        vals = [v for v, n in allpair_phi[pid] if v is not None]
        o3_pooled[pid] = sum(vals) / len(vals) if vals else None
    o3_all = [v for v in o3_pooled.values() if v is not None]
    o3_mean = sum(o3_all) / len(o3_all) if o3_all else None
    print(f"\n===== POOLED (model-balanced, {len(per)} models) =====")
    print(f"  oracle gap G(cert): per-model {['%+.2f%%'%(g*100) for g in gs]}  mean={sum(gs)/len(gs)*100:+.3f}%  "
          f"argmin-in-opt-class {sum(r['argmin_in_optclass'] for r in per.values())}/{len(per)}")
    print(f"  local-min D*-beats-all-perturb frac: mean={sum(locs)/len(locs):.3f}")
    print(f"  Phi-vs-Gamma pooled Phi-correct per pair: {[f'{p}={v:.3f}' for p,v in o3_pooled.items()]}")
    print(f"     -> Phi wins on {sum(1 for v in o3_all if v>0.5)}/{len(o3_all)} pairs; mean Phi-correct={o3_mean:.3f}")
    print(f"  coarse ladder frac0inv: mean={sum(coarses)/len(coarses):.3f}")
    if domfs:
        print(f"  DOM cert<Strassen: per-model {['%.3f'%d for d in domfs]}  mean={sum(domfs)/len(domfs):.3f}; "
              f"median error reduction mean={sum(domrs)/len(domrs)*100:.1f}%")
    out = {"phi_min": phi_min, "per_model": {m: {k: v for k, v in r.items() if k != "E"}
                                             for m, r in per.items()},
           "pooled": {"G_cert_mean": sum(gs) / len(gs),
                      "argmin_in_optclass": sum(r["argmin_in_optclass"] for r in per.values()),
                      "local_min_mean": sum(locs) / len(locs),
                      "o3_phi_correct_per_pair": o3_pooled, "o3_phi_correct_mean": o3_mean,
                      "coarse_frac0inv_mean": sum(coarses) / len(coarses) if coarses else None,
                      "strassen_dom_frac_mean": sum(domfs) / len(domfs) if domfs else None,
                      "strassen_red_median_mean": sum(domrs) / len(domrs) if domrs else None,
                      "n_models": len(per),
                      "n_tiles": sum(r["n_tiles"] for r in per.values())}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}")
    print("REALTILE_PANEL_PARSE_DONE")


if __name__ == "__main__":
    main()
