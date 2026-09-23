#!/usr/bin/env python3
"""check_arms.py -- verify the fixed 24-point panel against its own recorded description.

Standard library only. Reads arms24.json and re-derives the panel properties directly
from the coefficient triples.

Unlike the certificate checks, this file works in floating point: the panel arms are
orbit points reached by a numerical change of basis, so they are float64 by nature.  The
tolerances below are therefore real tolerances, and they are stated rather than hidden.

WHAT IS CHECKED, per arm

  * EXACTNESS.  sum_r U[r,i] V[r,j] W[r,k] reproduces the <2,2,2> multiplication tensor to
    within EXACT_TOL over all 64 entries.  Every panel arm must be a genuine rank-7
    algorithm; an arm that is not is not a competitor, it is a bug.
  * PHI and GAMMA.  Both recomputed from the coefficients and compared with the values
    recorded when the panel was frozen.
  * CANONICAL GAUGE.  ||u_r||_2 = ||v_r||_2 for each of the seven terms, which is what the
    canonical rebalanced gauge fixes.  This is the property that makes the arms comparable:
    Phi and Gamma are invariant under the rescaling, but realized fp8 error is not.

WHAT IS CHECKED, about the panel as a whole

  * The certified arm attains the certificate value 200/9.
  * The optimum class -- the arms whose Phi equals that minimum to the panel's resolution --
    is exactly {dps_acc, dps_int}, as Appendix E states.
  * The twelve perturbations sit at the design ratio Phi/Phi_min = 1.08.
  * The three disagreement pairs really do disagree: within each pair Gamma is tied to
    GAMMA_TIE_TOL while Phi is split, and the Phi and Gamma orderings point opposite ways.
    If a pair failed this, the rival-hypothesis test of Section 4.2 would be vacuous.

This script checks the panel's DESIGN.  It does not re-run the kernels, and it says nothing
about which arm won on hardware -- that is in the released measurements, see
scripts/analyze_realtile.py.
"""
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

EXACT_TOL = 1e-9          # max abs deviation allowed on the 64 tensor entries
VALUE_TOL = 1e-9          # relative agreement for recomputed Phi / Gamma
GAUGE_TOL = 1e-9          # relative agreement for ||u_r|| = ||v_r||
PHI_MIN_TOL = 1e-9        # absolute agreement of the certified arm with 200/9
OPT_CLASS_TOL = 1e-3      # "equals the minimum to the panel's resolution"
PERTURB_TOL = 1e-6        # agreement of Phi/Phi_min with the 1.08 design target
GAMMA_TIE_TOL = 1e-2      # relative Gamma spread allowed inside a disagreement pair

PHI_EXACT = 200.0 / 9.0
OPTIMUM_CLASS = {"dps_acc", "dps_int"}
EXPECTED_ARM_NAMES = {
    "dps_acc", "strassen_classic", "winograd_form", "dps_int", "dps_evenpow",
    "dps_smallrat",
    "pert_f0_diag_p", "pert_f0_diag_m", "pert_f0_shear_p", "pert_f0_shear_m",
    "pert_f1_diag_p", "pert_f1_diag_m", "pert_f1_shear_p", "pert_f1_shear_m",
    "pert_f2_diag_p", "pert_f2_diag_m", "pert_f2_shear_p", "pert_f2_shear_m",
    "disagree0_a", "disagree0_b", "disagree1_a", "disagree1_b",
    "disagree2_a", "disagree2_b",
}
EXPECTED_GROUP_COUNTS = {"named": 6, "perturb": 12, "disagree": 6}


def multiplication_tensor():
    T = [[[0.0] * 4 for _ in range(4)] for _ in range(4)]
    for c1 in range(2):
        for c2 in range(2):
            for m in range(2):
                T[2 * c1 + m][2 * m + c2][2 * c1 + c2] = 1.0
    return T


TENSOR = multiplication_tensor()


def norm(row):
    return math.sqrt(sum(v * v for v in row))


def tensor_error(U, V, W):
    worst = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                acc = sum(U[r][i] * V[r][j] * W[r][k] for r in range(len(U)))
                worst = max(worst, abs(acc - TENSOR[i][j][k]))
    return worst


def phi_of(U, V, W):
    return sum((norm(U[r]) * norm(V[r]) * norm(W[r])) ** 2 for r in range(len(U)))


def gamma_of(U, V, W):
    return sum(norm(U[r]) * norm(V[r]) * norm(W[r]) for r in range(len(U)))


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    with open(os.path.join(HERE, "arms24.json")) as f:
        panel = json.load(f)
    failures = []
    listed_arms = panel.get("arms")
    if not isinstance(listed_arms, list):
        print("  FAIL  panel field 'arms' is not a list")
        print("  1 failure(s)")
        return 1
    names = [a.get("name") for a in listed_arms if isinstance(a, dict)]
    if len(names) != len(listed_arms) or len(names) != len(set(names)):
        failures.append("panel contains a malformed or duplicate arm name")
    arms = {a["name"]: a for a in listed_arms if isinstance(a, dict) and "name" in a}
    print(f"check_arms.py -- fixed {panel['n_arms']}-point orbit-response panel "
          f"(seed {panel['seed']}, certified arm {panel['certified_arm']})")
    print()

    if panel.get("n_arms") != 24 or len(listed_arms) != 24:
        failures.append(
            f"frozen panel must declare/carry 24 arms, found "
            f"{panel.get('n_arms')!r}/{len(listed_arms)}")
    if set(names) != EXPECTED_ARM_NAMES:
        failures.append(
            f"arm inventory differs from frozen panel: "
            f"missing={sorted(EXPECTED_ARM_NAMES - set(names))}, "
            f"extra={sorted(set(names) - EXPECTED_ARM_NAMES)}")
    group_counts = Counter(
        a.get("group") for a in listed_arms if isinstance(a, dict))
    if dict(group_counts) != EXPECTED_GROUP_COUNTS:
        failures.append(
            f"group counts {dict(group_counts)!r} != frozen {EXPECTED_GROUP_COUNTS!r}")

    # ---- per-arm ------------------------------------------------------------------
    print(f"  {'arm':<20} {'group':<9} {'Phi':>10} {'Gamma':>10}  exactness   gauge")
    for a in listed_arms:
        U, V, W = a["U"], a["V"], a["W"]
        terr = tensor_error(U, V, W)
        p, g = phi_of(U, V, W), gamma_of(U, V, W)
        gauge = max(abs(norm(U[r]) - norm(V[r])) / max(1.0, norm(U[r])) for r in range(7))
        flags = []
        if terr > EXACT_TOL:
            flags.append(f"NOT an exact decomposition (max entry error {terr:.2e})")
        if not close(p, a["phi"], VALUE_TOL):
            flags.append(f"Phi recomputes to {p!r}, recorded {a['phi']!r}")
        if not close(g, a["gamma"], VALUE_TOL):
            flags.append(f"Gamma recomputes to {g!r}, recorded {a['gamma']!r}")
        if gauge > GAUGE_TOL:
            flags.append(f"not in the canonical rebalanced gauge (max |u|-|v| rel. {gauge:.2e})")
        status = "ok" if not flags else "FAIL"
        print(f"  {a['name']:<20} {a['group']:<9} {p:10.6f} {g:10.6f}  {terr:.1e}    "
              f"{gauge:.1e}  {status}")
        for msg in flags:
            failures.append(f"{a['name']}: {msg}")
    print()

    # ---- the certificate value ------------------------------------------------------
    if panel.get("certified_arm") not in arms:
        failures.append(f"certified arm {panel.get('certified_arm')!r} is absent")
        cert = None
    else:
        cert = arms[panel["certified_arm"]]
    if cert is None:
        cert_phi = float("nan")
    else:
        cert_phi = phi_of(cert["U"], cert["V"], cert["W"])
    if abs(cert_phi - PHI_EXACT) <= PHI_MIN_TOL:
        print(f"  certified arm attains Phi = {cert_phi!r}, i.e. 200/9 to {PHI_MIN_TOL:g}")
    else:
        failures.append(f"certified arm has Phi = {cert_phi!r}, not 200/9 = {PHI_EXACT!r}")

    # ---- the optimum class ------------------------------------------------------------
    found = {a["name"] for a in listed_arms
             if abs(phi_of(a["U"], a["V"], a["W"]) - PHI_EXACT) <= OPT_CLASS_TOL * PHI_EXACT}
    if found == OPTIMUM_CLASS:
        print(f"  optimum class at the panel's resolution = {sorted(found)}, as stated")
    else:
        failures.append(f"optimum class is {sorted(found)}, not {sorted(OPTIMUM_CLASS)}")

    # ---- the perturbation shell ---------------------------------------------------------
    perturbs = [a for a in listed_arms if a["group"] == "perturb"]
    target = panel["perturbation_phi_target"]
    ratios = [phi_of(a["U"], a["V"], a["W"]) / PHI_EXACT for a in perturbs]
    if len(perturbs) != 12:
        failures.append(f"expected 12 perturbations, found {len(perturbs)}")
    off = [r for r in ratios if abs(r - target) > PERTURB_TOL]
    if ratios and not off:
        print(f"  all {len(perturbs)} perturbations sit at Phi/Phi_min = {target} "
              f"(observed spread {max(ratios) - min(ratios):.2e})")
    else:
        failures.append(f"{len(off)} perturbations are off the {target} design ratio: "
                        f"{[round(r, 6) for r in off]}")

    # ---- the disagreement pairs -----------------------------------------------------------
    pairs = {}
    for a in listed_arms:
        if a["group"] == "disagree":
            pairs.setdefault(a["name"].rstrip("_ab").rstrip("_"), []).append(a)
    if len(pairs) != 3:
        failures.append(f"expected 3 disagreement pairs, found {len(pairs)}: {sorted(pairs)}")
    for key in sorted(pairs):
        members = pairs[key]
        if len(members) != 2:
            failures.append(f"pair {key} has {len(members)} members")
            continue
        x, y = members
        px, py = phi_of(x["U"], x["V"], x["W"]), phi_of(y["U"], y["V"], y["W"])
        gx, gy = gamma_of(x["U"], x["V"], x["W"]), gamma_of(y["U"], y["V"], y["W"])
        gamma_spread = abs(gx - gy) / max(abs(gx), abs(gy))
        phi_splits = abs(px - py) / max(abs(px), abs(py))
        orders_disagree = (px < py) != (gx < gy)
        detail = (f"  pair {key}: Gamma {gx:.4f} vs {gy:.4f} (spread {gamma_spread:.2e}), "
                  f"Phi {px:.4f} vs {py:.4f} (split {phi_splits:.1%})")
        if gamma_spread <= GAMMA_TIE_TOL and phi_splits > GAMMA_TIE_TOL and orders_disagree:
            print(detail + " -- tie in Gamma, split in Phi, orderings opposed")
        else:
            print(detail)
            if gamma_spread > GAMMA_TIE_TOL:
                failures.append(f"pair {key}: Gamma is not tied (spread {gamma_spread:.2e})")
            if phi_splits <= GAMMA_TIE_TOL:
                failures.append(f"pair {key}: Phi is not split (split {phi_splits:.2e})")
            if not orders_disagree:
                failures.append(f"pair {key}: Phi and Gamma order the pair the same way, "
                                f"so it does not separate the two hypotheses")

    print()
    for msg in failures:
        print(f"  FAIL  {msg}")
    print(f"  {len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
