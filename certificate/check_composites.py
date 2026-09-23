#!/usr/bin/env python3
"""check_composites.py -- exact certificates for the values built on top of Phi_min = 200/9.

Same discipline as check_floor.py: standard library only, exact arithmetic in Q(rho) with
rho = (4/3)^(1/4), no sampling and no tolerances.  Each value below is computed from the
transformed triple, not asserted.

WHAT IS CHECKED

  (G) THE CLASSICAL GROWTH FACTOR.  At T* the seven per-term Phi-masses
      ||u_r||^2 ||v_r||^2 ||w_r||^2 are the exact rational multiset {8} u {64/27 (x6)}.
      Because those masses are rational, the classical l1 growth factor
      Gamma = sum_r ||u_r|| ||v_r|| ||w_r|| = sqrt(8) + 6 sqrt(64/27) = 2 sqrt(2) + 16/sqrt(3)
      follows with no radical arithmetic at all.  That is the value Dumas, Pernet and
      Sedoglavic reach and describe as probably optimal with a gap of at most 2.6%.
      Gamma's own global optimality is the geodesic-convexity argument in Appendix C of the
      paper, cited rather than re-run here; what this check certifies is the VALUE.

  (K) MULTIPLICATIVITY UNDER THE KRONECKER PRODUCT.  The <4,4,4> rank-49 realization
      T* (x) T* has Phi = (200/9)^2 = 40000/81.  This is verified by actually building the
      49 x 16 Kronecker triple and summing its masses -- multiplicativity is computed, not
      assumed.

  (D) ADDITIVITY UNDER DIRECT SUM.  The block-diagonal sum T* (+) T* is a rank-14 <2,2,4>
      algorithm with Phi = 200/9 + 200/9 = 400/9.  This script constructs its 14
      coefficient rows, checks all 4*8*8 = 256 entries of the <2,2,4> multiplication
      tensor exactly, and computes Phi from those rows.

  For (K), exactness of the full Kronecker product follows from the standard tensor-product
  construction; this script re-checks the rank-7 factor's exactness, the part specific to
  T*.  For (D), the full rank-14 <2,2,4> construction itself is checked here.

NEGATIVE CONTROL
  NEGATIVE_CONTROL=1 replaces rho^4 = 4/3 by (4/3)(1 + 1/1000).  The mass multiset then
  leaves {8, 64/27 x6}, so (G), (K) and (D) all FAIL and the script exits 1.  The factor
  exactness check (0) and direct-sum exactness check (D0) must still PASS because the
  change-of-basis action remains exact.  If the exact marker vector changes, verify.py's
  negative-control contract fails.
"""
import sys
from fractions import Fraction as F

from exact import (NEGATIVE_CONTROL, ZERO, q, qadd, qmul, is_rational, qstr, approx,
                   transformed_triple, phi, row_mass, tensor_defects, Report)

MASS_MULTISET = sorted([F(8)] + [F(64, 27)] * 6)
PHI_KRONECKER = F(40000, 81)
PHI_DIRECT_SUM = F(400, 9)

Ut, Vt, Wt = transformed_triple()


def phi_masses():
    """The seven per-term Phi-masses, as rationals when they are rational."""
    out = []
    for r in range(7):
        m = qmul(qmul(row_mass(Ut[r]), row_mass(Vt[r])), row_mass(Wt[r]))
        out.append(m[0] if is_rational(m) else None)
    return out


def kronecker_square(rows):
    """(7 x 4) coefficient matrix -> the (49 x 16) Kronecker square, over Q(rho)."""
    return [[qmul(rows[i][k], rows[j][l]) for k in range(4) for l in range(4)]
            for i in range(7) for j in range(7)]


def pad_2x2_columns(row, block):
    """Embed a flattened 2x2 row into columns ``2*block:2*block+2`` of a 2x4 row."""
    out = [ZERO] * 8
    for i in range(2):
        for j in range(2):
            out[4 * i + 2 * block + j] = row[2 * i + j]
    return out


def direct_sum_224():
    """Construct the 14-row <2,2,4> algorithm obtained from two column blocks."""
    U = [row[:] for row in Ut] + [row[:] for row in Ut]
    V = [pad_2x2_columns(row, 0) for row in Vt]
    V += [pad_2x2_columns(row, 1) for row in Vt]
    W = [pad_2x2_columns(row, 0) for row in Wt]
    W += [pad_2x2_columns(row, 1) for row in Wt]
    return U, V, W


def tensor_defects_224(U, V, W):
    """Count wrong entries of the exact <2,2,4> multiplication tensor (256 total)."""
    bad = 0
    for i in range(4):                  # A: 2 x 2
        a_row, inner_a = divmod(i, 2)
        for j in range(8):              # B: 2 x 4
            inner_b, b_col = divmod(j, 4)
            for k in range(8):          # C: 2 x 4
                c_row, c_col = divmod(k, 4)
                target = 1 if (inner_a == inner_b and a_row == c_row
                               and b_col == c_col) else 0
                acc = ZERO
                for r in range(14):
                    acc = qadd(acc, qmul(qmul(U[r][i], V[r][j]), W[r][k]))
                if acc != q(target):
                    bad += 1
    return bad


def main():
    rep = Report("check_composites.py -- exact certificates for the composite Phi values"
                 + ("   [NEGATIVE CONTROL]" if NEGATIVE_CONTROL else ""))

    # sanity: the rank-7 factor these composites are built from is itself exact.
    defects = tensor_defects(Ut, Vt, Wt)
    rep.check("0", defects == 0,
              "the rank-7 factor is an exact <2,2,2> decomposition",
              f"the rank-7 factor is NOT exact ({defects}/64 tensor entries wrong); "
              "the composite results below rest on nothing")

    # (G) the mass multiset behind the classical growth factor -------------------------
    masses = phi_masses()
    got = sorted(masses) if None not in masses else None
    rep.check("G", got == MASS_MULTISET,
              "per-term Phi-mass multiset = {8, 64/27 x6} exactly, so "
              "Gamma = 2*sqrt(2) + 16/sqrt(3) (the DPS value)",
              f"per-term Phi-mass multiset is {got}, not {{8, 64/27 x6}}"
              if got is not None else
              "per-term Phi-masses are not rational, so Gamma does not close in this form")

    # (K) multiplicativity under Kronecker product --------------------------------------
    UK, VK, WK = (kronecker_square(Ut), kronecker_square(Vt), kronecker_square(Wt))
    phi_k = ZERO
    for r in range(49):
        phi_k = qadd(phi_k, qmul(qmul(row_mass(UK[r]), row_mass(VK[r])), row_mass(WK[r])))
    rep.check("K", phi_k == q(PHI_KRONECKER),
              "Phi(T* (x) T*) = (200/9)^2 = 40000/81 exactly, computed over all 49 terms "
              "of the <4,4,4> Kronecker square",
              f"Phi(T* (x) T*) = {qstr(phi_k)} (numerically {approx(phi_k):.6f}), "
              f"not 40000/81 = {float(PHI_KRONECKER):.6f}")

    # (D) additivity under direct sum ----------------------------------------------------
    UD, VD, WD = direct_sum_224()
    direct_defects = tensor_defects_224(UD, VD, WD)
    rep.check("D0", direct_defects == 0,
              "constructed rank-14 triple is an exact <2,2,4> decomposition "
              "(all 256 tensor entries)",
              f"constructed rank-14 triple is NOT an exact <2,2,4> decomposition "
              f"({direct_defects}/256 tensor entries wrong)")
    phi_d = phi(UD, VD, WD)
    rep.check("D", phi_d == q(PHI_DIRECT_SUM),
              "Phi(T* (+) T*) = 200/9 + 200/9 = 400/9 exactly, computed from the "
              "constructed rank-14 rows",
              f"Phi(T* (+) T*) = {qstr(phi_d)}, not 400/9")

    failures = rep.emit()

    print()
    if NEGATIVE_CONTROL:
        print("  negative control active: rho^4 = (4/3)(1 + 1/1000).")
        print("  Checks (G), (K) and (D) MUST FAIL; (0) and (D0) MUST PASS.")
        if failures == 0:
            print("  *** all checks passed under the control: they are testing nothing. ***")
    print(f"  {failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
