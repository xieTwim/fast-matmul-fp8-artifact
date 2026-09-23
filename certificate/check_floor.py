#!/usr/bin/env python3
"""check_floor.py -- the executable certificate for the class-wide Phi optimum.

This is the script behind Theorem 1 and behind the "Exact (symbolic)" cells of Table 1.
It runs in about a second on a laptop, uses the standard library only, and decides every
question by exact arithmetic in Q(rho), rho = (4/3)^(1/4).  Nothing is sampled and nothing
is compared against a tolerance.

WHAT IS CHECKED

  (a) VALUE.  At T* = [[rho^-1, rho/2], [0, rho]] the transformed Strassen triple has
      Phi = sum_r ||u_r||^2 ||v_r||^2 ||w_r||^2 = 200/9 exactly: the three irrational
      components of the Q(rho) result all vanish.

  (b) ADMISSIBILITY.  The transformed triple is still an exact rank-7 <2,2,2>
      decomposition -- all 64 tensor entries agree.  Without this the minimizer could be
      a degenerate point rather than a real algorithm.

  (c) STATIONARITY (the moment map).  The exact first variation of Phi at T* vanishes in
      all six noncompact directions: the moment-map components S_E, S_F, S_G are the zero
      2x2 matrix.  These are DERIVED from the transformed algorithm here, not asserted.

  (d) CURVATURE.  The 6x6 coordinate Hessian of Phi at T* is DERIVED here, exactly, and
      certified positive definite by exact LDL^T pivots.

      How (d) is derived, since this is the step a reader is most entitled to distrust.
      Phi is expanded along the geodesic (E,F,G) = (exp(tH), exp(tK), exp(tL)) with H, K, L
      symmetric traceless -- these six coordinates are all of Sym_0(2)^3, the full tangent
      complement to the gauge directions:

          Phi(t) = sum_r [u_r^T (E ox F^-1) u_r] [v_r^T (F ox G^-1) v_r] [w_r^T (E^-1 ox G) w_r]

      Each matrix entry is carried as a polynomial in t truncated at t^2 with coefficients
      in Q(rho), so Phi(t) = c0 + c1 t + c2 t^2 exactly.  Evaluating on the six coordinate
      directions and on their pairwise sums, and polarizing, gives the matrix M of the
      quadratic form c2.  The coordinate Hessian is Hess = 2M, because Phi(t) = c0 + c2 t^2
      means d^2Phi/dt^2 = 2 c2.  (That factor of two is the whole difference between the
      form's eigenvalues 64/9, 220/9 and the Hessian's 128/9, 440/9 reported in the paper;
      it is stated here so that no reader has to guess which convention a number is in.)

      Positive definiteness is then certified by exact LDL^T: six strictly positive
      pivots, by Sylvester's criterion.  Leading principal minors are printed alongside as
      a second, independent exact witness.  The eigenvalues 128/9 (multiplicity two) and
      440/9 (multiplicity four) are confirmed by exhibiting det(Hess - lambda I) = 0 for
      each -- again exactly, never by a floating-point eigensolver.

  Globality itself -- that this strict local minimum is the global one over the whole
  orbit, and that the value transfers to every exact real rank-7 decomposition -- is the
  Kempf-Ness / geodesic-convexity argument plus de Groote's classification.  Those are
  proved in Appendices A and B of the paper; they are not re-run here.  This script
  certifies the attained value, the admissibility, the exact stationarity, and the exact
  curvature that the argument is applied to.

NEGATIVE CONTROL
  NEGATIVE_CONTROL=1 replaces rho^4 = 4/3 by (4/3)(1 + 1/1000): a genuinely different
  algebraic point.  Expected behaviour, which the script asserts rather than merely
  describes:
    (a) FAILS   -- Phi leaves 200/9.
    (b) PASSES  -- exactness is invariant under ANY change of basis, by construction.
                   A control that broke (b) would mean the sandwich action was misimplemented.
    (c) FAILS   -- the moment map moves off zero.
    (d) FAILS   -- the first-order term is no longer zero, so the point is not critical and
                   "the Hessian is positive definite" no longer certifies a minimum of
                   anything.  The derived Hessian also stops matching the certified one.
  Under the control the script exits 1 and says so.  If it ever reports "0 failures" under
  the control, the checks are testing nothing and must not be believed.
"""
import sys
from fractions import Fraction as F

from exact import (NEGATIVE_CONTROL, ZERO, q, qadd, qsub, qmul, qscal, is_rational, qstr,
                   approx, transformed_triple, phi, tensor_defects, row_mass,
                   ldl, leading_principal_minors, determinant, Report)

PHI_TARGET = F(200, 9)
HESSIAN_EIGENVALUES = {F(128, 9): 2, F(440, 9): 4}

Ut, Vt, Wt = transformed_triple()


# --------------------------------------------------------------------------------------
# (c) the moment map
# --------------------------------------------------------------------------------------

def _MMt(u):
    """M M^T for the 2x2 reshape M of a flattened row u, returned flattened [00,01,10,11]."""
    return [qadd(qmul(u[0], u[0]), qmul(u[1], u[1])), qadd(qmul(u[0], u[2]), qmul(u[1], u[3])),
            qadd(qmul(u[2], u[0]), qmul(u[3], u[1])), qadd(qmul(u[2], u[2]), qmul(u[3], u[3]))]


def _MtM(u):
    """M^T M, same flattening."""
    return [qadd(qmul(u[0], u[0]), qmul(u[2], u[2])), qadd(qmul(u[0], u[1]), qmul(u[2], u[3])),
            qadd(qmul(u[1], u[0]), qmul(u[3], u[2])), qadd(qmul(u[1], u[1]), qmul(u[3], u[3]))]


def moment_map():
    """The three moment-map components (S_E, S_F, S_G) at the transformed point.

    With a_r = ||u_r||^2, b_r = ||v_r||^2, c_r = ||w_r||^2 and M(x) the 2x2 reshape of row x:
        S_E = sum_r  b_r c_r M(u_r)M(u_r)^T  -  b_r a_r M(w_r)M(w_r)^T
        S_F = sum_r  c_r a_r M(v_r)M(v_r)^T  -  c_r b_r M(u_r)^T M(u_r)
        S_G = sum_r  a_r b_r M(w_r)^T M(w_r) -  a_r c_r M(v_r)^T M(v_r)
    T* is a critical point of Phi in the noncompact directions iff all three vanish.
    """
    SE = [ZERO] * 4
    SF = [ZERO] * 4
    SG = [ZERO] * 4
    for r in range(7):
        u, v, w = Ut[r], Vt[r], Wt[r]
        a, b, c = row_mass(u), row_mass(v), row_mass(w)
        Mu_l, Mu_r = _MMt(u), _MtM(u)
        Mv_l, Mv_r = _MMt(v), _MtM(v)
        Mw_l, Mw_r = _MMt(w), _MtM(w)
        for i in range(4):
            SE[i] = qadd(SE[i], qsub(qmul(qmul(b, c), Mu_l[i]), qmul(qmul(b, a), Mw_l[i])))
            SF[i] = qadd(SF[i], qsub(qmul(qmul(c, a), Mv_l[i]), qmul(qmul(c, b), Mu_r[i])))
            SG[i] = qadd(SG[i], qsub(qmul(qmul(a, b), Mw_r[i]), qmul(qmul(a, c), Mv_r[i])))
    return SE, SF, SG


# --------------------------------------------------------------------------------------
# (d) the Hessian, derived
# --------------------------------------------------------------------------------------

from exact import PZERO, pconst, padd, pmul, kron2  # noqa: E402  (kept next to their use)


def _expm_trunc(m1, m2, sign):
    """exp(sign * t * M) mod t^3 for M = [[m1, m2], [m2, -m1]] with m1, m2 rational.

    For a symmetric traceless 2x2 matrix, M^2 = (m1^2 + m2^2) I, which is why the second
    order term is a scalar multiple of the identity and the series closes in three terms.
    """
    s = F(sign)
    entries = [[F(m1), F(m2)], [F(m2), -F(m1)]]
    half_sq = (F(m1) * F(m1) + F(m2) * F(m2)) / 2
    out = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            c0 = q(1) if i == j else ZERO
            c1 = q(s * entries[i][j])
            c2 = q(half_sq) if i == j else ZERO
            out[i][j] = (c0, c1, c2)
    return out


def _pkron2(A, B):
    return [[pmul(A[i][k], B[j][l]) for k in range(2) for l in range(2)]
            for i in range(2) for j in range(2)]


def _quad_form(vec, M4):
    """v^T M v where v has constant Q(rho) entries and M is 4x4 of t-polynomials."""
    acc = PZERO
    for i in range(4):
        if vec[i] == ZERO:
            continue
        for j in range(4):
            if vec[j] == ZERO:
                continue
            acc = padd(acc, pmul(pconst(qmul(vec[i], vec[j])), M4[i][j]))
    return acc


def phi_along(direction):
    """Phi(exp(tH), exp(tK), exp(tL)) mod t^3, for direction = (h1,h2,k1,k2,l1,l2)."""
    h1, h2, k1, k2, l1, l2 = direction
    E, Ei = _expm_trunc(h1, h2, +1), _expm_trunc(h1, h2, -1)
    Fm, Fi = _expm_trunc(k1, k2, +1), _expm_trunc(k1, k2, -1)
    G, Gi = _expm_trunc(l1, l2, +1), _expm_trunc(l1, l2, -1)
    EF, FG, EG = _pkron2(E, Fi), _pkron2(Fm, Gi), _pkron2(Ei, G)
    tot = PZERO
    for r in range(7):
        tot = padd(tot, pmul(pmul(_quad_form(Ut[r], EF), _quad_form(Vt[r], FG)),
                             _quad_form(Wt[r], EG)))
    return tot


def _unit(*indices):
    return tuple(F(1) if i in indices else F(0) for i in range(6))


def derive_hessian():
    """Return (Hessian, first_order_defects, irrational_defects).

    Hessian is the exact 6x6 rational matrix of d^2 Phi/dt^2 at T*.  first_order_defects
    collects the directions in which the t^1 coefficient failed to vanish -- nonempty means
    T* is not a critical point, which is what the negative control produces.
    """
    first_order_defects = []
    irrational = []
    c2 = {}

    def sample(key, direction):
        p = phi_along(direction)
        if p[1] != ZERO:
            first_order_defects.append((key, p[1]))
        if not is_rational(p[2]):
            irrational.append(key)
        return p[2][0]

    for i in range(6):
        c2[(i,)] = sample((i,), _unit(i))
    M = [[F(0)] * 6 for _ in range(6)]
    for i in range(6):
        M[i][i] = c2[(i,)]
    for i in range(6):
        for j in range(i + 1, 6):
            both = sample((i, j), _unit(i, j))
            off = (both - c2[(i,)] - c2[(j,)]) / 2
            M[i][j] = M[j][i] = off
    hess = [[2 * M[i][j] for j in range(6)] for i in range(6)]
    return hess, first_order_defects, irrational


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    rep = Report("check_floor.py -- exact certificate for Phi_min = 200/9 on <2,2,2;7>"
                 + ("   [NEGATIVE CONTROL]" if NEGATIVE_CONTROL else ""))

    # (a) value ---------------------------------------------------------------------
    value = phi(Ut, Vt, Wt)
    rep.check("a", value == q(PHI_TARGET),
              f"Phi(T*) = 200/9 exactly (all three irrational components vanish)",
              f"Phi(T*) != 200/9 -- got {qstr(value)}, numerically {approx(value):.12f}")

    # (b) admissibility -------------------------------------------------------------
    defects = tensor_defects(Ut, Vt, Wt)
    rep.check("b", defects == 0,
              "the transformed triple is an exact rank-7 <2,2,2> decomposition "
              "(all 64 tensor entries)",
              f"the transformed triple is NOT an exact decomposition "
              f"({defects}/64 tensor entries wrong)")

    # (c) stationarity ---------------------------------------------------------------
    SE, SF, SG = moment_map()
    nonzero = sum(1 for M in (SE, SF, SG) for e in M if e != ZERO)
    rep.check("c", nonzero == 0,
              "moment map S_E = S_F = S_G = 0 exactly -- T* is a critical point of Phi",
              f"moment map does NOT vanish: {nonzero}/12 entries nonzero "
              f"(S_E[0,0] = {qstr(SE[0])})")

    # (d) curvature -------------------------------------------------------------------
    hess, first_order, irrational = derive_hessian()
    if irrational:
        rep.fail("d", f"the t^2 coefficient is irrational in directions {irrational}; "
                      "the quadratic form is not rational and the derivation below is void")
    elif first_order:
        key, coeff = first_order[0]
        rep.fail("d", f"T* is NOT a critical point -- the t^1 coefficient is nonzero in "
                      f"{len(first_order)} of 21 sampled directions (first: e{key} with "
                      f"c1 = {qstr(coeff)}). Positive definiteness certifies nothing at a "
                      f"non-critical point.")
    else:
        L, pivots = ldl(hess)
        positive = L is not None and all(p > 0 for p in pivots)
        rep.check("d", positive,
                  "coordinate Hessian is positive definite -- six exact LDL^T pivots "
                  + ", ".join(str(p) for p in pivots) + ", all > 0 (Sylvester)",
                  "coordinate Hessian is NOT positive definite -- exact LDL^T pivots "
                  + ", ".join(str(p) for p in pivots))
        if positive:
            minors = leading_principal_minors(hess)
            rep.check("d2", all(m > 0 for m in minors),
                      "second independent witness: all six leading principal minors > 0",
                      "leading principal minors are not all positive: "
                      + ", ".join(str(m) for m in minors))
            # Multiplicities, certified rather than asserted.
            #
            # Checking only det(Hess - lambda I) == 0 for the two lambdas says they ARE
            # eigenvalues; it says nothing about the other four, so a matrix with spectrum
            # {128/9, 440/9, 1, 2, 3, 4} passes it.
            #
            # Two exact facts together pin the spectrum completely:
            #   (i)  (Hess - aI)(Hess - bI) == 0. The minimal polynomial then divides
            #        (x-a)(x-b), so the spectrum is CONTAINED in {a, b} -- no third
            #        eigenvalue can hide. (Hess is symmetric, hence diagonalisable.)
            #   (ii) trace(Hess) == 2a + 4b. With the spectrum confined to {a, b} and
            #        m_a + m_b = 6, the trace fixes m_a = 2 and m_b = 4 uniquely.
            (a, ma), (b, mb) = sorted(HESSIAN_EIGENVALUES.items())
            n = len(hess)
            A = [[hess[i][j] - (a if i == j else 0) for j in range(n)] for i in range(n)]
            B = [[hess[i][j] - (b if i == j else 0) for j in range(n)] for i in range(n)]
            prod = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
                    for i in range(n)]
            confined = all(prod[i][j] == 0 for i in range(n) for j in range(n))
            trace_ok = sum(hess[i][i] for i in range(n)) == ma * a + mb * b
            rep.check("d3", confined and trace_ok,
                      "spectrum certified exactly: (Hess - 128/9 I)(Hess - 440/9 I) = 0 "
                      "confines the spectrum to {128/9, 440/9}, and trace = 2(128/9) + "
                      "4(440/9) fixes the multiplicities at 2 and 4",
                      "the derived Hessian does not have the certified spectrum "
                      "{128/9 x2, 440/9 x4}: "
                      + ("spectrum not confined to the two certified eigenvalues"
                         if not confined else "trace disagrees with multiplicities 2 and 4"))

    failures = rep.emit()

    print()
    if NEGATIVE_CONTROL:
        print("  negative control active: rho^4 = (4/3)(1 + 1/1000).")
        print("  Checks (a), (c) and (d) MUST appear as FAIL above; (b) MUST still pass.")
        if failures == 0:
            print("  *** (a), (c) and (d) all passed under the control: these checks are "
                  "testing nothing. ***")
    print(f"  {failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
