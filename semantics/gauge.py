#!/usr/bin/env python3
"""gauge.py -- executable definitions of re-basing and the canonical rebalanced gauge.

Appendix E states both of these in prose.  This file is the same two definitions as code,
with no dependencies at all, so that a reader can execute them rather than reconstruct
them.  Running it prints a worked demonstration on classic Strassen; importing it gives
the four operations the paper's orbit machinery is built from.

    transform(U, V, W, X, Y, Z)   the sandwich action -- a change of basis on the
                                  <2,2,2> tensor.  RE-BASING is this action with a
                                  particular choice of (X, Y, Z); the intervention
                                  reported in the paper uses X = Y = Z = T*.
    rebalance(U, V)               the canonical rebalanced gauge.
    phi(U, V, W), gamma(U, V, W)  the two coefficient functionals.

THE CANONICAL REBALANCED GAUGE, and why fixing it is a measurement requirement.

A rank-one term (U_r, V_r, W_r) may be rescaled as

    (U_r, V_r, W_r)  ->  (U_r / s_r,  s_r V_r,  W_r)

for any s_r > 0 without changing the algorithm: the product U_r . V_r . W_r is untouched,
so the decomposition stays exact.  The canonical gauge picks

    s_r = ( ||U_r||_2 / ||V_r||_2 )^(1/2),

which equalizes the two encoder norms in every term, ||U_r|| = ||V_r||, and leaves the
decoder alone.  Phi and Gamma are invariant under this rescaling, because both depend only
on the products ||U_r|| ||V_r|| ||W_r||.  Realized fp8 error is NOT invariant: an extreme
split drives one encoded operand toward the e4m3 subnormal range, which would register as
accuracy the realization does not actually have.  So the gauge has to be fixed before two
realizations can be compared on hardware, and every arm in the panel is stored in it.

WHAT THIS FILE DOES NOT COVER.  The fp8 quantization semantics -- block scaling, the e4m3
cast, the fused encoding path -- are NOT reimplemented here.  Reimplementing them in pure
Python would produce a second, subtly different rounding path, which is worse than
shipping one authoritative implementation.  That implementation is
experiments/fp8_emulator.py, whose quantizer and algorithm functions are identical,
comments and docstrings aside, to the implementation that produced the paper's emulator
numbers; the rounding rule it applies is stated in Appendix E of the paper and in the
emulator's docstring.

Usage:  python3 gauge.py
"""
import math
import sys

# --------------------------------------------------------------------------------------
# Strassen's integer triple.  Rows are the seven products; each row is a 2x2 pattern
# flattened row-major.
# --------------------------------------------------------------------------------------

U_STRASSEN = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
              [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
V_STRASSEN = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
              [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
W_STRASSEN = [[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
              [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]

RHO = (4.0 / 3.0) ** 0.25
#: T* = [[rho^-1, rho/2], [0, rho]].  Section 3; exact coordinates in coordinates.json.
T_STAR = [[1.0 / RHO, RHO / 2.0], [0.0, RHO]]


# --------------------------------------------------------------------------------------
# 2x2 linear algebra, written out so the file needs nothing
# --------------------------------------------------------------------------------------

def inv2(M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if det == 0:
        raise ValueError("singular change of basis")
    return [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]]


def transpose2(M):
    return [[M[0][0], M[1][0]], [M[0][1], M[1][1]]]


def kron2(A, B):
    """Kronecker product of two 2x2 matrices -> 4x4."""
    return [[A[i][k] * B[j][l] for k in range(2) for l in range(2)]
            for i in range(2) for j in range(2)]


def rows_times(rows, M):
    """(n x 4) coefficient rows times a 4x4 matrix."""
    return [[sum(row[k] * M[k][c] for k in range(4)) for c in range(4)] for row in rows]


# --------------------------------------------------------------------------------------
# the two definitions
# --------------------------------------------------------------------------------------

def transform(U, V, W, X, Y, Z):
    """The sandwich action.  RE-BASING is this with the chosen (X, Y, Z).

        U -> U (X ox Y^-T),   V -> V (Y ox Z^-T),   W -> W (X^-T ox Z)

    The action preserves the <2,2,2> tensor, so the result is still an exact rank-7
    algorithm; what it moves is Phi.
    """
    Xi_T, Yi_T, Zi_T = transpose2(inv2(X)), transpose2(inv2(Y)), transpose2(inv2(Z))
    return (rows_times(U, kron2(X, Yi_T)),
            rows_times(V, kron2(Y, Zi_T)),
            rows_times(W, kron2(Xi_T, Z)))


def rebalance(U, V):
    """The canonical rebalanced gauge: s_r = sqrt(||U_r|| / ||V_r||), U_r/s_r, s_r V_r.

    Exactness-preserving and Phi/Gamma-neutral.  Its purpose is to remove the degenerate
    gauges in which one encoder is driven toward the fp8 subnormal range.
    """
    Uo, Vo = [], []
    for u, v in zip(U, V):
        nu, nv = norm(u), norm(v)
        s = math.sqrt(max(nu, 1e-30) / max(nv, 1e-30))
        Uo.append([x / s for x in u])
        Vo.append([x * s for x in v])
    return Uo, Vo


def rebase_to_optimum(U, V, W):
    """The intervention reported in the paper: re-base to T*, then fix the canonical gauge."""
    Ut, Vt, Wt = transform(U, V, W, T_STAR, T_STAR, T_STAR)
    Ut, Vt = rebalance(Ut, Vt)
    return Ut, Vt, Wt


# --------------------------------------------------------------------------------------
# the functionals and the exactness check
# --------------------------------------------------------------------------------------

def norm(row):
    return math.sqrt(sum(x * x for x in row))


def term_products(U, V, W):
    return [norm(U[r]) * norm(V[r]) * norm(W[r]) for r in range(len(U))]


def phi(U, V, W):
    """Phi = sum_r ||u_r||^2 ||v_r||^2 ||w_r||^2, the second moment of the term products."""
    return sum(p * p for p in term_products(U, V, W))


def gamma(U, V, W):
    """Gamma = sum_r ||u_r|| ||v_r|| ||w_r||, the classical l1 growth factor."""
    return sum(term_products(U, V, W))


def tensor_error(U, V, W):
    """Max deviation from the <2,2,2> multiplication tensor over all 64 entries."""
    worst = 0.0
    target = {}
    for c1 in range(2):
        for c2 in range(2):
            for m in range(2):
                target[(2 * c1 + m, 2 * m + c2, 2 * c1 + c2)] = 1.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                acc = sum(U[r][i] * V[r][j] * W[r][k] for r in range(len(U)))
                worst = max(worst, abs(acc - target.get((i, j, k), 0.0)))
    return worst


# --------------------------------------------------------------------------------------
# demonstration
# --------------------------------------------------------------------------------------

def main():
    failures = []
    print("gauge.py -- re-basing and the canonical rebalanced gauge, demonstrated")
    print()

    U, V, W = U_STRASSEN, V_STRASSEN, W_STRASSEN
    print(f"  classic Strassen        Phi = {phi(U, V, W):.6f}   Gamma = {gamma(U, V, W):.6f}"
          f"   exactness {tensor_error(U, V, W):.1e}")

    # re-basing moves Phi; it does not break the algorithm.
    Ur, Vr, Wr = rebase_to_optimum(U, V, W)
    err = tensor_error(Ur, Vr, Wr)
    print(f"  re-based to T*          Phi = {phi(Ur, Vr, Wr):.6f}   "
          f"Gamma = {gamma(Ur, Vr, Wr):.6f}   exactness {err:.1e}")
    print()

    if err > 1e-9:
        failures.append(f"re-basing broke exactness (max tensor entry error {err:.2e})")

    target = 200.0 / 9.0
    if abs(phi(Ur, Vr, Wr) - target) > 1e-9:
        failures.append(f"re-based Phi is {phi(Ur, Vr, Wr)!r}, not 200/9 = {target!r}")
    else:
        print(f"  re-basing takes Phi from 32 to 200/9 = {target:.6f}, a factor of "
              f"{32.0 / target:.4f}, with the algorithm still exact.")

    # the gauge equalizes the encoder norms ...
    spread = max(abs(norm(Ur[r]) - norm(Vr[r])) for r in range(7))
    if spread > 1e-9:
        failures.append(f"the canonical gauge did not equalize encoder norms (max gap {spread:.2e})")
    else:
        print(f"  in the canonical gauge ||u_r|| = ||v_r|| for all seven terms "
              f"(max gap {spread:.1e}).")

    # ... and leaves both functionals alone.
    Ug, Vg = rebalance(Ur, Vr)
    # Phi and Gamma agreeing is NOT idempotence -- they are invariant under every admissible
    # product-one rescaling, so this comparison passes for rescalings that move the representative.
    # The output states the stronger property, so test the stronger property:
    # (Ur, Vr) is already canonical, hence rebalancing it again must return it ELEMENTWISE.
    idem_gap = max(max(abs(a - b) for a, b in zip(g_row, r_row))
                   for g_rows, r_rows in ((Ug, Ur), (Vg, Vr))
                   for g_row, r_row in zip(g_rows, r_rows))
    if abs(phi(Ug, Vg, Wr) - phi(Ur, Vr, Wr)) > 1e-12 or \
       abs(gamma(Ug, Vg, Wr) - gamma(Ur, Vr, Wr)) > 1e-12:
        failures.append("rebalancing changed Phi or Gamma; the gauge is not neutral")
    elif idem_gap > 1e-12:
        failures.append(
            f"rebalancing an already-canonical representative moved it (max coefficient gap "
            f"{idem_gap:.2e}); it is not idempotent")
    else:
        print(f"  rebalancing is idempotent (max coefficient gap {idem_gap:.1e}) and leaves Phi")
        print("  and Gamma unchanged: the gauge is a choice of representative, not a change")
        print("  of algorithm.")

    # a degenerate gauge: same algorithm, same Phi, wildly different operand scales.
    scale = 1e4
    Ud = [[x * scale for x in row] for row in Ur]
    Vd = [[x / scale for x in row] for row in Vr]
    same_phi = abs(phi(Ud, Vd, Wr) - phi(Ur, Vr, Wr)) < 1e-6 * phi(Ur, Vr, Wr)
    # Scale-INDEPENDENT tolerance. This read `1e-9 * scale` (= 1e-5 here) while the intended
    # transformation's error is ~2.2e-16, so it admitted a mutation five orders of magnitude off
    # product-one. Harmless while the value was only printed; the moment it became a failure
    # condition it had to hold the weight it was being given.
    still_exact = tensor_error(Ud, Vd, Wr) < 1e-9
    print()
    print(f"  a degenerate gauge (encoder split by {scale:g}x) keeps the algorithm exact "
          f"({'yes' if still_exact else 'NO'})")
    print(f"  and keeps Phi identical ({'yes' if same_phi else 'NO'}) -- but would place one")
    print("  encoded operand near the e4m3 subnormal range on hardware.  That is why the")
    print("  gauge is fixed before any error is measured.")
    if not same_phi:
        failures.append("Phi was not invariant under a pure gauge change; check rebalance()")
    # `still_exact` was computed and PRINTED -- including the word NO -- without ever being able to
    # fail the script. A displayed negative that cannot turn the exit code is a report, not a check.
    if not still_exact:
        failures.append(
            "a pure gauge change broke exactness; a product-one rescaling must leave the "
            "product untouched")

    print()
    for msg in failures:
        print(f"  FAIL  {msg}")
    print(f"  {len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
