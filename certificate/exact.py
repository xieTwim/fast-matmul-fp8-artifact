#!/usr/bin/env python3
"""Exact arithmetic for the certificate checks.  STANDARD LIBRARY ONLY, BY DESIGN.

Everything the certificate scripts need lives here, in three layers:

  1. Q(rho), rho = (4/3)^(1/4).  An element is a 4-tuple of `fractions.Fraction`
     (the coefficients of 1, rho, rho^2, rho^3); multiplication reduces via rho^4 = 4/3.
     Nothing is floating point and nothing is sampled, so every equality below is a
     decision, not a tolerance test.

  2. Truncated polynomials in t (degree <= 2) with Q(rho) coefficients.  This is what
     lets the second-order expansion of Phi around T* be *derived* rather than asserted.

  3. The objects: Strassen's integer triple, the balanced transform T*, the sandwich
     action, and an exact LDL^T factorization used as the positive-definiteness
     certificate.

NEGATIVE CONTROL
  With NEGATIVE_CONTROL=1 in the environment, rho^4 is replaced by (4/3)(1 + 1/1000):
  a genuinely different algebraic point.  Every check that is really testing something
  must FAIL under the control.  Each script says in its own header which of its checks
  the control is expected to break, and why the remaining ones are invariant.
"""
from fractions import Fraction as F
import os

NEGATIVE_CONTROL = os.environ.get("NEGATIVE_CONTROL", "") == "1"

#: rho^4.  4/3 at the real algebraic point; perturbed under the negative control.
R4 = F(4, 3) * (F(1001, 1000) if NEGATIVE_CONTROL else 1)

# --------------------------------------------------------------------------------------
# Layer 1 -- Q(rho)
# --------------------------------------------------------------------------------------

ZERO = (F(0), F(0), F(0), F(0))


def q(a=0, b=0, c=0, d=0):
    """The element a + b*rho + c*rho^2 + d*rho^3."""
    return (F(a), F(b), F(c), F(d))


def qadd(x, y):
    return tuple(x[i] + y[i] for i in range(4))


def qneg(x):
    return tuple(-v for v in x)


def qsub(x, y):
    return tuple(x[i] - y[i] for i in range(4))


def qmul(x, y):
    z = [F(0)] * 7
    for i in range(4):
        if x[i] == 0:
            continue
        for j in range(4):
            if y[j] == 0:
                continue
            z[i + j] += x[i] * y[j]
    # reduce rho^4 = R4, rho^5 = R4*rho, rho^6 = R4*rho^2
    return (z[0] + R4 * z[4], z[1] + R4 * z[5], z[2] + R4 * z[6], z[3])


def qscal(k, x):
    return tuple(F(k) * v for v in x)


def is_rational(x):
    """True when the three irrational components vanish, i.e. x lies in Q."""
    return x[1] == 0 and x[2] == 0 and x[3] == 0


def approx(x):
    """Float image, for reporting only -- never for a decision."""
    r = float(R4) ** 0.25
    return float(x[0]) + float(x[1]) * r + float(x[2]) * r ** 2 + float(x[3]) * r ** 3


def qstr(x):
    """Readable form, e.g. '200/9' or '1/2 + 3*rho^2'."""
    if is_rational(x):
        return str(x[0])
    parts = []
    for power, coeff in enumerate(x):
        if coeff == 0:
            continue
        if power == 0:
            parts.append(str(coeff))
        elif power == 1:
            parts.append(f"{coeff}*rho")
        else:
            parts.append(f"{coeff}*rho^{power}")
    return " + ".join(parts) if parts else "0"


# --------------------------------------------------------------------------------------
# Layer 2 -- polynomials in t, truncated at t^2, coefficients in Q(rho)
# --------------------------------------------------------------------------------------

PZERO = (ZERO, ZERO, ZERO)


def pconst(x):
    return (x, ZERO, ZERO)


def padd(p, r):
    return tuple(qadd(p[i], r[i]) for i in range(3))


def pmul(p, r):
    """Product truncated at t^2 -- the orders above t^2 are not needed and are discarded."""
    out = [ZERO, ZERO, ZERO]
    for i in range(3):
        if p[i] == ZERO:
            continue
        for j in range(3 - i):
            if r[j] == ZERO:
                continue
            out[i + j] = qadd(out[i + j], qmul(p[i], r[j]))
    return tuple(out)


# --------------------------------------------------------------------------------------
# Layer 3 -- the objects
# --------------------------------------------------------------------------------------

def emb(rows):
    """Embed an integer coefficient matrix into Q(rho)."""
    return [[q(v) for v in row] for row in rows]


#: Strassen's rank-7 <2,2,2> triple, integer coefficients, rows indexed by the seven products.
#: Row r of U/V/W holds the 2x2 encoder/encoder/decoder pattern flattened row-major.
U_STRASSEN = emb([[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
                  [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]])
V_STRASSEN = emb([[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
                  [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
W_STRASSEN = emb([[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
                  [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

RHO = q(0, 1)
RHO_INV = qscal(F(1) / R4, q(0, 0, 0, 1))          # rho^-1 = rho^3 / rho^4

#: T* = [[rho^-1, rho/2], [0, rho]].  det T* = rho^-1 * rho = 1.
TSTAR = [[RHO_INV, qscal(F(1, 2), RHO)],
         [ZERO, RHO]]
#: (T*)^-1 = [[rho, -rho/2], [0, rho^-1]].
TSTAR_INV = [[RHO, qscal(F(-1, 2), RHO)],
             [ZERO, RHO_INV]]
#: ((T*)^-1)^T, the factor the sandwich action needs.
TSTAR_INV_T = [[TSTAR_INV[0][0], TSTAR_INV[1][0]],
               [TSTAR_INV[0][1], TSTAR_INV[1][1]]]


def kron2(A, B):
    """Kronecker product of two 2x2 matrices over Q(rho) -> 4x4."""
    return [[qmul(A[i][k], B[j][l]) for k in range(2) for l in range(2)]
            for i in range(2) for j in range(2)]


def rowsmul(rows, M):
    """(n x 4) rows over Q(rho) times a 4x4 matrix over Q(rho)."""
    return [[qadd(qadd(qmul(r[0], M[0][c]), qmul(r[1], M[1][c])),
                  qadd(qmul(r[2], M[2][c]), qmul(r[3], M[3][c]))) for c in range(4)]
            for r in rows]


def sandwich(U, V, W, X, Y, Zm, Xi_T, Yi_T, Zi_T):
    """The change-of-basis action (U,V,W) -> (U (X ox Y^-T), V (Y ox Z^-T), W (X^-T ox Z))."""
    return (rowsmul(U, kron2(X, Yi_T)),
            rowsmul(V, kron2(Y, Zi_T)),
            rowsmul(W, kron2(Xi_T, Zm)))


def transformed_triple():
    """Strassen's triple pushed to T*, i.e. X = Y = Z = T*.  This is D* = T* . Strassen."""
    return sandwich(U_STRASSEN, V_STRASSEN, W_STRASSEN,
                    TSTAR, TSTAR, TSTAR,
                    TSTAR_INV_T, TSTAR_INV_T, TSTAR_INV_T)


def row_mass(row):
    """||row||_2^2 in Q(rho)."""
    m = ZERO
    for e in row:
        m = qadd(m, qmul(e, e))
    return m


def phi(U, V, W):
    """Phi = sum_r ||u_r||^2 ||v_r||^2 ||w_r||^2, exactly."""
    tot = ZERO
    for r in range(len(U)):
        tot = qadd(tot, qmul(qmul(row_mass(U[r]), row_mass(V[r])), row_mass(W[r])))
    return tot


#: The exact <2,2,2> multiplication tensor.  C[c1,c2] = sum_m A[c1,m] B[m,c2] gives
#: T[(c1,m),(m,c2),(c1,c2)] = 1 and every other entry 0.
def multiplication_tensor():
    T = {}
    for c1 in range(2):
        for c2 in range(2):
            for m in range(2):
                T[(2 * c1 + m, 2 * m + c2, 2 * c1 + c2)] = F(1)
    return T


TENSOR_222 = multiplication_tensor()


def tensor_defects(U, V, W):
    """Number of the 64 tensor entries where sum_r U[r,i]V[r,j]W[r,k] differs from <2,2,2>."""
    bad = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                acc = ZERO
                for r in range(len(U)):
                    acc = qadd(acc, qmul(qmul(U[r][i], V[r][j]), W[r][k]))
                if acc != q(TENSOR_222.get((i, j, k), 0)):
                    bad += 1
    return bad


# --------------------------------------------------------------------------------------
# Exact LDL^T -- the positive-definiteness certificate
# --------------------------------------------------------------------------------------

def ldl(A):
    """Exact LDL^T of a symmetric rational matrix.

    Returns (L, D).  D is the list of pivots; A is positive definite iff every pivot is
    strictly positive (Sylvester).  Returns (None, D) if a zero pivot is reached, which
    is itself a proof that A is not positive definite.  All arithmetic is in Fraction,
    so the pivots are exact and their signs are decisions rather than tolerance tests.
    """
    n = len(A)
    L = [[F(0)] * n for _ in range(n)]
    D = [F(0)] * n
    for j in range(n):
        D[j] = A[j][j] - sum(L[j][k] * L[j][k] * D[k] for k in range(j))
        L[j][j] = F(1)
        if D[j] == 0:
            return None, D[:j + 1]
        for i in range(j + 1, n):
            L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k] * D[k] for k in range(j))) / D[j]
    return L, D


def leading_principal_minors(A):
    """The n leading principal minors, exactly, by fraction-free elimination on each block."""
    out = []
    for k in range(1, len(A) + 1):
        out.append(determinant([row[:k] for row in A[:k]]))
    return out


def determinant(A):
    """Exact determinant of a rational matrix by Gaussian elimination over Fraction."""
    n = len(A)
    M = [row[:] for row in A]
    det = F(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return F(0)
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            det = -det
        det *= M[col][col]
        inv = F(1) / M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] * inv
            if f:
                for c in range(col, n):
                    M[r][c] -= f * M[col][c]
    return det


# --------------------------------------------------------------------------------------
# Small reporting helpers shared by the check scripts
# --------------------------------------------------------------------------------------

class Report:
    """Collects PASS/FAIL lines so a script can print them all and exit once."""

    def __init__(self, title):
        self.title = title
        self.failures = 0
        self.lines = []

    def ok(self, tag, message):
        self.lines.append(f"  [{tag}] PASS  {message}")

    def fail(self, tag, message):
        self.failures += 1
        self.lines.append(f"  [{tag}] FAIL  {message}")

    def check(self, tag, condition, ok_message, fail_message):
        (self.ok if condition else self.fail)(tag, ok_message if condition else fail_message)
        return condition

    def emit(self):
        print(self.title)
        for line in self.lines:
            print(line)
        return self.failures
