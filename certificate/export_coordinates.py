#!/usr/bin/env python3
"""export_coordinates.py -- build T* and the certified realization D*, exactly.

Section 3 gives T* = [[rho^-1, rho/2], [0, rho]] with rho = (4/3)^(1/4) and X = Y = Z = T*.
That is enough to rebuild the certified realization. This script does it and writes
coordinates.json, which carries, for T* and for each of the 21 coefficient rows of the
transformed triple D* = T* . Strassen:

  * the EXACT value, as coefficients of 1, rho, rho^2, rho^3 over the rationals.  This is
    the primary form: it is what the certificate is proved about, and it is exact.
  * a 34-digit decimal, for anyone who wants to paste numbers into another tool.  This is
    a convenience, not the certificate.

Also written: the seven per-term Phi-masses (the multiset {8} u {64/27 x6}), Phi itself,
and the exact 6x6 coordinate Hessian with its LDL^T pivots -- the same objects
check_floor.py derives, serialized so they can be inspected without running anything.

Standard library only.

Usage
    python3 export_coordinates.py          write coordinates.json
    python3 export_coordinates.py --check  rebuild in memory and byte-compare the shipped file
"""
import argparse
import json
import os
import sys
from decimal import Decimal, getcontext
from fractions import Fraction as F

from exact import (NEGATIVE_CONTROL, R4, transformed_triple, phi, row_mass, qstr,
                   is_rational, TSTAR, TSTAR_INV, ldl)
from check_floor import derive_hessian

getcontext().prec = 40
HERE = os.path.dirname(os.path.abspath(__file__))


def rho_decimal():
    """rho = (4/3)^(1/4) to the current decimal precision."""
    return ((Decimal(4) / Decimal(3)).ln() / Decimal(4)).exp()


RHO_D = rho_decimal()


def to_decimal(x):
    """Decimal image of a Q(rho) element, at 34 significant digits."""
    total = Decimal(0)
    for power, coeff in enumerate(x):
        if coeff == 0:
            continue
        total += Decimal(coeff.numerator) / Decimal(coeff.denominator) * (RHO_D ** power)
    return f"{total:.34f}".rstrip("0").rstrip(".")


def serialize(x):
    return {
        "exact": qstr(x),
        "coefficients": [str(c) for c in x],   # of 1, rho, rho^2, rho^3
        "decimal": to_decimal(x),
    }


def build_payload():
    """Return the complete serializable coordinate payload without touching the filesystem."""
    Ut, Vt, Wt = transformed_triple()
    hess, first_order, irrational = derive_hessian()
    _, pivots = ldl(hess)

    masses = []
    for r in range(7):
        m = row_mass(Ut[r])
        n = row_mass(Vt[r])
        o = row_mass(Wt[r])
        from exact import qmul
        masses.append(qmul(qmul(m, n), o))

    payload = {
        "description": "The balanced transform T* and the certified realization "
                       "D* = T* . Strassen, exactly.  Elements of Q(rho) are given as "
                       "coefficients of [1, rho, rho^2, rho^3] with rho = (4/3)^(1/4); "
                       "the decimal forms are 34-digit conveniences, not the certificate.",
        "rho": {
            "definition": "rho^4 = 4/3",
            "decimal": f"{RHO_D:.34f}",
        },
        "T_star": {
            "definition": "[[rho^-1, rho/2], [0, rho]]; determinant 1; applied as X = Y = Z = T*",
            "entries": [[serialize(TSTAR[i][j]) for j in range(2)] for i in range(2)],
            "inverse": [[serialize(TSTAR_INV[i][j]) for j in range(2)] for i in range(2)],
        },
        "D_star": {
            "definition": "rows indexed by the seven products; each row is a 2x2 pattern "
                          "flattened row-major.  U and V are the encoders, W the decoder.",
            "U": [[serialize(e) for e in row] for row in Ut],
            "V": [[serialize(e) for e in row] for row in Vt],
            "W": [[serialize(e) for e in row] for row in Wt],
        },
        "phi_masses": {
            "definition": "||u_r||^2 ||v_r||^2 ||w_r||^2 for each of the seven terms",
            "values": [qstr(m) for m in masses],
            "all_rational": all(is_rational(m) for m in masses),
        },
        "phi": serialize(phi(Ut, Vt, Wt)),
        "hessian": {
            "definition": "the exact 6x6 coordinate Hessian d^2 Phi/dt^2 at T* over "
                          "Sym_0(2)^3, derived in check_floor.py",
            "is_critical_point": not first_order,
            "matrix": [[str(v) for v in row] for row in hess],
            "ldl_pivots": [str(p) for p in pivots],
            "eigenvalues": {"128/9": 2, "440/9": 4},
        },
    }
    return payload, masses, pivots, Ut, Vt, Wt


def encoded(payload):
    """The canonical on-disk representation used by both write and check modes."""
    return (json.dumps(payload, indent=1) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and byte-compare coordinates.json")
    args = ap.parse_args()

    if NEGATIVE_CONTROL:
        print("refusing to export coordinates under NEGATIVE_CONTROL "
              "(they would describe the perturbed point, not T*)", file=sys.stderr)
        return 2

    payload, masses, pivots, Ut, Vt, Wt = build_payload()
    expected = encoded(payload)

    dest = os.path.join(HERE, "..", "coordinates.json")
    if args.check:
        try:
            with open(dest, "rb") as f:
                shipped = f.read()
        except OSError as exc:
            print(f"FAIL  cannot read shipped coordinates.json: {exc}", file=sys.stderr)
            return 1
        if shipped != expected:
            print("FAIL  shipped coordinates.json does not byte-match the exact in-memory rebuild",
                  file=sys.stderr)
            print(f"  rebuilt bytes: {len(expected)}; shipped bytes: {len(shipped)}",
                  file=sys.stderr)
            return 1
    else:
        with open(dest, "wb") as f:
            f.write(expected)

    print("exact coordinates of T* and the certified realization D*")
    print()
    print(f"  rho = (4/3)^(1/4) = {RHO_D:.20f}...")
    print()
    print("  T* =")
    for i in range(2):
        cells = "   ".join(f"{qstr(TSTAR[i][j]):>18}" for j in range(2))
        print(f"      [ {cells} ]")
    print()
    print(f"  {'r':>2}  {'||u_r||^2 ||v_r||^2 ||w_r||^2':>30}")
    for r, m in enumerate(masses):
        print(f"  {r:>2}  {qstr(m):>30}")
    print(f"      {'Phi = ' + qstr(phi(Ut, Vt, Wt)):>36}")
    print()
    print(f"  exact LDL^T pivots of the Hessian: {', '.join(str(p) for p in pivots)}")
    print()
    if args.check:
        print(f"  byte-for-byte match: {os.path.normpath(dest)} ({len(expected)} bytes)")
    else:
        print(f"  wrote {os.path.normpath(dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
