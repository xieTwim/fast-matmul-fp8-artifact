#!/usr/bin/env python3
"""build_arms24.py -- deterministically rebuild arms24.json from the frozen panel archive.

The fixed 24-point orbit-response panel was frozen to results/orbit_panel.{npz,json}
BEFORE the real-kernel runs, and that archive ships in this package as
artifacts/results/orbit_panel.npz (coefficients) and .../orbit_panel.json (metadata).
This script joins the two into one human-readable file, arms24.json, so that reading the
panel needs neither numpy nor an archive reader.

The shipped NPZ contains only little-endian float64 C-order arrays of shape (7,4), so this
script reads that deliberately narrow NPY subset with the standard library.  This makes a
byte-for-byte rebuild check part of the dependency-free verification path.

Usage
    python3 build_arms24.py          write arms24.json next to this file
    python3 build_arms24.py --check  rebuild in memory and byte-compare the shipped file
"""
import argparse
import ast
import collections
import json
import os
import struct
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "..", "artifacts")
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


def load_npy_f8(zf, member):
    """Read the exact NPY subset used by orbit_panel.npz: C-order ``<f8`` 2-D arrays."""
    raw = zf.read(member)
    if not raw.startswith(b"\x93NUMPY") or len(raw) < 10:
        raise ValueError(f"{member}: not an NPY file")
    major, minor = raw[6], raw[7]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        header_at = 10
    elif major in (2, 3):
        if len(raw) < 12:
            raise ValueError(f"{member}: truncated NPY v{major}.{minor} header")
        header_len = struct.unpack("<I", raw[8:12])[0]
        header_at = 12
    else:
        raise ValueError(f"{member}: unsupported NPY version {major}.{minor}")
    header_end = header_at + header_len
    if header_end > len(raw):
        raise ValueError(f"{member}: truncated NPY header")
    header = ast.literal_eval(raw[header_at:header_end].decode("latin1"))
    if header.get("descr") != "<f8" or header.get("fortran_order") is not False:
        raise ValueError(f"{member}: expected little-endian float64 C order, got {header!r}")
    shape = header.get("shape")
    if shape != (7, 4):
        raise ValueError(f"{member}: expected shape (7, 4), got {shape!r}")
    count = shape[0] * shape[1]
    payload = raw[header_end:]
    if len(payload) != count * 8:
        raise ValueError(f"{member}: expected {count * 8} data bytes, got {len(payload)}")
    flat = struct.unpack("<" + "d" * count, payload)
    return [list(flat[i * shape[1]:(i + 1) * shape[1]]) for i in range(shape[0])]


def build_payload():
    """Return the human-readable panel assembled entirely in memory."""
    archive = os.path.join(ARTIFACTS, "results", "orbit_panel.npz")
    with zipfile.ZipFile(archive) as npz, \
            open(os.path.join(ARTIFACTS, "results", "orbit_panel.json")) as f:
        meta = json.load(f)
        points = meta.get("points")
        if not isinstance(points, list):
            raise ValueError("orbit_panel.json field 'points' is not a list")
        names = [point.get("name") for point in points if isinstance(point, dict)]
        if len(points) != 24 or meta.get("n_points") != 24:
            raise ValueError(
                f"the frozen panel must contain/declare 24 points; "
                f"found {len(points)} / {meta.get('n_points')!r}")
        if len(names) != len(set(names)):
            raise ValueError("orbit_panel.json contains duplicate or malformed arm names")
        if set(names) != EXPECTED_ARM_NAMES:
            raise ValueError(
                f"arm-name inventory differs from the frozen panel: "
                f"missing={sorted(EXPECTED_ARM_NAMES - set(names))}, "
                f"extra={sorted(set(names) - EXPECTED_ARM_NAMES)}")
        group_counts = collections.Counter(point.get("group") for point in points)
        if dict(group_counts) != EXPECTED_GROUP_COUNTS:
            raise ValueError(
                f"group counts {dict(group_counts)!r} != frozen {EXPECTED_GROUP_COUNTS!r}")
        expected_members = {
            f"{name}|{factor}.npy" for name in EXPECTED_ARM_NAMES for factor in ("U", "V", "W")
        }
        archive_member_list = npz.namelist()
        archive_members = set(archive_member_list)
        if len(archive_member_list) != len(archive_members):
            duplicates = sorted({
                name for name in archive_members if archive_member_list.count(name) > 1
            })
            raise ValueError(
                f"NPZ archive contains duplicate member names: {duplicates}")
        if len(archive_member_list) != len(expected_members):
            raise ValueError(
                f"NPZ archive contains {len(archive_member_list)} members; "
                f"expected exactly {len(expected_members)}")
        if archive_members != expected_members:
            raise ValueError(
                f"NPZ member inventory differs from the frozen panel: "
                f"missing={sorted(expected_members - archive_members)}, "
                f"extra={sorted(archive_members - expected_members)}")

        arms = []
        for point in points:
            name = point["name"]
            arms.append({
                "name": name,
                "group": point["group"],
                "note": point["note"],
                "U": load_npy_f8(npz, f"{name}|U.npy"),
                "V": load_npy_f8(npz, f"{name}|V.npy"),
                "W": load_npy_f8(npz, f"{name}|W.npy"),
                "phi": point["phi"],
                "gamma": point["gamma"],
                "coeff_max": point["coeff_max"],
            })

    return {
        "description": "The fixed 24-point orbit-response panel of Section 4.2, frozen "
                       "before the real-kernel runs.  Every arm is an exact rank-7 "
                       "<2,2,2> decomposition in the canonical rebalanced gauge.  Rows of "
                       "U, V and W are indexed by the seven products; each row is a 2x2 "
                       "pattern flattened row-major.",
        "seed": meta["seed"],
        "certified_arm": meta["cert"],
        "phi_min": meta["phi_min"],
        "perturbation_phi_target": meta["phi_target"],
        "n_arms": meta["n_points"],
        "groups": {
            "named": "the certified Phi-optimum, the other DPS form, the two small "
                     "rational forms, classic Strassen and Winograd",
            "perturb": "twelve signed perturbations of the certified arm: three factors "
                       "crossed with diagonal or shear directions and two signs",
            "disagree": "three Gamma-tie / Phi-split pairs, the rival-hypothesis test",
        },
        "arms": arms,
    }


def encoded(payload):
    """The canonical on-disk representation used by both write and check modes."""
    return (json.dumps(payload, indent=1) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and byte-compare arms24.json")
    args = ap.parse_args()

    try:
        out = build_payload()
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f"FAIL  could not rebuild the frozen panel: {exc}", file=sys.stderr)
        return 1
    expected = encoded(out)
    dest = os.path.join(HERE, "arms24.json")
    if args.check:
        try:
            with open(dest, "rb") as f:
                shipped = f.read()
        except OSError as exc:
            print(f"FAIL  cannot read shipped arms24.json: {exc}", file=sys.stderr)
            return 1
        if shipped != expected:
            print("FAIL  shipped arms24.json does not byte-match the frozen NPZ/JSON rebuild",
                  file=sys.stderr)
            print(f"  rebuilt bytes: {len(expected)}; shipped bytes: {len(shipped)}",
                  file=sys.stderr)
            return 1
        print(f"byte-for-byte match: {dest} -- {len(out['arms'])} arms, {len(expected)} bytes")
    else:
        with open(dest, "wb") as f:
            f.write(expected)
        print(f"wrote {dest} -- {len(out['arms'])} arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
