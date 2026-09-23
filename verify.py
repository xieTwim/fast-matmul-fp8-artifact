#!/usr/bin/env python3
"""Run the dependency-free certificate, panel, and released-data checks."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKS = [
    ("certificate/check_floor.py", [], r"^\s*0 failure\(s\)\s*$"),
    ("certificate/check_composites.py", [], r"^\s*0 failure\(s\)\s*$"),
    ("certificate/export_coordinates.py", ["--check"], r"^\s*byte-for-byte match:"),
    ("panel/build_arms24.py", ["--check"], r"^\s*byte-for-byte match:"),
    ("panel/check_arms.py", [], r"^\s*0 failure\(s\)\s*$"),
    ("semantics/gauge.py", [], r"^\s*0 failure\(s\)\s*$"),
]
CONTROL_CHECKS = [
    ("certificate/check_floor.py",
     {"a": "FAIL", "b": "PASS", "c": "FAIL", "d": "FAIL"}),
    ("certificate/check_composites.py",
     {"0": "PASS", "G": "FAIL", "K": "FAIL", "D0": "PASS", "D": "FAIL"}),
]
MARKER = re.compile(r"^\s+\[([^\]]+)\]\s+(PASS|FAIL)\s", re.MULTILINE)
PRIVATE_PATH = re.compile(r"(?:^|[\s'\"])(?:/Users/|/home/|/root/|/mnt/|~/)")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(rel, args, env=None):
    path = os.path.join(ROOT, rel)
    child_env = dict(env or os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, os.path.basename(path)] + args,
        cwd=os.path.dirname(path),
        env=child_env,
        capture_output=True,
        text=True,
    )


def result_vector(text):
    result = {}
    duplicate = False
    for tag, value in MARKER.findall(text):
        duplicate |= tag in result
        result[tag] = value
    return result, duplicate


def walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)


def check_data():
    errors = []
    manifest_path = os.path.join(ROOT, "data", "checksums.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = manifest.get("files", {})
    if not isinstance(records, dict) or not records:
        errors.append("data/checksums.json has no file records")
    actual = set()
    for base in (os.path.join(ROOT, "data"), os.path.join(ROOT, "artifacts", "results")):
        for directory, _, names in os.walk(base):
            for name in names:
                path = os.path.join(directory, name)
                rel = os.path.relpath(path, ROOT)
                if rel != "data/checksums.json":
                    actual.add(rel)
    if actual != set(records):
        for rel in sorted(set(records) - actual):
            errors.append(f"manifested data file is missing: {rel}")
        for rel in sorted(actual - set(records)):
            errors.append(f"unmanifested data file is present: {rel}")
    for rel, expected in sorted(records.items()):
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            errors.append(f"missing data file: {rel}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {rel}")

    realtile_dir = os.path.join(ROOT, "data", "realtile")
    realtile_files = sorted(
        name for name in os.listdir(realtile_dir)
        if name.startswith("realtile_panel_") and name.endswith(".json")
    )
    if len(realtile_files) != 5:
        errors.append(f"expected five real-tile JSON files, found {len(realtile_files)}")
    tile_count = 0
    for name in realtile_files:
        path = os.path.join(realtile_dir, name)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if "model_path" in payload or "panel" in payload:
            errors.append(f"private source-location field remains in {name}")
        tile_count += len(payload.get("tiles", []))
        for value in walk_strings(payload):
            if PRIVATE_PATH.search(value):
                errors.append(f"private absolute path remains in {name}")
                break
    if tile_count != 2352:
        errors.append(f"expected 2352 real tiles, found {tile_count}")

    required_arms = {
        "clean", "cubic_fp8 all", "acc_fp8 all", "stras_fp8 all", "wino_fp8 all"
    }
    for name in sorted(os.listdir(os.path.join(ROOT, "data", "nll"))):
        if not name.endswith(".csv"):
            continue
        counts = {}
        with open(os.path.join(ROOT, "data", "nll", name), newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        if len(rows) != 256:
            errors.append(f"expected 256 NLL rows in {name}, found {len(rows)}")
        for arm in required_arms:
            if counts.get(arm) != 32:
                errors.append(f"expected 32 rows for {arm!r} in {name}")
    return errors


def positive_run():
    failures = 0
    data_errors = check_data()
    print("data integrity")
    if data_errors:
        failures += 1
        for error in data_errors:
            print(f"  FAIL  {error}")
    else:
        print("  PASS  checksums, five model panels, 2352 tiles, and two NLL tables")

    for rel, args, success_pattern in CHECKS:
        proc = run(rel, args)
        combined = "\n".join(part for part in (proc.stdout.rstrip(), proc.stderr.rstrip()) if part)
        print(f"\n--- {rel}")
        print(combined)
        if proc.returncode != 0 or re.search(success_pattern, combined, re.MULTILINE) is None:
            failures += 1
            print(f"  VERIFY FAIL: exit={proc.returncode}, required marker missing")
    return failures


def control_run():
    failures = 0
    env = dict(os.environ, NEGATIVE_CONTROL="1")
    for rel, expected in CONTROL_CHECKS:
        proc = run(rel, [], env=env)
        combined = "\n".join(part for part in (proc.stdout.rstrip(), proc.stderr.rstrip()) if part)
        got, duplicate = result_vector(combined)
        print(f"\n--- negative control: {rel}")
        print(combined)
        if proc.returncode != 1 or duplicate or got != expected:
            failures += 1
            print(f"  CONTROL FAIL: exit={proc.returncode}, vector={got}, expected={expected}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 8):
        print("Python 3.8 or newer is required", file=sys.stderr)
        return 2

    failures = control_run() if args.control else positive_run()
    if failures:
        print(f"\n{failures} verification group(s) failed")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
