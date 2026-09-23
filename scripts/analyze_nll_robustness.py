#!/usr/bin/env python3
"""Dependence-aware robustness analysis of the excess-NLL comparison.

The script reports paired bootstrap intervals, moving-block intervals, exact
Wilcoxon signed-rank tests, and non-overlapping block summaries. Block lengths
are fixed at 2 and 4.
"""
import csv, math, json, os, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CSVS = {
    "Qwen2.5-72B": os.path.join(ROOT, "data", "nll", "qwen25_72b_per_chunk.csv"),
    "Llama-3.3-70B": os.path.join(ROOT, "data", "nll", "llama3_70b_per_chunk.csv"),
}
ACC, STRAS, N = "acc_fp8 all", "stras_fp8 all", 32
B = 20000
# Reference point estimates used as an input-integrity check.
REFERENCE_SUMMARY = {"Qwen2.5-72B": (55.0, 30), "Llama-3.3-70B": (10.2, 22)}


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["arm"], {})[int(r["chunk_id"])] = float(r["ppl"])
    return rows


def excess(rows, arm):
    return [math.log(rows[arm][c]) - math.log(rows["clean"][c]) for c in range(N)]


def reduction_pct(acc, stras):
    """Percent reduction computed as a ratio of mean excess NLL values."""
    return 100.0 * (1.0 - (sum(acc) / len(acc)) / (sum(stras) / len(stras)))


def wilcoxon_signed_rank(diffs):
    """Exact two-sided Wilcoxon signed-rank via full DP over the null distribution."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                                     # average ranks within ties
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    # exact null: every sign assignment equally likely (integer ranks only)
    if all(float(r).is_integer() for r in ranks):
        counts = {0: 1}
        for r in (int(r) for r in ranks):
            nxt = {}
            for s, c in counts.items():
                nxt[s] = nxt.get(s, 0) + c
                nxt[s + r] = nxt.get(s + r, 0) + c
            counts = nxt
        total = 2 ** n
        tot = sum(ranks)
        lo, hi = min(w_plus, tot - w_plus), max(w_plus, tot - w_plus)
        p = sum(c for s, c in counts.items() if s <= lo or s >= hi) / total
        return w_plus, min(1.0, p)
    return w_plus, float("nan")


def sign_test_p(k, n):
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(xs, q):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))]


def moving_block_ci(diffs, L, rng, stat=lambda d: sum(d) / len(d)):
    """Moving-block bootstrap: resample contiguous blocks, preserving local serial structure."""
    nb = math.ceil(N / L)
    starts = list(range(N - L + 1))
    out = []
    for _ in range(B):
        s = []
        for _ in range(nb):
            b = rng.choice(starts)
            s.extend(diffs[b:b + L])
        out.append(stat(s[:N]))
    return pct(out, 0.025), pct(out, 0.975)


def main():
    report = {}
    for model, path in CSVS.items():
        rows = load(path)
        acc, stras = excess(rows, ACC), excess(rows, STRAS)
        diffs = [a - s for a, s in zip(acc, stras)]          # negative = Phi-optimal better
        wins = sum(1 for d in diffs if d < 0)
        red = reduction_pct(acc, stras)

        g_red, g_wins = REFERENCE_SUMMARY[model]
        assert abs(red - g_red) < 0.05, (
            f"{model}: reduction {red:.2f} != reference {g_red}"
        )
        assert wins == g_wins, f"{model}: wins {wins} != reference {g_wins}"

        rng = random.Random(20260729)
        # (1) iid paired bootstrap — resample chunks, recompute BOTH statistics
        b_abs, b_rel = [], []
        for _ in range(B):
            idx = [rng.randrange(N) for _ in range(N)]
            a2, s2 = [acc[i] for i in idx], [stras[i] for i in idx]
            b_abs.append(sum(d for d in (x - y for x, y in zip(a2, s2))) / N)
            b_rel.append(reduction_pct(a2, s2))

        # (2) moving-block bootstrap on the PRIMARY absolute paired difference
        mb = {L: moving_block_ci(diffs, L, random.Random(20260729 + L)) for L in (2, 4)}

        # (3) nonoverlapping block aggregation -> sign test on block means
        blocks = {}
        for L in (2, 4):
            bm = [sum(diffs[i:i + L]) / L for i in range(0, N, L)]
            bw = sum(1 for d in bm if d < 0)
            blocks[L] = {"n_blocks": len(bm), "blocks_favoring_phi_opt": bw,
                         "sign_test_p": sign_test_p(len(bm) - bw, len(bm)),
                         "mean_block_diff_nats": sum(bm) / len(bm)}

        w, wp = wilcoxon_signed_rank(diffs)
        report[model] = {
            "summary": {"excess_nll_reduction_pct": round(red, 2),
                        "paired_wins": f"{wins}/{N}"},
            "primary_absolute_diff_nats": {
                "mean": sum(diffs) / N,
                "iid_bootstrap_ci95": [pct(b_abs, .025), pct(b_abs, .975)],
                "moving_block_ci95": {f"L={L}": list(v) for L, v in mb.items()},
            },
            "secondary_relative_reduction_pct": {
                "point": red, "iid_bootstrap_ci95": [pct(b_rel, .025), pct(b_rel, .975)]},
            "sign_test_p_chunkwise": sign_test_p(N - wins, N),
            "wilcoxon_exact": {"W_plus": w, "p_two_sided": wp},
            "nonoverlapping_block_aggregation": blocks,
        }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "nll_robustness.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
