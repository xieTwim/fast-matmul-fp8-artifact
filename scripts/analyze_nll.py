"""Recompute the paired excess-NLL statistics from the released per-chunk CSVs.

The calculation uses only the Python standard library.  Per chunk ``c`` and
arm ``a``, excess NLL is ``log(ppl[a,c]) - log(ppl[clean,c])``.
"""

import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSVS = {
    "qwen72": os.path.join(ROOT, "data", "nll", "qwen25_72b_per_chunk.csv"),
    "llama70": os.path.join(ROOT, "data", "nll", "llama3_70b_per_chunk.csv"),
}
# Recorded KL reductions used as an input-integrity check.
KL_REFERENCE = {"qwen72": -47.58, "llama70": -7.99}  # percent, ±0.1pp
ARMS = ["cubic_fp8 all", "acc_fp8 all", "stras_fp8 all", "wino_fp8 all"]
N = 32


def sign_test_p(k_less, n):
    """Exact two-sided binomial sign test P(X <= min(k, n-k) or X >= max(...)), p=0.5."""
    k = min(k_less, n - k_less)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_t(diffs):
    n = len(diffs)
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return m / se if se > 0 else float("inf"), se


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["arm"], {})[int(r["chunk_id"])] = (float(r["ppl"]), float(r["kl"]))
    return rows


def main():
    out = {}
    for model, path in CSVS.items():
        rows = load(path)
        assert all(len(rows[a]) == N for a in ARMS + ["clean"]), f"{model}: missing chunks"
        clean_nll = [math.log(rows["clean"][c][0]) for c in range(N)]

        # Check that the released table reproduces the recorded KL reduction.
        kl_acc = sum(rows["acc_fp8 all"][c][1] for c in range(N)) / N
        kl_str = sum(rows["stras_fp8 all"][c][1] for c in range(N)) / N
        kl_red = 100.0 * (kl_acc / kl_str - 1.0)
        assert abs(kl_red - KL_REFERENCE[model]) < 0.1, (
            f"{model}: KL reduction {kl_red:.2f} vs reference {KL_REFERENCE[model]}"
        )

        arms = {}
        for a in ARMS:
            ex = [math.log(rows[a][c][0]) - clean_nll[c] for c in range(N)]
            arms[a] = {
                "mean_excess_nll": sum(ex) / N,
                "geo_ppl_ratio": math.exp(sum(ex) / N),
                "excess": ex,
            }

        acc, stras = arms["acc_fp8 all"]["excess"], arms["stras_fp8 all"]["excess"]
        diffs = [a - s for a, s in zip(acc, stras)]
        wins = sum(1 for d in diffs if d < 0)  # acc lower excess NLL = win
        t, se = paired_t(diffs)
        red = 1.0 - (sum(acc) / N) / (sum(stras) / N)
        out[model] = {
            "kl_reduction_pct": kl_red,
            "arms": {a: {k: v for k, v in arms[a].items() if k != "excess"} for a in ARMS},
            "acc_vs_stras": {
                "mean_excess_nll_acc": sum(acc) / N,
                "mean_excess_nll_stras": sum(stras) / N,
                "excess_nll_reduction_pct": 100.0 * red,
                "acc_wins_chunks": wins,
                "n": N,
                "sign_test_p": sign_test_p(N - wins, N),
                "paired_t": t,
                "mean_diff": sum(diffs) / N,
                "se_diff": se,
            },
            "ordering_excess_nll": sorted(ARMS, key=lambda a: arms[a]["mean_excess_nll"]),
        }

        print(f"== {model} ==  (KL reduction: {kl_red:+.2f}%)")
        for a in ARMS:
            print(f"  {a:16s} mean excess NLL {arms[a]['mean_excess_nll']:.6f}  "
                  f"geo ppl ratio {arms[a]['geo_ppl_ratio']:.4f}")
        h = out[model]["acc_vs_stras"]
        print(f"  acc vs stras: excess-NLL reduction {h['excess_nll_reduction_pct']:+.2f}%  "
              f"wins {h['acc_wins_chunks']}/{N}  sign-p {h['sign_test_p']:.2e}  t {h['paired_t']:.2f}")

    output_dir = os.path.join(ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, "nll_summary.json")
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
