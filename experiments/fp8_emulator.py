#!/usr/bin/env python3
"""FP8 emulator for one level of a rank-seven 2x2 block algorithm.

The quantizer uses 1x128 groups, an absolute-maximum scale, e4m3
round-to-nearest-even conversion, and fp32 accumulation.  Running this file
compares the certified realization and classic Strassen on seeded operands.
"""
import argparse
import json
import os

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F8 = torch.float8_e4m3fn; QG = 128


def q8g(x, rowwise):
    dim = 1 if rowwise else 0; xt = x if dim == 1 else x.t(); R, C = xt.shape; pad = (-C) % QG
    if pad: xt = torch.nn.functional.pad(xt, (0, pad))
    xg = xt.reshape(R, -1, QG); s = xg.abs().amax(2, keepdim=True).clamp(min=1e-12) / 448.0
    out = ((xg / s).to(F8).to(torch.float32) * s).reshape(R, -1)[:, :C]
    return out if dim == 1 else out.t()


def fast_mm(A, B, U, V, W, fused):
    """One level of a <2,2,2;7> bilinear algorithm in fp8 blk128 (fp32-emulated). B is k-major (K,N).
    fused=True quantizes only each encoded combination.  fused=False also
    quantizes the input quadrants before encoding."""
    M, K = A.shape; N = B.shape[1]
    Mp, Kp, Np = M + (M & 1), K + (K & 1), N + (N & 1)
    Af = torch.zeros(Mp, Kp, device=A.device, dtype=torch.float32); Af[:M, :K] = A.float()
    Bf = torch.zeros(Kp, Np, device=B.device, dtype=torch.float32); Bf[:K, :N] = B.float()
    hm, hk, hn = Mp // 2, Kp // 2, Np // 2
    aq = [Af[:hm, :hk], Af[:hm, hk:], Af[hm:, :hk], Af[hm:, hk:]]
    bq = [Bf[:hk, :hn], Bf[:hk, hn:], Bf[hk:, :hn], Bf[hk:, hn:]]
    if not fused:
        aq = [q8g(t, True) for t in aq]; bq = [q8g(t, False) for t in bq]
    C = [torch.zeros(hm, hn, device=A.device, dtype=torch.float32) for _ in range(4)]
    for i in range(7):
        ea = eb = None
        for j in range(4):
            if U[i][j] != 0: ea = aq[j] * float(U[i][j]) if ea is None else ea + aq[j] * float(U[i][j])
            if V[i][j] != 0: eb = bq[j] * float(V[i][j]) if eb is None else eb + bq[j] * float(V[i][j])
        m = q8g(ea, True) @ q8g(eb, False)
        for p in range(4):
            if W[i][p] != 0: C[p] += float(W[i][p]) * m
    return torch.cat([torch.cat([C[0], C[1]], 1), torch.cat([C[2], C[3]], 1)], 0)[:M, :N]


def cubic_fp8(A, B):
    return q8g(A.float(), True) @ q8g(B.float(), False)


# ---- matrix corpus --------------------------------------------------------------------------------
# ballard_d2 makes the second
#   K-half of A and the first K-half of B tiny, stressing the REDUCTION dimension (which is what makes
#   it adversarial for the input-sum encodings).
def gen(kind, m, k, n, g):
    if kind == "uniform_pm1":
        return torch.rand(m, k, generator=g, device=DEV) * 2 - 1, torch.rand(k, n, generator=g, device=DEV) * 2 - 1
    if kind == "uniform_01":
        return torch.rand(m, k, generator=g, device=DEV), torch.rand(k, n, generator=g, device=DEV)
    if kind == "ballard_d2":
        s = 1.0 / (max(m, n) ** 2)
        A = torch.rand(m, k, generator=g, device=DEV); B = torch.rand(k, n, generator=g, device=DEV)
        A[:, k // 2:] *= s; B[:k // 2, :] *= s
        return A, B
    if kind == "logunif_rowcol":
        A = torch.rand(m, k, generator=g, device=DEV) * 2 - 1
        B = torch.rand(k, n, generator=g, device=DEV) * 2 - 1
        ra = torch.pow(2.0, torch.randint(-12, 13, (m, 1), generator=g, device=DEV).float())
        cb = torch.pow(2.0, torch.randint(-12, 13, (1, n), generator=g, device=DEV).float())
        return A * ra, B * cb
    raise ValueError(kind)


def load_arm(name):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "panel", "arms24.json"), encoding="utf-8") as handle:
        payload = json.load(handle)
    for arm in payload["arms"]:
        if arm["name"] == name:
            return arm
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--distribution", default="uniform_pm1",
                        choices=["uniform_pm1", "uniform_01", "ballard_d2", "logunif_rowcol"])
    parser.add_argument("--double-quant", action="store_true")
    args = parser.parse_args()

    arms = [load_arm("dps_acc"), load_arm("strassen_classic")]
    values = {arm["name"]: [] for arm in arms}
    for seed in range(args.seeds):
        generator = torch.Generator(device=DEV).manual_seed(20260723 + seed)
        A, B = gen(args.distribution, args.size, args.size, args.size, generator)
        reference = A.double() @ B.double()
        for arm in arms:
            result = fast_mm(A, B, arm["U"], arm["V"], arm["W"], not args.double_quant).double()
            values[arm["name"]].append(float((result - reference).norm() / reference.norm()))

    print(f"device={DEV} size={args.size} seeds={args.seeds} distribution={args.distribution}")
    for name, errors in values.items():
        print(f"{name}: mean relative Frobenius error = {sum(errors) / len(errors):.8e}")


if __name__ == "__main__":
    main()
