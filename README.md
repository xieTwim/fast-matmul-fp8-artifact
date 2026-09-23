<h1 align="center">Fast Matrix Multiplication in fp8</h1>

<p align="center">
  <b>Certified Coefficient Optimization and Measured Error</b>
</p>

<p align="center">
  <i>Same product, same multiplication count, different fp8 error.</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.26077"><img src="https://img.shields.io/badge/arXiv-2609.26077-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f.svg" alt="MIT license"></a>
  <a href="https://github.com/xieTwim/fast-matmul-fp8-artifact/actions/workflows/verify.yml"><img src="https://github.com/xieTwim/fast-matmul-fp8-artifact/actions/workflows/verify.yml/badge.svg" alt="Verify artifact"></a>
</p>

<p align="center">
  <a href="#news">News</a> &bull;
  <a href="#overview">Overview</a> &bull;
  <a href="#artifact-contents">Artifact</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#reproducibility-scope">Scope</a> &bull;
  <a href="#citation">Citation</a>
</p>

---

This repository is the public research artifact for
[*Fast Matrix Multiplication in fp8: Certified Coefficient Optimization and Measured Error*](https://arxiv.org/abs/2609.26077).
It contains the exact certificate checkers and the certified coordinates, the
fixed 24-arm orbit-response panel, the real-tile measurements, and the
per-chunk inputs and analysis scripts for the model-level experiments.

## News

- **[2026/09]** Research artifact released with the certificate checkers and the real-kernel measurements.
- **[2026/09]** Paper released on [arXiv](https://arxiv.org/abs/2609.26077).

## Overview

A Strassen-type algorithm can be realized in many ways that compute the same
exact product with the same number of multiplications. Their fp8 error still
differs, because a change of basis reshapes the coefficients. The paper asks
which realization to run. It assigns each realization a coefficient functional
Φ = Σ_r ‖u_r‖² ‖v_r‖² ‖w_r‖², where u_r, v_r and w_r are the coefficient vectors
of the r-th product. It then minimizes Φ over the change-of-basis orbit and
certifies the global minimum rather than searching for it.

The central results are:

1. **Class-wide certificate.** Minimizing Φ over the orbit is a Kempf–Ness
   problem on a Hadamard manifold. An exact moment-map zero fixes
   Φ_min = 200/9, and de Groote's classification extends this optimum to every
   exact real rank-7 2×2 decomposition. Every such realization therefore has a
   Φ-predicted RMS constant at least 5/3 times that of the cubic algorithm, at
   fixed noise coefficient.
2. **An explicit fp8 model.** In a block-scaled e4m3 second-moment model, Φ is
   the leading-order coefficient of relative expected mean-squared error. The
   paper tests the Φ-predicted ordering against realized fp8 error instead of
   assuming it.
3. **Real matmul tiles.** On 2,352 activation-by-weight matmuls from five models
   in four architecture families, run on real `deep_gemm` kernels, the
   Φ-optimal realization falls in the fp8 low-error region of a fixed 24-arm
   panel of equivalent realizations.
4. **Model-level effect.** On Qwen2.5-72B-Instruct and Llama-3.3-70B-Instruct,
   also on real `deep_gemm` kernels, the Φ-optimal realization removes 55%
   (Qwen) and 10% (Llama) of classic Strassen's excess NLL over the clean model.
   The Qwen result is strongly significant; the Llama result is marginal
   (sign test p = 0.0501).

The certificate applies only to Φ-optimality. It neither inherits the fp8
model's physical uncertainty nor gives a lower bound on realized fp8 error.
The paper also reports the limits of that prediction: on ±1 Rademacher
operands classic Strassen wins all six designed cells, and the fine ordering
among nearby realizations is not resolved.

## Artifact contents

| Path | Contents |
|---|---|
| [`verify.py`](verify.py) | One command that runs the certificate, panel, and data-integrity checks; `--control` runs the negative control |
| [`certificate/`](certificate/) | Exact arithmetic over Q(ρ), ρ = (4/3)^(1/4): the Φ_min = 200/9 certificate, the composite values, and the coordinate export |
| [`coordinates.json`](coordinates.json) | The balanced transform T* and the certified realization D* = T*·Strassen, exactly and as 34-digit decimals |
| [`panel/`](panel/) | The fixed 24-arm panel (`arms24.json`), its byte-for-byte rebuild, and checks of its design |
| [`artifacts/results/`](artifacts/results/) | The frozen panel archive from which `arms24.json` is rebuilt |
| [`semantics/gauge.py`](semantics/gauge.py) | Executable definitions of re-basing and of the canonical rebalanced gauge |
| [`data/realtile/`](data/realtile/) | 2,352 real-tile error measurements from five models |
| [`data/nll/`](data/nll/) | Per-chunk perplexity and KL for the two model-level experiments |
| [`data/reference/`](data/reference/) | Frozen outputs of the analysis scripts |
| [`scripts/`](scripts/) | Analyses of the real-tile and model-level data |
| [`experiments/fp8_emulator.py`](experiments/fp8_emulator.py) | Optional PyTorch emulator of the block-scaled e4m3 path |

## Quick start

Clone the repository and run the checks:

```bash
git clone https://github.com/xieTwim/fast-matmul-fp8-artifact.git
cd fast-matmul-fp8-artifact

python3 verify.py            # certificate, panel, and data-integrity checks
python3 verify.py --control  # negative control on a perturbed algebraic point
```

Both commands need only Python 3.8 or newer, with no GPU, network access, or
third-party package. Each finishes in about a second and ends with
`All checks passed.` The first also checks every file under `data/` and
`artifacts/results/` against the SHA-256 checksums in `data/checksums.json`.
The second replaces ρ⁴ = 4/3 by (4/3)(1 + 1/1000) and passes only if the
certificate checks fail at that point while the exactness checks still hold.

To recompute the paper's statistics from the released measurements:

```bash
python3 scripts/analyze_realtile.py        # real-tile panel (Section 4.2)
python3 scripts/analyze_nll.py             # excess-NLL comparison (Section 4.3)
python3 scripts/analyze_nll_robustness.py  # dependence-aware re-analysis (Appendix E)
```

These also use only the standard library. Each writes a JSON file under
`results/` whose values agree with its frozen copy in `data/reference/` up to
floating-point rounding.

The layer-clustered bootstrap needs NumPy, and the pinned version requires
Python 3.12 or newer. With 4,000 replicates it takes about five minutes on a
laptop:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/bootstrap_realtile.py 4000
```

`experiments/fp8_emulator.py` compares the Φ-optimal realization with classic
Strassen on seeded synthetic operands, one recursion level deep, under the
paper's quantization rule: 1×128 groups, an absolute-maximum scale, e4m3
round-to-nearest-even conversion, and fp32 accumulation. It needs a PyTorch
build with `torch.float8_e4m3fn` and runs on CPU or CUDA:

```bash
python3 experiments/fp8_emulator.py --size 512 --seeds 4
```

## Reproducibility scope

This release supports:

- exact, dependency-free verification of the Φ certificate (Theorem 1 and
  Table 1 of the paper) and of its negative control;
- inspection of the certified coordinates and of the fixed 24-arm panel;
- independent re-analysis of the real-tile panel and of the model-level
  excess-NLL comparison, including the bootstrap and robustness checks.

The scripts check the value Φ = 200/9, exactness, the moment-map zero, and the
positive-definite Hessian at the certified point. The two remaining steps of
the proof, global optimality on the orbit by geodesic convexity and the
extension to the whole rank-7 class by de Groote's classification, are
analytic arguments given in Appendices A and B of the paper; they are not
re-run here.

This release does **not** redistribute model checkpoints, WikiText-2 text,
activations, prompts, token IDs, or DeepGEMM and CUTLASS source. Repeating the
GPU measurements therefore requires the named checkpoints, WikiText-2, and a
compatible DeepGEMM installation. The original model and tokenizer revisions
and the upstream revision of the H20 DeepGEMM build were not recorded, so
bit-for-bit regeneration of those measurements is not claimed. The paper's
other experiments, such as the twenty-shape emulator study and the
cross-hardware throughput comparison, are reported in full in its Appendix E;
their data are not part of this release.

## Data layout

- `data/realtile/realtile_panel_<model>.json` holds one model's measurements
  (Qwen2.5-7B, Qwen2.5-32B, Qwen3-8B, Yi-6B, and Llama-3.3-70B-Instruct). Each
  entry of `tiles` is one activation-by-weight matmul for one linear layer and
  one WikiText-2 context. Its `rows` give, for each of the 24 panel arms and
  for cubic fp8, the relative error of the real `deep_gemm` path (`err_real`)
  and of the fp32 emulator (`err_emu`) against an fp64 reference; `ref_norm`
  is the Frobenius norm of that reference.
- `data/nll/<model>_per_chunk.csv` holds, for 32 consecutive WikiText-2 chunks,
  the perplexity (`ppl`) of each arm and its KL divergence to the clean model
  (`kl`).
- `panel/arms24.json` lists each arm's coefficient triple (U, V, W) with its Φ
  and Γ values.
- `coordinates.json` gives each coefficient exactly, as rational coefficients
  of 1, ρ, ρ², ρ³, alongside its decimal value.

No model weights, activations, prompts, token IDs, or dataset text are included.

## Citation

If you use this artifact, please cite the accompanying paper:

```bibtex
@misc{xie2026fastmatmulfp8,
  title         = {Fast Matrix Multiplication in fp8: Certified Coefficient
                   Optimization and Measured Error},
  author        = {Xie, Shuxiao and Xie, Shuyang and Cao, Yuan and
                   Ran, Dezhi and Yang, Wei and Xie, Tao},
  year          = {2026},
  eprint        = {2609.26077},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2609.26077},
  url           = {https://arxiv.org/abs/2609.26077}
}
```

The same metadata is available in [`CITATION.cff`](CITATION.cff).

## License

Author-created code, released measurements, and documentation are available
under the [MIT License](LICENSE). No third-party software, model checkpoints, or
datasets are redistributed; see [`NOTICE`](NOTICE).
