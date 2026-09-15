# Architecture

How the Metal backend is built and what it is checked against. Results live
in [BENCHMARKS.md](BENCHMARKS.md); rejected approaches, with numbers, live in
[NEGATIVE_RESULTS.md](../bench/results/NEGATIVE_RESULTS.md).

## 🔬 Correctness

| check | result |
|---|---|
| forward vs float64-gradcheck oracle | max abs ≤ 2×10⁻³ |
| gradients vs torch autograd of the oracle | rel ≤ 3.5×10⁻⁶, cosine 1.000000 (all five tensors) |
| Metal binning vs torch binning | **bit-identical** image and alpha |
| fused SSIM vs the `F.conv2d` expression | value bit-identical, gradient cosine 1.00000012 |
| fused Adam vs `torch.optim.Adam` | rel 4.7×10⁻⁷ after 30 steps |

Run the suite with `pytest tests`. `tests/test_runner.py` guards the
benchmark harness rather than the kernels — every wrong number this project has published came
from a harness, and the two that did the most damage are now regression tests.

## ⚙️ How it works

- **Projection + SH + activations** fused into one kernel, thread-per-gaussian, analytic backward.
- **Tile binning in Metal**, including an exact ellipse-vs-tile test that drops 38.7 % of
  tile-gaussian pairs while leaving the image bit-identical (gsplat's AccuTile, in-kernel).
- **Rasterization** with threadgroup-cooperative attribute staging and simdgroup-aggregated
  gradient atomics (32 lanes → 1 atomic).
- **SSIM entirely in Metal** — stack construction, both separable blur passes and the tail. The
  symmetric gaussian is self-adjoint, so one kernel serves the forward and both adjoint passes.
- **Adam in one pass** instead of torch's five, at the measured memory-bandwidth floor.
- **MCMC densification** with gradient-targeted relocation, exact multiplicity correction and
  Adam-moment resets.
- Metal compiles at **runtime** via `newLibraryWithSource` — hence no Xcode.
