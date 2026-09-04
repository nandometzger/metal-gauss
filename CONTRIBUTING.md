# Contributing

Issues and pull requests are welcome. Two things about this repo are unusual
enough to be worth knowing before you start.

## You need an Apple Silicon Mac

The kernels are Metal and compile at runtime. There is no CPU fallback and no
CUDA path, so the rasteriser tests cannot run anywhere else.

The provenance job is CPU-only: it runs the harness tests and checks that the
generated tables still match their JSONs. The separate `metal` job runs the
kernel suite on a hosted Apple Silicon runner, but remains opt-in while the
repository evaluates its cost and long-term stability. So **the default CI
job passing does not mean the kernels are fine** — run the full suite locally,
or opt in to the hosted Apple Silicon job.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[bench,train]" pytest lpips scikit-image imageio tqdm torchvision
.venv/bin/python -m pytest -q          # 188 tests, needs an Apple GPU
```

The `metal` job in `.github/workflows/checks.yml` is opt-in during its initial
evaluation. It uses the hosted `macos-15` arm64 runner. A repository variable
named `METAL_GAUSS_ENABLE_MPS_CI` set to `true` enables it for pull requests;
the `run_metal` workflow-dispatch input enables a one-off run. The job checks
that the runner is `arm64`, verifies that PyTorch can see MPS, installs the
complete test dependencies, and runs all tests. It contains no timing
benchmarks.

The Metal extension compiles its `.metal` sources at runtime. The runtime
compiler on the hosted runner does not accept the default language level for
the repository's floating-point atomics, so `rasterize.mm` explicitly
requests Metal Shading Language 3.0 before creating the library. This keeps
the existing native `atomic_float` gradient path and avoids a slower software
accumulator fallback.

The job was validated on September 4, 2026 using the GitHub-hosted
`macos-15-arm64` image, Python 3.12, and PyTorch 2.14.0: the runner reported
`machine=arm64`, `mps_built=True`, and `mps_available=True`, and the complete
suite passed (`188 passed in 92.92s`). The hosted run is recorded [in the
Actions log](https://github.com/Neilus03/metal-gauss/actions/runs/33887832871).

## Performance claims need a warm machine and an idle one

Apple Silicon runs short bursts at boost clock. The same kernel times ~12 ms or
~19 ms depending on how warm the laptop is, so the first measurement of anything
is optimistic. `bench/quick.py` ramps for 2 seconds before timing; if you write
a new benchmark, do the same.

Wall-clock numbers also assume the GPU is yours alone. `bench/runner.py` exposes
`require_gpu_exclusive()` — call it at the top of anything that reports timings.

Before quoting a difference, check it against the run-to-run spread. Ours is
0.19 dB worst case; a competitor's can be far wider. A difference smaller than
the spread is not a result. `bench/compare/variance.py` repeats a configuration
and reports mean, stdev and range.

## Tables and figures are generated

Every table in `README.md` and `docs/` is produced from a JSON in
`bench/results/`. Do not hand-edit them between the `<!-- BEGIN:x -->` markers.

```bash
.venv/bin/python bench/readme_tables.py          # regenerate
.venv/bin/python bench/readme_tables.py --check  # fails if a table drifted
```

The figures come from `bench/compare/plot_pareto_scenes.py` and
`plot_margin.py`. `docs/BENCHMARKS.md` lists every script and what it is for.

## Where things live

| | |
|---|---|
| `metal_gauss/` | the trainer and the Metal kernels (`csrc/*.metal`) |
| `bench/` | benchmarks, table generation, provenance checks |
| `bench/compare/` | competitor harnesses and the figures |
| `tests/` | correctness, mostly against a torch reference |
| `bench/results/NEGATIVE_RESULTS.md` | levers measured and rejected — read this before optimising |

## Scope

Competitor benchmarks run their shipped configuration, scored by one evaluator
on the official split, strictly sequentially. If you add an implementation,
follow that and say which configuration you used — a comparison is only worth
publishing if someone else can reproduce the protocol.
