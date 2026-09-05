# Apple-Silicon 3DGS implementations: what is actually comparable

Every Apple-native Gaussian-splatting project we could find, what it turned out to be, and
whether it can appear in a training benchmark. **A repo that does not build, or that turns out
not to train scenes, is a result — not an omission.**

Checked on macOS 26, Apple M5, 24 GB.

## Two different axes, often conflated

"Who are our competitors" has two answers, and mixing them produces bad comparisons:

| axis | who | belongs in |
|---|---|---|
| **Apple-native training** | Brush, msplat, splat-apple, OpenSplat, spirula-studio | the Pareto front |
| **Algorithms we borrow** | 3DGS-MCMC, Mip-Splatting, Taming 3DGS, Speedy-Splat, gsplat, LiteGS | the adoption ledger in [`NEGATIVE_RESULTS.md`](../results/NEGATIVE_RESULTS.md) |
| **Rasteriser libraries** | gsplat-mlx, gsplat-mps | a forward-only table, not training results |
| **Renderers** | MetalSplatter, mlx-splat, 3DGS.cpp | nothing; they cannot train |

Only the first group is a rival. The second is where "what are we missing" is actually answered,
and those are CUDA papers rather than Mac projects -- adopting one is engineering work, not a
benchmark. The distinction matters because a technique from row two can be adopted, while a
project in row one can only be beaten or lost to.

## Installing them

```bash
python bench/compare/setup_competitors.py --check   # what is present
python bench/compare/setup_competitors.py           # install what is not
```

Versions are pinned in that script, not resolved to "latest" at install time,
and what it installed is written to `$METAL_GAUSS_THIRD_PARTY/versions.json`
(default `~/third_party`). `bench/compare/pareto.py` stamps that record onto
every competitor row, so a published number can name the build that produced
it.

This exists because it did not. `msplat_bin()` defaulted to
`/tmp/cmp_msplat/bin/msplat-train`; /tmp does not survive a reboot, and by the
time anyone tried to reproduce the competitor half of the README, all three
binaries were gone. The install knowledge below was prose, not a script.

spirula's build has one non-obvious flag. `SS_BACKEND` defaults to `cuda`
(`cmake/SsOptions.cmake`), so a plain `cmake -S . -B build` on a Mac fails
inside `CMakeDetermineCUDACompiler` hunting a CUDA toolkit that cannot exist
here. `-DSS_BACKEND=vulkan` selects the MoltenVK path, which is what the
benchmarked build used. The installer passes it; this paragraph exists because
that flag was not written down anywhere and had to be recovered from the
project's own CMake.

## Real scene trainers (benchmarkable)

| project | install | status |
|---|---|---|
| **metal-gauss** (this repo) | `pip install -e .` | — |
| [Brush](https://github.com/ArthurBrussee/brush) 0.3 | prebuilt binary | **works.** wgpu/Burn, the existing reference |
| [rayanht/msplat](https://github.com/rayanht/msplat) 1.1.4 | `pip install msplat[cli]` | **works.** All-Metal fused trainer, numpy-only dependency. COLMAP/Nerfstudio in, `.ply` out, `--eval` for held-out scoring. ADC densification (`--densify-grad-thresh 0.0002`) |
| [ghif/splat-apple](https://github.com/ghif/splat-apple) | `pip install -r requirements.txt`, then `setup.py` **and** `setup_mlx.py build_ext` | **works, both backends built.** torch 2.10 + MLX 0.30. `train_torch.py` / `train_mlx.py`, COLMAP via `--data_dir` |
| [OpenSplat](https://github.com/pierotofy/OpenSplat) | CMake + **OpenCV 4** + libtorch | **builds, but GPU support cannot be built on this machine.** Two separate traps, both below** |
| [spirula-studio](https://github.com/harry7557558/spirula-studio) | **benchmarked** | Cross-vendor trainer (Vulkan/MoltenVK). Builds from source here, no Xcode, no CUDA; real `spirula train` CLI; reads Nerfstudio layout so it takes the same seed cloud. **Strongest competitor measured: 29.22 dB mean over 8 scenes at 15k, ahead of Brush's 26.94, at 28.4 min vs our 6.0.** Dominated at 45/48 points on PSNR, 41/48 on PSNR+SSIM. Its apparent wins on lego and mic were our 100k capacity default, not its method: matched to its 1M cap we score 35.41 vs 35.53 on mic and 35.72 on lego, 2.7 dB above its best. Noise floor is scene-dependent (0.079 dB on mic, 1.27 on lego). Traps: `--keep-viewer-alive` defaults on (times the viewer, not the run); `ply_file_path` wins over `sparse_pc.ply`, so the seed cloud must be named to match or init parity is silently lost. See [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) |
| [splat-local](https://github.com/michael-L-i/splat-local) | not attempted | video-to-splat pipeline for Apple Silicon that appears to **wrap Brush** rather than implement a trainer. If so it is a pipeline row, not a competitor, and Brush already has a row |

### OpenSplat: two traps, one of them fatal here

1. `brew install opencv` now gives OpenCV **5**, which renamed `calib3d`; the link fails with
   ``library 'opencv_calib3d' not found``. Fix: `brew install opencv@4` and point
   `CMAKE_PREFIX_PATH` at `/opt/homebrew/opt/opencv@4`. libtorch can come from any pip `torch`.
2. It then builds and **runs on CPU only**. Its MPS path is gated on
   `xcrun -sdk macosx metal --version`, i.e. the Metal compiler from **full Xcode**. This machine
   has Command Line Tools only, so cmake silently downgrades `GPU_RUNTIME=MPS` to `CPU` with a
   warning, and the binary exits with "GPU support not built, use --cpu".

That second point is worth stating because it is the exact constraint metal-gauss was designed
around: we compile our `.metal` sources at **runtime** with `newLibraryWithSource`, so no Xcode
and no `.metallib` step is required. OpenSplat cannot be GPU-benchmarked here without installing
Xcode, and a CPU number would not be comparable, so it has no row in the training tables.

### A third trap: duplicate OpenMP runtimes

Once built, OpenSplat aborted with ``OMP Error #15: Initializing libomp.dylib, but found
libomp.dylib already initialized``. libtorch bundles a libomp whose install name is
*`/opt/homebrew/opt/libomp/lib/libomp.dylib`* -- the same path Homebrew's own copy occupies -- so
two different files claim one identity.

The usual workaround, `KMP_DUPLICATE_LIB_OK=TRUE`, is declined here: the runtime itself warns it
"can degrade performance or cause incorrect results", and a wrong benchmark number is worse than
a missing one. Instead the two libraries were compared symbol by symbol (1640 vs 1644 exports;
the six that differ are all OpenMP *target-offload* internals, unused on Metal) and torch's copy
symlinked to Homebrew's. A missing symbol would fail loudly at load time rather than silently
corrupt results, which is why this trade was acceptable and the env-var was not.

### splat-apple needs PINHOLE cameras

Both backends abort with `NotImplementedError: Camera model 4 not implemented` on a COLMAP
reconstruction using the **OPENCV** model. `train_torch.py` catches this and **exits 0**, so a
naive harness records a successful run with a missing PSNR -- ours now treats exit-0-with-error-text
as failure. Running `colmap image_undistorter` to produce PINHOLE cameras fixes both backends,
and that undistorted scene is what every implementation is benchmarked on, so the input is
identical for all.

## Not scene trainers, despite appearances

These are rasterizer libraries. Both ship a file called `simple_trainer.py`, and in both cases it
fits a **synthetic 2D target**, not a reconstruction. They cannot produce a PSNR on a real scene,
so they are excluded from the training tables rather than reported as slow or broken.

| project | what its "trainer" actually does |
|---|---|
| [RobotFlow-Labs/gsplat-mlx](https://github.com/RobotFlow-Labs/gsplat-mlx) | **This entry may be stale.** When checked, `examples/simple_trainer.py` optimised **200 random gaussians to reproduce a 32x32 synthetic gradient image** (`--num-steps 300 --num-gaussians 200 --width 32 --height 32`), despite a README describing a "full training pipeline". It now also advertises 2DGS and 405 tests, so it may have gained a real trainer since. **Needs re-checking before being cited either way** |
| [iffyloop/gsplat-mps](https://github.com/iffyloop/gsplat-mps) | `examples/simple_trainer.py` fits gaussians to **one 256x256 image with red and blue quadrants**. A gsplat 0.1.3 fork ported to MPS |

Their forward rasterizers could be timed against ours in a render-only comparison; that is a
different table and is not mixed into training results.

## Renderers only

[MetalSplatter](https://github.com/scier/MetalSplatter), [mlx-splat](https://github.com/daikiad/mlx-splat),
[3DGS.cpp](https://github.com/shg8/3DGS.cpp) — viewers. They render a trained `.ply` and cannot
train, so they are outside the scope of a training benchmark.

## The common evaluator: why it looked impossible, and what it was

Comparing trainers is harder than it looks, and this is the trap that cost the most time here.

Each trainer holds out its own views and scores them with its own code, so their reported PSNRs
are not strictly comparable. NeRF-synthetic has an *official* test split, so the obvious fix was:
everyone trains on the official train frames, exports a `.ply`, and **one** evaluator
(`score_ply.py`) renders the same 200 official test views for all of them.

That evaluator is correct. It reproduces our own trainer's reported figure exactly (14.0740 vs
14.07) and the float64 gradcheck oracle disagrees with it by 0.00033 dB.

It still could not score msplat. On one `.ply`, **msplat's own eval reports 20.25 dB and ours
reports 9.58**. Everything checkable checks out:

- `load_ply` maps their file correctly — SH DC matches the raw `f_dc_*` columns exactly, scales
  are exp'd, opacity sigmoid'd, quaternions normalised, positions verbatim.
- Their geometry is sane: 123k splats centred near the origin with a 2.6-unit extent, which is
  lego-sized.
- Their splats render: alpha mean 0.40, 248k tile intersections. They are simply in the wrong
  place relative to our test cameras, so the image comes out ~white.
- Flipping the world frame, the camera frame, or inverting the pose changes nothing (8.7-8.9 dB).
- No colour-encoding interpretation recovers it either (best 10.6 dB).

That diagnosis was right, and it had a one-flag fix. msplat's Nerfstudio loader applies
Nerfstudio's scene transform -- auto-orient and auto-scale -- so its world frame is a rotated and
rescaled version of the one our `transforms_test.json` cameras live in. Its own eval stays
self-consistent and looks fine; any external evaluator reads ~9 dB instead of ~20.

**`--keep-crs` disables that transform, and the common evaluator works.** Every row of the
published Pareto front -- ours and msplat's alike -- is now scored by `score_ply.py` on the same
official 200-view test split, from the same random initialisation. That is what makes it a Pareto
front rather than two curves drawn on one axis.

Two things worth keeping from the wrong turn. The flag is **required**, not cosmetic: without it
the numbers are not merely noisy, they are in a different coordinate system, and msplat looks
catastrophically bad through no fault of its own. And the elimination trail above -- checking the
ply mapping, the geometry, the alpha coverage, then flipping frames and colour encodings -- is
what located the frame mismatch. The decisive test was running msplat's *own* eval on the same
file: 20.25 against our 9.58. When two correct-looking pipelines disagree by 10 dB, the fault is
usually a frame, not a metric.

## Fairness rules for every number in the comparison

- Same machine, same scene, same iteration count, same held-out split.
- Every implementation is scored by **one** evaluator (`score_ply.py`) on the **official**
  NeRF-synthetic test split, from an identical random initialisation. This requires `--keep-crs`
  for msplat; see above. Where a project's own eval is also available it is reported alongside,
  never mixed into the same column.
- Where a project publishes numbers we cite them and link the source, and we state the hardware:
  several publish M4 Max figures; ours are M5.
- Timings use a sustained-load GPU ramp. Apple Silicon runs short bursts at a boost clock, which
  flatters burst benchmarks by up to 40% — see `bench/results/NEGATIVE_RESULTS.md`.

## What msplat's kernels actually do

Its `default.metallib` and `_core.so` are precompiled with no source shipped,
but the AIR metadata names every kernel and its arguments, so the architecture
is readable without guessing. Recorded here because "why are they faster per
step" had been answered with speculation for weeks.

Their kernel set:

| kernel | what it implies |
|---|---|
| `project_and_sh_forward_kernel` / `_backward_kernel` | projection and SH fused, as ours are |
| **`proj_sh_bwd_adam`** | **the projection/SH backward and Adam in ONE kernel** |
| `fused_adam_kernel` | a standalone fused Adam as well |
| `rasterize_forward_kernel`, `rasterize_backward_kernel` | plus `_chunked` variants and a `_merge` pass |
| **`prefix_sort_pack` → `packed_xy_opac`, `packed_conic`, `packed_rgb`** | **attributes gathered into sorted, contiguous per-intersection arrays once** |
| **`scatter_to_prealloc_bins_kernel`** with an `overflow_flag` | **single-pass scatter into preallocated bins** |
| `radix_sort_histogram/scan/scatter_kernel` | their own radix sort in Metal |
| `bitonic_sort_per_tile_kernel` | exact per-tile depth sort |
| `fused_loss_forward_kernel` | loss in Metal, as ours is |

Three differences are candidate explanations for the measured per-step gap, and
all three are structural rather than micro-optimisations:

1. **Adam folded into the backward.** Ours is a separate one-pass kernel costing
   ~20 % of the step. Folding it into the backward removes a whole pass over
   the parameters and its memory traffic.
2. **Pack once, then stream.** They pay one gather to build sorted contiguous
   arrays, after which the rasteriser reads coalesced. We gather scattered by
   gaussian id into threadgroup memory in every tile, which is the
   staging path several already-rejected levers were trying to make cheaper --
   possibly the wrong target, since the packing removes the gather rather than
   accelerating it.
3. **Single-pass binning into preallocated bins.** We count, prefix-sum, then
   write. They preallocate and scatter once, with an overflow flag as the
   escape hatch. Note this is NOT the per-tile cap already measured and
   rejected here (0.009 dB, 18 % slower): that was about DROPPING gaussians,
   this is about how the list is built.

None of these is adopted yet, and none should be until profiling at the
matched operating point says which part of our step is actually large there.
Every profile in this repo is at 600k splats @ 152x270; the gap was measured at
100k @ 800x800, and this project has twice been caught assuming a profile
transfers between regimes.

## Brush: on the front, and what its short runs actually do

Brush is now benchmarked through `bench/compare/pareto.py --impls brush`, not
cited from a stale JSON. Two things had to be established first.

**Its ply is readable by our evaluator, and its frame is correct.** The header
carries `comment Vertical axis: y`, which is a hardcoded string with no `z`
variant anywhere in the binary and therefore carries no information. The
geometry settles it: every splat above alpha 0.1 sits within r < 1.5 of the
origin with DC colour [0.74, 0.63, 0.26] -- lego yellow -- and the point cloud's
one-sided vertical extent matches the camera origins. No `--keep-crs`
equivalent is needed, unlike msplat.

Its ply also lists properties ALPHABETICALLY (`f_rest_0, f_rest_1, f_rest_10,
...`) with `x, y, z` LAST, where the INRIA convention puts `x, y, z` first and
numbers `f_rest` in order. That is harmless here only because `metal_gauss/io.py`
indexes by field NAME through plyfile. A reader using fixed column offsets
would produce garbage from this file.

**Its stock defaults are degenerate at short step counts.** At 1000 steps it
exports 10,540 splats of which ~10,283 are a near-transparent grey veil filling
the camera volume (median alpha 0.010); only ~159 carry the object. Growth has
barely fired, because the defaults target its own 30000-step default.

The scale of that: Brush's OWN saved eval renders score **11.44 dB** against
ground truth composited over black -- *worse than a constant black image*, which
scores 10.43. Our evaluator gives its exported ply **14.74 dB**, which beats a
constant white image at 9.95. We are flattering it, not penalising it.

By 7000 steps it is healthy: 24.69 dB, 13,866 splats, 6.41 min.

Reported as measured. Running a competitor at a step count its schedule was not
designed for is a real property of that competitor under those settings, and
the alternative -- retuning it -- is the mistake already recorded here, where
"fairness" rescaling of msplat's schedules cost it up to 4.4 dB.

**Brush renders over BLACK and has no background option.** NeRF-synthetic is
scored over white by convention, and its saved PNGs are RGB with no alpha, so
they cannot be recomposited. This is why the front scores Brush's exported ply
with our common evaluator rather than its own renders -- the ply can be
rendered over white, its own PNGs cannot.
