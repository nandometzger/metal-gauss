# Benchmarks

Full results, protocol and competitor setup. The short version, with the
figure, is in the [README](../README.md).

Every table here is generated from a committed JSON by
`python bench/readme_tables.py`; `--check` fails if any drifts from its data.

# 📊 Benchmarks

Measured on one machine: **Apple M5, 24 GB, macOS 26**. Every number traces to a JSON in
[`bench/results/`](../bench/results/) and a command in [Reproducing](#-reproducing).

> **On timing methodology.** Apple Silicon runs short bursts at a boost clock and settles lower
> under sustained load — the same kernel measured 11.6 ms and 19.8 ms on alternating runs, a 70 %
> swing that once made a no-op change look like a 26 % win here. Every timing below is taken
> after a 2-second sustained-load ramp, which is the clock a real training run actually sees.
> Burst numbers would be up to 40 % more flattering.

## Apple-Silicon head-to-head

Real capture, 162 images, COLMAP undistorted to PINHOLE so every implementation reads identical
data. Runs sequential, never concurrent.

**room1, 10 000 iterations, 600 k splats, 1600 px:**

| | PSNR | wall |
|---|---:|---:|
| **metal-gauss** (current defaults) | **27.62** | **11.9 min** |
| metal-gauss (as published 2026-08-26) | 27.43 | 28.5 min |
| [Brush](https://github.com/ArthurBrussee/brush) ¹ | 27.55 | ~40–44 min |

¹ *Brush's number is from an earlier measurement whose JSON cannot state its own protocol
(`provenance: incomplete`); it is cited as a reference point, not as evidence.*

**Read this as "matches Brush at roughly 3.5× the speed", not "beats Brush".** The 0.07 dB gap to
Brush and the 0.19 dB gain over our own earlier run are both **inside the measured 0.49 dB
run-to-run spread at 10 k** and mean nothing. The wall-clock change is the real result, and it
comes from coarse-to-fine plus kernel work, not from the capacity fix — this run pins budget at
600 k deliberately so the two rows stay comparable.

The msplat and splat-apple tables that stood here were produced under the harness bug below and
are quarantined rather than reprinted; the **Pareto front at the top of this README is the
current, provenanced comparison**. What those measurements did establish, and what still holds:

- **msplat's speed is largely its progressive resolution schedule** (full resolution only after
  iteration 6 000), and that schedule is a real technique rather than a benchmark artifact —
  forced to full resolution its own score *drops*, 21.37 → 20.78. **We now implement this**
  (`--num-downscales`, default 2). Isolated from the capacity change it is worth about +0.07 dB
  mean across the 8 Blender scenes at ~1.6× the speed — a wall-clock win, not a quality one. (That
  isolation comes from two pre-fix sweeps whose JSONs are quarantined; the direction is clear, the
  exact figures are not re-measured.)
- **splat-apple exits 0 on failure.** It prints "Error loading colmap data" and returns success,
  which would land in any table as a completed run with a missing PSNR. The harness treats
  exit-0-with-error-text, and implausibly fast completion, as failure.

## Paired margins with confidence intervals

The README carries this as a forest plot; these are the numbers behind it.

<!-- BEGIN:margin -->
| vs | iters | mean Δ | 95% CI | scenes won |
|---|---:|---:|---|---:|
| spirula-studio | 500 | +7.78 dB | [+6.29, +9.27] | 8/8 |
| spirula-studio | 7,000 | +8.43 dB | [+5.47, +11.39] | 8/8 |
| spirula-studio | 15,000 | +2.65 dB | **[-0.13, +5.44]** — includes 0 | 7/8 |
| Brush | 500 | +8.70 dB | [+6.12, +11.29] | 8/8 |
| Brush | 7,000 | +6.68 dB | [+4.26, +9.11] | 8/8 |
| Brush | 15,000 | +4.93 dB | [+1.99, +7.87] | 8/8 |
| msplat | 500 | +8.01 dB | [+6.44, +9.58] | 8/8 |
| msplat | 7,000 | +8.27 dB | [+5.45, +11.09] | 8/8 |
| msplat | 15,000 | +10.28 dB | [+7.58, +12.97] | 8/8 |

*Per-scene difference across all 8 scenes, 95% CI from Student-t (df 7). A CI containing zero means the quality margin at that point is not resolved by 8 scenes, whatever the mean says.*
<!-- END:margin -->

## The full 8-scene ladder

Every implementation at every rung, sorted by wall-clock. The README carries the compact
"what do I get in N minutes" summary; this is the measurement behind it.

<!-- BEGIN:pareto-scenes -->
| implementation | iters | wall (8-scene mean) | PSNR | SSIM |
|---|---:|---:|---:|---:|
| msplat (stock) | 500 | 0.10 min | 12.06 | 0.7986 |
| msplat (stock) | 2,000 | 0.15 min | 19.65 | 0.8537 |
| msplat (stock) | 1,000 | 0.16 min | 13.50 | 0.8108 |
| metal-gauss | 500 | 0.23 min | 21.55 | 0.8501 |
| msplat (stock) | 4,000 | 0.34 min | 19.13 | 0.8754 |
| spirula-studio | 500 | 0.34 min | 13.77 | 0.8167 |
| brush | 500 | 0.43 min | 12.85 | 0.7967 |
| msplat (scaled) | 500 | 0.44 min | 13.54 | 0.8113 |
| metal-gauss | 1,000 | 0.60 min | 22.76 | 0.8842 |
| spirula-studio | 1,000 | 0.61 min | 15.12 | 0.8328 |
| msplat (scaled) | 2,000 | 0.73 min | 18.26 | 0.8690 |
| brush | 1,000 | 0.79 min | 14.35 | 0.8042 |
| msplat (scaled) | 1,000 | 0.88 min | 17.53 | 0.8330 |
| msplat (stock) | 7,000 | 0.90 min | 21.19 | 0.9067 |
| metal-gauss | 2,000 | 1.04 min | 26.31 | 0.9206 |
| spirula-studio | 2,000 | 1.32 min | 17.36 | 0.8665 |
| brush | 2,000 | 1.58 min | 18.44 | 0.8340 |
| msplat (scaled) | 4,000 | 1.59 min | 20.22 | 0.8977 |
| metal-gauss | 4,000 | 1.82 min | 28.82 | 0.9440 |
| msplat (scaled) | 7,000 | 2.74 min | 20.60 | 0.9021 |
| metal-gauss | 7,000 | 2.82 min | 30.38 | 0.9547 |
| brush | 4,000 | 3.19 min | 21.37 | 0.8677 |
| spirula-studio | 4,000 | 3.60 min | 19.60 | 0.8973 |
| msplat (stock) | 15,000 | 4.14 min | 20.58 | 0.9084 |
| brush | 7,000 | 5.67 min | 23.70 | 0.8979 |
| msplat (scaled) | 15,000 | 5.78 min | 21.04 | 0.9084 |
| metal-gauss | 15,000 | 5.93 min | 31.87 | 0.9639 |
| spirula-studio | 7,000 | 9.78 min | 21.95 | 0.9082 |
| brush | 15,000 | 12.57 min | 26.94 | 0.9358 |
| spirula-studio | 15,000 | 28.38 min | 29.22 | 0.9506 |

*All 8 NeRF-synthetic scenes, official 200-view test split, one evaluator, identical seed point cloud per scene, strictly sequential. metal-gauss dominates (faster **and** better on PSNR / on PSNR **and** SSIM): **48/48** / **48/48** of Brush, **47/48** / **45/48** of spirula-studio, **68/96** / **51/96** of msplat (both variants). Single runs per cell; noise floors differ per implementation AND per scene (see NEGATIVE_RESULTS.md): metal-gauss 0.22 dB, spirula 0.15-1.27, Brush 0.74, msplat up to 3.35 -- msplat's counts are indicative only.*
<!-- END:pareto-scenes -->

## lego, as a worked example

The headline front in the README is the mean over all 8 scenes. This is the same measurement on a
single scene, kept because a per-scene ladder shows the shape that an average hides — in particular
how far msplat's fast configuration reaches before quality stops following wall-clock.

A single scene is an **example, not a claim about a method** — the headline front is the 8-scene
mean above.

<!-- BEGIN:pareto -->
| implementation | iters | wall | PSNR | splats |
|---|---:|---:|---:|---:|
| msplat | 500 | 0.10 min | 10.86 | 100,000 |
| msplat | 1,000 | 0.17 min | 12.64 | 100,000 |
| msplat | 2,000 | 0.17 min | 20.38 | 19,615 |
| metal-gauss | 500 | 0.30 min | 20.52 | 27,857 |
| msplat | 4,000 | 0.40 min | 20.23 | 50,021 |
| brush | 500 | 0.52 min | 12.28 | 10,540 |
| metal-gauss | 1,000 | 0.68 min | 23.67 | 100,000 |
| brush | 1,000 | 0.83 min | 14.24 | 10,540 |
| msplat | 7,000 | 1.01 min | 24.33 | 64,492 |
| metal-gauss | 2,000 | 1.16 min | 26.28 | 100,000 |
| brush | 2,000 | 1.81 min | 18.47 | 3,126 |
| metal-gauss | 4,000 | 1.94 min | 28.79 | 100,000 |
| metal-gauss | 7,000 | 3.13 min | 30.81 | 100,000 |
| brush | 4,000 | 3.45 min | 21.51 | 5,529 |
| msplat | 15,000 | 4.52 min | 24.35 | 126,384 |
| metal-gauss | 15,000 | 6.11 min | 33.01 | 100,000 |
| brush | 7,000 | 6.13 min | 25.03 | 14,620 |
| brush | 15,000 | 15.01 min | 28.74 | 44,838 |

*One evaluator (`bench/compare/score_ply.py`) on the official 200-view test split, identical random init, strictly sequential. msplat variant: **stock**.*
<!-- END:pareto -->

![Pareto front](../bench/results/pareto_lego.svg)

## spirula-studio

[spirula-studio](https://github.com/harry7557558/spirula-studio) validated macOS/Apple-Silicon
training in August 2026, after the original competitor survey was written, and was carried in
`STATUS.md` as a known gap rather than quietly omitted. It is now measured.

It builds from source on this machine (CMake fetches MoltenVK; no Xcode, no CUDA), exposes a real
`spirula train` CLI, and reads the Nerfstudio layout, so it takes the **same seed point cloud**
as metal-gauss and msplat — verified byte-identical by SHA-256, not assumed.

<!-- BEGIN:spirula -->
| iters | wall | PSNR | SSIM | LPIPS | beaten on all three by |
|---:|---:|---:|---:|---:|---|
| 500 | 0.37 min | 13.64 | 0.7769 | 0.4048 | 500 it — 0.30 min |
| 1,000 | 0.66 min | 15.93 | 0.8082 | 0.2924 | **holds the front** |
| 2,000 | 1.36 min | 17.59 | 0.8655 | 0.1765 | 2,000 it — 1.16 min |
| 4,000 | 3.58 min | 20.56 | 0.9110 | 0.1138 | 4,000 it — 1.94 min |
| 7,000 | 10.02 min | 27.63 | 0.9524 | 0.0496 | 15,000 it — 6.11 min |
| 15,000 | 30.20 min | 32.05 | 0.9703 | 0.0199 | **holds the front** |

*spirula-studio 2026.8.28, `synthetic` preset, shipped defaults (1M splat cap), same seed point cloud, one evaluator on the official 200-view test split. **Single runs on one scene, superseded by the 8-scene front in the README.** Kept because it is the only table here carrying LPIPS. Read its 'holds the front' rows with the correction above in mind: repeating lego gives spirula 33.02 and 31.74 (1.27 dB spread), and the remaining gap is our 100k capacity default rather than its method. Dominated at **6/6** on PSNR, **4/6** on PSNR+SSIM+LPIPS.*
<!-- END:spirula -->

**It is the strongest competitor measured here.** Across all 8 scenes it averages 29.22 dB at
15 000 iterations, ahead of Brush's 26.94, and is dominated at 45/48 points on PSNR and 41/48 once
SSIM must hold too. It spends **28.4 min against our 6.0**.

**It briefly appeared to beat us, and the correction is instructive.** On the single-scene lego
ladder it scored 32.05 dB / SSIM 0.9703 / LPIPS 0.0199 against our 33.01 / 0.9694 / 0.0258 — winning
SSIM and LPIPS. Two things dissolved that:

- **Its noise floor is scene-dependent.** Repeating lego gives 33.02 and 31.74, a 1.27 dB spread,
  while mic repeats to 0.079 dB over three runs. The lego "tie" was a high draw; we average ahead.
- **The remaining gap was our capacity default, not its method.** It grows to a 1 M splat cap where
  `auto_budget` gives us 100 k. Matched to 1 M we score 35.41 on mic against its 35.53, and **35.72
  on lego — 2.7 dB above its best** — in slightly less wall-clock.

So spirula does not reconstruct better; it buys quality with 10× the splats and ~5× the time. That
is a legitimate trade and our default declines it deliberately.

Three configuration traps, each of which would have produced a wrong number:

- **`--keep-viewer-alive` defaults to 1.** The process serves an HTTP viewer forever after
  training ends, so a naive wall-clock measures the viewer rather than the run. `--disable-viewer 1`
  is mandatory for any timing.
- **Its parser honours `ply_file_path` from `transforms.json` before falling back** to
  `sparse_pc.ply` / `pointcloud.ply`. Stage the seed cloud under the wrong name and the run starts
  from a *different initialisation* than every other row while still looking healthy.
- **The `synthetic` preset is the correct one for NeRF-synthetic** — it disables the bilateral grid
  and per-pixel ISP, which exist for real captures. Running the generic preset here would benchmark
  a misconfiguration, the same reasoning that made us scale msplat's schedules.

Two checks that came back clean, recorded because their failure modes have bitten this repo before:

- **Coordinate frame verified by geometry, not by score** — splats centre on the origin
  (0, −0.01, 0) with radius median 1.40. No `--keep-crs` equivalent is needed, unlike msplat.
- **Background convention is irrelevant to its score.** It trains with `background-mode black`
  against white-background images, which is what made Brush score below a constant image. Rendered
  over white it scores 31.72 dB and over black 31.71 — a 0.01 dB difference, because it paints its
  own background opaquely. Our white-background evaluation does not penalise it.

In its favour, and worth stating plainly: its parser **throws** when the seed cloud is missing in
trainer mode rather than silently random-initialising, and its schedules already scale to
`--num-iterations`. Nothing here is hand-tuned; this is stock.

## NeRF-synthetic, all 8 scenes

<!-- BEGIN:nerf-synthetic -->
| scene | PSNR | wall | splats | budget | 3DGS @30k | gap |
|---|---:|---:|---:|---:|---:|---:|
| chair | 32.43 | 2.7 min | 100,000 | 100,000 | 35.8 | 3.4 |
| drums | 24.94 | 2.7 min | 100,000 | 100,000 | 26.1 | 1.2 |
| ficus | 28.68 | 2.6 min | 100,000 | 100,000 | 34.8 | 6.1 |
| hotdog | 36.47 | 3.4 min | 100,000 | 100,000 | 37.7 | 1.2 |
| lego | 30.75 | 2.8 min | 100,000 | 100,000 | 35.8 | 5.0 |
| materials | 27.95 | 2.6 min | 100,000 | 100,000 | 30.0 | 2.1 |
| mic | 32.28 | 2.6 min | 100,000 | 100,000 | 35.4 | 3.1 |
| ship | 29.43 | 3.1 min | 100,000 | 100,000 | 30.9 | 1.5 |
| **mean** | **30.37** | **22.6 min total** | | | 33.3 | **2.9** |

*7000 steps @ 800px, official Blender train/test split, white background, budget `auto`, num_downscales `default`. Built from `e262019`.*
<!-- END:nerf-synthetic -->

**Read the gap column against the cost of the short protocol, not against zero.** lego is the one
scene run at both protocols: 30.75 dB at 7 k against 35.88 at 30 k, so **the short protocol itself
costs 5.1 dB**. The mean gap above is 2.9 dB — that is, at 7 k iterations most scenes land *better*
than the protocol penalty alone would predict, and none is now an outlier beyond it.

Run-to-run spread on this protocol, measured over two full sweeps of all 8 scenes: **0.07 dB
typical, 0.19 dB worst** (`python bench/reproducibility.py`). The 0.49 dB figure quoted elsewhere in
this README was measured on room1 at 10 k/1600 px and does not apply here.

Every row is generated from a committed JSON carrying the trainer's own resolved configuration
(`python bench/readme_tables.py --check` fails if this table and the data disagree). That is not
ceremony. An earlier version of this table was produced by a harness that forced a 300 k budget on
every scene while reporting that it had not, which suppressed the capacity fix worth **+8.3 dB on
materials and +6.6 dB on ficus** and left both scenes documented as "unexplained outliers" across
two plan revisions. The post-mortem is in
[`bench/results/NEGATIVE_RESULTS.md`](../bench/results/NEGATIVE_RESULTS.md).

## Absolute calibration

A private scene has no ground truth to be below, so a systematic deficit there is invisible. The
Blender scenes have published numbers for exactly this algorithm:

<!-- BEGIN:calibration -->
| lego, 30 k steps | PSNR |
|---|---:|
| 3DGS-MCMC (paper, random init) | 36.01 |
| **metal-gauss** | **35.88** |
| 3DGS baseline (paper, random init) | 35.84 |

*500,000 splats, 35.3 min, scored by `score_ply.py` over 200 official test views. Built from `4927e7b`.*
<!-- END:calibration -->

**0.13 dB below the method we implement, and above the 3DGS baseline.** The previously published
figure here was 35.48 dB, 0.53 dB short; the difference is the capacity schedule, whose tail above
300 k splats was an untested extrapolation until this run. The MCMC implementation is sound.

## Raster speed

1.96 M gaussians, float32, steady-state clock, median of 11 trimmed.

<!-- BEGIN:raster -->
| gaussians | resolution | forward | forward + backward |
|---|---|---:|---:|
| 600,000 | 270x480 | 10.8 ms | **29.9 ms** |
| 600,000 | 900x1600 | 20.0 ms | **59.0 ms** |
| 1,000,000 | 270x480 | 17.8 ms | **48.4 ms** |
| 1,000,000 | 900x1600 | 33.8 ms | **90.4 ms** |
| 1,955,058 | 270x480 | 37.9 ms | **93.8 ms** |
| 1,955,058 | 900x1600 | 64.8 ms | **167.6 ms** |

*tile 16, after a 2 s sustained-load clock ramp. Built from `8065b0f`.*
<!-- END:raster -->

The pure-torch oracle (`torch_ref`) needs 4.81 s for the same forward at 270×480 and **runs out of
memory at 30.2 GB** on the backward.

That the reference cannot finish a backward pass at production scale is the point: `torch_ref` is
the float64-gradcheck **correctness oracle**, not a fallback. There is no degraded mode.

## Where a training step goes

600 k gaussians @ 270×480. Phases timed with `mps.synchronize()` between them, so the total is an
upper bound — use it to compare phases, not to predict wall-clock.

<!-- BEGIN:step-profile -->
| phase | ms | share |
|---|---:|---:|
| backward | 22.6 | 46 % |
| adam | 9.8 | 20 % |
| render_fwd | 9.6 | 19 % |
| mcmc | 5.6 | 11 % |
| loss_fwd | 1.7 | 4 % |
| **total** | **49.3** | 283 ms when this work started |

*600,000 gaussians @ 152×270. Phases separated by `mps.synchronize()`, so the total is an upper bound: compare phases with it, do not predict wall-clock. **`backward` is the whole autograd pass** — raster, projection/SH and loss backward together — not the raster backward alone. Built from `8065b0f`.*
<!-- END:step-profile -->

## 🔁 Reproducing

### Environment and staged inputs

The dev environment lives in `.venv/` **inside the repo**, not in `/tmp`. macOS purges `/tmp` files
older than three days, and it does so file-by-file: a venv there loses `pyvenv.cfg` and individual
`__init__.py` files while its 357 MB of package data sits intact, so imports fail in ways that look
like a broken install rather than a deleted one.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[bench,train]" pytest lpips scikit-image imageio tqdm torchvision
```

Competitor binaries and the private capture are located by environment variable, with a documented
default. None of these paths are hardcoded:

| variable | what | default |
|---|---|---|
| `METAL_GAUSS_THIRD_PARTY` | root holding competitor checkouts | `~/third_party` |
| `METAL_GAUSS_BRUSH` | Brush binary | `$THIRD_PARTY/brush/brush-app-aarch64-apple-darwin/brush_app` |
| `METAL_GAUSS_SPIRULA` | spirula-studio binary | `$THIRD_PARTY/spirula-studio/build/spirula` |
| `METAL_GAUSS_MSPLAT` | msplat-train | `/tmp/cmp_msplat/bin/msplat-train` |
| `METAL_GAUSS_ROOM1` | private real capture; benchmarks needing it are skipped when unset | unset |

Competitor inputs are staged under `/tmp/cmp_data/<scene>_ns` and **are** disposable: the seed point
cloud comes from `np.random.default_rng(0)`, so re-staging reproduces it byte-for-byte (verified:
lego is always `ed443e71…`). Rebuild all eight with

```bash
for s in lego drums ficus chair hotdog materials mic ship; do
  python bench/compare/blender_to_nerfstudio.py data/nerf_synthetic/$s /tmp/cmp_data/${s}_ns
done
```

The Brush and spirula binaries live under `third_party/` and persist; msplat is installed to
`/tmp/cmp_msplat` and needs reinstalling if that is purged.

### Tools

Every script under `bench/`, and what it is for. Eight of these were undocumented until this table
existed, which is how two dead ones (`plot_pareto.py`, superseded; `metal_gauss/quat_utils.py`,
zero callers) survived as long as they did.

| script | purpose |
|---|---|
| `bench/compare/sweep8.py` | the 8-scene x 4-implementation sweep; resumable, skips completed cells |
| `bench/compare/pareto.py` | one implementation across the iteration ladder on one scene |
| `bench/compare/run_spirula.py` | spirula-studio driver (viewer disabled, seed cloud matched) |
| `bench/compare/variance.py` | repeat one configuration N times; reports mean, stdev, range |
| `bench/compare/plot_pareto_scenes.py` | the Pareto SVG; `--scenes lego` for the single-scene version |
| `bench/compare/collect_timelapse.py` | checkpoints from all four trainers on an equal wall-clock budget |
| `bench/compare/render_timelapse.py` | scores those checkpoints into a frames JSON |
| `bench/compare/build_timelapse_page.py` | the interactive timelapse page |
| `bench/compare/render_timelapse_gif.py` | the README GIF and MP4 |
| `bench/compare/blender_to_nerfstudio.py` | stage a scene for msplat/spirula with a deterministic seed cloud |
| `bench/compare/score_ply.py` | the single evaluator every implementation is scored by |
| `bench/compare/front_summary.py` | per-metric Pareto ownership across scored scenes |
| `bench/compare/rescore_front.py` | add SSIM/LPIPS to an existing front |
| `bench/readme_tables.py` | regenerate every table in README/docs; `--check` fails on drift |
| `bench/quarantine.py` | list results that cannot state the protocol they ran under |
| `bench/tile_sweep.py` | tile size at a given operating point |

### Benchmarks

```bash
python bench/quick.py stages                  # raster speed table
python bench/step_profile.py --res 270        # step breakdown
python bench/nerf_synthetic_sweep.py          # 8-scene table   (22.6 min measured)
python bench/compare/pareto.py --msplat-stock  # Pareto front vs msplat  (~35 min)
python bench/readme_tables.py --check         # fail if a README table drifts from its JSON
python bench/quarantine.py                    # list results that cannot state their own protocol
python bench/reproducibility.py a.json b.json # per-scene run-to-run spread from repeated sweeps
python bench/startup_profile.py               # how much of a short run is startup, and what caches
python bench/compare/run_compare.py --scene <colmap> --iters 7000   # head-to-head
```

Competitor install status — including three traps that make projects look broken when they are
not — is in [`bench/compare/STATUS.md`](../bench/compare/STATUS.md).

### How results are kept honest

Every wrong number this project ever published came from a benchmark harness, never from a kernel.
Twice, a harness supplied its own default for the very setting under test and then recorded its own
arguments as "the protocol" — which silently suppressed a **+3 dB** improvement and invented two
"unexplained outlier" scenes that were never anomalous. So:

- **`train.py --report` writes the *resolved* config**, all of it, after `auto_budget`, the
  steps-scaler and every other rewrite have been applied. Not a curated subset: curation is how a
  knob goes unrecorded, and an unrecorded knob is how both bugs stayed invisible.
- **`bench/runner.py` is the only way a benchmark starts the trainer, and it has no defaults.** A
  setting the caller does not pass produces no flag, so the trainer's own default applies *and is
  recorded*. A setting the caller does pass is checked against what actually ran, and a mismatch
  raises instead of quietly producing a number attributed to the wrong protocol.
- **Results that cannot state their own protocol are quarantined** (`bench/quarantine.py`) and the
  README generator refuses to render from them.
- **Every table above is generated** from a committed JSON; `bench/readme_tables.py --check` fails
  if any of them drifts from its data. It caught this table mislabelling its own resolution.
- **Deltas are compared against a measured floor**, not a remembered one
  (`bench/reproducibility.py`).

Both historical bugs are now regression tests in `tests/test_runner.py`.
