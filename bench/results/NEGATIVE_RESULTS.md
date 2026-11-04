# Levers measured and rejected

Techniques that looked promising here and did not survive measurement, kept so
they are cheaper to read than to rediscover. Every number is from an idle
machine, one evaluator, and the protocol stated in each entry.

Several published speedups measure at or near zero on this hardware. That is not
a criticism of the papers: a speedup is a statement about someone else's
bottleneck profile, and the right move before porting one is to re-derive the
bottleneck locally.

### Exact compositing is free; msplat's per-tile cap would cost us speed

msplat prints `per-tile overflow (>2048 gaussians in a tile). Some gaussians were dropped from
overfull tiles.` -- it caps per-tile occupancy and discards the rest. We composite exactly with
depth slabs, and the obvious worry was that exactness is why we are slower.

It is not. Same model (lego, 300k splats, 15k steps), same 25 test views, cap on vs off:

| | PSNR | ms/view | intersections |
|---|---:|---:|---:|
| exact (default) | **30.431** | **21.07** | 25,251,767 |
| cap at 2048 | 30.422 | 24.83 | 24,424,783 |

The cap costs 0.009 dB -- far below the noise floor -- and is **18% SLOWER**. Only 3.1% of tiles
exceed 2048 (max observed 4522), so it removes 3.3% of rasterisation work while the filtering
pass that implements it (gather, mask, re-searchsorted) costs more than it saves.

Two things follow. Our exactness is not a speed handicap, so the depth-slab design costs nothing
to keep. And msplat's cap is not the source of their advantage at these tile densities -- their
speed came from schedules (coarse-to-fine resolution) and capacity, both of which we have since adopted.

`--max-per-tile` stays in `build_tile_lists_metal`, off by default, because it is the honest way
to have asked the question and would matter on far denser scenes.

### The fast end is lost to startup cost, not to throughput

msplat owns every wall-clock budget under ~0.3 min on lego. Two probes into why.

**Capacity below 1000 steps.** `auto_budget` returns a flat 100 k for anything
up to 10 k steps. Below ~500 steps that is too much:

| iters | 30 k budget | auto (100 k) | winner |
|---:|---|---|---|
| 100 | 8.78 @ 0.22 min | 8.15 @ 0.27 min | 30 k, both axes |
| 200 | **15.35** @ 0.23 min | 11.53 @ 0.29 min | 30 k, both axes |
| 300 | **18.18** @ 0.25 min | 17.18 @ 0.33 min | 30 k, both axes |
| 500 | 20.50 @ **0.29** min | 20.45 @ 0.41 min | 30 k on time, tie on quality |
| 1000 | 22.89 | **23.89** | 100 k |
| 2000 | 25.10 | **26.28** | 100 k |

**Shipped, after the 8-scene treatment.** The lego-only result above was not
trusted on its own -- the last capacity rule inferred from lego alone varied
from +0.65 to +8.14 dB across the other seven scenes -- so it was re-run on all
eight at two step counts:

| steps | 100k mean | 30k mean | delta | scenes 30k wins | wall |
|---:|---:|---:|---:|---:|---:|
| 200 | 11.43 | **15.65** | +4.21 | 8/8 | 0.74x |
| 500 | 18.85 | **21.53** | +2.69 | 7/8 | 0.71x |

Better and ~27% faster, so `auto_budget` now returns 30 k below 1000 steps.
Caution was warranted: on lego, the weakest scene for this effect, 500 steps
gives +0.14 dB, against +8.01 dB on mic. A threshold set from lego alone would
have been placed at 200 and left most of the gain on the table. The single
regression is hotdog at 500 steps, -0.66 dB.

**The real limit is fixed startup, now measured** (`bench/startup_profile.py`,
`bench/results/startup_warm.json`):

| | startup | train (50 steps) | total |
|---|---:|---:|---:|
| warm, 4 consecutive runs | 8.3-8.7 s | 5.5-5.6 s | ~14 s |
| forced extension rebuild | 8.40 s | **17.6 s** | 26.0 s |

Startup is 8.4 s and barely varies (0.4 s over four runs). It is Python and
torch import plus scene load; it does not fall however few steps are asked for,
against msplat's ~6 s total.

Two things this corrected. The extension rebuild costs ~12 s but is paid **once
ever** and shows up in `train_s`, not startup, because torch loads the
extension lazily on first kernel use -- the first version of this measurement
computed the cost from `startup_s` and duly reported it as -0.2 s, hiding a
12 s effect. No sweep run ever paid a compile cost,
so that is not why the first scene was slow.

The consequence is that **the sub-0.3-minute regime is not a rasterizer
problem at all** -- no kernel work can win it, because no kernel runs during
the 13 s that decides it. The levers are a compiled-kernel cache, a lazier
import graph, and a faster scene loader. None of them is a splatting
optimisation, which is presumably why none was on the list.

### Packing intersections msplat-style: measured ceiling 2.3%, rejected before building it

Reading msplat's metallib showed `prefix_sort_pack` producing
`packed_xy_opac` / `packed_conic` / `packed_rgb`: they gather attributes into
sorted contiguous arrays once, so the rasteriser streams coalesced. We gather
scattered by gaussian id into threadgroup memory in every tile. With
rasterisation dominating the step, this looked like the lever.

Rather than rewrite the hottest kernel to find out, the ceiling was measured
first. `pack_intersections` performs exactly the gather the staging loop
performs -- one scattered read of uv/conic/opacity/colour per intersection --
so its runtime bounds what packing can save:

| | ms | share |
|---|---:|---:|
| full fwd+bwd, 100k @ 800x800, 956k intersections | 43.98 | — |
| `pack_intersections` alone | **1.02** | **2.3%** |

**The gather is 2.3% of the step.** Packing would replace the backward's share
of it with a contiguous read while paying 1.02 ms for the pack itself, so the
net is zero at best and negative at worst.

The reason is already in the forward kernel's own comment: cooperative staging
turns "5 scattered device loads per (pixel, gaussian)" into "one per
(threadgroup, gaussian)". The gather was amortised years of GPU folklore ago;
packing amortises it again and there is nothing left to win. msplat's design is
sound, it simply buys them far less than its prominence suggests -- and this is
the second time their most eye-catching kernel has turned out to target something small here.

The kernel stays in `csrc/rasterize.metal`, unused, because it is the honest
way to have asked the question and it re-measures in one command on a machine
where the answer might differ.

**So the backward is compositing arithmetic and atomics, not data movement.** Closing the remaining per-step gap needs an algorithmic change, not a memory-layout one -- msplat's
`rasterize_backward_chunked_kernel` with its `prefix_T_buf` / `chunk_T_buf` /
`after_C_buf` checkpointing is the visible candidate, and it is a large change
with no measurement behind it yet.

### Antialiasing: a wash on the benchmark, transformative off it

`--antialias` (Mip-Splatting's 2D Mip filter, as in gsplat's antialiased mode)
rescales opacity by `sqrt(det_before / det_after)` so the 0.3px low-pass
dilation preserves energy instead of eroding sub-pixel gaussians.

**On the standard 8-scene protocol it is worth +0.01 dB.** Two wins, three
losses, three inside the noise floor, and free in wall-clock (-0.1%):

| scene | base | antialias | delta |
|---|---:|---:|---:|
| ficus | 28.68 | **29.92** | +1.23 |
| chair | 32.43 | 32.64 | +0.21 |
| hotdog | 36.47 | 36.59 | +0.12 |
| drums | 24.94 | 24.96 | +0.02 |
| lego | 30.75 | 30.60 | -0.16 |
| materials | 27.95 | 27.60 | -0.35 |
| ship | 29.43 | 28.96 | -0.48 |
| mic | 32.28 | 31.78 | -0.50 |
| **mean** | **30.37** | **30.38** | **+0.01** |

On that evidence it would have been rejected. **The protocol was the problem.**
It trains at 800px and scores at 800px, and the filter exists for rendering at a
resolution the model was *not* trained at. Scoring the same exported model at
several resolutions -- Mip-Splatting's own protocol -- gives a different answer:

| delta (antialias - baseline) | 800px (trained) | 400px | 200px |
|---|---:|---:|---:|
| ficus | +1.09 | +3.05 | +6.14 |
| mic | -0.39 | +4.23 | +7.24 |
| lego | -0.06 | +3.07 | +6.66 |
| **mean** | **+0.21** | **+3.45** | **+6.68** |

The mechanism is scale robustness. Degradation from 800px down to 200px:

| scene | baseline | antialias |
|---|---:|---:|
| ficus | -7.01 dB | -1.96 dB |
| mic | -10.71 dB | -3.08 dB |
| lego | -8.73 dB | -2.01 dB |

**3.7x more robust, for free.** A baseline model is fitted to one sampling rate
and falls apart away from it; the compensated model does not.

**Why it is off by default.** The compensation is applied at **render time**, and the exported `.ply` stores
the RAW opacity. A model trained with `--antialias` therefore renders ~2.3 dB
worse in any viewer that does not apply the same filter, which is every viewer
that is not this one. Mip-Splatting avoids that by *fusing* its filter into the
exported gaussians -- but only its **3D** filter can be fused, because it is
view-independent. The 2D Mip filter is a screen-space quantity and is
inherently render-time, so ours cannot be baked into a ply at all.

So: off by default, worth turning on whenever the renders will not be at
training resolution, and the honest path to making it default is the 3D
smoothing filter, which is exportable. The other half of Mip-Splatting, its 3D filter, is measured in the next entry.

**Lesson.** 12. **A negative result is only as good as the regime it was measured in.** The
    8-scene protocol said +0.01 dB and would have retired this technique. The
    same code is worth +6.7 dB one axis over. Before rejecting a technique, ask
    what regime it was designed for and whether the benchmark exercises it --
    ours did not, because train and test resolution were identical by
    construction.

### Mip-Splatting's 3D filter: the bakeable half is the weak half

`--antialias` (the 2D Mip filter) is worth +3.45 dB at 400px and +6.68 dB at
200px render resolution for no wall-clock cost, and cannot be the default: it
is applied at render time while the ply stores raw opacity, so such a model
renders ~2.3 dB worse in any viewer without it.

The 3D smoothing filter was built specifically to solve that. It band-limits
each gaussian to the sampling rate of the views that see it, which is a
property of the gaussian rather than the camera, so it folds into the exported
scales and opacity.

**The baking works.** A lego model trained with `--filter-3d` and exported
scores 20.944 dB rendered with NO filter applied, against 20.94 from the
trainer's own filtered eval -- 0.004 dB apart. Any viewer renders it correctly
with no cooperation. That part of the premise held.

**The benefit does not.** Eight scenes at training resolution:

| | mean PSNR | total wall |
|---|---:|---:|
| baseline | 30.37 | 22.6 min |
| `--filter-3d` | **29.91** | 32.5 min |

-0.46 dB, negative on all eight scenes, and +44% wall-clock for the periodic
camera pass. And off-resolution, where it was supposed to pay for itself:

| render res | ficus | lego | mean | 2D filter, for comparison |
|---|---:|---:|---:|---:|
| 800px (trained) | -0.43 | -0.29 | **-0.36** | +0.21 |
| 400px | +0.26 | +0.60 | **+0.43** | **+3.45** |
| 200px | +0.53 | +0.50 | **+0.52** | **+6.68** |

**About 13x weaker than the 2D filter at 200px**, while costing what the 2D
filter does not: quality at training resolution and 44% of wall-clock.

So the premise that motivated building it is wrong. The half that can be baked
into a ply is the half that does not do the work. There is no route to making
antialiasing a default through the 3D filter, and `--antialias` stays opt-in.

`--filter-3d` is kept as a flag rather than removed: it is a correct
implementation, it does help slightly off-resolution, it is the honest record
of the question, and it re-measures in one command if a future operating point
changes the answer. It is off by default and should stay off.

**Untried, and the obvious next question:** Mip-Splatting uses BOTH filters
together, and only the combination has ever been published. The two may be
more than additive. But the combination still cannot be baked, because the 2D
half is screen-space, so it would not solve the default problem either -- it
could only be a better opt-in.

**Lesson.** 17. **Check that the property you are buying is attached to the benefit you
    want.** The 3D filter was chosen over the 2D one entirely because it
    exports, and it does export -- exactly as designed, verified to 0.004 dB.
    It simply does not carry the benefit that made exporting worth wanting.
    Two hours of implementation would have been avoided by measuring the 3D
    filter alone before building the bake path, which was the expensive part.

### msplat's run-to-run noise floor is up to 3.35 dB, and grows with training

msplat exposes **no seed flag**, so identical configurations are independent
samples. Measuring 36 paired repeats (3 scenes x 2 variants x 6 rungs):

| rung | median delta | max delta |
|---:|---:|---:|
| 500 | 0.01 | 0.03 |
| 1,000 | 0.03 | 0.31 |
| 2,000 | 0.21 | 2.28 |
| 4,000 | 0.40 | 1.96 |
| 7,000 | 0.84 | 2.35 |
| 15,000 | **1.06** | **3.35** |

It is essentially deterministic before densification engages and increasingly
divergent after, which points at its adaptive density control as the source.

For contrast, on the same protocol: **metal-gauss 0.22 dB** worst case across two
scenes and six rungs, **Brush 0.74 dB** on lego. 

spirula's spread is likewise scene-dependent, and measured on mic alone
(stdev 0.079 dB over three runs) it looks far tighter than it is:

| spirula, 15000 it | spread |
|---|---:|
| mic (n=3) | 0.148 |
| ficus | 0.34 |
| drums | 0.67 |
| **lego** | **1.27** |

A noise floor is a property of an **(implementation, scene)** pair, not of an
implementation. 

**Lesson.** A noise floor is a property of an implementation, not of a protocol.
Before comparing two numbers from a third-party trainer, measure what that
trainer does when asked the same question twice: a difference smaller than its
own spread is not a result.

The one piece of luck: msplat's front-holding points sit at 500-2000 iterations
where its spread is smallest, so the noise does not undermine the part of its
result that actually reaches the Pareto front.

## Levers rejected, in brief

One line of evidence each. The entries above are the ones that needed more.

Kept here because a measured negative is cheaper to read than to rediscover.
Every number below is from `bench/quick.py` on an idle machine.

| lever | idea | measured | verdict |
|---|---|---|---|
| **Selective Adam, as first built** | update only frustum-visible gaussians (gsplat) | In torch: slower than dense Adam (15 gather/scatter launches, or a chain of `where` temporaries). As one Metal kernel it halves Adam (room1 600k: 9.60 -> 4.75 ms), but skipping **every** group with **per-gaussian** bias correction cost **-0.86 dB, worse on 8/8** NeRF-synthetic scenes at 7k (`nerf_synthetic_7000_selective_adam_all_row.json` vs `_control.json`). Per-gaussian correction gives a relocated gaussian (steps reset to 0) ~3x smaller first steps than dense Adam does; skipping geometry freezes its momentum and the MCMC regularisers while hidden. Probes on chair / hotdog: SH-only per-gaussian +0.08 / -1.14, all groups global -0.47 / -0.43 | superseded: `--selective-adam` now skips only `sh_dc`/`sh_rest`, bias-corrected with the global step count: **+0.03 dB [-0.04, +0.10]** over 8 scenes (`nerf_synthetic_7000_selective_adam.json`), Adam -51% |
| **Per-image appearance correction** | learnable gain+bias per training image to absorb the capture's documented auto-exposure (gsplat bilateral grid / NeRF-W) | **−0.81 dB @t1, −2.70 dB @t2**, +57–66% wall. Gets *worse* with more steps: the 6 per-image parameters absorb signal the gaussians should learn | behind `--appearance`, default off |
| **fp16 threadgroup staging** | halve staged attribute traffic (36B→18B/gaussian) | forward **15% slower** (133 vs 115 ms @900×1600), backward neutral, plus 3.3e-3 precision loss. `half3` is 6 bytes — unaligned — and the pack/unpack in the inner loop costs more than the bandwidth saved | reverted |

| **Analytic SSIM gradient (in torch)** | replace autograd's conv-chain backward with the closed form (fused-ssim approach) | value bit-identical, gradient cosine 1.00000000 -- and **69 ms vs 43 ms**. Each elementwise term of the closed form materialises a full-res temporary; torch's conv backward is better optimised than the hand-rolled chain | **SUPERSEDED** -- the prediction in that entry held. Ported to a fused Metal kernel (`csrc/ssim.metal`, tail only, convs left in torch) the same derivation gives 69.2 -> 38.4 ms fwd+bwd at 900x1600, 1.80x, cosine still 1.00000000. A negative result in one execution model, not a wrong derivation. |
| **Morton reordering** | re-sort gaussians along a space-filling curve for cache locality (LichtFeld #1753) | not attempted: **their own PR reports −1.85%** (167.5s → 164.4s). Our wall-clock noise floor is ±8%, so the effect is unmeasurable here | skipped on their evidence |

| **Two pixels per thread (backward)** | halve gradient write-backs by giving each thread two pixels (LichtFeld PR #1780 "GUT", worth 25% of their step) | **+15% slower** (23.62 vs 20.52 ms isolated, 13% IQR). Their win assumes PER-PIXEL atomics; we already reduce 32 lanes to 1 atomic via `simd_sum`, so halving simdgroups saves little, while each thread now carries two transmittances and two suffix accumulators - register pressure costs more than the atomics saved | reverted |

| **Threadgroup-level gradient reduction** | cut atomics 8x (256 threads = 8 simdgroups, each issuing its own atomic per gaussian) | Not built. Diagnostic first: replaced the atomic adds with plain racy stores (same memory traffic, zero contention) and measured **13.02 vs 15.16 ms** -- atomics are only **14%** of `rasterize_backward`. A threadgroup reduction cannot reach zero, so realistic gain is ~7%. Also explains why two-pixels-per-thread failed | not attempted, measured ceiling too low |
| **Skipping staged batches above the stop index** | the backward walks the FULL tile list for every pixel and only predicates; whole batches beyond the threadgroup's max stop-index contribute to no lane | **0.1% of staged batches are skippable.** Mean tile list is 128.3 intersections and the mean per-tile max stop-index is 126.0 -- within a 16x16 tile some pixel is almost always in a gap where transmittance never saturates | not attempted |
| **Command-buffer batching** | ~20 `dispatch_sync` + `commit()` per step (Adam alone dispatches 6x, one per param group) looked like pure driver overhead the small 270p kernels cannot amortise | Measured the MARGINAL cost per dispatch: 20 back-to-back trivial dispatches cost 1.15-1.31 ms, so **~60 us each**, not the ~500 us a single-dispatch measurement suggests (that figure is the `synchronize` itself). 20 dispatches is **~1.2 ms of a 52 ms step, 2.3%** -- and batching would not remove all of it | not built |
| **Smaller staged tile buffer (STAGE 256 -> 128)** | halve threadgroup memory from 11.0 to 5.5 KB to raise threadgroup residency above two | Appeared to win 23%, then measured **3% SLOWER** once the GPU clock ramp was controlled. See the bimodality section above | reverted |
| **Cached `torch.arange` in binning** | torch.profiler put `aten::arange` at 11 ms/step across 6 calls | on/off measured **66.8 vs 66.0 ms** -- a wash. The intersection count changes every step, so the one arange that matters never hits the cache | reverted |

| **Exact ellipse-vs-tile test in torch** (gsplat AccuTile, PR #927) | our tile bounds are the ellipse's axis-aligned box, so rotated/elongated gaussians claim corner tiles their ellipse never reaches | The work reduction is REAL and bit-exact: **37.5% of tile-gaussian pairs removed**, 9.88 -> 6.06 tiles per gaussian, image and alpha bit-identical. But in torch the test costs more than it saves -- binning **15.41 -> 30.02 ms** (+14.61 for the gathers and the quadratic minimisation) against raster fwd -2.28 and bwd -5.27, a net **+7 ms** | kept, default OFF; enable once binning is a kernel |

| **Speedy-Splat soft/hard pruning** | their two biggest wins (3.14x and 6.71x overall), removing low-contribution gaussians during and after densification | **Does not apply to a fixed-budget MCMC trainer.** Measured on our own 10k room1 model: only **0.28%** of 600k gaussians sit below the rasteriser's 1/255 opacity cutoff. There is no dead weight to prune, because MCMC relocation already recycles it every 100 steps. Their gains come from 3DGS-ADC over-densifying; we cap the count by construction, and cutting further would be a capacity decision, not a free speedup | not applicable |

**On `rasterize_backward`, the largest single kernel (86 ms in situ at 1600p, 35% of the
step): it is compositing-loop bound.** Atomics are 14%, skippable work is 0.1%, and giving each
thread two pixels made it slower. There is no kernel-level lever left; the only remaining
direction is doing fewer gaussian-pixel evaluations, which means tighter bounds or opacity
culling, and those change the image rather than just the speed.

