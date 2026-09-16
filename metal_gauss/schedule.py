"""Training schedules. Pure arithmetic, no third-party imports.

Kept in its own module precisely because it has no dependencies. The tests that
pin this schedule run in CI on a machine with no GPU and no torch -- CI installs
pytest and nothing else, deliberately, since a CI job that cannot pass is worse
than no CI job. While `auto_budget` lived in train.py those tests had to import
it through a module that imports numpy and torch at load time, so they failed on
every push with ModuleNotFoundError while the other 18 passed.

The rule is measured, and the evidence is in the docstring below rather than in
a commit message, because this is the number a user silently inherits.
"""
from __future__ import annotations

import math


def _scale_step(value: int, factor: float, *, zero_is_disabled: bool = False) -> int:
    """Scale a step count or interval while preserving disabled zero values."""
    if value == 0 and zero_is_disabled:
        return 0
    return max(1, int(value * factor))


def resolve_training_schedule(*, steps: int, steps_scaler: float,
                              budget: int | None, start_active: int,
                              relocate_every: int, eval_every: int,
                              sh_warmup: int, resolution_schedule: int | None,
                              filter_3d_every: int,
                              export_every: int) -> dict[str, int | float]:
    """Resolve the step-dependent training settings in one place.

    ``--steps-scaler`` is intended to make a short run a proportional miniature
    of a longer run. That means it must be applied before defaults derived from
    the run length, especially ``auto_budget`` and the default resolution
    schedule. Explicit capacity remains an override because it is not a
    step-domain schedule.

    Zero is a meaningful sentinel for ``sh_warmup``, ``filter_3d_every`` and
    ``export_every``: it disables that feature or requests the corresponding
    fallback, so those values are not rounded up to one.
    """
    if not math.isfinite(steps_scaler) or steps_scaler <= 0:
        raise ValueError("steps_scaler must be finite and greater than zero")

    resolved_steps = _scale_step(steps, steps_scaler)
    resolved_resolution = (
        _scale_step(resolution_schedule, steps_scaler)
        if resolution_schedule is not None
        else max(1, resolved_steps // 3)
    )
    resolved_budget = auto_budget(resolved_steps) if budget is None else budget

    resolved_start_active = start_active
    if resolved_start_active > resolved_budget:
        # The parameter tensors are preallocated at `budget`; an active count
        # above that would read past the end.
        resolved_start_active = max(1000, resolved_budget // 2)

    return {
        "steps": resolved_steps,
        "budget": resolved_budget,
        "start_active": resolved_start_active,
        "relocate_every": _scale_step(relocate_every, steps_scaler),
        "eval_every": _scale_step(eval_every, steps_scaler),
        "sh_warmup": _scale_step(sh_warmup, steps_scaler,
                                  zero_is_disabled=True),
        "resolution_schedule": resolved_resolution,
        "filter_3d_every": _scale_step(filter_3d_every, steps_scaler,
                                         zero_is_disabled=True),
        "export_every": _scale_step(export_every, steps_scaler,
                                     zero_is_disabled=True),
    }


def step_cost(step: int, *, steps: int, num_downscales: int, resolution_schedule: int,
              grow: bool, start_active: int, budget: int, grow_until_frac: float) -> float:
    """Relative cost of one training step: pixels times active splats.

    Both factors are normalised to the end of the run, where the cost is 1.
    They are the trainer's own curriculum: `--num-downscales` starts at
    1/2^N resolution and doubles on `--resolution-schedule`, and `--grow` ramps
    capacity from `--start-active` to `--budget` over the first
    `--grow-until-frac` of the run.

    A downscale level costs HALF the next one, not a quarter, even though it
    has a quarter of the pixels. Measured on lego (600 steps, 30k splats), the
    three levels ran at 13, 28 and 40 ms per step against pixel ratios of
    1:4:16 -- a small frame is bound by dispatch, Adam and Python rather than
    by pixels, so the cost tracks the side length. Scaling by pixels predicted
    208 ms for a step that took 40, and a 2.8x too-long ETA.

    Still an approximation: Adam runs over the whole budget rather than the
    active slice, and the constant differs with splat count and resolution.
    """
    lvl = (max(0, num_downscales - step // max(1, resolution_schedule))
           if num_downscales > 0 else 0)
    pixels = 0.5 ** lvl
    if grow and budget > 0:
        grow_until = max(1, int(grow_until_frac * steps))
        active = start_active + (budget - start_active) * min(1.0, step / grow_until)
        return pixels * active / budget
    return pixels


def remaining_s(step: int, *, steps: int, recent_step_s: float, eval_every: int,
                eval_s: float, **schedule) -> float | None:
    """Seconds of training left, with the curriculum applied. None if not yet known.

    Recent steps set the price of one unit of work and `step_cost` says how many
    units are left, so the constant cancels and nothing needs calibrating. Evals
    are added separately: a 200-view evaluation is not a step.
    """
    if recent_step_s <= 0.0:
        return None
    now = step_cost(step, steps=steps, **schedule)
    if now <= 0.0:
        return None
    work = sum(step_cost(s, steps=steps, **schedule) for s in range(step + 1, steps + 1))
    evals = len({s for s in range(step + 1, steps + 1)
                 if (eval_every > 0 and s % eval_every == 0) or s == steps})
    return recent_step_s * work / now + evals * eval_s


def format_duration(seconds: float) -> str:
    """A duration for a progress line: 45s, 3m 20s, 1h 04m."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def auto_budget(steps: int) -> int:
    """Capacity that suits the step budget.

    Measured on lego, scored on the official test split (bench/results/
    capacity_sweep.json). A fixed 300k -- the old default -- is not merely slow
    at short budgets, it is WORSE, because that many randomly-initialised
    splats cannot organise in a few thousand steps:

        iters    30k budget       100k budget      300k budget
         1000  22.89 @ 0.88min  23.89 @ 1.13min  17.09 @ 2.25min
         2000  25.11 @ 1.41     26.32 @ 1.80     22.72 @ 3.96
         4000  26.93 @ 2.39     28.49 @ 2.99     26.48 @ 6.57
         7000       --          30.26 @ 5.10     29.61 @ 9.63
        15000       --          32.83 @ 10.92    33.73 @ 18.27

    100k dominates 300k on BOTH axes everywhere up to 7k steps. Only at 15k
    does the larger capacity earn its keep, and then it buys 0.9 dB for 1.7x
    the wall-clock -- a genuine trade rather than a free win, so both remain on
    the Pareto front and the user can pick with --budget.

    The rising tail was an extrapolation when written -- the table above stops
    at 15k steps and 300k splats -- and has since been tested at its far end:

        lego, 30k steps    300k budget  35.48 dB @ 42.9 min
                           500k budget  35.88 dB @ 35.3 min   <- this function

    +0.40 dB, and above the published 3DGS baseline (35.84). The wall-clock
    difference is kernel work between the two runs, not capacity. The tail
    between 15k and 30k steps remains interpolated rather than measured.
    """
    if steps < 1_000:
        # Very short runs want far less capacity than the 100k floor. Measured
        # on all 8 Blender scenes at two step counts, scored on the official
        # test split (bench/results/shortrun_*.json):
        #
        #     steps   100k mean   30k mean   delta   scenes 30k wins   wall
        #       200     11.43       15.65    +4.21        8/8          0.74x
        #       500     18.85       21.53    +2.69        7/8          0.71x
        #
        # Better AND ~27% faster. It flips by 1000 steps (30k 22.89 vs 100k
        # 23.89 on lego), which is where the threshold sits. The one regression
        # is hotdog at 500 steps, -0.66 dB.
        #
        # This is deliberately 8-scene evidence: the same rule inferred from
        # lego alone would have put the threshold at 200, because lego is the
        # scene where the effect is weakest (+0.14 dB at 500 against +8.01 on
        # mic). Capacity effects here vary several-fold between scenes.
        return 30_000
    # The plateau used to end at 10_000, which put 300k splats on a 15_000-step
    # run. Measured at 15_000 steps with budget as the ONLY variable, three
    # scenes, all three metrics:
    #
    #   scene   100k PSNR/SSIM/LPIPS      300k PSNR/SSIM/LPIPS      wall
    #   ficus   30.66 / 0.980 / 0.019     26.76 / 0.968 / 0.041     2.0x
    #   drums   25.50 / 0.949 / 0.052     25.16 / 0.948 / 0.050     2.0x
    #   lego    33.01 / 0.969 / 0.026     34.24 / 0.977 / 0.016     2.0x
    #
    # 100k is +1.0 dB better on the 3-scene mean AND 2x faster. ficus prefers
    # it on every metric by a wide margin; drums is a tie at half the cost;
    # lego is the one scene that genuinely wants 300k, and pays double the
    # wall-clock for +1.2 dB.
    #
    # Moving the plateau to 15_000 rather than deleting the ramp is deliberate:
    # it leaves auto_budget(30_000) at the 500k cap, which is the configuration
    # the calibration anchor was measured under (lego 35.88 dB against
    # 3DGS-MCMC's published 36.01). Deleting the ramp would have invalidated a
    # published number to fix a different one.
    #
    # STILL UNMEASURED above 15_000 except on lego at 30_000. The ramp's shape
    # between 15k and 30k is interpolation, and the earlier version of this
    # tail was validated on lego alone -- which is exactly how it came to put
    # 300k on ficus and cost 3.9 dB.


# ------------------------------------------------- shared step (see note)

# bench/step_profile.py drifted from this file THREE times -- it profiled
# torch.optim.Adam after FusedAdam became the default, kept a per-step SH cat
# the split layout had removed, and left the pose on the GPU after render()
# started wanting it on the host. Each time it under-reported the very lever
# that had just landed. The construction below is the single definition both
# use, so a profile cannot describe an optimiser or a call the trainer does not
# make.
    if steps <= 15_000:
        return 100_000
    return min(500_000, 100_000 + (steps - 15_000) * 40)
