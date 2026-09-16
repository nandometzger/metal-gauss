"""CPU-only tests for the trainer's step-dependent schedule resolution."""

from __future__ import annotations

import pytest

from metal_gauss.schedule import format_duration, remaining_s, resolve_training_schedule


def _resolve(**overrides):
    values = {
        "steps": 30_000,
        "steps_scaler": 0.1,
        "budget": None,
        "start_active": 150_000,
        "relocate_every": 100,
        "eval_every": 1_000,
        "sh_warmup": 1_000,
        "resolution_schedule": None,
        "filter_3d_every": 0,
        "export_every": 0,
    }
    values.update(overrides)
    return resolve_training_schedule(**values)


# ------------------------------------------------------------------- the ETA

def _eta(**overrides):
    """A flat run by default: constant resolution, constant capacity, no evals."""
    values = {
        "steps": 1_000, "recent_step_s": 0.01, "eval_every": 10_000, "eval_s": 0.0,
        "num_downscales": 0, "resolution_schedule": 1_000, "grow": False,
        "start_active": 50_000, "budget": 100_000, "grow_until_frac": 0.7,
    }
    values.update(overrides)
    step = values.pop("step")
    return remaining_s(step, **values)


def test_a_flat_run_is_just_the_steps_that_are_left():
    assert _eta(step=0) == pytest.approx(10.0)
    assert _eta(step=900) == pytest.approx(1.0)


def test_the_last_step_has_nothing_left():
    assert _eta(step=1_000) == 0.0


def test_coarse_to_fine_makes_the_early_steps_a_bad_guide():
    """Training starts downscaled, so a flat guess is wildly optimistic.

    Steps 1-999 run at level 2, 1000-1999 at level 1 and the rest at full size,
    and a level costs half the next one, so from step 10 the work left is
    (989/4 + 1000/2 + 1001) / (1/4) = 6993 steps at the current price.
    """
    eta = _eta(step=10, steps=3_000, num_downscales=2, resolution_schedule=1_000)
    assert eta == pytest.approx(69.93, abs=0.05)
    assert 2_990 * 0.01 == pytest.approx(29.9), "what a flat guess would have said"


def test_capacity_growth_raises_the_estimate():
    """Splats ramp from 50k to 100k over the first 70% of the run."""
    assert _eta(step=10, grow=True) == pytest.approx(16.17, abs=0.05)


def test_the_evaluations_still_to_come_are_counted():
    """A 200-view eval is not a step; four of them at 20 s is 80 s on top."""
    assert _eta(step=100, steps=2_000, eval_every=500, eval_s=20.0) == pytest.approx(99.0)


def test_an_unknown_step_time_gives_no_estimate():
    assert remaining_s(10, steps=1_000, recent_step_s=0.0, eval_every=100, eval_s=0.0,
                       num_downscales=0, resolution_schedule=1_000, grow=False,
                       start_active=1, budget=1, grow_until_frac=0.7) is None


@pytest.mark.parametrize("seconds, want", [
    (0.4, "0s"), (5.0, "5s"), (65.0, "1m 05s"), (599.0, "9m 59s"), (3725.0, "1h 02m"),
])
def test_format_duration(seconds, want):
    assert format_duration(seconds) == want


def test_steps_scaler_resolves_defaults_from_scaled_steps():
    got = _resolve()

    assert got == {
        "steps": 3_000,
        "budget": 100_000,
        "start_active": 50_000,
        "relocate_every": 10,
        "eval_every": 100,
        "sh_warmup": 100,
        "resolution_schedule": 1_000,
        "filter_3d_every": 0,
        "export_every": 0,
    }


def test_steps_scaler_scales_explicit_step_intervals_but_not_budget():
    got = _resolve(budget=500_000, resolution_schedule=2_000,
                   filter_3d_every=250, export_every=100)

    assert got["steps"] == 3_000
    assert got["budget"] == 500_000
    assert got["resolution_schedule"] == 200
    assert got["filter_3d_every"] == 25
    assert got["export_every"] == 10


def test_zero_intervals_remain_disabled():
    got = _resolve(sh_warmup=0, filter_3d_every=0, export_every=0)

    assert got["sh_warmup"] == 0
    assert got["filter_3d_every"] == 0
    assert got["export_every"] == 0


def test_unit_scaler_keeps_original_defaults_and_explicit_values():
    got = _resolve(steps_scaler=1.0, resolution_schedule=777,
                   budget=123_000, start_active=100_000, relocate_every=37,
                   eval_every=211,
                   sh_warmup=59, filter_3d_every=31, export_every=43)

    assert got == {
        "steps": 30_000,
        "budget": 123_000,
        "start_active": 100_000,
        "relocate_every": 37,
        "eval_every": 211,
        "sh_warmup": 59,
        "resolution_schedule": 777,
        "filter_3d_every": 31,
        "export_every": 43,
    }


def test_steps_scaler_must_be_positive_and_finite():
    for value in (0.0, -0.5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="greater than zero"):
            _resolve(steps_scaler=value)
