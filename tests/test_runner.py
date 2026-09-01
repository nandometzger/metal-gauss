"""The two harness bugs that produced wrong published numbers, as tests.

Neither bug was in a kernel. Both were a harness telling the trainer one thing,
the trainer doing another, and the JSON recording the harness's intent. These
tests run without a GPU: they exercise argv construction and the requested-vs-
resolved check directly, which is exactly where both bugs lived.

**Nothing here may import `metal_gauss.train`.** CI installs pytest and nothing
else -- no numpy, no torch -- because a CI job that needs a GPU cannot pass and
is worse than no job at all. The auto_budget tests below once imported the
schedule through `train`, which pulls numpy at module load, so they failed on
every push for days while the other 18 passed. The schedule now lives in
`metal_gauss.schedule`, which has no third-party imports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.runner import (RunDiverged, RunFailed, build_cmd,  # noqa: E402
                          check, gpu_competitors, require_gpu_exclusive, run)


def _report(**resolved):
    base = {"steps": 7000, "budget": 100_000, "num_downscales": 2,
            "start_active": 50_000, "seed": 0, "steps_scaler": 1.0}
    base.update(resolved)
    return {"schema": 1, "resolved": base, "env": {}, "metrics": {"psnr": 30.0}}


# --------------------------------------------------------------------------
# The regression: nerf_synthetic_sweep.py declared --budget default=300_000 and
# forwarded it to every child, so auto_budget() never ran and an 8-scene table
# labelled "old vs new defaults" held budget fixed at 300k in both arms.
# --------------------------------------------------------------------------

def test_unspecified_knob_is_not_forwarded():
    """A harness must not inject a default for a knob it was not asked about.

    This is the bug itself. build_cmd has no defaults of its own precisely so
    that "I didn't ask for a budget" cannot turn into "--budget 300000".
    """
    cmd = build_cmd({"blender": "/data/lego", "steps": 7000}, Path("/tmp/r.json"))
    assert "--budget" not in cmd
    assert "--num-downscales" not in cmd
    assert cmd.count("--report") == 1


def test_divergence_between_requested_and_actual_is_fatal():
    """Request 100k, have the trainer report 300k -> must be caught.

    The historical failure produced a plausible number under the wrong
    protocol and nothing complained for a week.
    """
    bad = check({"budget": 100_000}, _report(budget=300_000))
    assert bad and "budget" in bad[0]
    assert "100000" in bad[0].replace("_", "") or "100000" in bad[0]


def test_agreement_is_silent():
    assert check({"budget": 100_000, "steps": 7000}, _report()) == []


def test_unspecified_knob_is_still_recorded():
    """"Unspecified" must mean "trainer chose, and we wrote down what", not
    "unknown". Recording only what the harness passed is what hid both bugs."""
    rep = _report(budget=100_000)
    assert check({"steps": 7000}, rep) == []          # not requested, not checked
    assert rep["resolved"]["budget"] == 100_000        # but still on the record


# --------------------------------------------------------------------------
# Documented rewrites main() performs, which must be surfaced but not fatal.
# --------------------------------------------------------------------------

def test_steps_scaler_rewrite_is_a_transform_not_a_bug(capsys):
    bad = check({"steps": 7000, "steps_scaler": 0.5}, _report(steps=3500,
                                                             steps_scaler=0.5))
    assert bad == []
    assert "transform" in capsys.readouterr().out


def test_start_active_clamp_is_a_transform():
    assert check({"budget": 10_000, "start_active": 150_000},
                 _report(budget=10_000, start_active=5_000)) == []


def test_string_and_numeric_forms_compare_equal():
    """Values cross the process boundary as argv strings; "7000" == 7000."""
    assert check({"steps": "7000"}, _report(steps=7000)) == []


# --------------------------------------------------------------------------
# Structural requirements on the report.
# --------------------------------------------------------------------------

def test_report_without_resolved_block_is_refused():
    with pytest.raises(RunFailed):
        check({"budget": 1}, {"schema": 1, "metrics": {"psnr": 30.0}})


def test_missing_report_raises_rather_than_returning_none(tmp_path):
    """splat-apple exited 0 on failure and got counted as a run. A missing
    report is a failure regardless of exit code."""
    with pytest.raises(RunFailed):
        run({"steps": 1}, report=tmp_path / "never_written.json",
            cwd=tmp_path)          # no trainer here, so nothing is written


def test_stale_report_is_not_reused(tmp_path, monkeypatch):
    """A leftover report from a previous run must never be read as this run's
    result -- that would silently republish an old number."""
    stale = tmp_path / "r.json"
    stale.write_text(json.dumps(_report(budget=999)))
    with pytest.raises(RunFailed):
        run({"steps": 1}, report=stale, cwd=tmp_path)
    assert not stale.exists()


def test_bool_knobs_become_flag_or_no_flag():
    assert "--fused-adam" in build_cmd({"fused_adam": True}, Path("/tmp/r"))
    assert "--no-fused-adam" in build_cmd({"fused_adam": False}, Path("/tmp/r"))


def test_boolean_divergence_true_to_false_is_fatal():
    """A requested true flag must not be reported as successfully resolved false."""
    bad = check({"fused_adam": True}, _report(fused_adam=False))
    assert bad and "fused_adam" in bad[0]


def test_boolean_divergence_false_to_true_is_fatal():
    """A requested false flag must not be reported as successfully resolved true."""
    bad = check({"fused_adam": False}, _report(fused_adam=True))
    assert bad and "fused_adam" in bad[0]


def test_boolean_agreement_is_silent():
    assert check({"fused_adam": True}, _report(fused_adam=True)) == []


def test_steps_divergence_without_a_scaler_is_NOT_excused():
    """The allow-list must check that its trigger actually fired.

    Listing `steps` as "the steps-scaler may rewrite it" would otherwise wave
    through a run that used 3500 steps when 7000 were requested and no scaler
    was set -- the exact bug class this module exists to catch.
    """
    bad = check({"steps": 7000}, _report(steps=3500, steps_scaler=1.0))
    assert bad and "steps" in bad[0]


def test_start_active_clamp_only_excused_when_it_actually_exceeded_budget():
    bad = check({"budget": 300_000, "start_active": 150_000},
                _report(budget=300_000, start_active=999))
    assert bad, "an unexplained start_active change must not be excused"


# --------------------------------------------------------------------------
# auto_budget's capacity schedule. Both ends of it have shipped wrong once.
# --------------------------------------------------------------------------

def test_auto_budget_plateau_covers_15000_steps():
    """The plateau must reach 15000, not stop at 10000.

    Measured at 15000 steps with budget as the only variable: 100k beats 300k
    by +3.90 dB on ficus and +0.34 on drums and is 2x faster on both, losing
    only on lego (-1.23). The three-scene mean favours 100k by 1.0 dB. A
    plateau ending at 10000 put 300k on every 15000-step run.
    """
    from metal_gauss.schedule import auto_budget
    assert auto_budget(10_000) == 100_000
    assert auto_budget(15_000) == 100_000


def test_auto_budget_still_reaches_500k_at_30000():
    """The calibration anchor must not move.

    lego at 30000 steps and the 500k cap scores 35.88 dB against 3DGS-MCMC's
    published 36.01. Flattening the ramp to fix the 15000-step case would have
    invalidated that number, so the plateau was moved rather than the ramp
    deleted.
    """
    from metal_gauss.schedule import auto_budget
    assert auto_budget(30_000) == 500_000


def test_auto_budget_short_run_tier_intact():
    from metal_gauss.schedule import auto_budget
    assert auto_budget(999) == 30_000
    assert auto_budget(1_000) == 100_000


# --------------------------------------------------------------------------
# Timing hygiene. A guard that can never pass is worse than no guard.
# --------------------------------------------------------------------------

def test_exclusivity_guard_does_not_match_itself():
    """The naive form matched the wrapper shell and the benchmark runner,
    reporting 4 competitors during a run that had one. Matching argv instead of
    the executable name is the bug; this pins the fix."""
    assert gpu_competitors() == [] or all(
        "python" not in h and "zsh" not in h for h in gpu_competitors())


def test_exclusivity_guard_raises_only_when_a_competitor_exists(monkeypatch):
    """State-independent: the first draft of this test called the real guard and
    failed the moment a benchmark was legitimately running in another shell,
    which is a test asserting "the machine is idle", not "the guard works"."""
    import bench.runner as R
    monkeypatch.setattr(R, "gpu_competitors", lambda *a, **k: [])
    R.require_gpu_exclusive()                       # silent when alone

    monkeypatch.setattr(R, "gpu_competitors", lambda *a, **k: ["brush (pid 1)"])
    with pytest.raises(RunFailed) as e:
        R.require_gpu_exclusive()
    assert "brush" in str(e.value) and "exclusive" in str(e.value)


# --------------------------------------------------------------------------
# Export filenames. A collision here does not raise -- it silently attaches
# one run's metrics to another run's row, so the rule gets a test.
# --------------------------------------------------------------------------

def test_export_tag_separates_msplat_variants():
    """Running the stock arm of an A/B overwrote the scaled arm's plys on drums
    and ficus, leaving only PSNR (already in JSON) and losing SSIM/LPIPS."""
    from bench.compare.pareto import export_tag
    stock = export_tag("ficus", "msplat", 7000, msplat_stock=True)
    scaled = export_tag("ficus", "msplat", 7000, msplat_stock=False)
    assert stock != scaled
    assert "stock" in stock and "scaled" in scaled


def test_export_tag_separates_scenes_and_iters():
    from bench.compare.pareto import export_tag
    t = lambda s, n: export_tag(s, "metal-gauss", n, msplat_stock=False)
    assert len({t("lego", 7000), t("drums", 7000), t("lego", 15000)}) == 3


def test_export_tag_has_no_variant_suffix_for_non_msplat():
    """The variant is an msplat knob; stamping it on other rows would repeat
    the `budget: 300000` mistake of recording a setting that had no effect."""
    from bench.compare.pareto import export_tag
    assert "stock" not in export_tag("lego", "brush", 7000, msplat_stock=True)
