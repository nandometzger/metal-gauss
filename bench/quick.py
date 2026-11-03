"""Tiered benchmark runner: fast feedback beats big runs.

    python bench/quick.py t1                 # ~3 min gate, every change
    python bench/quick.py t2                 # ~15 min, before crediting a dB
    python bench/quick.py stages             # isolated per-stage timings, no quality

Tier runs write JSON to bench/results/ and print a markdown row. Timing runs
never compute quality; quality runs never claim timing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ROOT on sys.path so `bench.*` imports when this file is run directly. Without
# it sys.path[0] is bench/, which contains bench.py, and `import bench` resolves
# to that module instead of this package -- so this has to come BEFORE the
# first `from bench import`, not after it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import paths as _paths  # noqa: E402

ROOM1_COLMAP = _paths.room1("colmap")
ROOM1_IMAGES = _paths.room1("images")
RESULTS = Path(__file__).parent / "results"

# Reference PSNRs are Brush's saved eval renders RESCORED AT EACH TIER'S
# RESOLUTION (bench/results/brush_reference_multires.json). Comparing a 640p
# run against a 1600p reference is meaningless -- PSNR rises as resolution
# falls (Brush: 27.55 @1600p, 28.70 @640p, 30.12 @270p for the same model).
# Caveat kept in view: Brush trained at 1600p and is downscaled for these
# rows, which if anything flatters it, since downscaling averages away error.
TIERS = {
    "t1": dict(steps=1000, budget=600_000, res=270, ref=21.91, mins=3),
    "t2": dict(steps=3000, budget=600_000, res=640, ref=25.77, mins=15),
}


def run_tier(name: str, extra: list[str]) -> dict:
    cfg = TIERS[name]
    out = RESULTS / f"{name}_latest.json"
    # The tiers deliberately fix capacity so t1/t2 stay comparable across
    # commits, so --budget IS passed here on purpose. It still goes through
    # bench/runner.py, which verifies the child actually used it: a deliberate
    # override and an accidental one look identical in a result file, and the
    # accidental kind is what produced a week of mislabelled numbers.
    from bench.runner import run as run_trainer
    spec = {"colmap": ROOM1_COLMAP, "images": ROOM1_IMAGES,
            "steps": cfg["steps"], "budget": cfg["budget"],
            "max_resolution": cfg["res"], "eval_every": cfg["steps"] // 2,
            "steps_scaler": 1.0}
    print(f"[{name}] ~{cfg['mins']} min: {cfg['steps']} steps @ {cfg['res']}p, "
          f"budget {cfg['budget']:,}\n")
    t0 = time.perf_counter()
    res = run_trainer(spec, report=out)
    wall = time.perf_counter() - t0
    psnr = res["metrics"]["psnr"]

    base_p = RESULTS / f"{name}_baseline.json"
    base = json.loads(base_p.read_text()) if base_p.exists() else None
    print(f"\n| tier | PSNR | Brush ref | wall | ms/step |")
    print(f"|---|---|---|---|---|")
    print(f"| {name} | **{psnr:.2f} dB** | {cfg['ref']:.2f} | {wall/60:.1f} min | "
          f"{1000*res['wall_clock_s']/cfg['steps']:.0f} |")
    if base:
        d = psnr - base["final_psnr"]
        print(f"\nvs baseline: {d:+.2f} dB, "
              f"{100*(res['wall_clock_s']/base['wall_clock_s']-1):+.0f}% wall")
    if psnr < cfg["ref"] - 0.05:
        print(f"\n!! below the {name} gate ({cfg['ref']:.2f} dB)")
    return res


def run_stages() -> dict:
    """Isolated per-stage timings at several scales. Quality never touched."""
    import numpy as np
    import torch

    from metal_gauss import render
    from metal_gauss.io import load_ply

    ply = _paths.room1("ply")
    sp_full = load_ply(ply, device="mps")
    rows = []
    for n in (600_000, 1_000_000, len(sp_full)):
        idx = torch.randperm(len(sp_full), device="mps")[:n]
        sub = sp_full.subset(idx)
        for W, H in ((270, 480), (900, 1600)):
            K = torch.tensor([[0.8 * H, 0, W / 2], [0, 0.8 * H, H / 2], [0, 0, 1]],
                             dtype=torch.float32, device="mps")
            vm = torch.eye(4, device="mps")
            vm[2, 3] = 4.0

            def fwd():
                return render(sub.means, sub.quats, sub.scales, sub.opacities, sub.sh,
                              K, vm, W, H, sh_degree=3, backend="metal")

            m = sub.means.clone().requires_grad_(True)

            def fb():
                rgb, _, _ = render(m, sub.quats, sub.scales, sub.opacities, sub.sh,
                                   K, vm, W, H, sh_degree=3, backend="metal")
                rgb.square().mean().backward()

            def t(fn, k=11, ramp_s=2.0):
                """Median of k, trimmed, after ramping the GPU to steady state.

                The ramp is not optional. Apple Silicon runs short bursts at a
                boost clock and settles lower under sustained load, which makes
                timings BIMODAL: the same kernel measured 11.6 and 19.8 ms on
                alternating runs, a 70% swing that made a no-op change look
                like a 26% win. Two seconds of continuous work first collapses
                that to ~4% and reports the clock a real training run sees.

                Training is sustained load, so the steady state is the honest
                number; boost-clock figures flatter the benchmark."""
                import time as _t
                end = _t.perf_counter() + ramp_s
                while _t.perf_counter() < end:
                    fn()
                torch.mps.synchronize()
                fn(); torch.mps.synchronize()
                ts = []
                for _ in range(k):
                    t0 = time.perf_counter(); fn(); torch.mps.synchronize()
                    ts.append(time.perf_counter() - t0)
                ts = sorted(ts)[1:-1]
                return float(np.median(ts)) * 1000

            rows.append({"gaussians": n, "res": f"{W}x{H}", "tile": 16,
                         "fwd_ms": round(t(fwd), 1), "fwd_bwd_ms": round(t(fb), 1)})
            print(f"  {n:>9,} @ {W}x{H:<5} fwd {rows[-1]['fwd_ms']:>7.1f} ms   "
                  f"fwd+bwd {rows[-1]['fwd_bwd_ms']:>7.1f} ms")
    out = RESULTS / "stages_latest.json"
    from bench.provenance import stamp
    out.write_text(json.dumps(stamp(rows, benchmark="stages"), indent=2))
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", choices=["t1", "t2", "stages"])
    args, extra = ap.parse_known_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_stages() if args.tier == "stages" else run_tier(args.tier, extra)


if __name__ == "__main__":
    main()
