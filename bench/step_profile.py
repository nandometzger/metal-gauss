"""Where does a training step actually go?

Times the five phases of a real training step separately, with an explicit
mps.synchronize() between each so the async queue cannot smear one phase's
cost into the next. Reports quality NEVER -- this is a timing run.

    python bench/step_profile.py --budget 600000 --res 270
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from metal_gauss.dataset import load_scene
from metal_gauss.mcmc import add_noise, grow, relocate
from metal_gauss.metal_backend import render
from metal_gauss.train import (_gaussian_kernel, init_params, make_optimizer,
                               render_view, scene_extent, split_sh, ssim)

# ROOT on sys.path so `bench.*` imports when this file is run directly. Without
# it sys.path[0] is bench/, which contains bench.py, and `import bench` resolves
# to that module instead of this package -- so this has to come BEFORE the
# first `from bench import`, not after it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import paths as _paths  # noqa: E402

RESULTS = Path(__file__).parent / "results"
DEF_COLMAP = _paths.room1("colmap")
DEF_IMAGES = _paths.room1("images")


def sync():
    torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=None,
                    help="NeRF-synthetic scene dir. Without this the profile "
                         "runs on room1, which is PORTRAIT: --res 800 there "
                         "gives 450x800, not 800x800, so a profile meant to "
                         "match a square Blender scene silently lands at 56%% "
                         "of the pixel count on different geometry.")
    ap.add_argument("--colmap", default=DEF_COLMAP)
    ap.add_argument("--images", default=DEF_IMAGES)
    ap.add_argument("--budget", type=int, default=600_000)
    ap.add_argument("--res", type=int, default=270)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--relocate-every", type=int, default=100,
                    help="must match the trainer: relocating every step perturbs "
                         "scales and inflates every downstream phase")
    ap.add_argument("--out", default=None)
    ap.add_argument("--torch-adam", action="store_true",
                    help="profile torch.optim.Adam instead of the trainer's FusedAdam")
    ap.add_argument("--selective-adam", action=argparse.BooleanOptionalAction, default=True,
                    help="the trainer's default: only the SH of gaussians each view "
                         "sees is updated; --no-selective-adam profiles dense Adam")
    ap.add_argument("--selective-adam-groups", choices=["all", "sh"], default="sh")
    ap.add_argument("--selective-adam-steps", choices=["row", "global"], default="global")
    args = ap.parse_args()

    dev = "mps"
    if args.blender:
        from metal_gauss.blender import load_blender
        scene = load_blender(args.blender, args.res)
    else:
        scene = load_scene(args.colmap, args.images, args.res, 8)
    p = init_params(scene, args.budget, dev)
    p = split_sh(p)
    active = args.budget
    kernel = _gaussian_kernel(device=dev)
    extent = scene_extent(scene)

    # Same construction the trainer uses -- see the note in train.py.
    opt = make_optimizer(p, 2e-4 * extent, fused=not args.torch_adam,
                         selective=args.selective_adam,
                         selective_groups=args.selective_adam_groups,
                         selective_steps=args.selective_adam_steps)

    # mcmc is amortised: relocate fires once per relocate_every steps, so its
    # per-step share is the amortised mean, not the cost of a relocate step.
    t = {k: [] for k in ("render_fwd", "loss_fwd", "backward", "adam", "mcmc")}
    views = scene.train

    # Warmup here doubles as the GPU clock ramp -- see the note in
    # bench/quick.py. With --warmup 8 the first timed steps run at boost clock
    # and under-report a sustained run; at 270p a step is ~50 ms so 40 warmup
    # steps is the ~2 s needed to settle.
    for i in range(args.iters + args.warmup):
        v = views[i % len(views)]
        H, W = v.image.shape[:2]
        rec = i >= args.warmup

        sync(); t0 = time.perf_counter()
        rgb, alpha, info = render_view(p, v, active)
        sync(); t1 = time.perf_counter()

        gt = v.image.to(dev).float() / 255.0
        l1 = (rgb - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim(rgb, gt, kernel))
        loss = loss + 0.01 * torch.sigmoid(p["logit_opac"][:active]).mean() \
                    + 0.01 * torch.exp(p["log_scales"][:active]).mean()
        sync(); t2 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync(); t3 = time.perf_counter()

        opt.step(info["valid_mask"]) if args.selective_adam else opt.step()
        sync(); t4 = time.perf_counter()

        with torch.no_grad():
            add_noise(p, 2e-4 * extent, 4e4, active=active)
            if i % args.relocate_every == 0:
                relocate(p, opt=opt, active=active)
        sync(); t5 = time.perf_counter()

        if rec:
            t["render_fwd"].append((t1 - t0) * 1e3)
            t["loss_fwd"].append((t2 - t1) * 1e3)
            t["backward"].append((t3 - t2) * 1e3)
            t["adam"].append((t4 - t3) * 1e3)
            t["mcmc"].append((t5 - t4) * 1e3)

    med = {k: float(np.median(v)) for k, v in t.items()}
    total = sum(med.values())
    print(f"\n{active:,} gaussians @ {W}x{H}   ({args.iters} steps, warmup {args.warmup})")
    print(f"{'phase':<12} {'ms':>8} {'share':>8}")
    for k, v in sorted(med.items(), key=lambda kv: -kv[1]):
        print(f"{k:<12} {v:8.2f} {100 * v / total:7.1f}%")
    print(f"{'TOTAL':<12} {total:8.2f}")

    out = Path(args.out) if args.out else RESULTS / f"step_profile_{args.budget}_{args.res}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from bench.provenance import env as _env
    out.write_text(json.dumps(
        {"schema": 1, "env": _env(),
         "scene": (args.blender or args.colmap), "budget": active,
         "W": W, "H": H, "median_ms": med, "total_ms": total,
         "iqr_pct": {k: float(100 * (np.percentile(v, 75) - np.percentile(v, 25))
                             / max(np.median(v), 1e-9)) for k, v in t.items()}},
        indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
