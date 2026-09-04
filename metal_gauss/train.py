"""3DGS trainer: metal-gauss rasterizer + MCMC densification.

    python -m metal_gauss.train --colmap sparse/0 --images images/ --steps 10000

Design notes, measurement-first: loss (0.8 L1 + 0.2 (1-SSIM)) and Adam run in
torch on MPS -- they are fused later only if profiling shows them hot. The
rasterizer is the fused-Metal fwd+bwd path. Densification is MCMC (fixed
budget, relocation + exploration noise) rather than ADC growth, so step time
stays flat.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from metal_gauss import render
from metal_gauss.dataset import Scene, downscaled, load_scene
from metal_gauss.schedule import (auto_budget, resolve_training_schedule)  # noqa: F401
from metal_gauss.appearance import AppearanceModel
from metal_gauss.mcmc import add_noise, grow, relocate
from metal_gauss.mipfilter import apply_3d_filter, compute_3d_filter

C0 = 0.28209479177387814


# ---------------------------------------------------------------- SSIM (torch)

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device="mps", groups: int = 15):
    """1D gaussian window, shaped for a batched separable grouped conv2d.

    `groups` is 15 because SSIM needs five blurred quantities (x, y, x^2, y^2,
    x*y) of three channels each, and they are blurred in ONE grouped
    convolution rather than five separate calls.
    """
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return (g / g.sum()).view(1, 1, 1, size).expand(groups, 1, 1, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """a, b: (H,W,3) in [0,1]. SSIM, 11-tap sigma=1.5 gaussian.

    Two optimisations over the textbook form, neither of which changes the
    result: the gaussian window is separable, so an 11x11 convolution becomes
    two 11-tap passes; and the five quantities SSIM needs are stacked into one
    15-channel grouped convolution, so the whole thing costs 2 dispatches
    instead of 10. Profiling put this loss at 39% of a training step -- more
    than the rasteriser -- almost all of it dispatch overhead rather than
    arithmetic.
    """
    return _SSIMFused.apply(a.contiguous(), b.contiguous(),
                            kernel[0, 0, 0].contiguous()).mean()


class _SSIMFused(torch.autograd.Function):
    """Whole SSIM in Metal: stack construction, both blur passes, and the tail.

    Replaces cat + two grouped conv2d + the tail. Three things go away:
      * the (1,15,H,W) stack, 86 MB at 900x1600, materialised every step;
      * the permute to (1,3,H,W) and its contiguous copy for BOTH inputs, since
        the kernels read (H,W,3) -- the layout render() and the ground truth
        already have;
      * torch's four convolution dispatches and their autograd graph.

    A symmetric gaussian is self-adjoint, so `ssim_blur15` is reused unchanged
    for the forward vertical pass and for both adjoint passes in the backward.
    Only the stack construction needs a separate forward and chain kernel.

    Zero padding matches F.conv2d exactly -- taps off the edge contribute
    nothing. A mismatch there only shows in a 5px border, which a mean-reduced
    loss hides well.
    """

    @staticmethod
    def forward(ctx, a, b, w):
        from metal_gauss.metal_backend import _load
        ext = _load()
        H, W = a.shape[0], a.shape[1]
        inter = ext.ssim_stack_blur_h(a, b, w, H, W)
        blurred = ext.ssim_blur15(inter, w, 1, H, W)
        ctx.save_for_backward(a, b, w, blurred)
        return ext.ssim_tail_forward(blurred, H, W)

    @staticmethod
    def backward(ctx, grad_out):
        from metal_gauss.metal_backend import _load
        ext = _load()
        a, b, w, blurred = ctx.saved_tensors
        H, W = a.shape[0], a.shape[1]
        d_blur = ext.ssim_tail_backward(blurred, grad_out.contiguous(), H, W)
        d_inter = ext.ssim_blur15(d_blur, w, 1, H, W)     # vertical adjoint
        d_stack = ext.ssim_blur15(d_inter, w, 0, H, W)    # horizontal adjoint
        return ext.ssim_chain_backward(a, b, d_stack, H, W), None, None


# ---------------------------------------------------------------- init

def scene_extent(scene: Scene) -> float:
    """Radius of the camera cloud -- the scale every position LR is relative to."""
    C = np.stack([(-v.viewmat[:3, :3].T @ v.viewmat[:3, 3]).numpy()
                  for v in scene.train])
    return float(np.linalg.norm(C - C.mean(0), axis=1).max())


def init_params(scene: Scene, budget: int, device: str) -> dict:
    pts = scene.points
    cols = scene.colors
    if len(pts) > budget:
        sel = np.random.default_rng(0).choice(len(pts), budget, replace=False)
        pts, cols = pts[sel], cols[sel]
    n = len(pts)

    # knn-ish scale init: median distance to 3 nearest of a random subset
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=4)
    s0 = np.clip(d[:, 1:].mean(1), 1e-4, None)
    # Isolated sparse points get huge knn distances -> screen-covering splats
    # that made early steps ~10x slower (every tile walks them every frame).
    s0 = np.minimum(s0, np.quantile(s0, 0.9))

    means = torch.tensor(pts, dtype=torch.float32, device=device)
    if n < budget:                      # fill the budget with jittered copies
        extra = budget - n
        idx = torch.randint(0, n, (extra,), device=device)
        jit = torch.randn(extra, 3, device=device) * \
            torch.tensor(s0, dtype=torch.float32, device=device)[idx][:, None]
        means = torch.cat([means, means[idx] + jit])
        cols = np.concatenate([cols, cols[np.asarray(idx.cpu())]])
        s0 = np.concatenate([s0, s0[np.asarray(idx.cpu())]])

    sh = torch.zeros(budget, 16, 3, device=device)
    sh[:, 0] = (torch.tensor(cols, dtype=torch.float32, device=device) - 0.5) / C0

    return {
        "means": means.requires_grad_(True),
        "log_scales": torch.log(torch.tensor(
            np.repeat(s0[:, None], 3, 1), dtype=torch.float32, device=device)
        ).requires_grad_(True),
        "quats": torch.cat([torch.ones(budget, 1, device=device),
                            torch.zeros(budget, 3, device=device)], 1).requires_grad_(True),
        "logit_opac": torch.full((budget,), -2.0, device=device).requires_grad_(True),
        "sh": sh.requires_grad_(True),
    }



def split_sh(p: dict) -> dict:
    """DC band and bands 1+ as separate leaves, for their different LRs."""
    p["sh_dc"] = p["sh"][:, :1].detach().clone().requires_grad_(True)
    p["sh_rest"] = p["sh"][:, 1:].detach().clone().requires_grad_(True)
    del p["sh"]
    return p


def make_optimizer(p: dict, lr_means0: float, *, lr_opac: float = 1e-2,
                   sh_lr_div: float = 20.0, selective: bool = False,
                   fused: bool = True):
    groups = [
        {"params": [p["means"]], "lr": lr_means0, "name": "means"},
        {"params": [p["log_scales"]], "lr": 5e-3, "name": "scales"},
        {"params": [p["quats"]], "lr": 1e-3, "name": "quats"},
        {"params": [p["logit_opac"]], "lr": lr_opac, "name": "opac"},
        {"params": [p["sh_dc"]], "lr": 2.5e-3, "name": "sh_dc"},
        {"params": [p["sh_rest"]], "lr": 2.5e-3 / sh_lr_div, "name": "sh_rest"},
    ]
    if selective:
        from metal_gauss.selective_adam import SelectiveAdam
        return SelectiveAdam(groups, eps=1e-15)
    if fused:
        from metal_gauss.fused_adam import FusedAdam
        return FusedAdam(groups, eps=1e-15)
    return torch.optim.Adam(groups, eps=1e-15)


def render_view(p: dict, v, active: int, sh_deg: int = 3,
                background=(0.0, 0.0, 0.0), antialias: bool = False,
                absgrad_out=None, filter_3d=None):
    """One rendered view, exactly as training renders it.

    v.K and v.viewmat stay on the HOST on purpose: the kernel reads both there,
    and passing MPS tensors makes every call drain the queue.
    """
    H, W = v.image.shape[:2]
    scales = torch.exp(p["log_scales"][:active])
    opac = torch.sigmoid(p["logit_opac"][:active])
    if filter_3d is not None:
        # Mip-Splatting's 3D low-pass. Applied here rather than in the kernel
        # because it is a plain reparameterisation of scale and opacity, so
        # autograd carries it and no adjoint has to be written.
        scales, opac = apply_3d_filter(scales, opac, filter_3d[:active])
    return render(
        p["means"][:active], p["quats"][:active],
        scales, opac, p["sh_dc"][:active],
        v.K, v.viewmat, W, H,
        sh_degree=sh_deg, backend="metal", sh_rest=p["sh_rest"][:active],
        background=background, antialias=antialias, absgrad_out=absgrad_out)


# ---------------------------------------------------------------- training

def train(args) -> dict:
    device = "mps"
    # Seed every torch RNG the run touches: the init jitter, relocate/grow's
    # multinomial, and add_noise's randn. The numpy streams (point subsample,
    # view order) were already fixed at 0.
    #
    # The point is PAIRED comparison. Two policies run at the same seed share
    # their draws, so the only difference between them is the thing under test.
    # Unpaired, the 10k/1600p configuration has a 0.49 dB run-to-run spread --
    # wider than any lever measured here, and wide enough that a single sample
    # produced two confidently wrong conclusions in one session.
    #
    # This does NOT make runs bit-reproducible. rasterize_backward accumulates
    # through atomics whose order is not controllable, worth ~1e-10 a step, and
    # 10k steps of that compounds. Seeding removes the large term, not all of it.
    torch.manual_seed(args.seed)
    if hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(args.seed)
    if args.blender:
        from metal_gauss.blender import load_blender
        scene = load_blender(args.blender, args.max_resolution)
        # The Blender PNGs are composited over WHITE, so the renderer must
        # fill unoccupied pixels with white too. Leaving the default black
        # background here cost ~10 dB on lego and looks exactly like a broken
        # model rather than a convention mismatch.
        bg = (1.0, 1.0, 1.0)
    else:
        scene = load_scene(args.colmap, args.images, args.max_resolution,
                           args.eval_split_every)
        bg = (0.0, 0.0, 0.0)
    print(f"{len(scene.train)} train views, {len(scene.heldout)} held out, "
          f"{len(scene.points):,} sparse points, budget {args.budget:,}")

    # Selective Adam was built, validated, and MEASURED SLOWER here: at this
    # scene's per-view visibility (well above 50%), 15 gather/scatter launches
    # (or the masked-dense where-chain) cost more than dense Adam's 69 ms.
    # It stays available via --selective-adam for low-visibility regimes;
    # measurement-first means the default is the thing that is actually fast.
    p = init_params(scene, args.budget, device)
    # Position LR is relative to scene scale, and SH bands 1+ train 20x colder
    # than DC: with ~160 views, hot high-band SH memorises per-view shading and
    # costs held-out PSNR. Both conventions come from Inria 3DGS / Brush.
    extent = scene_extent(scene)
    lr_means0 = args.lr_means * extent if args.lr_scene_scaled else args.lr_means
    print(f"scene extent {extent:.2f}, initial means lr {lr_means0:.2e}")

    p = split_sh(p)
    opt = make_optimizer(p, lr_means0, lr_opac=args.lr_opac,
                         sh_lr_div=args.sh_lr_div,
                         selective=args.selective_adam, fused=args.fused_adam)

    appearance = None
    if args.appearance != "off":
        appearance = AppearanceModel(len(scene.train), args.appearance, device)
        opt.add_param_group({"params": list(appearance.parameters()),
                             "lr": args.lr_appearance, "name": "appearance"})
        print(f"appearance correction: {args.appearance} "
              f"({sum(x.numel() for x in appearance.parameters())} params "
              f"over {len(scene.train)} training images)")

    kernel = _gaussian_kernel(device=device)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(scene.train))
    oi = 0
    t0 = time.perf_counter()
    log = []

    # Capacity ramp: start small (cheap early steps) and grow to the cap by
    # grow_until. This is ADC's curriculum without ADC's bookkeeping -- and it
    # is a speed lever as much as a quality one.
    active = min(args.start_active, args.budget) if args.grow else args.budget
    grow_until = int(args.grow_until_frac * args.steps)
    lr_decay = (args.lr_means_end / max(args.lr_means, 1e-12)) ** (1.0 / args.steps)
    relocate_until = int(args.relocate_until_frac * args.steps)

    # Screen-space positional gradient, accumulated per gaussian between
    # densification events. 3DGS-ADC's signal: a gaussian whose projected
    # centre is being pulled hard is straddling detail it cannot represent, and
    # is the one worth splitting. Averaged over the views that actually saw it,
    # because a gaussian visible in 3 views must not outrank one visible in 30.
    use_absgrad = args.densify_weight == "absgrad"
    filter_3d = None
    f3d_every = args.filter_3d_every or args.relocate_every
    if args.filter_3d:
        filter_3d = compute_3d_filter(p["means"], scene.train)
    grad_sum = torch.zeros(args.budget, device=device)
    grad_hits = torch.zeros(args.budget, device=device)

    step_times: list[float] = []
    for step in range(1, args.steps + 1):
        t_step = time.perf_counter()
        if oi >= len(order):
            order = rng.permutation(len(scene.train))
            oi = 0
        view_idx = int(order[oi])
        v = scene.train[view_idx]
        oi += 1

        # anneal the position LR (and with it the exploration noise)
        cur_lr_means = lr_means0 * (lr_decay ** step)
        opt.param_groups[0]["lr"] = cur_lr_means
        # SH degree warmup: +1 band per warmup interval
        sh_deg = min(3, step // args.sh_warmup) if args.sh_warmup else 3

        # Coarse-to-fine: start at 1/2^num_downscales and double on a schedule.
        # Off by default (num_downscales=0) until the Pareto curve says it wins;
        # msplat's speed is roughly half this and half per-step cost.
        if args.num_downscales > 0:
            lvl = max(0, args.num_downscales - step // max(1, args.resolution_schedule))
            v = downscaled(v, 1 << lvl)
        H, W = v.image.shape[:2]
        # absgrad is accumulated by the backward kernel straight into grad_sum,
        # which is exactly the accumulator the other criteria fill by hand and
        # is zeroed on the same schedule.
        if filter_3d is not None and step % f3d_every == 0:
            # the gaussians have moved since the last refresh
            filter_3d = compute_3d_filter(p["means"], scene.train)
        rgb, alpha, info = render_view(p, v, active, sh_deg, bg,
                                       antialias=args.antialias,
                                       absgrad_out=grad_sum if use_absgrad else None,
                                       filter_3d=filter_3d)
        gt = v.image.to(device).float() / 255.0
        # Correct the RENDER (not the ground truth) so the exported splat stays
        # in the true photometric space and held-out views need no transform.
        rgb_c = appearance(rgb, view_idx) if appearance is not None else rgb
        l1 = (rgb_c - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim(rgb_c, gt, kernel))
        if appearance is not None:
            loss = loss + args.appearance_reg * appearance.regulariser()
        # MCMC regularisers: keep opacity and scale mass in check
        # Regularisers ramp down: they exist to keep early growth honest, not
        # to fight the reconstruction at convergence (Brush's aux_loss_weight).
        aux = max(0.0, 1.0 - step / (0.9 * args.steps))
        loss = loss + aux * (args.opac_reg * torch.sigmoid(p["logit_opac"][:active]).mean()
                             + args.scale_reg * torch.exp(p["log_scales"][:active]).mean())

        opt.zero_grad(set_to_none=True) if not args.selective_adam else opt.zero_grad()
        loss.backward()
        opt.step(info["valid_mask"]) if args.selective_adam else opt.step()

        if args.densify_weight != "opacity":
            with torch.no_grad():
                if use_absgrad:
                    # grad_sum was filled by the kernel; only the view count is
                    # left to record.
                    grad_hits[:active] += info["valid_mask"].to(grad_sum.dtype)
                else:
                    guv = info["uv"].grad
                    if guv is not None:
                        grad_sum[:active] += guv.norm(dim=1)
                        grad_hits[:active] += info["valid_mask"].to(grad_sum.dtype)

        with torch.no_grad():
            add_noise(p, cur_lr_means, args.noise_weight, active=active)
            densify_now = step % args.relocate_every == 0
            w = None
            if densify_now and args.densify_weight != "opacity":
                avg_grad = grad_sum / grad_hits.clamp_min(1.0)
                if args.densify_weight in ("grad", "absgrad"):
                    # same consumer: absgrad only changes how grad_sum was
                    # filled, not what relocation does with it.
                    w = avg_grad
                elif args.densify_weight == "opacity_grad":
                    # Opacity keeps MCMC's requirement that a target be a real
                    # gaussian worth splitting; the gradient says where the
                    # reconstruction is straining. Normalised so the product is
                    # not dominated by whichever term happens to be larger.
                    g = avg_grad / avg_grad.max().clamp_min(1e-12)
                    w = torch.sigmoid(p["logit_opac"]) * g
                else:                                   # "grad_gate"
                    # ADC densifies everything ABOVE a gradient threshold, and
                    # treats those equally. Sampling proportional to the
                    # gradient instead is far more concentrated, and gets worse
                    # with resolution: the p99/p50 ratio of the per-gaussian
                    # gradient is 93 at 270p and 212 at 1600p, so the top 1% of
                    # gaussians absorb 25% of the sampling mass at 270p but 37%
                    # at 1600p. That is the mechanism behind opacity_grad
                    # winning at 270p and losing at 1600p. A gate keeps the
                    # "where" signal without the concentration.
                    live = avg_grad[avg_grad > 0]
                    if live.numel() > 0:
                        thr = torch.quantile(live.float(),
                                             1.0 - args.densify_gate_frac)
                        w = torch.sigmoid(p["logit_opac"]) * (avg_grad > thr)
                    else:
                        w = None
            if densify_now and step < relocate_until:
                moved = relocate(p, opt=opt, active=active, weights=w)
                if moved:
                    log.append({"step": step, "relocated": moved})
            if args.grow and densify_now and step <= grow_until:
                target = int(args.start_active + (args.budget - args.start_active)
                             * min(1.0, step / max(grow_until, 1)))
                new_active = grow(p, target, active, opt=opt, weights=w)
                if new_active != active:
                    log.append({"step": step, "active": new_active})
                    active = new_active
            if densify_now:
                grad_sum.zero_(); grad_hits.zero_()

        # MPS driver-side allocations grow without bound if the queue never
        # drains: a 10k-step run died at ~25GB of "other allocations" before
        # the first eval. Sync + drain periodically; costs ~ms, saves the run.
        if step % 50 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()
            if step % 200 == 0:
                da = torch.mps.driver_allocated_memory() / 1e9
                if da > 20.0:
                    print(f"  [mem] driver {da:.1f} GB at step {step}", flush=True)

        # Stall detector. A step that suddenly takes far longer than the
        # running median means something outside the model changed: the machine
        # swapping, another job taking the GPU, or -- as happened here -- the
        # Mac going to sleep mid-run and the log simply stopping. Any of those
        # is worth a loud line, because a process that has quietly stopped
        # logging looks identical to one that is merely slow, and that
        # ambiguity already produced one confidently wrong diagnosis.
        # Long unattended runs should go under `caffeinate -i`.
        if step > 20:
            dt_step = time.perf_counter() - t_step
            step_times.append(dt_step)
            if len(step_times) > 50:
                step_times.pop(0)
            med = sorted(step_times)[len(step_times) // 2]
            if dt_step > max(10.0 * med, 30.0):
                print(f"  [STALL] step {step} took {dt_step:.1f}s against a "
                      f"{med * 1000:.0f} ms median. The machine is very likely "
                      f"swapping: {active:,} active gaussians at {W}x{H}. "
                      f"Reduce --budget or --max-resolution.", flush=True)

        if step == 100 or step % 500 == 0 and step % args.eval_every != 0:
            dt = time.perf_counter() - t0
            print(f"step {step:>6}  loss {loss.item():.4f}  ({1000 * dt / step:.0f} ms/step)",
                  flush=True)
        if args.export and args.export_every and step % args.export_every == 0:
            # written before the eval so the mtime reflects training time, not
            # training plus a 200-view evaluation
            ck = Path(args.export).with_suffix(f".step{step:06d}.ply")
            export_ply({k: (v[:active] if torch.is_tensor(v) else v)
                        for k, v in p.items()}, str(ck), filter_3d=filter_3d)

        if step % args.eval_every == 0 or step == args.steps:
            torch.mps.empty_cache()
            psnr = evaluate(p, scene, device, sh_degree=sh_deg, active=active,
                            background=bg, antialias=args.antialias,
                            filter_3d=filter_3d)
            dt = time.perf_counter() - t0
            print(f"step {step:>6}  loss {loss.item():.4f}  heldout PSNR {psnr:.2f} dB  "
                  f"{active/1000:.0f}k splats  {dt:.0f}s  ({1000 * dt / step:.0f} ms/step)",
                  flush=True)
            log.append({"step": step, "psnr": psnr, "wall_s": round(dt, 1),
                        "active": active})

    out = _run_report(args, log, time.perf_counter() - t0, active)
    for dest in (args.out, getattr(args, "report", None)):
        if dest:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text(json.dumps(out, indent=2, default=str))
    if args.export:
        export_ply({k: (v[:active] if torch.is_tensor(v) else v) for k, v in p.items()},
                   args.export, filter_3d=filter_3d)
        print(f"exported {args.export}")
    return out


@torch.no_grad()
def evaluate(p, scene: Scene, device: str, max_views: int | None = None,
             background=(0.0, 0.0, 0.0),
             sh_degree: int = 3, active: int | None = None,
             antialias: bool = False, filter_3d=None) -> float:
    """Held-out PSNR over ALL held-out views by default.

    This used to default to the first 10 of 21 views, which biased every
    number we reported by up to ~0.5 dB depending on which views happened to
    be easy. Metrics get fixed before levers get credited.
    """
    psnrs = []
    for v in (scene.heldout if max_views is None else scene.heldout[:max_views]):
        H, W = v.image.shape[:2]
        a = len(p["means"]) if active is None else active
        sh_full = torch.cat([p["sh_dc"], p["sh_rest"]], dim=1) if "sh_dc" in p else p["sh"]
        sc_e = torch.exp(p["log_scales"][:a])
        op_e = torch.sigmoid(p["logit_opac"][:a])
        if filter_3d is not None:
            sc_e, op_e = apply_3d_filter(sc_e, op_e, filter_3d[:a])
        rgb, _, _ = render(p["means"][:a], p["quats"][:a], sc_e, op_e, sh_full[:a],
                           v.K, v.viewmat, W, H,
                           sh_degree=sh_degree, backend="metal", background=background,
                           antialias=antialias)
        gt = v.image.to(device).float() / 255.0
        mse = ((rgb.clamp(0, 1) - gt) ** 2).mean().item()
        psnrs.append(-10 * math.log10(max(mse, 1e-10)))
    return float(np.mean(psnrs))


@torch.no_grad()
def export_ply(p, path: str, filter_3d=None) -> None:
    """Write an INRIA-convention ply.

    With `filter_3d`, Mip-Splatting's 3D low-pass is BAKED IN: the widened
    scales and dimmed opacity are written in the file's pre-activation space,
    so a viewer that knows nothing about the filter renders the model exactly
    as this trainer does. That is the property the 2D Mip filter lacks -- it is
    screen-space and cannot be folded into a ply at all, which is why
    --antialias cannot be a default while this can.
    """
    import plyfile

    n = len(p["means"])
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.zeros(n, dtype=[(nm, "f4") for nm in names])
    m = p["means"].detach().cpu().numpy()
    data["x"], data["y"], data["z"] = m.T
    sh = (torch.cat([p["sh_dc"], p["sh_rest"]], dim=1) if "sh_dc" in p
          else p["sh"]).detach().cpu().numpy()
    for c in range(3):
        data[f"f_dc_{c}"] = sh[:, 0, c]
        for b in range(15):
            data[f"f_rest_{c * 15 + b}"] = sh[:, b + 1, c]
    logit = p["logit_opac"].detach()
    log_scales = p["log_scales"].detach()
    if filter_3d is not None:
        f = filter_3d[:n].detach()
        sc, op = apply_3d_filter(torch.exp(log_scales), torch.sigmoid(logit), f)
        log_scales = torch.log(sc.clamp_min(1e-12))
        # back to logit space; clamp keeps a fully suppressed gaussian finite
        op = op.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(op / (1.0 - op))
    data["opacity"] = logit.cpu().numpy()
    ls = log_scales.cpu().numpy()
    for i in range(3):
        data[f"scale_{i}"] = ls[:, i]
    q = p["quats"].detach().cpu().numpy()
    for i in range(4):
        data[f"rot_{i}"] = q[:, i]
    plyfile.PlyData([plyfile.PlyElement.describe(data, "vertex")]).write(path)


def _run_report(args, log, wall_s, active):
    """Everything needed to reproduce this run, recorded by the process that ran it.

    Records ALL of vars(args), not a curated subset. Curation is how knobs go
    unrecorded, and an unrecorded knob is how this project twice published a
    number produced by settings other than the ones it claimed: --steps-scaler
    once, then --budget, where a harness forwarded 300k to every child and
    auto_budget() never ran in any 8-scene sweep. Both were invisible because
    the harness recorded its own argparse namespace as "the protocol".

    args is read AFTER main() has resolved auto_budget, resolution_schedule,
    start_active clamping and the steps-scaler, so these are the values that
    actually ran, by construction rather than by convention.
    """
    def _git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=Path(__file__).resolve().parent,
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except Exception:
            return None

    resolved = {k: v for k, v in sorted(vars(args).items())}
    ms = (1000.0 * wall_s / args.steps) if args.steps else None
    return {
        "schema": 1,
        "resolved": resolved,
        "env": {
            "git": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "metrics": {
            "psnr": log[-1]["psnr"] if log else None,
            "wall_s": round(wall_s, 1),
            "n_splats": int(active),
            "ms_per_step": round(ms, 2) if ms else None,
        },
        # kept at the old top-level keys so existing readers of --out still work
        "steps": args.steps,
        "budget": args.budget,
        "wall_clock_s": round(wall_s, 1),
        "final_psnr": log[-1]["psnr"] if log else None,
        "log": log,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap")
    ap.add_argument("--blender", help="NeRF-synthetic scene dir (transforms_*.json). "
                                      "Uses the published train/test split, for "
                                      "absolute calibration against reported numbers.")
    ap.add_argument("--images")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--budget", type=int, default=None,
                    help="gaussian capacity. Default scales with --steps; see "
                         "auto_budget(). Set explicitly to override.")
    ap.add_argument("--max-resolution", type=int, default=1600)
    ap.add_argument("--eval-split-every", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--relocate-every", type=int, default=100)
    ap.add_argument("--lr-means", type=float, default=2e-4)
    ap.add_argument("--noise-weight", type=float, default=4e4)
    ap.add_argument("--opac-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--appearance", choices=["off", "gain_bias", "affine"],
                    default="off",
                    help="per-training-image photometric correction; held-out "
                         "views always render with the identity transform")
    ap.add_argument("--lr-appearance", type=float, default=1e-3)
    ap.add_argument("--appearance-reg", type=float, default=1e-2)
    ap.add_argument("--lr-means-end", type=float, default=2e-6)
    ap.add_argument("--lr-scene-scaled", action="store_true", default=True)
    ap.add_argument("--lr-opac", type=float, default=1e-2)
    ap.add_argument("--sh-lr-div", type=float, default=20.0)
    ap.add_argument("--sh-warmup", type=int, default=1000,
                    help="steps per SH band; 0 disables warmup")
    ap.add_argument("--grow", action="store_true", default=True)
    ap.add_argument("--no-grow", dest="grow", action="store_false")
    ap.add_argument("--start-active", type=int, default=150_000)
    ap.add_argument("--grow-until-frac", type=float, default=0.7)
    ap.add_argument("--relocate-until-frac", type=float, default=0.85)
    # LichtFeld's steps_scaler: scale EVERY schedule together so a short run is
    # a proportional miniature of a long one, not a truncation. Without this,
    # rankings measured on a 1k-step run do not transfer to 10k.
    ap.add_argument("--steps-scaler", type=float, default=1.0)
    ap.add_argument("--selective-adam", action="store_true")
    ap.add_argument("--densify-weight",
                    choices=["opacity", "grad", "opacity_grad", "grad_gate",
                             "absgrad"],
                    default="opacity_grad",
                    help="what relocation and growth sample proportional to. "
                         "'opacity' is the 3DGS-MCMC policy; 'grad' is ADC's "
                         "screen-space positional gradient; 'opacity_grad' is "
                         "their product; 'grad_gate' restricts opacity "
                         "sampling to the top --densify-gate-frac by gradient, "
                         "which is closer to what ADC actually does.")
    ap.add_argument("--filter-3d", action="store_true",
                    help="Mip-Splatting's 3D smoothing filter: band-limit each "
                         "gaussian to the sampling rate of the views that see "
                         "it. Unlike --antialias this is view-independent, so "
                         "it is BAKED INTO THE EXPORTED PLY and any viewer "
                         "renders it correctly without cooperation.")
    ap.add_argument("--filter-3d-every", type=int, default=0,
                    help="steps between recomputing the 3D filter; 0 means use "
                         "--relocate-every. Gaussians move under MCMC "
                         "relocation, so a filter computed once at init goes "
                         "stale, but recomputing every step is waste.")
    ap.add_argument("--antialias", action="store_true",
                    help="Mip-Splatting / gsplat antialiased rasterisation: "
                         "rescale opacity so the low-pass dilation preserves "
                         "energy instead of eroding sub-pixel gaussians. "
                         "Matters most when render and training resolution "
                         "differ, which --num-downscales makes the default.")
    ap.add_argument("--num-downscales", type=int, default=2,
                    help="start training at 1/2^N resolution and double on "
                         "--resolution-schedule. 0 disables (full res throughout).")
    ap.add_argument("--resolution-schedule", type=int, default=None,
                    help="steps between resolution doublings. Default steps//3, "
                         "so full resolution is reached two thirds through. "
                         "--steps-scaler scales this too, explicit value "
                         "included, so a scaled run keeps the same curriculum "
                         "in proportion.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds torch RNG. Use the SAME seed on both arms of an "
                         "A/B so they share draws and only the tested variable "
                         "differs; atomics still make runs non-bit-reproducible.")
    ap.add_argument("--densify-gate-frac", type=float, default=0.30,
                    help="grad_gate: fraction of gaussians, by gradient, that "
                         "are eligible densification targets.")
    ap.add_argument("--fused-adam", action="store_true", default=True,
                    help="Adam in one Metal pass instead of torch's five (2.2x)")
    ap.add_argument("--no-fused-adam", dest="fused_adam", action="store_false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None,
                    help="write the resolved config, env and metrics as JSON. "
                         "Benchmarks must read this instead of parsing stdout.")
    ap.add_argument("--export", default=None)
    ap.add_argument("--export-every", type=int, default=0,
                    help="also write a checkpoint ply every N steps, named "
                         "<export>.stepNNNNNN.ply. Used to build the wall-clock "
                         "convergence comparison: the checkpoint's mtime gives "
                         "its real elapsed time, which beats interpolating from "
                         "the step count when per-step cost is not constant.")
    args = ap.parse_args()
    if not args.blender and not (args.colmap and args.images):
        ap.error("need --blender, or --colmap with --images")
    budget_was_auto = args.budget is None
    try:
        resolved = resolve_training_schedule(
            steps=args.steps,
            steps_scaler=args.steps_scaler,
            budget=args.budget,
            start_active=args.start_active,
            relocate_every=args.relocate_every,
            eval_every=args.eval_every,
            sh_warmup=args.sh_warmup,
            resolution_schedule=args.resolution_schedule,
            filter_3d_every=args.filter_3d_every,
            export_every=args.export_every,
        )
    except ValueError as e:
        ap.error(str(e))

    for name, value in resolved.items():
        setattr(args, name, value)

    if budget_was_auto:
        print(f"budget {args.budget:,} (auto, from {args.steps} steps)")
    if args.steps_scaler != 1.0:
        print(f"steps_scaler {args.steps_scaler}: {args.steps} steps, relocate every "
              f"{args.relocate_every}, eval every {args.eval_every}, "
              f"resolution every {args.resolution_schedule}, "
              f"sh warmup {args.sh_warmup}")
    train(args)


if __name__ == "__main__":
    main()
