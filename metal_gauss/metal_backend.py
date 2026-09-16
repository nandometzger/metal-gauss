"""Metal backend: custom kernels driven from PyTorch MPS tensors.

Division of labour, chosen deliberately:

  * Projection, SH evaluation, tile binning and the depth sort stay in PyTorch.
    They are O(N) or handled well by torch.sort on MPS, and keeping them in
    torch means autograd chains through them for free -- gradients reach means,
    quaternions, scales and SH without a line of hand-written backward code.
  * The per-pixel compositing loop is in Metal. It is O(pixels x gaussians in
    tile) and is where the pure-torch reference spends effectively all of its
    time, because torch has to materialise a (tile, gaussian, pixel) tensor
    that a Metal thread can just iterate.

The custom autograd Function therefore only needs to bridge four tensors
(uv, conic, opacity, colour); everything upstream is ordinary torch autograd.

Correctness is defined by metal_gauss.torch_ref, which is gradcheck-verified in
float64. These kernels are never trusted on their own.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).parent
_ext = None

# Longer than any build of this extension takes; past it, a lock is a leftover.
_STALE_LOCK_S = 180.0


def stale_lock_message(build_dir, now: float | None = None) -> str | None:
    """What to say about a lock sitting in the extension's build directory.

    torch's `load` waits on `<build_dir>/lock` with a bare
    `while os.path.exists(lock): sleep(0.1)` and prints nothing, because it
    builds its FileBaton without `warn_after_seconds`. A lock left behind by an
    interrupted build therefore stops every later run forever, with no output
    at all -- indistinguishable from a hung GPU until you sample the process
    and find it asleep in `time_sleep`.

    Nothing is deleted here. Two processes may legitimately build at once, and
    deleting another one's lock would corrupt its build; saying what the file
    is costs nothing and is what the wait itself never says.
    """
    try:
        lock = Path(build_dir) / "lock"
        age = (now if now is not None else time.time()) - lock.stat().st_mtime
    except OSError:                      # no lock, or no build directory at all
        return None
    where = f"waiting for the kernel build lock at {lock} ({age:.0f}s old)"
    if age < _STALE_LOCK_S:
        return f"{where}; another process is probably compiling."
    return (f"{where}. A build takes a minute or two, so this one is stale --"
            f" left by an interrupted build. Delete it to carry on:\n"
            f"    rm {lock}")


def _warn_about_lock() -> None:
    """Say what the wait inside torch's `load` never says. Never fatal."""
    try:
        from torch.utils.cpp_extension import _get_build_directory

        message = stale_lock_message(_get_build_directory("metal_gauss_metal", False))
    except Exception:                    # a private torch helper; not worth dying for
        return
    if message is not None:
        print(f"metal-gauss: {message}", file=sys.stderr, flush=True)


def _load():
    """Build and cache the extension. Runtime-compiles the .metal source."""
    global _ext
    if _ext is not None:
        return _ext
    import os
    import sys

    # torch finds ninja via PATH, but pip installs it into the venv's bin,
    # which is absent when python is invoked by absolute path. Prepend it so
    # `pip install metal-gauss` works without shell activation.
    venv_bin = str(Path(sys.exec_prefix) / "bin")
    if venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    from torch.utils.cpp_extension import load

    _warn_about_lock()
    _ext = load(
        name="metal_gauss_metal",
        sources=[str(_HERE / "csrc" / "rasterize.mm")],
        extra_cflags=["-std=c++20", "-ObjC++"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )
    _ext.init((_HERE / "csrc" / "rasterize.metal").read_text() + "\n"
              + (_HERE / "csrc" / "preprocess.metal").read_text() + "\n"
              + (_HERE / "csrc" / "adam.metal").read_text() + "\n"
              + (_HERE / "csrc" / "ssim.metal").read_text() + "\n"
              + (_HERE / "csrc" / "binning.metal").read_text())
    return _ext


def build_tile_lists(uv, radius, depth, valid, W, H, tile, ry=None,
                     conic=None, opacity=None):
    """Flat depth-sorted Gaussian list plus per-tile offsets.

    `radius` is the x half-extent; `ry` the y half-extent (defaults to radius
    for the circular callers). Sorting is ONE int64 sort on a packed
    (tile_id << 32 | depth_bits) key instead of two chained argsorts --
    float32 depths reinterpreted as sortable uint32 (all positive here, so
    the IEEE bit pattern is already monotonic).
    """
    device = uv.device
    tx, ty = math.ceil(W / tile), math.ceil(H / tile)
    n_tiles = tx * ty
    if ry is None:
        ry = radius

    idx = torch.nonzero(valid, as_tuple=True)[0]
    if idx.numel() == 0:
        return (torch.zeros(0, dtype=torch.int32, device=device),
                torch.zeros(n_tiles + 1, dtype=torch.int32, device=device), tx)

    u, v = uv[idx, 0], uv[idx, 1]
    rx_, ry_ = radius[idx], ry[idx]
    x0 = ((u - rx_) / tile).floor().clamp(0, tx - 1).long()
    x1 = ((u + rx_) / tile).floor().clamp(0, tx - 1).long()
    y0 = ((v - ry_) / tile).floor().clamp(0, ty - 1).long()
    y1 = ((v + ry_) / tile).floor().clamp(0, ty - 1).long()

    nx, ny = (x1 - x0 + 1), (y1 - y0 + 1)
    counts = nx * ny
    total = int(counts.sum().item())

    g = torch.repeat_interleave(torch.arange(idx.numel(), device=device), counts)
    start = torch.cumsum(counts, 0) - counts
    within = torch.arange(total, device=device) - start[g]
    dx, dy = within % nx[g], within // nx[g]
    tix, tiy = x0[g] + dx, y0[g] + dy

    if conic is not None and opacity is not None:
        # Exact ellipse-vs-tile test (gsplat's AccuTile, PR #927). The bounds
        # above are the ellipse's axis-aligned box, so a rotated or elongated
        # gaussian claims corner tiles its ellipse never reaches: measured
        # 38.7% of all pairs at 600k/900x1600, 9.88 -> 6.06 tiles per gaussian.
        #
        # This is EXACT, not conservative-approximate. The rasteriser skips a
        # gaussian at a pixel when alpha < 1/255, and
        #     alpha >= 1/255  <=>  Q <= 2 ln(255 * opacity)
        # with Q the conic's quadratic form. So a pair whose MINIMUM Q over the
        # whole tile rectangle already exceeds that bound cannot contribute to
        # any pixel in the tile, in the forward or the backward. Dropping it
        # leaves the image bit-identical.
        #
        # The minimum of a positive-definite quadratic over a rectangle is
        # either 0 (centre inside) or attained on an edge, where it is a 1-D
        # quadratic with a closed-form minimiser.
        gi = idx[g]
        a_, b_, c_ = conic[gi, 0], conic[gi, 1], conic[gi, 2]
        ux, uy = uv[gi, 0], uv[gi, 1]
        thr = 2.0 * torch.log(torch.clamp(255.0 * opacity[gi], min=1.0 + 1e-6))
        X0 = (tix * tile).to(uv.dtype)
        X1 = torch.clamp((tix + 1) * tile, max=W).to(uv.dtype)
        Y0 = (tiy * tile).to(uv.dtype)
        Y1 = torch.clamp((tiy + 1) * tile, max=H).to(uv.dtype)

        def _q(px, py):
            ex, ey = px - ux, py - uy
            return a_ * ex * ex + 2.0 * b_ * ex * ey + c_ * ey * ey

        best = None
        for yc in (Y0, Y1):                      # horizontal edges, minimise in x
            xs = torch.clamp(ux - b_ * (yc - uy) / torch.clamp(a_, min=1e-12), X0, X1)
            q = _q(xs, yc)
            best = q if best is None else torch.minimum(best, q)
        for xc in (X0, X1):                      # vertical edges, minimise in y
            ys = torch.clamp(uy - b_ * (xc - ux) / torch.clamp(c_, min=1e-12), Y0, Y1)
            best = torch.minimum(best, _q(xc, ys))
        inside = (ux >= X0) & (ux <= X1) & (uy >= Y0) & (uy <= Y1)
        keep = inside | (best < thr)

        g, tix, tiy = g[keep], tix[keep], tiy[keep]

    tile_id = tiy * tx + tix

    d = depth[idx][g]
    dbits = d.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = (tile_id << 32) | dbits
    order = torch.argsort(key)
    tile_sorted = tile_id[order]
    gauss_sorted = idx[g][order].to(torch.int32)

    # tile_sorted is already non-decreasing (tile_id occupies the key's high
    # bits), so the per-tile offsets ARE the searchsorted boundaries -- there is
    # nothing to count. This is not a micro-optimisation: torch.bincount cost
    # 205 ms of a 283 ms training step (73%) while the rasteriser it feeds took
    # 4 ms. bincount HAS an MPS kernel -- it is not a CPU fallback -- but it
    # atomically increments one counter per element, and at 270p there are only
    # 90 tiles, so 389k atomics contend over 90 addresses. That is also why
    # binning measured a mere 15% at 1600p, where ~2800 tiles spread the
    # contention ~30x thinner: the pathology scales with 1/n_tiles, so it is
    # WORST exactly in the small fast configs used for quick iteration.
    # searchsorted is O(n_tiles log n), contention-free, and bit-identical.
    offsets = torch.searchsorted(
        tile_sorted, torch.arange(n_tiles + 1, device=device, dtype=tile_sorted.dtype))
    return gauss_sorted.contiguous(), offsets.to(torch.int32).contiguous(), tx


def build_tile_lists_metal(uv, rxy, depth, valid, conic, opacity, W, H, tile,
                           max_per_tile: int = 0):
    """Tile binning in Metal, with the exact ellipse test applied for free.

    Same contract as build_tile_lists: (gauss_ids, tile_offsets, tiles_x), with
    gauss_ids sorted by (tile, depth). The torch version spends ~15 ms at
    600k/900x1600 on nonzero, four floors, repeat_interleave, arange, modulo and
    gathers before any filtering; here the whole expansion is two kernels, and
    the ellipse test that cost 14.6 ms in torch is a few ALU ops on registers.

    Only the sort stays in torch -- MPS has no exposed device radix sort, and
    torch.sort on the packed key is the fastest thing available. It now sorts
    ~37% fewer keys.
    """
    ext = _load()
    device = uv.device
    tx, ty = math.ceil(W / tile), math.ceil(H / tile)
    n_tiles = tx * ty

    counts = ext.bin_count(uv.contiguous(), rxy.contiguous(), conic.contiguous(),
                           opacity.contiguous(), valid.to(torch.int32).contiguous(),
                           W, H, tile, tx, ty)
    offs = torch.cumsum(counts, 0, dtype=torch.int64) - counts
    total = int(offs[-1].item()) + int(counts[-1].item())
    if total == 0:
        return (torch.zeros(0, dtype=torch.int32, device=device),
                torch.zeros(n_tiles + 1, dtype=torch.int32, device=device), tx)

    keys, ids = ext.bin_write(uv.contiguous(), rxy.contiguous(), conic.contiguous(),
                              opacity.contiguous(), valid.to(torch.int32).contiguous(),
                              depth.contiguous(), offs.to(torch.int32).contiguous(),
                              total, W, H, tile, tx, ty)
    order = torch.argsort(keys)
    tile_sorted = (keys[order] >> 32)
    gauss_sorted = ids[order]
    offsets = torch.searchsorted(
        tile_sorted, torch.arange(n_tiles + 1, device=device, dtype=tile_sorted.dtype))

    if max_per_tile > 0:
        # Keep at most `max_per_tile` gaussians per tile, nearest first.
        #
        # This is what msplat does -- its own warning reads "per-tile overflow
        # (>2048 gaussians in a tile). Some gaussians were dropped from overfull
        # tiles." It is an APPROXIMATION: an overfull tile loses its farthest
        # contributors, so the image changes. This project composites exactly
        # with depth slabs because an earlier per-tile budget here silently
        # discarded 1,035,247 of 1,165,807 intersections (89%) and produced a
        # flat grey image that still looked like a plausible render.
        #
        # The list is already sorted by (tile, depth), so the first N entries of
        # each tile are exactly its nearest N -- the ones that dominate
        # front-to-back compositing. Off by default; see NEGATIVE_RESULTS.md.
        rank = (torch.arange(tile_sorted.numel(), device=device)
                - offsets[:-1].gather(0, tile_sorted))
        keep = rank < max_per_tile
        tile_sorted = tile_sorted[keep]
        gauss_sorted = gauss_sorted[keep]
        offsets = torch.searchsorted(
            tile_sorted,
            torch.arange(n_tiles + 1, device=device, dtype=tile_sorted.dtype))

    return gauss_sorted.contiguous(), offsets.to(torch.int32).contiguous(), tx


class _PreprocessMetal(torch.autograd.Function):
    """Fused projection + SH + activation, forward and analytic backward.

    Exists because profiling showed the torch-side projection eating 2.47s
    forward and its autograd 5.3s backward per step at 1.96M gaussians, while
    the rasterization kernels took 157ms combined. Gradients validated against
    torch autograd of torch_ref: rel <=3.5e-6, cosine 1.000000 on all of
    means/quats/scales/sh.

    SH arrives as two tensors. The trainer keeps the DC band and bands 1+
    separate so Adam can give them different learning rates, and concatenating
    them every step cost 11.2 ms fwd+bwd at 600k. The kernel reads either
    layout, so callers holding a single (N,16,3) tensor pass it as both and the
    gradient comes back as one tensor.
    """

    @staticmethod
    def forward(ctx, means, quats, scales, sh, sh_rest, ctx_opac, viewmat,
                fx, fy, cx, cy, W, H,
                near, far, blur, max_radius, cam_center, sh_degree):
        ext = _load()
        fused = sh_rest is sh
        args = [t.contiguous() for t in (means, quats, scales, sh)]
        rest = args[3] if fused else sh_rest.contiguous()
        uv, conic, depth, rxy, valid, color = ext.preprocess_forward(
            *args, rest, ctx_opac.contiguous(), viewmat, fx, fy, cx, cy, W, H,
            near, far, blur, max_radius, cam_center.contiguous(), sh_degree)
        ctx.save_for_backward(*args, rest, valid, viewmat, cam_center)
        ctx.fused = fused
        ctx.meta = (fx, fy, cx, cy, W, H, near, far, blur, max_radius, sh_degree)
        return uv, conic, depth, rxy, valid, color

    @staticmethod
    def backward(ctx, d_uv, d_conic, d_depth, d_rxy, d_valid, d_color):
        ext = _load()
        means, quats, scales, sh, rest, valid, viewmat, cam_center = ctx.saved_tensors
        fx, fy, cx, cy, W, H, near, far, blur, max_radius, sh_degree = ctx.meta
        d_means, d_quats, d_scales, d_sh, d_sh_rest = ext.preprocess_backward(
            means, quats, scales, sh, rest,
            d_uv.contiguous(), d_conic.contiguous(), d_color.contiguous(), valid,
            viewmat, fx, fy, cx, cy, W, H, near, far, blur, max_radius,
            cam_center.contiguous(), sh_degree)
        # When fused, d_sh and d_sh_rest are the same tensor; returning it twice
        # would double the gradient, so the second slot gets None.
        d_rest = None if ctx.fused else d_sh_rest
        return (d_means, d_quats, d_scales, d_sh, d_rest) + (None,) * 14


class _RasterizeMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uv, conic, opacity, color, gauss_ids, tile_offsets, W, H, tile,
                tiles_x, absgrad_out=None):
        ext = _load()
        args = [t.contiguous() for t in (uv, conic, opacity, color)]
        rgb, alpha, T, ncontrib = ext.rasterize_forward(
            *args, gauss_ids, tile_offsets, W, H, tile, tiles_x)
        ctx.save_for_backward(*args, gauss_ids, tile_offsets, T, ncontrib)
        ctx.dims = (W, H, tile, tiles_x)
        # absgrad is a statistic produced by the backward, not a gradient, so
        # autograd has no way to return it. The caller hands in a buffer and the
        # backward accumulates into it.
        ctx.absgrad_out = absgrad_out
        return rgb, alpha

    @staticmethod
    def backward(ctx, grad_rgb, grad_alpha):
        ext = _load()
        uv, conic, opacity, color, gauss_ids, tile_offsets, T, ncontrib = ctx.saved_tensors
        W, H, tile, tiles_x = ctx.dims
        d_uv, d_conic, d_opacity, d_color, d_absuv = ext.rasterize_backward(
            uv, conic, opacity, color, gauss_ids, tile_offsets, T, ncontrib,
            grad_rgb.contiguous(), grad_alpha.contiguous(), W, H, tile, tiles_x,
            ctx.absgrad_out is not None)
        if ctx.absgrad_out is not None:
            ctx.absgrad_out[:d_absuv.numel()] += d_absuv
        return (d_uv, d_conic, d_opacity, d_color,
                None, None, None, None, None, None, None)


def antialias_scale(conic: torch.Tensor, blur: float) -> torch.Tensor:
    """Mip-Splatting / gsplat "antialiased" opacity compensation.

    Dilating the 2D covariance by `blur` without touching opacity spreads the
    same alpha over a larger footprint, so sub-pixel Gaussians are blurred AND
    dimmed. Rescaling opacity by sqrt(det_before / det_after) makes the dilation
    energy-preserving instead. It is <= 1 always and -> 1 once a Gaussian is
    comfortably wider than a pixel, so it only touches the eroded ones.

    Recovered from the conic the fused preprocess already emits, rather than by
    changing the kernel: conic is the inverse of the DILATED covariance C, so

        det(C)     = 1 / det(conic)
        a, b, c    = conic.z * det(C), -conic.y * det(C), conic.x * det(C)

    and the undilated determinant is (a - blur)(c - blur) - b^2. Gradients flow
    through torch autograd because `conic` carries them out of the Metal
    backward, so no extra adjoint has to be written or validated.
    """
    cxx, cxy, cyy = conic[:, 0], conic[:, 1], conic[:, 2]
    det_conic = (cxx * cyy - cxy * cxy).clamp_min(1e-12)
    det_after = 1.0 / det_conic
    a = cyy * det_after
    b = -cxy * det_after
    c = cxx * det_after
    det_before = (a - blur) * (c - blur) - b * b
    return (det_before.clamp_min(0.0) / det_after).clamp(0.0, 1.0).sqrt()


def render(means, quats, scales, opacities, sh, K, viewmat, W, H,
           sh_degree: int = 3, tile: int | None = None,
           near: float = 0.01, far: float = 100.0,
           background=(0.0, 0.0, 0.0), colors=None, max_radius_frac: float = 1.0,
           sh_rest=None, antialias: bool = False, absgrad_out=None,
           **_ignored):
    """Same positional signature as torch_ref.render. Returns (rgb, alpha, info).

    `sh_rest` is appended LAST rather than beside `sh` on purpose: inserting it
    mid-signature would rebind any caller passing `near`/`far` positionally.
    Pass it (with `sh` as the (N,1,3) DC band) to skip the per-step concatenation
    -- see _PreprocessMetal.
    """
    from metal_gauss.sh import eval_sh, num_sh_bases
    from metal_gauss.torch_ref import build_cov3d, project

    # `colors` is REASSIGNED below by the fused preprocess (it returns the
    # SH-evaluated colours under the same name), so the caller's intent has to
    # be captured here. Branching on `colors is None` after that point silently
    # always took the else-branch.
    explicit_colors = colors is not None

    if not means.device.type == "mps":
        raise RuntimeError(
            f"metal backend requires MPS tensors, got device '{means.device}'. "
            "Refusing to silently move data or fall back to another backend."
        )

    # 16x16, always. An earlier version selected 32x32 for frames above 512k
    # pixels, on the strength of a FORWARD measurement (110 vs 157 ms at
    # 900x1600, 2M splats). That was the wrong measurement to default on:
    # training is dominated by the backward, and 32x32 loses there in every
    # configuration tested, by 32-73% (interleaved A/B, fwd+bwd):
    #
    #     600k @ 270x480     31.5 vs  54.4 ms
    #     600k @ 900x1600    72.2 vs  95.2 ms
    #     1.96M @ 270x480   107.8 vs 182.1 ms
    #     1.96M @ 900x1600  215.1 vs 301.2 ms
    #
    # Forward is a wash (93.8 vs 95.9 at 1.96M/900x1600), so 16 costs nothing
    # even for pure inference. The backward keeps two accumulators and a
    # per-gaussian gradient in registers, so a 32x32 threadgroup runs at much
    # lower occupancy than the forward's staging loop does.
    if tile is None:
        tile = 16

    # Four .item() calls, each a device->host sync that drains the queue when K
    # lives on the GPU: measured 2.69 ms per forward with work in flight, vs
    # 0.01 ms when K is already on the host. The intrinsics are read as Python
    # floats and never touch a kernel as a tensor, so there is no reason for
    # them to be on the device at all.
    K_h = K if K.device.type == "cpu" else K.detach().cpu()
    fx, fy, cx, cy = K_h[0, 0].item(), K_h[1, 1].item(), K_h[0, 2].item(), K_h[1, 2].item()

    if colors is not None:
        # explicit-colour path (used by depth rendering etc.): torch projection.
        # Unlike the fused preprocess, this one runs the projection in torch, so
        # the pose has to be on the same device as the gaussians -- callers may
        # now hand it over on the host, which is what the metal path wants.
        cov3d = build_cov3d(quats, scales)
        vm_dev = viewmat if viewmat.device == means.device else viewmat.to(means.device)
        uv, conic, depth, radius, valid, opacity_scale = project(
            means, cov3d, vm_dev, fx, fy, cx, cy, W, H, near, far,
            max_radius_frac=max_radius_frac, antialias=antialias)
        opacities = opacities * opacity_scale
        valid_b = valid
        ry_extent = None          # circular bound on the explicit-colour path
    else:
        # fused Metal preprocess: projection + SH + activation in two kernels
        #
        # viewmat and cam_center are read on the HOST -- makeParams copies them
        # into the kernel's constant buffer. Handing them over as MPS tensors
        # made that a device->host copy, which drains the queue: once in the
        # forward and again in the backward, ~2 ms each at 600k. Normalising to
        # CPU here means a caller who keeps the pose on CPU (as train.py now
        # does) pays nothing, and a caller who passes an MPS tensor pays one
        # copy instead of two.
        vm_h = viewmat.detach()
        if vm_h.device.type != "cpu":
            vm_h = vm_h.cpu()
        cam_center = -vm_h[:3, :3].T @ vm_h[:3, 3]
        # Two SH layouts. Callers with one (N,16,3) tensor pass sh alone and it
        # is handed to the kernel as both buffers. The trainer keeps the DC band
        # and bands 1+ as separate leaves (different Adam learning rates) and
        # passes both, which is why the kernel takes a layout: concatenating
        # them every step cost 11.2 ms fwd+bwd at 600k.
        if sh_rest is None:
            sh_in = sh
            if sh_in.shape[1] < 16:   # kernel expects 16 bases; zero-pad
                pad = torch.zeros(sh_in.shape[0], 16 - sh_in.shape[1], 3,
                                  device=sh_in.device, dtype=sh_in.dtype)
                sh_in = torch.cat([sh_in, pad], dim=1)
            rest_in = sh_in
        else:
            if sh.shape[1] != 1:
                raise ValueError(
                    f"with sh_rest given, sh must be the DC band (N,1,3); got {tuple(sh.shape)}")
            sh_in, rest_in = sh, sh_rest
            if rest_in.shape[1] < 15:      # zero-pad bands 1+ to a full set
                pad = torch.zeros(rest_in.shape[0], 15 - rest_in.shape[1], 3,
                                  device=rest_in.device, dtype=rest_in.dtype)
                rest_in = torch.cat([rest_in, pad], dim=1)
        uv, conic, depth, rxy, valid_i, colors = _PreprocessMetal.apply(
            means, quats, scales, sh_in, rest_in, opacities.detach(), vm_h.contiguous(),
            fx, fy, cx, cy, W, H, near, far, 0.3,
            max_radius_frac * max(W, H), cam_center, sh_degree)
        radius = rxy[:, 0]
        ry_extent = rxy[:, 1].detach()
        valid_b = valid_i.bool()
        valid = valid_b
        if antialias:
            # Before binning on purpose: the SnugBox bound is derived from
            # opacity, so a compensated (lower) opacity gives a tighter and
            # still-exact bound. Applying it after binning would bin against an
            # opacity no longer used for compositing.
            opacities = opacities * antialias_scale(conic, 0.3)

    # Binning in Metal, with the exact ellipse-vs-tile test applied inside the
    # expansion where it is nearly free. Measured at 600k/900x1600 on an idle
    # GPU (an earlier set of numbers here was taken while a training run held
    # the device and overstated the gain):
    #     binning    12.82 ->  5.70 ms
    #     raster fwd 10.72 ->  8.95 ms
    #     raster bwd 29.65 -> 23.84 ms
    #     bin+raster 53.19 -> 38.50 ms  (1.38x)
    #     end-to-end 78.2  -> 60.1  ms  (1.30x, interleaved A/B)
    # The image is bit-identical: the test drops only pairs whose conic cannot
    # reach alpha 1/255 anywhere in the tile, which is the rasteriser's own
    # cutoff. The torch path stays for the explicit-colour route (no conic from
    # the fused preprocess) and as the reference the kernel is checked against.
    if not explicit_colors:
        gauss_ids, tile_offsets, tiles_x = build_tile_lists_metal(
            uv.detach(), rxy.detach(), depth.detach(), valid_i,
            conic.detach(), opacities.detach(), W, H, tile)
    else:
        gauss_ids, tile_offsets, tiles_x = build_tile_lists(
            uv.detach(), radius.detach(), depth.detach(), valid_b, W, H, tile,
            ry=ry_extent)

    rgb, alpha = _RasterizeMetal.apply(
        uv, conic, opacities, colors, gauss_ids, tile_offsets, W, H, tile, tiles_x,
        absgrad_out)

    if background is not None:
        bg = torch.as_tensor(background, device=rgb.device, dtype=rgb.dtype)
        rgb = rgb + (1.0 - alpha).clamp_min(0)[..., None] * bg

    # uv is the SCREEN-SPACE position. Its gradient is what 3DGS-ADC densifies
    # on -- large ||dL/duv|| means the gaussian is being pulled hard in image
    # space, i.e. it is straddling detail it cannot represent. Retained here so
    # the trainer can accumulate it; it is (N,2), ~5 MB at 600k, and only kept
    # when grad is actually enabled.
    if uv.requires_grad:
        uv.retain_grad()

    return rgb, alpha, {
        "uv": uv,
        "tiles": int(tile_offsets.numel() - 1),
        "isect_total": int(gauss_ids.numel()),
        "isect_dropped": 0,
        "isect_dropped_frac": 0.0,
        "valid_mask": valid_b,          # tensor, no sync; selective-Adam hook
        "backend": "metal",
    }

