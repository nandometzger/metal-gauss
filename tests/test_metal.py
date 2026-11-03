"""Metal backend correctness, judged entirely against the torch_ref oracle.

A hand-written GPU kernel with nothing to check it against is a random number
generator. Every assertion here compares Metal output to torch_ref, which is
itself gradcheck-verified in float64.
"""

from __future__ import annotations

import pytest
import torch

from metal_gauss import render

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal backend requires MPS"
)


def intrinsics(W, H, f=None):
    f = f or 0.8 * max(W, H)
    K = torch.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, W / 2, H / 2
    return K


def scene(n=400, seed=0, device="mps", requires_grad=False):
    torch.manual_seed(seed)
    means = (torch.randn(n, 3) * 0.6 + torch.tensor([0.0, 0.0, 4.0])).to(device)
    quats = torch.randn(n, 4).to(device)
    scales = (torch.rand(n, 3) * 0.10 + 0.03).to(device)
    opac = (torch.rand(n) * 0.7 + 0.15).to(device)
    cols = torch.rand(n, 3).to(device)
    if requires_grad:
        for t in (means, quats, scales, opac, cols):
            t.requires_grad_(True)
    return means, quats, scales, opac, cols


def test_an_empty_scene_renders_the_background():
    """A crop box can keep nothing, and so can a frustum on an empty file.

    The torch binning path has always returned empty lists for this; the Metal
    one read `offs[-1]` off a zero-length count and raised IndexError. The
    viewer contained it, but an empty box has an answer: the background.
    """
    W, H = 48, 32
    empty = torch.zeros(0, 3, device="mps")
    rgb, alpha, _ = render(empty, torch.zeros(0, 4, device="mps"),
                           torch.zeros(0, 3, device="mps"), torch.zeros(0, device="mps"),
                           torch.zeros(0, 16, 3, device="mps"),
                           intrinsics(W, H), torch.eye(4), W, H,
                           backend="metal", background=(1.0, 0.0, 0.0))
    assert rgb.shape == (H, W, 3) and alpha.shape == (H, W)
    assert torch.equal(rgb, torch.tensor([1.0, 0.0, 0.0], device="mps").expand(H, W, 3))
    assert float(alpha.max()) == 0.0


def test_metal_forward_matches_reference():
    W, H = 96, 64
    K, vm = intrinsics(W, H).to("mps"), torch.eye(4, device="mps")
    m, q, s, o, c = scene()

    ref = render(m, q, s, o, None, K, vm, W, H, colors=c, backend="torch_ref")
    met = render(m, q, s, o, None, K, vm, W, H, colors=c, backend="metal")

    for i, name in [(0, "rgb"), (1, "alpha")]:
        a, b = ref[i].cpu(), met[i].cpu()
        err = (a - b).abs().max().item()
        assert err < 2e-3, f"{name} max abs diff {err:.3e} vs torch_ref"


def test_metal_forward_matches_reference_with_sh():
    W, H = 64, 64
    K, vm = intrinsics(W, H).to("mps"), torch.eye(4, device="mps")
    m, q, s, o, _ = scene(n=250, seed=3)
    sh = (torch.randn(250, 16, 3) * 0.25).to("mps")

    ref = render(m, q, s, o, sh, K, vm, W, H, sh_degree=3, backend="torch_ref")
    met = render(m, q, s, o, sh, K, vm, W, H, sh_degree=3, backend="metal")
    err = (ref[0].cpu() - met[0].cpu()).abs().max().item()
    assert err < 2e-3, f"rgb max abs diff {err:.3e} vs torch_ref"


def test_metal_reports_no_dropped_intersections():
    W, H = 96, 64
    K, vm = intrinsics(W, H).to("mps"), torch.eye(4, device="mps")
    m, q, s, o, c = scene(n=800, seed=5)
    _, _, info = render(m, q, s, o, None, K, vm, W, H, colors=c, backend="metal")
    assert info["backend"] == "metal"
    assert info["isect_dropped"] == 0
    assert info["isect_total"] > 0


def test_metal_gradients_match_reference():
    """The whole point of writing the backward kernel: gradients must agree."""
    W, H = 64, 48
    K, vm = intrinsics(W, H).to("mps"), torch.eye(4, device="mps")

    grads = {}
    for backend in ("torch_ref", "metal"):
        m, q, s, o, c = scene(n=300, seed=11, requires_grad=True)
        rgb, alpha, _ = render(m, q, s, o, None, K, vm, W, H, colors=c, backend=backend)
        loss = rgb.square().mean() + 0.1 * alpha.square().mean()
        loss.backward()
        grads[backend] = {k: v.grad.detach().cpu().clone() for k, v in
                          dict(means=m, quats=q, scales=s, opacities=o, colors=c).items()}

    for name in grads["torch_ref"]:
        a, b = grads["torch_ref"][name], grads["metal"][name]
        assert torch.isfinite(b).all(), f"{name}: metal gradient has non-finite values"
        scale = a.abs().max().clamp_min(1e-8)
        rel = (a - b).abs().max() / scale
        assert rel < 5e-2, (
            f"{name}: metal vs torch_ref relative grad error {rel:.3e} "
            f"(ref max {a.abs().max():.3e}, metal max {b.abs().max():.3e})"
        )


def test_metal_sh_gradients_match_reference_with_offscreen_gaussians():
    """The fused preprocess skips SH for gaussians the view never sees.

    The test above passes explicit colours, which bypasses the fused
    preprocess entirely, so this is the only check of its SH and projection
    backward against the oracle. Half the scene is pushed out of frame: those
    take the skip path in both kernels and must come back exactly zero, while
    the visible half must still match torch_ref.
    """
    W, H = 64, 48
    K, vm = intrinsics(W, H).to("mps"), torch.eye(4, device="mps")
    n = 300
    m0, q0, s0, o0, _ = scene(n=n, seed=13)
    off = torch.arange(n, device="mps") % 2 == 1
    m0 = m0.clone()
    m0[off, 0] += 20.0                    # far outside the frustum at z ~ 4
    sh0 = (torch.randn(n, 16, 3) * 0.25).to("mps")

    grads = {}
    for backend in ("torch_ref", "metal"):
        m, q, s, sh = (t.clone().requires_grad_(True) for t in (m0, q0, s0, sh0))
        rgb, alpha, _ = render(m, q, s, o0, sh, K, vm, W, H, sh_degree=3, backend=backend)
        (rgb.square().mean() + 0.1 * alpha.square().mean()).backward()
        grads[backend] = dict(means=m.grad, quats=q.grad, scales=s.grad, sh=sh.grad)

    for name, b in grads["metal"].items():
        a = grads["torch_ref"][name]
        assert torch.isfinite(b).all(), f"{name}: metal gradient has non-finite values"
        assert float(b[off].abs().max()) == 0.0, f"{name}: off-screen gaussian got a gradient"
        rel = (a - b).abs().max() / a.abs().max().clamp_min(1e-8)
        assert rel < 5e-2, f"{name}: metal vs torch_ref relative grad error {rel:.3e}"


def test_metal_refuses_cpu_tensors():
    """No silent device migration: a CPU tensor must be an error, not a copy."""
    from metal_gauss import metal_backend
    W = H = 16
    m, q, s, o, c = scene(n=10, device="cpu")
    with pytest.raises(RuntimeError, match="requires MPS tensors"):
        metal_backend.render(m, q, s, o, None, intrinsics(W, H), torch.eye(4), W, H, colors=c)


def elongated_scene(n=300, seed=17, requires_grad=False):
    """Long, thin, rotated splats: an ellipse that clears its tile's bounds
    but reaches only some of the tile's simdgroups, which is the case the
    rasteriser's per-simdgroup cull has to get right."""
    m, q, s, o, c = scene(n=n, seed=seed)
    s = s * torch.tensor([4.0, 0.5, 1.0], device=s.device)
    if requires_grad:
        for t in (m, q, s, o, c):
            t.requires_grad_(True)
    return m, q, s, o, c


@pytest.mark.parametrize("tile", [8, 16, 32])   # 2, 8 and 32 simdgroups
def test_metal_matches_reference_at_every_tile_size(tile):
    """The cull keeps one mask bit per simdgroup; 32x32 uses all 32 bits."""
    W, H = 96, 64
    K, vm = intrinsics(W, H).to("mps"), torch.eye(4, device="mps")

    out, grads = {}, {}
    for backend in ("torch_ref", "metal"):
        m, q, s, o, c = elongated_scene(requires_grad=True)
        # torch_ref's image depends on its own tile size too, so match it.
        rgb, alpha, _ = render(m, q, s, o, None, K, vm, W, H, colors=c, backend=backend,
                               tile=tile)
        (rgb.square().mean() + 0.1 * alpha.square().mean()).backward()
        out[backend] = (rgb.detach().cpu(), alpha.detach().cpu())
        grads[backend] = {k: v.grad.detach().cpu() for k, v in
                          dict(means=m, quats=q, scales=s, opacities=o, colors=c).items()}

    for i, name in [(0, "rgb"), (1, "alpha")]:
        err = (out["torch_ref"][i] - out["metal"][i]).abs().max().item()
        assert err < 2e-3, f"tile {tile}: {name} max abs diff {err:.3e} vs torch_ref"
    for name, a in grads["torch_ref"].items():
        b = grads["metal"][name]
        rel = (a - b).abs().max() / a.abs().max().clamp_min(1e-8)
        assert rel < 5e-2, f"tile {tile}: {name} relative grad error {rel:.3e}"


def test_metal_refuses_a_tile_the_gpu_cannot_dispatch():
    """A threadgroup over the pipeline's limit is dropped by Metal without an
    error, so the image and every gradient come back as zeros. One tile is one
    threadgroup: 64x64 is 4096 threads, past any Apple GPU's 1024."""
    W, H = 96, 64
    m, q, s, o, c = scene(n=50)
    with pytest.raises(RuntimeError, match="threads per threadgroup"):
        render(m, q, s, o, None, intrinsics(W, H).to("mps"), torch.eye(4, device="mps"),
               W, H, colors=c, backend="metal", tile=64)

