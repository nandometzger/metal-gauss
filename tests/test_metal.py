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


def test_metal_refuses_cpu_tensors():
    """No silent device migration: a CPU tensor must be an error, not a copy."""
    from metal_gauss import metal_backend
    W = H = 16
    m, q, s, o, c = scene(n=10, device="cpu")
    with pytest.raises(RuntimeError, match="requires MPS tensors"):
        metal_backend.render(m, q, s, o, None, intrinsics(W, H), torch.eye(4), W, H, colors=c)
