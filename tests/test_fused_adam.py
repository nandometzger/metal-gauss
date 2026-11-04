"""FusedAdam must be torch.optim.Adam, not merely Adam-like.

Checked over many steps rather than one: the bias corrections change with the
step count, so a formula error in bc1/bc2 (or eps entering before the
sqrt(bc2) division rather than after) is invisible at step 1 and grows later.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(),
                                reason="requires MPS")


def _params(seed=7):
    torch.manual_seed(seed)
    shapes = [(5000, 3), (5000, 4), (5000, 1, 3), (5000, 15, 3), (5000,)]
    return [torch.randn(*s, device="mps", requires_grad=True) for s in shapes]


def test_matches_torch_adam_over_many_steps():
    from metal_gauss.fused_adam import FusedAdam
    lrs = [1.6e-4, 1e-3, 2.5e-3, 1.25e-4, 1e-2]
    A, B = _params(), _params()
    oa = torch.optim.Adam([{"params": [t], "lr": l} for t, l in zip(A, lrs)], eps=1e-15)
    ob = FusedAdam([{"params": [t], "lr": l} for t, l in zip(B, lrs)], eps=1e-15)
    torch.manual_seed(11)
    for _ in range(30):
        gs = [torch.randn_like(t) for t in A]
        for t, g in zip(A, gs):
            t.grad = g.clone()
        for t, g in zip(B, gs):
            t.grad = g.clone()
        oa.step()
        ob.step()
    for a, b in zip(A, B):
        rel = ((a - b).abs().max() / a.abs().max()).item()
        assert rel < 1e-5, f"diverged from torch Adam: rel {rel:.2e}"
        # The moments too: the parameter only sees sqrt(v), which halves an
        # error in v -- (1 - b2) rounded in float hid there at 1.3e-5.
        for key in ("exp_avg", "exp_avg_sq"):
            x, y = oa.state[a][key], ob.state[b][key]
            rel = ((x - y).abs().max() / x.abs().max()).item()
            assert rel < 1e-5, f"{key} diverged from torch Adam: rel {rel:.2e}"


def test_state_layout_matches_torch():
    """mcmc.reset_adam_state reaches into opt.state[p]['exp_avg'] directly."""
    from metal_gauss.fused_adam import FusedAdam
    p = _params()[:1]
    o = FusedAdam([{"params": p, "lr": 1e-3}], eps=1e-15)
    p[0].grad = torch.randn_like(p[0])
    o.step()
    st = o.state[p[0]]
    assert {"step", "exp_avg", "exp_avg_sq"} <= set(st)
    assert st["exp_avg"].shape == p[0].shape


def test_relocation_reset_still_works():
    from metal_gauss.fused_adam import FusedAdam
    from metal_gauss.mcmc import reset_adam_state
    p = _params()[:1]
    params = {"means": p[0]}
    o = FusedAdam([{"params": p, "lr": 1e-3}], eps=1e-15)
    p[0].grad = torch.randn_like(p[0])
    o.step()
    assert o.state[p[0]]["exp_avg"].abs().sum() > 0
    idx = torch.arange(100, device="mps")
    reset_adam_state(o, params, idx)
    assert o.state[p[0]]["exp_avg"][idx].abs().sum() == 0
