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


def _selective_pair(n=5000, row_steps=True):
    """The trainer's six gaussian groups plus a dense appearance-like group,
    once through SelectiveAdam's torch path and once through the Metal one."""
    from metal_gauss.selective_adam import SelectiveAdam
    torch.manual_seed(3)
    shapes = [(n, 3), (n, 3), (n, 4), (n,), (n, 1, 3), (n, 15, 3), (4, 3)]
    lrs = [1.6e-4, 5e-3, 1e-3, 1e-2, 2.5e-3, 1.25e-4, 3e-3]
    init = [torch.randn(*s, device="mps") for s in shapes]
    A = [t.clone().requires_grad_(True) for t in init]
    B = [t.clone().requires_grad_(True) for t in init]
    opts = []
    for ts, fused in ((A, False), (B, True)):
        o = SelectiveAdam([{"params": [t], "lr": lr} for t, lr in zip(ts[:-1], lrs)],
                          eps=1e-15, fused=fused, row_steps=row_steps)
        o.add_param_group({"params": [ts[-1]], "lr": lrs[-1], "rowwise": False})
        opts.append(o)
    return opts[0], opts[1], A, B


@pytest.mark.parametrize("row_steps", [True, False])
def test_fused_selective_adam_matches_the_torch_path_over_many_steps(row_steps):
    """Per-row bias correction only shows once rows' step counts diverge, so
    this runs 40 steps with visibility from 10% to 100% (both of the torch
    path's branches), an active set smaller than the capacity, and an MCMC
    reset of some rows' history halfway through."""
    from metal_gauss.mcmc import reset_adam_state
    n, active = 5000, 4000
    ref, fus, A, B = _selective_pair(n, row_steps)
    gen = torch.Generator().manual_seed(5)
    for step in range(40):
        frac = (0.1, 0.3, 0.8, 1.0)[step % 4]
        vis = (torch.rand(active, generator=gen) < frac).to("mps")
        grads = [torch.randn(t.shape, generator=gen).to("mps") for t in A]
        for ts in (A, B):
            for t, g in zip(ts, grads):
                t.grad = g.clone()
        ref.step(vis)
        fus.step(vis)
        if step == 20:
            idx = torch.arange(0, active, 7, device="mps")
            for o, ts in ((ref, A), (fus, B)):
                reset_adam_state(o, dict(enumerate(ts[:-1])), idx)

    for i, (a, b) in enumerate(zip(A, B)):
        sa, sb = ref.state[a], fus.state[b]
        for name, x, y in (("param", a, b), ("exp_avg", sa["exp_avg"], sb["exp_avg"]),
                           ("exp_avg_sq", sa["exp_avg_sq"], sb["exp_avg_sq"])):
            rel = ((x - y).abs().max() / x.abs().max().clamp_min(1e-30)).item()
            assert rel < 1e-5, f"group {i} {name}: rel {rel:.2e} vs the torch path"
        if "steps" in sa:
            assert torch.equal(sa["steps"], sb["steps"]), f"group {i}: step counts differ"


def test_fused_selective_adam_leaves_hidden_rows_bit_identical():
    """A hidden row's parameter AND moments must not move at all: not decayed,
    not touched by a regulariser's gradient, not rounded."""
    from metal_gauss.selective_adam import SelectiveAdam
    torch.manual_seed(8)
    t = torch.randn(3000, 15, 3, device="mps", requires_grad=True)
    p0 = t.detach().clone()
    o = SelectiveAdam([{"params": [t], "lr": 1e-2}], eps=1e-15, fused=True)
    vis = torch.zeros(2000, dtype=torch.bool, device="mps")   # 1000 rows beyond active
    vis[::3] = True
    for _ in range(3):
        t.grad = torch.randn_like(t)
        o.step(vis)
    hidden = torch.ones(3000, dtype=torch.bool, device="mps")
    hidden[:2000] = ~vis
    st = o.state[t]
    assert torch.equal(t[hidden], p0[hidden])
    assert torch.equal(st["exp_avg"][hidden], torch.zeros_like(st["exp_avg"][hidden]))
    assert torch.equal(st["exp_avg_sq"][hidden], torch.zeros_like(st["exp_avg_sq"][hidden]))
    assert not torch.equal(t[~hidden], p0[~hidden])
    assert torch.equal(st["steps"], hidden.logical_not().float() * 3)


def test_fused_selective_adam_with_global_steps_is_adam_on_the_visible_rows():
    """row_steps=False is gsplat's selective Adam: a visible row gets Adam's
    update at the optimiser's own step count t, a hidden row is frozen. The
    reference is that sentence written out in torch."""
    from metal_gauss.selective_adam import SelectiveAdam
    torch.manual_seed(9)
    n, active, lr, b1, b2, eps = 3000, 2500, 1e-2, 0.9, 0.999, 1e-15
    p0 = torch.randn(n, 15, 3, device="mps")
    x = p0.clone().requires_grad_(True)
    o = SelectiveAdam([{"params": [x], "lr": lr}], eps=eps, fused=True, row_steps=False)
    p, m, v = p0.clone(), torch.zeros_like(p0), torch.zeros_like(p0)
    gen = torch.Generator().manual_seed(4)
    for t in range(1, 31):
        vis = (torch.rand(active, generator=gen) < 0.3).to("mps")
        g = torch.randn(p0.shape, generator=gen).to("mps")
        x.grad = g.clone()
        o.step(vis)
        rows = torch.zeros(n, dtype=torch.bool, device="mps")
        rows[:active] = vis
        r = rows.view(-1, 1, 1)
        m = torch.where(r, b1 * m + (1 - b1) * g, m)
        v = torch.where(r, b2 * v + (1 - b2) * g * g, v)
        step = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
        p = torch.where(r, p - step * m / (v.sqrt() + eps), p)
    rel = ((x.detach() - p).abs().max() / p.abs().max()).item()
    assert rel < 1e-5, f"global-step selective Adam: rel {rel:.2e} vs the reference"
    capacity = torch.zeros(n, dtype=torch.bool, device="mps")
    capacity[active:] = True                 # past the mask: hidden every step
    assert torch.equal(x.detach()[capacity], p0[capacity])


def test_fused_selective_adam_dense_groups_are_torch_adam():
    """--selective-adam-groups sh keeps geometry and opacity dense. Those groups
    must be torch.optim.Adam exactly -- hidden rows keep their momentum -- not
    a selective update with every row marked visible."""
    from metal_gauss.selective_adam import SelectiveAdam
    lrs = [1.6e-4, 1e-3, 2.5e-3, 1.25e-4, 1e-2]
    A, B = _params(), _params()
    oa = torch.optim.Adam([{"params": [t], "lr": lr} for t, lr in zip(A, lrs)], eps=1e-15)
    ob = SelectiveAdam([{"params": [t], "lr": lr, "rowwise": False} for t, lr in zip(B, lrs)],
                       eps=1e-15, fused=True)
    vis = torch.zeros(5000, dtype=torch.bool, device="mps")
    vis[::5] = True
    torch.manual_seed(12)
    for _ in range(30):
        gs = [torch.randn_like(t) for t in A]
        for t, g in zip(A, gs):
            t.grad = g.clone()
        for t, g in zip(B, gs):
            t.grad = g.clone()
        oa.step()
        ob.step(vis)
    for a, b in zip(A, B):
        rel = ((a - b).abs().max() / a.abs().max()).item()
        assert rel < 1e-5, f"dense group diverged from torch Adam: rel {rel:.2e}"


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
