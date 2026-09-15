"""Index-based Adam that only updates visible gaussians.

Per view only ~20-40% of gaussians project into the frustum; the rest have
exactly-zero gradients, yet a dense Adam still reads and writes all N x 64
parameter/state floats every step. Updating the visible subset by index cuts
optimizer time and memory traffic proportionally (gsplat's selective_adam
idea). Bias correction uses a PER-GAUSSIAN step count, matching what dense
Adam would have done had the invisible steps simply not occurred.

The trainer treats this optimizer like a regular torch optimizer: it changes
param_groups[0]["lr"], adds an appearance-parameter group, and asks MCMC
to reset state after relocating or growing gaussians. The implementation keeps
that interface and uses torch Adam's state names so those operations work for
selective Adam as well.
"""

from __future__ import annotations

import torch


class SelectiveAdam(torch.optim.Optimizer):
    """Adam with per-row updates for Gaussian parameters.

    Groups marked rowwise (the default) use the visibility mask passed to
    step() and maintain one Adam step count per Gaussian. Dense groups
    are updated with ordinary Adam. The latter is needed for parameters such
    as the optional per-training-image appearance correction, whose first
    dimension is the number of images rather than the number of Gaussians.
    """

    def __init__(self, groups: list[dict], eps: float = 1e-15,
                 betas: tuple[float, float] = (0.9, 0.999)):
        defaults = {"lr": 1e-3, "eps": eps, "betas": betas, "rowwise": True}
        super().__init__(groups, defaults)
        # Keep the old attribute as a compatibility alias for callers that
        # used the initial implementation before it matched torch's API.
        self.groups = self.param_groups

    def _state_for(self, p: torch.Tensor, rowwise: bool) -> dict:
        st = self.state[p]
        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)
            if rowwise:
                st["steps"] = torch.zeros(p.shape[0], device=p.device)
            else:
                st["step"] = 0
        return st

    @staticmethod
    def _full_visible(visible: torch.Tensor, size: int,
                      device: torch.device) -> torch.Tensor:
        """Pad an active-set mask to the preallocated Gaussian capacity."""
        if visible.numel() > size:
            raise ValueError(
                f"visibility mask has {visible.numel()} rows, but parameter "
                f"has only {size}")
        visible = visible.to(device=device)
        if visible.numel() == size:
            return visible
        full = torch.zeros(size, dtype=torch.bool, device=device)
        full[:visible.numel()] = visible
        return full

    @staticmethod
    def _shape_for_rows(p: torch.Tensor) -> list[int]:
        return [-1] + [1] * (p.dim() - 1)

    @staticmethod
    def _check_gradient(p: torch.Tensor) -> torch.Tensor:
        grad = p.grad
        if grad is None:
            raise AssertionError("gradient checked only for non-None gradients")
        if grad.is_sparse:
            raise RuntimeError("SelectiveAdam does not support sparse gradients")
        if grad.shape != p.shape:
            raise ValueError(
                f"gradient shape {tuple(grad.shape)} does not match parameter "
                f"shape {tuple(p.shape)}")
        return grad

    @staticmethod
    def _hyper(group: dict) -> tuple[float, float, float, float]:
        b1, b2 = group["betas"]
        return group["lr"], b1, b2, group["eps"]

    @torch.no_grad()
    def step(self, visible: torch.Tensor, closure=None):
        """Update visible Gaussian rows and any dense auxiliary groups."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if visible.ndim != 1 or visible.dtype is not torch.bool:
            raise TypeError("visible must be a one-dimensional boolean tensor")

        # The Metal renderer normally returns a mask on the same device as the
        # Gaussian parameters. Keep the CPU-only/unit-test case convenient too.
        for group in self.param_groups:
            params = group["params"]
            if params:
                visible = visible.to(device=params[0].device)
                break

        n = visible.numel()
        nvis = int(visible.sum().item())
        if nvis > n // 2:
            self._step_masked(visible)
        else:
            self._step_indexed(visible)
        return loss

    @torch.no_grad()
    def _step_indexed(self, visible: torch.Tensor) -> None:
        """Use gathers/scatters when a minority of active rows is visible."""
        idx = visible.nonzero(as_tuple=True)[0]
        for group in self.param_groups:
            if not group["rowwise"]:
                self._step_dense(group)
                continue
            lr, b1, b2, eps = self._hyper(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() == 0 or p.shape[0] < visible.numel():
                    raise ValueError(
                        "rowwise SelectiveAdam parameters must have at least "
                        "one row per visibility-mask entry")
                grad = self._check_gradient(p)
                if idx.numel() == 0:
                    continue
                st = self._state_for(p, rowwise=True)
                steps = st["steps"][idx] + 1
                gsel = grad[idx]
                m = st["exp_avg"][idx] * b1 + gsel * (1 - b1)
                v = (st["exp_avg_sq"][idx] * b2
                     + gsel * gsel * (1 - b2))
                st["steps"][idx] = steps
                st["exp_avg"][idx] = m
                st["exp_avg_sq"][idx] = v
                k = steps
                bc1 = 1 - b1 ** k
                bc2 = 1 - b2 ** k
                shape = self._shape_for_rows(p)
                step_size = lr * (bc2.sqrt() / bc1).reshape(shape)
                p[idx] -= step_size * m / (v.sqrt() + eps)

    @torch.no_grad()
    def _step_masked(self, visible: torch.Tensor) -> None:
        """Use masked dense operations when most active rows are visible."""
        for group in self.param_groups:
            if not group["rowwise"]:
                self._step_dense(group)
                continue
            lr, b1, b2, eps = self._hyper(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() == 0 or p.shape[0] < visible.numel():
                    raise ValueError(
                        "rowwise SelectiveAdam parameters must have at least "
                        "one row per visibility-mask entry")
                grad = self._check_gradient(p)
                full_visible = self._full_visible(
                    visible, p.shape[0], p.device)
                shape = self._shape_for_rows(p)
                mask = full_visible.reshape(shape)
                st = self._state_for(p, rowwise=True)
                st["steps"] += full_visible.to(st["steps"].dtype)
                masked_grad = grad * mask.to(p.dtype)
                st["exp_avg"] = torch.where(
                    mask, st["exp_avg"] * b1 + masked_grad * (1 - b1),
                    st["exp_avg"])
                st["exp_avg_sq"] = torch.where(
                    mask,
                    st["exp_avg_sq"] * b2
                    + masked_grad * masked_grad * (1 - b2),
                    st["exp_avg_sq"])
                k = st["steps"].clamp_min(1.0)
                bc1 = 1 - b1 ** k
                bc2 = 1 - b2 ** k
                step_size = (lr * bc2.sqrt() / bc1).reshape(shape)
                p -= (mask.to(p.dtype) * step_size * st["exp_avg"]
                      / (st["exp_avg_sq"].sqrt() + eps))

    @torch.no_grad()
    def _step_dense(self, group: dict) -> None:
        """Apply ordinary Adam to an auxiliary, non-Gaussian group."""
        lr, b1, b2, eps = self._hyper(group)
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = self._check_gradient(p)
            st = self._state_for(p, rowwise=False)
            st["step"] += 1
            st["exp_avg"].mul_(b1).add_(grad, alpha=1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(
                grad, grad, value=1 - b2)
            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            step_size = lr * (bc2 ** 0.5) / bc1
            p.addcdiv_(st["exp_avg"],
                       st["exp_avg_sq"].sqrt().add(eps),
                       value=-step_size)
