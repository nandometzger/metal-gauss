"""The fused Metal SSIM tail must equal the torch expression it replaced.

Value AND gradient: the tail is a quotient of four terms that each depend on
mx/my, so a dropped chain-rule path shows up only in the gradient, and only
for some inputs. Random images are used rather than smooth ones so sxx/syy/sxy
are well away from zero.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(),
                                reason="requires MPS")

C1, C2 = 0.01 ** 2, 0.03 ** 2


def _reference(a, b, kernel):
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    stack = torch.cat([x, y, x * x, y * y, x * y], dim=1)
    pad = kernel.shape[-1] // 2
    t = F.conv2d(stack, kernel, padding=(0, pad), groups=15)
    bl = F.conv2d(t, kernel.transpose(2, 3), padding=(pad, 0), groups=15)
    mx, my, exx, eyy, exy = bl.split(3, dim=1)
    sxx, syy, sxy = exx - mx ** 2, eyy - my ** 2, exy - mx * my
    return (((2 * mx * my + C1) * (2 * sxy + C2))
            / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))).mean()


@pytest.mark.parametrize("H,W", [(64, 64), (200, 300), (97, 131)])
def test_value_and_gradient_match_torch(H, W):
    from metal_gauss.train import _gaussian_kernel, ssim
    torch.manual_seed(H * W)
    a = torch.rand(H, W, 3, device="mps", requires_grad=True)
    b = torch.rand(H, W, 3, device="mps")
    ker = _gaussian_kernel(device="mps")

    v1 = ssim(a, b, ker)
    g1, = torch.autograd.grad(v1, a)
    a2 = a.detach().clone().requires_grad_(True)
    v2 = _reference(a2, b, ker)
    g2, = torch.autograd.grad(v2, a2)

    assert abs(v1.item() - v2.item()) < 1e-6
    rel = ((g1 - g2).abs().max() / g2.abs().max()).item()
    cos = F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
    assert rel < 1e-4, f"gradient rel err {rel:.2e}"
    # MPS convolution/reduction order can vary slightly across supported
    # PyTorch and macOS releases. The relative-error check above remains
    # strict; keep this directional check tolerant to the last few ULPs.
    assert cos > 1 - 1e-6, f"gradient cosine {cos:.8f}"


def test_identical_images_give_ssim_one():
    from metal_gauss.train import _gaussian_kernel, ssim
    torch.manual_seed(3)
    a = torch.rand(80, 80, 3, device="mps")
    ker = _gaussian_kernel(device="mps")
    assert abs(ssim(a, a, ker).item() - 1.0) < 1e-5
