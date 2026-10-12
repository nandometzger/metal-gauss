"""Benchmark metal-gauss on any 3DGS ply + COLMAP reconstruction.

    python bench/bench.py --ply scene.ply --colmap sparse/0 --width 900 --height 1600
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metal_gauss import render
from metal_gauss.colmap import read_model
from metal_gauss.io import load_ply


def timeit(fn, n=5):
    fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--colmap", required=True, help="COLMAP sparse model dir")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--ref", action="store_true", help="also run the (slow) torch_ref oracle")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rec = read_model(args.colmap)
    im = sorted(rec.images.values(), key=lambda i: i.name)[args.frame_index]
    cam = rec.cameras[im.camera_id]
    W, H = args.width, args.height
    sx, sy = W / cam.width, H / cam.height
    K = torch.tensor([[cam.fx * sx, 0, cam.cx * sx],
                      [0, cam.fy * sy, cam.cy * sy],
                      [0, 0, 1.0]], dtype=torch.float32, device="mps")
    vm = torch.eye(4, device="mps")
    vm[:3, :3] = torch.as_tensor(im.R, dtype=torch.float32, device="mps")
    vm[:3, 3] = torch.as_tensor(im.t, dtype=torch.float32, device="mps")

    sp = load_ply(args.ply, device="mps")
    print(f"{len(sp):,} gaussians @ {W}x{H}")
    out = {"gaussians": len(sp), "width": W, "height": H}

    for backend in (["metal", "torch_ref"] if args.ref else ["metal"]):
        fwd = timeit(lambda: render(sp.means, sp.quats, sp.scales, sp.opacities, sp.sh,
                                    K, vm, W, H, sh_degree=sp.sh_degree, backend=backend))
        m = sp.means.clone().requires_grad_(True)
        sh = sp.sh.clone().requires_grad_(True)

        def fb():
            rgb, _, _ = render(m, sp.quats, sp.scales, sp.opacities, sh, K, vm, W, H,
                               sh_degree=sp.sh_degree, backend=backend)
            rgb.square().mean().backward()

        fbwd = timeit(fb)
        out[backend] = {"forward_s": round(fwd, 4), "forward_backward_s": round(fbwd, 4)}
        print(f"  {backend:10} forward {fwd*1000:7.1f}ms   fwd+bwd {fbwd*1000:7.1f}ms")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
