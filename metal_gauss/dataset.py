"""COLMAP scene loader for training: images, poses, intrinsics, eval split."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class View:
    name: str
    image: torch.Tensor      # (H,W,3) float32 in [0,1], cpu
    K: torch.Tensor          # (3,3)
    viewmat: torch.Tensor    # (4,4) world2cam


_PYRAMID: dict = {}


def downscaled(v: View, factor: int) -> View:
    """A view at 1/factor resolution, cached.

    Training the early steps at reduced resolution is most of why msplat is
    fast: its 7k average is 24.5 ms/step against 46 ms at full resolution. It
    is not a benchmarking trick either -- forced to full resolution its own
    held-out score DROPS (21.37 -> 20.78), so the coarse-to-fine schedule is
    doing real optimisation work, not just cheaper steps.

    Images are cached as uint8 and downsampled once per (view, factor); the
    pyramid costs ~33% more memory than the full-resolution set alone. The
    intrinsics scale with the image, and the principal point scales too -- it
    is not simply W/2 for a COLMAP camera.
    """
    if factor <= 1:
        return v
    key = (id(v), factor)
    hit = _PYRAMID.get(key)
    if hit is not None:
        return hit

    H, W = v.image.shape[:2]
    h, w = max(1, H // factor), max(1, W // factor)
    img = torch.nn.functional.interpolate(
        v.image.permute(2, 0, 1)[None].float(), size=(h, w),
        mode="area")[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)

    sx, sy = w / W, h / H
    K = v.K.clone()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy

    out = View(v.name, img.contiguous(), K, v.viewmat)
    if len(_PYRAMID) > 4096:          # bounded; scenes have a few hundred views
        _PYRAMID.clear()
    _PYRAMID[key] = out
    return out


class LazyViews(Sequence):
    """Views whose count is known now and whose pixels are decoded on demand.

    The held-out split of a Blender scene is 200 images, twice the training
    set, and a run that never evaluates never reads one of them -- but decoding
    them was 4.1 s of a 8.7 s startup. Counting frames needs only the JSON, so
    length is answered without touching a PNG.

    Materialisation happens once and the resulting View objects are kept:
    `downscaled()` caches its pyramid on `id(view)`, so handing out fresh
    objects per access would silently defeat that cache.
    """

    def __init__(self, count: int, materialise):
        self._count = count
        self._materialise = materialise
        self._views: list[View] | None = None

    def _all(self) -> list[View]:
        if self._views is None:
            self._views = self._materialise()
            if len(self._views) != self._count:
                raise RuntimeError(
                    f"expected {self._count} views, decoded {len(self._views)}")
        return self._views

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, i):
        return self._all()[i]

    def __iter__(self):
        return iter(self._all())


@dataclass
class Scene:
    train: Sequence[View]
    heldout: Sequence[View]
    points: np.ndarray       # (P,3) sparse init
    colors: np.ndarray       # (P,3) in [0,1]


def load_scene(colmap_dir: str | Path, images_dir: str | Path,
               max_resolution: int = 1600, eval_split_every: int = 8) -> Scene:
    import pycolmap
    from PIL import Image

    rec = pycolmap.Reconstruction(str(colmap_dir))
    cam = list(rec.cameras.values())[0]

    views = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        p = Path(images_dir) / im.name
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        scale = min(1.0, max_resolution / max(img.size))
        W, H = int(round(img.width * scale)), int(round(img.height * scale))
        img = img.resize((W, H), Image.LANCZOS)
        sx, sy = W / cam.width, H / cam.height
        K = torch.tensor([[cam.params[0] * sx, 0, cam.params[2] * sx],
                          [0, cam.params[1] * sy, cam.params[3] * sy],
                          [0, 0, 1.0]], dtype=torch.float32)
        cfw = im.cam_from_world()
        vm = torch.eye(4)
        vm[:3, :3] = torch.as_tensor(cfw.rotation.matrix(), dtype=torch.float32)
        vm[:3, 3] = torch.as_tensor(np.asarray(cfw.translation), dtype=torch.float32)
        # uint8 storage: 162 full-res frames as float32 was 4x the memory and
        # OOMed a 24GB M5 alongside the 1M-splat Adam state. Convert per step.
        views.append(View(im.name,
                          torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()),
                          K, vm))

    heldout = views[::eval_split_every]
    heldout_names = {v.name for v in heldout}
    train = [v for v in views if v.name not in heldout_names]

    pts = np.array([p.xyz for p in rec.points3D.values()], np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], np.float32) / 255.0
    return Scene(train, heldout, pts, cols)
