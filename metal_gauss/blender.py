"""NeRF-synthetic (Blender) loader, for ABSOLUTE calibration.

room1 has no ground-truth reference, so a systematic 1 dB deficit there is
invisible -- there is nothing to be 1 dB below. The Blender scenes have
published numbers for exactly this algorithm: 3DGS-MCMC reports 35.75 dB on
lego. Landing far off that means a real bug, not a tuning gap.

Two conversions are the whole job, and both are silent if wrong:

  * `transform_matrix` is camera-to-world in OpenGL convention (x right, y UP,
    z BACKWARD). The renderer wants world-to-camera in OpenCV (x right, y down,
    z forward). Flipping the y and z columns before inverting handles it. Get
    this wrong and training still converges -- to a mirrored scene at a few dB
    lower, which reads like a quality problem rather than a bug.
  * The PNGs are RGBA with a transparent background. The 3DGS convention is to
    composite over WHITE. Compositing over black instead costs about a dB and
    looks like a tuning issue too.

There is no sparse point cloud, so initialisation is a uniform random cube
scaled to the scene, which is what the 3DGS papers do for these scenes.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from metal_gauss.dataset import LazyViews, Scene, View

# OpenGL -> OpenCV: negate the y and z basis vectors.
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _frames(root: Path, split: str) -> tuple[float, list[tuple[Path, dict]]]:
    """Everything about a split that can be known without decoding a pixel.

    Existence filtering happens here so a caller can be told how many views a
    split has -- and be told the truth about a scene with missing files --
    while the images themselves are still on disk.
    """
    meta = json.loads((root / f"transforms_{split}.json").read_text())
    angle_x = float(meta["camera_angle_x"])
    out = []
    for fr in meta["frames"]:
        p = root / (fr["file_path"].lstrip("./") + ".png")
        if p.exists():
            out.append((p, fr))
    return angle_x, out


def _decode(p: Path, fr: dict, angle_x: float, max_resolution: int) -> View:
    from PIL import Image

    img = Image.open(p)
    scale = min(1.0, max_resolution / max(img.size))
    W, H = int(round(img.width * scale)), int(round(img.height * scale))
    if (W, H) != img.size:
        img = img.resize((W, H), Image.LANCZOS)

    a = np.asarray(img, dtype=np.float32) / 255.0
    if a.shape[-1] == 4:                       # composite over white
        rgb = a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4])
    else:
        rgb = a[..., :3]

    focal = 0.5 * W / math.tan(0.5 * angle_x)
    K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1.0]],
                     dtype=torch.float32)

    c2w = np.array(fr["transform_matrix"], dtype=np.float32) @ _GL2CV
    vm = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32))
    return View(p.name, torch.from_numpy((rgb * 255).astype(np.uint8)), K, vm)


def _decode_all(angle_x: float, frames: list[tuple[Path, dict]],
                max_resolution: int) -> list[View]:
    """Decode a split in parallel, in JSON order.

    PIL releases the GIL for both decode and resize, so this is a real 3.4x on
    ten cores. Order is the map's, not completion order: the trainer indexes
    the training list with a seeded permutation, so a reordering here would
    silently change which views a "reproducible" run trains on.
    """
    if not frames:
        return []
    workers = min(8, len(frames), os.cpu_count() or 1)
    if workers == 1:
        return [_decode(p, fr, angle_x, max_resolution) for p, fr in frames]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(
            lambda a: _decode(a[0], a[1], angle_x, max_resolution), frames))


def load_blender(root: str | Path, max_resolution: int = 800,
                 n_init: int = 100_000, seed: int = 0) -> Scene:
    root = Path(root)
    train_angle, train_frames = _frames(root, "train")
    test_angle, test_frames = _frames(root, "test")
    train = _decode_all(train_angle, train_frames, max_resolution)
    if not train:
        raise FileNotFoundError(f"no training frames under {root}")

    # Held out until something asks. A benchmark run with `eval_every` past
    # `steps` never does, and on lego that is 200 images and 4.1 s.
    heldout = LazyViews(
        len(test_frames),
        lambda: _decode_all(test_angle, test_frames, max_resolution))

    # No sparse cloud in these scenes. The 3DGS papers initialise from a
    # uniform cube; its extent is set from the camera ring so the scale is not
    # a magic number.
    C = np.stack([(-v.viewmat[:3, :3].T @ v.viewmat[:3, 3]).numpy() for v in train])
    radius = float(np.linalg.norm(C - C.mean(0), axis=1).max())
    rng = np.random.default_rng(seed)
    pts = (rng.random((n_init, 3), dtype=np.float32) - 0.5) * (radius * 0.6)
    cols = rng.random((n_init, 3), dtype=np.float32) * 0.5 + 0.25
    return Scene(train, heldout, pts.astype(np.float32), cols.astype(np.float32))
