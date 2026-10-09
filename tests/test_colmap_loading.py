"""Loading a COLMAP scene: intrinsics must be read per camera model.

`Camera.params` is laid out differently per model -- PINHOLE is (fx, fy, cx, cy)
but SIMPLE_PINHOLE / SIMPLE_RADIAL are (f, cx, cy, ...). Indexing params as
(fx, fy, cx, cy) read cx as fy for the SIMPLE_* models, which is what COLMAP's
default single-camera reconstruction produces.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("pycolmap")

from metal_gauss.dataset import load_scene

W, H = 64, 48


def _write_model(root, model, params, n_images=2):
    """Smallest text model pycolmap reads: one camera, identity-ish poses, no points."""
    from PIL import Image

    sparse, images = root / "sparse", root / "images"
    sparse.mkdir(parents=True)
    images.mkdir()
    (sparse / "cameras.txt").write_text(f"1 {model} {W} {H} {' '.join(map(str, params))}\n")
    lines = []
    for i in range(n_images):
        name = f"{i:03d}.png"
        Image.new("RGB", (W, H), (i * 40, 80, 120)).save(images / name)
        lines.append(f"{i + 1} 1 0 0 0 {0.1 * i} 0 0 1 {name}\n\n")
    (sparse / "images.txt").write_text("".join(lines))
    (sparse / "points3D.txt").write_text("")
    return sparse, images


def _K(scene):
    return (scene.train[0] if scene.train else scene.heldout[0]).K


@pytest.mark.parametrize("model,params,expected", [
    ("PINHOLE", [50.0, 55.0, 32.0, 24.0], (50.0, 55.0, 32.0, 24.0)),
    ("SIMPLE_PINHOLE", [50.0, 31.0, 23.0], (50.0, 50.0, 31.0, 23.0)),
    ("SIMPLE_RADIAL", [50.0, 31.0, 23.0, 0.0], (50.0, 50.0, 31.0, 23.0)),
    ("OPENCV", [50.0, 55.0, 32.0, 24.0, 0, 0, 0, 0], (50.0, 55.0, 32.0, 24.0)),
])
def test_intrinsics_follow_the_camera_model(tmp_path, model, params, expected):
    sparse, images = _write_model(tmp_path, model, params)
    K = _K(load_scene(sparse, images, max_resolution=W))  # no rescale
    fx, fy, cx, cy = expected
    torch.testing.assert_close(K, torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))


def test_intrinsics_rescale_with_the_image(tmp_path):
    sparse, images = _write_model(tmp_path, "SIMPLE_PINHOLE", [50.0, 31.0, 23.0])
    K = _K(load_scene(sparse, images, max_resolution=W // 2))
    torch.testing.assert_close(K, torch.tensor([[25.0, 0, 15.5], [0, 25.0, 11.5], [0, 0, 1.0]]))


def test_distortion_is_reported_not_silently_dropped(tmp_path):
    sparse, images = _write_model(tmp_path, "SIMPLE_RADIAL", [50.0, 31.0, 23.0, 0.05])
    with pytest.warns(UserWarning, match="ignores distortion"):
        load_scene(sparse, images, max_resolution=W)
