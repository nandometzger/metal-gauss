"""Loading a COLMAP scene: intrinsics must be read per camera model.

`Camera.params` is laid out differently per model -- PINHOLE is (fx, fy, cx, cy)
but SIMPLE_PINHOLE / SIMPLE_RADIAL are (f, cx, cy, ...). Indexing params as
(fx, fy, cx, cy) read cx as fy for the SIMPLE_* models, which is what COLMAP's
default single-camera reconstruction produces.

The model is read by metal_gauss.colmap, not pycolmap: pycolmap's wheel bundles
its own libomp, and importing it next to torch aborts the process (OMP Error
#15). pycolmap is still the reference for what the files mean, so the
cross-check runs it in a subprocess that never imports torch.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from metal_gauss.colmap import MODELS, read_model
from metal_gauss.dataset import load_scene

W, H = 64, 48

# Two images seeing two points, so points2D lines and point tracks are not empty.
IMAGES_TXT = (
    "# Image list with two lines of data per image:\n"
    "1 0.9238795 0 0.3826834 0 0.1 -0.2 3.0 1 000.png\n"
    "10.5 20.5 1 30.5 40.5 2\n"
    "2 1 0 0 0 0.5 0 2.0 1 001.png\n"
    "11.5 21.5 1\n"
)
POINTS_TXT = (
    "1 0.1 0.2 3.0 255 0 10 0.5 1 0 2 0\n"
    "2 -1.0 0.5 4.0 0 128 255 0.25 1 1\n"
)


def _write_model(root, model, params, with_images=True):
    """Small text model: one camera, two posed images, two observed points."""
    from PIL import Image

    sparse, images = root / "sparse", root / "images"
    sparse.mkdir(parents=True)
    images.mkdir()
    (sparse / "cameras.txt").write_text(
        "# Camera list\n" f"1 {model} {W} {H} {' '.join(map(str, params))}\n")
    (sparse / "images.txt").write_text(IMAGES_TXT)
    (sparse / "points3D.txt").write_text(POINTS_TXT)
    if with_images:
        for i, name in enumerate(["000.png", "001.png"]):
            Image.new("RGB", (W, H), (i * 40, 80, 120)).save(images / name)
    return sparse, images


def _K(scene):
    return (scene.train[0] if scene.train else scene.heldout[0]).K


@pytest.mark.parametrize("model,params,expected", [
    ("PINHOLE", [50.0, 55.0, 32.0, 24.0], (50.0, 55.0, 32.0, 24.0)),
    ("SIMPLE_PINHOLE", [50.0, 31.0, 23.0], (50.0, 50.0, 31.0, 23.0)),
    ("SIMPLE_RADIAL", [50.0, 31.0, 23.0, 0.0], (50.0, 50.0, 31.0, 23.0)),
    ("RADIAL", [50.0, 31.0, 23.0, 0.0, 0.0], (50.0, 50.0, 31.0, 23.0)),
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


def test_poses_and_points_are_read(tmp_path):
    sparse, images = _write_model(tmp_path, "PINHOLE", [50.0, 55.0, 32.0, 24.0])
    scene = load_scene(sparse, images, max_resolution=W, eval_split_every=1000)
    views = {v.name: v for v in [*scene.train, *scene.heldout]}
    # 45 degrees about y, then the translation
    c = s = np.sqrt(0.5)
    want = torch.tensor([[c, 0, s, 0.1], [0, 1, 0, -0.2], [-s, 0, c, 3.0], [0, 0, 0, 1]])
    torch.testing.assert_close(views["000.png"].viewmat, want.float(), atol=1e-6, rtol=0)
    np.testing.assert_allclose(scene.points, [[0.1, 0.2, 3.0], [-1.0, 0.5, 4.0]], rtol=1e-6)
    np.testing.assert_allclose(scene.colors, np.array([[255, 0, 10], [0, 128, 255]]) / 255.0,
                               rtol=1e-6)


def test_distortion_is_reported_not_silently_dropped(tmp_path):
    sparse, images = _write_model(tmp_path, "SIMPLE_RADIAL", [50.0, 31.0, 23.0, 0.05])
    with pytest.warns(UserWarning, match="ignores distortion"):
        load_scene(sparse, images, max_resolution=W)


def test_fisheye_is_reported_even_without_distortion_params(tmp_path):
    sparse, images = _write_model(tmp_path, "SIMPLE_FISHEYE", [50.0, 31.0, 23.0])
    with pytest.warns(UserWarning, match="not a pinhole projection"):
        load_scene(sparse, images, max_resolution=W)


def test_wrong_param_count_is_an_error(tmp_path):
    sparse, _ = _write_model(tmp_path, "PINHOLE", [50.0, 31.0, 23.0], with_images=False)
    with pytest.raises(ValueError, match="PINHOLE has 4 params, got 3"):
        read_model(sparse)


# --- cross-check against pycolmap -------------------------------------------

_PYCOLMAP_DUMP = textwrap.dedent("""
    import json, sys
    import pycolmap  # never torch in this process: two libomp copies abort

    src, out = sys.argv[1], sys.argv[2]
    rec = pycolmap.Reconstruction(src)
    rec.write_binary(out + "/bin")
    rec.write_text(out + "/txt")

    def pose(im):
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        return [cfw.rotation.matrix().tolist(), list(cfw.translation)]

    print(json.dumps({
        "cameras": {cid: [c.model.name, c.width, c.height, c.focal_length_x, c.focal_length_y,
                          c.principal_point_x, c.principal_point_y]
                    for cid, c in rec.cameras.items()},
        "images": {iid: [im.name, im.camera_id, *pose(im)] for iid, im in rec.images.items()},
        "points": sorted([*p.xyz.tolist(), *p.color.tolist()] for p in rec.points3D.values()),
    }))
""")


def _models_with_params():
    """One parameter vector per model, distinct values so a mixed-up index shows."""
    for name, (_, n, single_focal) in MODELS.items():
        base = [50.0, 31.0, 23.0] if single_focal else [50.0, 55.0, 32.0, 24.0]
        yield name, base + [0.01 * (k + 1) for k in range(n - len(base))]


@pytest.mark.skipif(importlib.util.find_spec("pycolmap") is None, reason="pycolmap not installed")
@pytest.mark.parametrize("model,params", list(_models_with_params()))
def test_binary_and_text_match_pycolmap(tmp_path, model, params):
    sparse, _ = _write_model(tmp_path, model, params, with_images=False)
    (tmp_path / "bin").mkdir()
    (tmp_path / "txt").mkdir()
    res = subprocess.run([sys.executable, "-c", _PYCOLMAP_DUMP, str(sparse), str(tmp_path)],
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    ref = json.loads(res.stdout)

    for fmt in ("bin", "txt"):
        rec = read_model(tmp_path / fmt)
        for cid, (name, w, h, fx, fy, cx, cy) in ref["cameras"].items():
            cam = rec.cameras[int(cid)]
            assert (cam.model, cam.width, cam.height) == (name, w, h), fmt
            np.testing.assert_allclose([cam.fx, cam.fy, cam.cx, cam.cy], [fx, fy, cx, cy],
                                       err_msg=f"{model} {fmt}")
        for iid, (name, cid, R, t) in ref["images"].items():
            im = rec.images[int(iid)]
            assert (im.name, im.camera_id) == (name, cid), fmt
            np.testing.assert_allclose(im.R, R, atol=1e-12, err_msg=fmt)
            np.testing.assert_allclose(im.t, t, atol=1e-12, err_msg=fmt)
        ours = sorted(np.hstack([rec.points, rec.colors]).tolist())
        np.testing.assert_allclose(ours, ref["points"], err_msg=fmt)
