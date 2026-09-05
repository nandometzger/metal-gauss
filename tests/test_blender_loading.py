"""Loading a Blender scene: what gets decoded, when, and that it is unchanged.

Startup was 8.7 s on lego, of which 6.1 s was PNG decode -- and 4.1 s of that
decoded the 200 held-out views, which a run with no evaluation never reads.
These tests pin the two properties that buys: the held-out split costs nothing
until something touches it, and threading the decode does not change a pixel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from metal_gauss.blender import load_blender

_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _write_scene(root, n_train=3, n_test=5, size=16):
    """A miniature NeRF-synthetic scene: RGBA PNGs plus the two transforms."""
    from PIL import Image

    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        (root / split).mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(n):
            px = rng.integers(0, 256, (size, size, 4), dtype=np.uint8)
            Image.fromarray(px, mode="RGBA").save(root / split / f"r_{i}.png")
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = [float(i), 1.0, 2.0]
            frames.append({"file_path": f"./{split}/r_{i}",
                           "transform_matrix": c2w.tolist()})
        (root / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": 0.69, "frames": frames}))
    return root


@pytest.fixture
def scene_root(tmp_path):
    return _write_scene(tmp_path / "mini")


def _count_opens(monkeypatch):
    """Count PIL decodes, by name, without changing what they return."""
    from PIL import Image

    opened = []
    real = Image.open

    def spy(fp, *a, **kw):
        # the split directory, not the whole path: pytest's own tmp dirs are
        # named after the test, and "test_heldout_..." contains "test".
        opened.append(f"{Path(fp).parent.name}/{Path(fp).name}")
        return real(fp, *a, **kw)

    monkeypatch.setattr(Image, "open", spy)
    return opened


def test_loading_a_scene_does_not_decode_the_heldout_split(scene_root, monkeypatch):
    opened = _count_opens(monkeypatch)
    load_blender(scene_root, max_resolution=16, n_init=8)
    assert not [p for p in opened if p.startswith("test/")], \
        "held-out PNGs were decoded during load"
    assert len([p for p in opened if p.startswith("train/")]) == 3


def test_heldout_length_is_known_without_decoding(scene_root, monkeypatch):
    scene = load_blender(scene_root, max_resolution=16, n_init=8)
    opened = _count_opens(monkeypatch)
    assert len(scene.heldout) == 5
    assert opened == [], "len() decoded the held-out split"


def test_heldout_decodes_on_first_access(scene_root, monkeypatch):
    scene = load_blender(scene_root, max_resolution=16, n_init=8)
    opened = _count_opens(monkeypatch)
    first = scene.heldout[0]
    assert len([p for p in opened if p.startswith("test/")]) == 5
    assert first.image.shape == (16, 16, 3)


def test_heldout_decodes_only_once(scene_root, monkeypatch):
    scene = load_blender(scene_root, max_resolution=16, n_init=8)
    list(scene.heldout)
    opened = _count_opens(monkeypatch)
    list(scene.heldout)
    assert opened == [], "held-out split decoded twice"


def test_heldout_views_keep_their_identity(scene_root):
    """`downscaled()` caches its pyramid on id(view); new objects would leak."""
    scene = load_blender(scene_root, max_resolution=16, n_init=8)
    assert scene.heldout[2] is scene.heldout[2]
    assert list(scene.heldout)[2] is scene.heldout[2]


def test_heldout_matches_a_straight_serial_decode(scene_root):
    """The threaded, lazy path returns exactly the eager one's pixels."""
    from PIL import Image

    scene = load_blender(scene_root, max_resolution=16, n_init=8)
    meta = json.loads((scene_root / "transforms_test.json").read_text())
    angle_x = float(meta["camera_angle_x"])

    for fr, got in zip(meta["frames"], scene.heldout):
        p = scene_root / (fr["file_path"].lstrip("./") + ".png")
        a = np.asarray(Image.open(p), dtype=np.float32) / 255.0
        rgb = a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4])
        want = torch.from_numpy((rgb * 255).astype(np.uint8))
        assert got.name == p.name
        assert torch.equal(got.image, want)

        W = H = 16
        focal = 0.5 * W / math.tan(0.5 * angle_x)
        K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1.0]])
        assert torch.allclose(got.K, K)
        c2w = np.array(fr["transform_matrix"], dtype=np.float32) @ _GL2CV
        assert torch.allclose(got.viewmat,
                              torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)))


def test_train_split_order_is_the_json_order(scene_root):
    """rng.permutation(len(train)) indexes this list; order must be stable."""
    scene = load_blender(scene_root, max_resolution=16, n_init=8)
    assert [v.name for v in scene.train] == ["r_0.png", "r_1.png", "r_2.png"]


def test_missing_frames_are_skipped_not_faked(scene_root):
    (scene_root / "test" / "r_3.png").unlink()
    scene = load_blender(scene_root, max_resolution=16, n_init=8)
    assert len(scene.heldout) == 4
    assert [v.name for v in scene.heldout] == \
        ["r_0.png", "r_1.png", "r_2.png", "r_4.png"]
