"""Camera keyframes and the path through them, checked without a GPU.

A flythrough is decided entirely before a pixel is drawn: where the camera is
at each frame, how it is turned, and how wide the lens is. That is arithmetic,
so it is tested here against hand-derived values rather than by looking at the
video afterwards -- a path that drifts, flips or fails to close reads as a bad
render and is very hard to diagnose from frames.
"""

from __future__ import annotations

import math

import pytest
import torch

from metal_gauss.keyframes import Keyframe, interpolate, preset
from metal_gauss.render_path import camera_path, world_to_camera
from metal_gauss.viewer import quat_to_matrix

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def key(position, wxyz=IDENTITY, fov=1.0):
    return Keyframe(tuple(float(v) for v in position), tuple(float(v) for v in wxyz), fov)


# ------------------------------------------------------------------ positions

def test_two_keyframes_pass_through_their_midpoint():
    """An open path starts on the first key, ends on the last, and with clamped
    end tangents its halfway frame is the plain midpoint."""
    path = interpolate([key((0.0, 0.0, 0.0)), key((2.0, 4.0, -6.0))], frames=3, loop=False)

    assert len(path) == 3
    assert path[0].position == pytest.approx((0.0, 0.0, 0.0))
    assert path[1].position == pytest.approx((1.0, 2.0, -3.0))
    assert path[2].position == pytest.approx((2.0, 4.0, -6.0))


def test_a_looped_path_lands_on_every_keyframe_and_returns():
    """One frame per key means the path IS the keys, and frame 0 is the start
    again after the last -- the wrap has to be an ordinary step, as for the
    wiggle and orbit paths this reuses."""
    keys = [key((0.0, 0.0, 0.0)), key((1.0, 0.0, 0.0)), key((1.0, 1.0, 0.0))]
    path = interpolate(keys, frames=3, loop=True)

    assert [p.position for p in path] == pytest.approx([k.position for k in keys])


def test_a_loop_has_no_seam_where_it_wraps():
    """The step from the last frame back to the first is an ordinary step.

    Phrased as `test_render_path.py` phrases it for the wiggle and orbit paths:
    the wrap is no larger than the largest step inside the path. A loop that
    merely repeated its first key would jump here. (Uniform Catmull-Rom varies
    its speed between keys, so the wrap is not the MEAN step.)
    """
    keys = [key((1.0, 0.0, 0.0)), key((0.0, 1.0, 0.0)),
            key((-1.0, 0.0, 0.0)), key((0.0, -1.0, 0.0))]
    path = interpolate(keys, frames=32, loop=True)

    steps = [math.dist(path[i].position, path[i + 1].position) for i in range(31)]
    wrap = math.dist(path[-1].position, path[0].position)
    assert 0.0 < wrap <= 1.05 * max(steps)


def test_one_keyframe_holds_still():
    path = interpolate([key((1.0, 2.0, 3.0))], frames=4, loop=True)
    assert [p.position for p in path] == [(1.0, 2.0, 3.0)] * 4


def test_a_path_needs_a_keyframe():
    with pytest.raises(ValueError, match="at least one keyframe"):
        interpolate([], frames=4)


# ------------------------------------------------------------------ rotations

def test_rotations_take_the_short_way_round():
    """q and -q are the same orientation, and a lerp between them as written
    swings almost the whole way round instead of turning gently.

    Identity to -(45 degrees about +Y): the halfway frame must be about 22.5
    degrees about +Y, not the 157-degree lurch the unflipped lerp gives.
    """
    turned = (-math.cos(math.pi / 8), 0.0, -math.sin(math.pi / 8), 0.0)
    path = interpolate([key((0.0, 0.0, 0.0)), key((0.0, 0.0, 0.0), turned)],
                       frames=3, loop=False)

    w, x, y, z = path[1].wxyz
    assert w > 0.9 and y > 0.0, "turned the long way round"
    assert (x, z) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert math.degrees(2.0 * math.acos(min(1.0, abs(w)))) == pytest.approx(22.5, abs=0.5)


def test_every_rotation_stays_a_unit_quaternion():
    keys = [key((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            key((1.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)),
            key((2.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))]
    for frame in interpolate(keys, frames=16, loop=True):
        assert sum(c * c for c in frame.wxyz) == pytest.approx(1.0, abs=1e-6)


def test_the_field_of_view_follows_the_keys():
    path = interpolate([key((0.0, 0.0, 0.0), fov=1.0), key((1.0, 0.0, 0.0), fov=2.0)],
                       frames=3, loop=False)
    assert [p.fov for p in path] == pytest.approx([1.0, 1.5, 2.0])


# -------------------------------------------------------------------- presets

@pytest.mark.parametrize("kind", ["orbit", "wiggle"])
def test_a_preset_is_the_path_metal_gauss_render_would_draw(kind):
    """One implementation of orbit and wiggle, not two: the preset's poses are
    exactly `camera_path`'s, so a video from the viewer and one from
    metal-gauss-render move the same way."""
    eye, target = torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3)
    keys = preset(kind, eye, target, frames=8, sweep_deg=20.0, fov=1.0)
    want = camera_path(eye, target, 8, 20.0, kind)

    assert len(keys) == 8
    for k, vm in zip(keys, want):
        got = world_to_camera(quat_to_matrix(k.wxyz),
                              torch.tensor(k.position, dtype=torch.float32))
        assert torch.allclose(got, vm, atol=1e-5)


def test_an_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        preset("dolly", torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]), frames=4,
               sweep_deg=5.0, fov=1.0)
