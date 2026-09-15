"""The viewer's camera maths and render scheduling, checked without a browser.

viser is an optional dependency and a GPU is not always there, so everything
that decides WHAT to render is kept in pure functions and tested here: how a
viser pose becomes a view matrix, how its vertical FOV becomes intrinsics, how
big a frame is, and what the render thread does next as the camera moves,
settles and refines. The one MPS test pins the viewer's pixels to
metal-gauss-render's for the same camera, so the two tools cannot drift apart.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from metal_gauss.render_path import _GL2CV, intrinsics, render_frames
from metal_gauss.viewer import (
    RenderScheduler,
    frame_size,
    import_viser,
    intrinsics_vfov,
    pose_to_viewmat,
    quat_to_matrix,
    vertical_fov,
)

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

S = math.sqrt(0.5)


# ---------------------------------------------------------------- camera maths

def test_identity_quaternion_is_the_identity_rotation():
    assert torch.allclose(quat_to_matrix((1.0, 0.0, 0.0, 0.0)), torch.eye(3), atol=1e-7)


def test_quarter_turn_about_y():
    """wxyz order: viser's, and the one a swapped xyzw reading gets wrong."""
    R = quat_to_matrix((S, 0.0, S, 0.0))
    assert torch.allclose(R, torch.tensor([[0.0, 0.0, 1.0],
                                           [0.0, 1.0, 0.0],
                                           [-1.0, 0.0, 0.0]]), atol=1e-6)


def test_a_viser_pose_puts_its_forward_axis_on_camera_z():
    """viser poses are camera-to-world in OpenCV axes: +Z forward, +Y down.

    Turned a quarter about Y, the camera looks along world +X, so a point two
    units that way sits on the optical axis and a point below it (world +Y) is
    at positive camera-y.
    """
    position = torch.tensor([0.5, -1.0, 2.0])
    vm = pose_to_viewmat((S, 0.0, S, 0.0), position)

    def cam(p):
        return (vm @ torch.cat([p, torch.ones(1)]))[:3]

    assert torch.allclose(cam(position), torch.zeros(3), atol=1e-6)
    assert torch.allclose(cam(position + torch.tensor([2.0, 0.0, 0.0])),
                          torch.tensor([0.0, 0.0, 2.0]), atol=1e-6)
    assert torch.allclose(cam(position + torch.tensor([0.0, 1.0, 0.0])),
                          torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_an_opengl_file_is_flipped_after_the_pose():
    """The camera moves in the flipped world the framing used; the flip is last."""
    q, p = (S, 0.0, S, 0.0), torch.tensor([0.5, -1.0, 2.0])
    assert torch.equal(pose_to_viewmat(q, p, world_flip=_GL2CV), pose_to_viewmat(q, p) @ _GL2CV)


def test_intrinsics_come_from_the_vertical_fov():
    """viser reports a vertical FOV; a 90 degree one over 100 rows is f = 50."""
    K = intrinsics_vfov(200, 100, math.radians(90.0))
    assert torch.allclose(K, torch.tensor([[50.0, 0.0, 100.0],
                                           [0.0, 50.0, 50.0],
                                           [0.0, 0.0, 1.0]]), atol=1e-5)


def test_vertical_fov_matches_the_renders_horizontal_one():
    """--fov is horizontal like metal-gauss-render's; the pixels must agree."""
    W, H = 320, 160
    vfov = vertical_fov(60.0, W / H)
    assert torch.allclose(intrinsics_vfov(W, H, vfov), intrinsics(W, H, 60.0), atol=1e-4)
    assert vertical_fov(90.0, 1.0) == pytest.approx(math.radians(90.0))


@pytest.mark.parametrize("aspect, max_side, scale, want", [
    (16 / 9, 1024, 1.0, (1024, 576)),
    (0.5, 1024, 1.0, (512, 1024)),
    (16 / 9, 1024, 0.25, (256, 144)),
    (2.0, 1000, 1.0, (992, 496)),
    (10.0, 64, 0.25, (16, 16)),
])
def test_frame_size_follows_the_aspect_in_multiples_of_16(aspect, max_side, scale, want):
    """Few distinct sizes, so the MPS allocator reuses buffers instead of churning."""
    assert frame_size(aspect, max_side, scale) == want


# ---------------------------------------------------------------- scheduling

def lens_ranges(s, now, samples):
    """Drain every lens job at `now`, completing each instantly."""
    out = []
    while (job := s.next_job(now, samples)) is not None:
        assert job.lens is not None
        out.append(job.lens)
        s.done(job, 0.01)
    return out


def test_a_new_client_gets_one_full_frame_and_then_nothing():
    """Idle means idle: no GPU work until something changes."""
    s = RenderScheduler()
    job = s.next_job(0.0, samples=0)
    assert (job.scale, job.lens, job.quality) == (1.0, None, 90)
    s.done(job, 0.02)
    assert s.next_job(0.5, samples=0) is None
    assert s.wait_s(0.5) is None


def test_moving_renders_previews_then_one_full_frame_once_settled():
    s = RenderScheduler()
    s.done(s.next_job(0.0, 0), 0.02)

    s.touch(1.0)
    preview = s.next_job(1.01, 0)
    assert preview.lens is None and preview.quality == 60
    s.done(preview, 0.01)
    assert s.next_job(1.02, 0) is None, "nothing moved since the last preview"
    assert s.wait_s(1.02) == pytest.approx(0.13)

    s.touch(1.05)
    s.done(s.next_job(1.06, 0), 0.01)
    assert s.next_job(1.15, 0) is None, "still inside the settle window"
    full = s.next_job(1.21, 0)
    assert (full.scale, full.lens, full.quality) == (1.0, None, 90)
    s.done(full, 0.02)
    assert s.next_job(2.0, 0) is None


def test_depth_of_field_refines_at_8_16_32_64_then_all_samples():
    """Once settled, no sharp pinhole frame first: bokeh would pop in over it."""
    s = RenderScheduler()
    assert lens_ranges(s, 0.0, 96) == [(0, 8), (8, 16), (16, 32), (32, 64), (64, 96)]
    assert s.next_job(1.0, 96) is None


def test_a_small_sample_count_stops_at_its_own_total():
    assert lens_ranges(RenderScheduler(), 0.0, 20) == [(0, 8), (8, 16), (16, 20)]


def test_a_change_mid_refinement_starts_the_lens_over():
    s = RenderScheduler()
    for _ in range(2):
        s.done(s.next_job(0.0, 96), 0.01)
    s.touch(1.0)
    assert s.next_job(1.01, 96).lens is None, "moving previews are pinhole"
    assert lens_ranges(s, 1.2, 96)[0] == (0, 8)


def test_a_frame_for_an_old_camera_does_not_advance_the_new_one():
    """The render thread can finish a job after the camera has already moved on."""
    s = RenderScheduler()
    stale = s.next_job(0.0, 96)
    s.touch(0.001)
    s.done(stale, 0.01)
    assert lens_ranges(s, 0.2, 96)[0] == (0, 8)


def test_preview_scale_steps_down_for_slow_frames_and_back_up():
    """Pixels cost time, so a frame at scale s costs full * s^2; hold ~30 fps."""
    s = RenderScheduler()
    s.touch(0.0)
    first = s.next_job(0.0, 0)
    assert first.scale == 1.0
    s.done(first, 0.1)                       # a full frame costs 100 ms
    s.touch(0.01)
    slow = s.next_job(0.01, 0)
    assert slow.scale == 0.5                 # 0.5^2 * 100 ms = 25 ms, 0.75^2 is 56
    s.done(slow, 0.005)                      # now 20 ms per full frame
    s.touch(0.02)
    assert s.next_job(0.02, 0).scale == 1.0


def test_a_failed_frame_waits_for_the_next_change():
    """An error must not become a hot retry loop hammering the GPU."""
    s = RenderScheduler()
    s.failed(s.next_job(0.0, 96))
    assert s.next_job(1.0, 96) is None
    assert s.wait_s(1.0) is None
    s.touch(2.0)
    assert s.next_job(2.0, 96) is not None


def test_missing_viser_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "viser", None)
    with pytest.raises(SystemExit, match=r"metal-gauss\[viewer\]"):
        import_viser()


# ---------------------------------------------------------------- pixels

@mps
def test_the_viewer_renders_exactly_what_metal_gauss_render_does():
    from metal_gauss.io import Splats
    from metal_gauss.render_path import camera_path
    from metal_gauss.viewer import render_view

    g = torch.Generator().manual_seed(0)
    n = 4000
    means = torch.randn(n, 3, generator=g) * 0.4 + torch.tensor([0.0, 0.0, 3.0])
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    sp = Splats(means, quats, torch.rand(n, 3, generator=g) * 0.05 + 0.01,
                torch.rand(n, generator=g) * 0.8 + 0.1,
                torch.randn(n, 16, 3, generator=g) * 0.3, 3).to("mps")
    W = H = 96
    vm = camera_path(torch.zeros(3), torch.tensor([0.0, 0.0, 3.0]), 8, 10.0, "orbit")[2]
    K = intrinsics(W, H, 50.0)

    ours = render_view(sp, vm, K, W, H, background=(1.0, 1.0, 1.0), far=100.0)
    theirs = next(iter(render_frames(sp, [vm], K, W, H, background=(1.0, 1.0, 1.0))))
    assert torch.equal(ours, theirs)
