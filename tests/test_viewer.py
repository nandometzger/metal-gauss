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

from types import SimpleNamespace

import numpy as np

from metal_gauss.dataset import LazyViews
from metal_gauss.render_path import _GL2CV, intrinsics, look_at, render_frames, world_to_camera
from metal_gauss.viewer import (
    PreviewBudget,
    RenderScheduler,
    SnapState,
    TrainClock,
    frame_size,
    view_frame_size,
    import_viser,
    infer_up,
    intrinsics_vfov,
    letterbox,
    matrix_to_quat,
    pose_matches,
    pose_to_viewmat,
    quat_to_matrix,
    scale_K,
    vertical_fov,
    vfov_from_K,
    frustums_at_eyes,
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


@pytest.mark.parametrize("W0, H0, max_side, scale", [
    (800, 800, 1024, 1.0), (1000, 667, 1024, 1.0), (1000, 667, 1024, 0.25),
    (667, 1000, 512, 0.5), (1600, 900, 1024, 0.75),
])
def test_a_snapped_view_keeps_its_own_aspect_to_half_a_pixel(W0, H0, max_side, scale):
    """Rounding the short side to 16 too stretched a 1000x667 view by up to 3%.

    The long side still snaps to 16; the short side follows the photograph, so
    the scaled intrinsics stay square-pixelled to within rounding.
    """
    W, H = view_frame_size(W0, H0, max_side, scale)
    assert max(W, H) % 16 == 0 and max(W, H) <= max_side
    short, short0 = min(W, H), min(W0, H0)
    assert abs(short - max(W, H) * short0 / max(W0, H0)) <= 0.5


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


def test_a_changed_model_gets_a_new_full_frame_not_a_preview():
    """Training moves the splats under a still camera; that is not camera motion."""
    s = RenderScheduler()
    s.done(s.next_job(0.0, 0), 0.02)
    assert s.next_job(1.0, 0) is None
    s.invalidate()
    job = s.next_job(1.0, 0)
    assert (job.scale, job.lens, job.preview) == (1.0, None, False)


def test_a_frame_of_the_old_model_does_not_satisfy_the_new_one():
    s = RenderScheduler()
    job = s.next_job(0.0, 0)
    s.invalidate()
    s.done(job, 0.02)
    assert s.next_job(0.1, 0) is not None


def test_a_changed_model_does_not_retry_a_failed_render():
    """Every training step changes the model; a broken render must stay stopped."""
    s = RenderScheduler()
    s.failed(s.next_job(0.0, 0))
    s.invalidate()
    assert s.next_job(1.0, 0) is None


# ---------------------------------------------------------------- budget

def test_the_budget_refuses_once_its_share_is_spent_and_recovers():
    """10% of a 1 s window is 100 ms of preview; older frames age out."""
    b = PreviewBudget(0.1, window_s=1.0)
    assert b.allow(0.0)
    b.spend(0.5, 0.05)
    assert b.allow(0.5)
    b.spend(0.6, 0.05)
    assert not b.allow(0.6)
    assert not b.allow(1.5), "the frame at 0.5 s is still inside the window"
    assert b.allow(1.55)


def test_a_zero_budget_never_renders():
    assert not PreviewBudget(0.0).allow(0.0)


def test_the_budget_follows_its_slider():
    b = PreviewBudget(0.1, window_s=1.0)
    b.spend(0.0, 0.1)
    assert not b.allow(0.1)
    b.fraction = 0.5
    assert b.allow(0.1)


# ---------------------------------------------------------------- training

def test_paused_time_is_not_training_time():
    """Wall-clock, ms/step and the stall detector must not count a pause."""
    t = [0.0]
    clock = TrainClock(lambda: t[0])
    t[0] = 10.0
    assert clock.elapsed() == 10.0
    clock.pause()
    t[0] = 25.0
    assert clock.elapsed() == 10.0
    clock.pause()                            # a second pause changes nothing
    t[0] = 30.0
    clock.resume()
    assert clock.paused_s == 20.0
    t[0] = 31.0
    assert clock.elapsed() == 11.0
    clock.resume()                           # nor does a second resume
    assert clock.paused_s == 20.0


def test_a_z_up_rig_infers_z_up():
    """Blender cameras ring the object looking down at it; their ups lean inward
    and the horizontal parts cancel."""
    views = []
    for k in range(8):
        a = 2.0 * math.pi * k / 8
        eye = torch.tensor([3.0 * math.cos(a), 3.0 * math.sin(a), 1.5])
        views.append(world_to_camera(look_at(eye, torch.zeros(3), up="+z"), eye))
    assert torch.allclose(infer_up(views), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_up_is_the_normalised_mean_of_the_camera_ups():
    front_z_up = torch.tensor([0.0, -2.0, 0.0])
    front_y_up = torch.tensor([0.0, 0.0, -2.0])
    views = [world_to_camera(look_at(front_z_up, torch.zeros(3), up="+z"), front_z_up),
             world_to_camera(look_at(front_y_up, torch.zeros(3), up="+y"), front_y_up)]
    assert torch.allclose(infer_up(views), torch.tensor([0.0, S, S]), atol=1e-6)


def _fake_live(monkeypatch, cls_name="LiveView"):
    """A LiveView with no server behind it: one client, instant fake frames."""
    import threading
    from types import SimpleNamespace as NS

    from metal_gauss import viewer as V

    live = getattr(V, cls_name).__new__(getattr(V, cls_name))
    live.clients_lock = threading.Lock()
    live._rr = 0
    live.preview_s = 0.0
    live.samples, live.aperture = NS(value=96), NS(value=0.0)
    live.status, live.readout = NS(content=""), NS(content="")
    live.source = lambda: [0] * 10
    live.wake = threading.Event()
    client = V._Client(NS(scene=NS(set_background_image=lambda image, **kw: None)))
    client.pose = V._Pose((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    live.clients = {1: client}
    monkeypatch.setattr(V.torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(live, "_frame", lambda *a: (16, 16, None))
    return live, client


def test_only_frames_taken_from_running_training_count_as_preview_time(monkeypatch):
    """Paused and after-training renders cost the trainer nothing.

    Counting them put 'preview 23% of wall-clock' on screen against a 10% budget:
    the numerator had paused renders in it, the wall-clock did not.
    """
    live, client = _fake_live(monkeypatch)
    assert live.pump() == 1                  # paused / finished: no budget
    assert live.preview_s == 0.0
    client.scheduler.invalidate()
    assert live.pump(max_jobs=1, budget=PreviewBudget(1.0)) == 1
    assert live.preview_s > 0.0


def test_the_sync_a_frame_forces_is_charged_to_the_budget(monkeypatch):
    """Measured: frames alone were 9% of wall-clock, training 17% slower.

    The flush before a frame stalls the trainer's CPU/GPU pipelining, so the
    budget pays for it. The adaptive preview scale still sees render time only.
    """
    import time as _time

    from metal_gauss import viewer as V

    live, client = _fake_live(monkeypatch)
    monkeypatch.setattr(V.torch.mps, "synchronize", lambda: _time.sleep(0.03))
    render_times = []
    real_done = client.scheduler.done
    monkeypatch.setattr(client.scheduler, "done",
                        lambda job, s: (render_times.append(s), real_done(job, s)))
    budget = PreviewBudget(1.0)
    assert live.pump(max_jobs=1, budget=budget) == 1
    assert live.preview_s >= 0.03
    assert sum(s for _, s in budget._frames) >= 0.03
    assert render_times and render_times[0] < 0.03


def test_reading_the_loss_is_charged_to_the_budget(monkeypatch):
    """loss.item() synchronises too; it happens only because a browser watches."""
    import time as _time
    from types import SimpleNamespace as NS

    live, _ = _fake_live(monkeypatch, "TrainingView")
    live.budget = PreviewBudget(1.0)
    live._stats, live.stats = {}, NS(content="")
    live._loss_at = live._stats_at = -math.inf
    monkeypatch.setattr(live, "invalidate_all", lambda: None)
    monkeypatch.setattr(live, "pump", lambda **kw: 0)
    slow_loss = NS(item=lambda: (_time.sleep(0.03), 0.5)[1])
    live.after_step(10, 5, slow_loss, TrainClock())
    assert live._stats["loss"] == 0.5
    assert live.preview_s >= 0.03
    assert sum(s for _, s in live.budget._frames) >= 0.03


def test_finishing_releases_a_pause_that_arrived_too_late(monkeypatch):
    """Pause clicked after the last step must not leave a live 'Resume' button."""
    from types import SimpleNamespace as NS

    live, _ = _fake_live(monkeypatch, "TrainingView")
    live.paused = True
    live.pause_button = NS(label="Resume", disabled=False)
    live._stats, live.stats = {}, NS(content="")
    monkeypatch.setattr(live, "invalidate_all", lambda: None)
    monkeypatch.setattr(live, "serve_forever", lambda: None)
    live.finish()
    assert live.paused is False
    assert (live.pause_button.label, live.pause_button.disabled) == ("Pause", True)
# ---------------------------------------------------------------- cameras

@pytest.mark.parametrize("wxyz, R", [
    ((1.0, 0.0, 0.0, 0.0), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ((S, S, 0.0, 0.0), [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),        # quarter turn about x
    ((S, 0.0, S, 0.0), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),        # about y
    ((S, 0.0, 0.0, S), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),        # about z
    ((0.0, 1.0, 0.0, 0.0), [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),   # half turn: w = 0
    ((0.5, 0.5, 0.5, 0.5), [[0, 0, 1], [1, 0, 0], [0, 1, 0]]),     # 120 deg about (1,1,1)
])
def test_matrix_to_quat_recovers_known_rotations(wxyz, R):
    """wxyz order, w >= 0. Frustums are placed with this, so a sign slip
    points every camera the wrong way."""
    got = matrix_to_quat(torch.tensor(R, dtype=torch.float32))
    assert got == pytest.approx(wxyz, abs=1e-6)


def test_matrix_to_quat_round_trips_arbitrary_rotations():
    g = torch.Generator().manual_seed(7)
    for _ in range(50):
        q = torch.nn.functional.normalize(torch.randn(4, generator=g), dim=0)
        R = quat_to_matrix(tuple(q.tolist()))
        assert torch.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-5)


def test_vertical_fov_from_intrinsics():
    K = torch.tensor([[70.0, 0.0, 64.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
    assert vfov_from_K(K, 100) == pytest.approx(math.pi / 2)


def test_scale_K_follows_each_axis_separately():
    """Principal point included: a frame shrunk unevenly moves its centre."""
    K = torch.tensor([[100.0, 0.0, 64.0], [0.0, 110.0, 48.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(scale_K(K, (128, 96), (64, 24)),
                          torch.tensor([[50.0, 0.0, 32.0], [0.0, 27.5, 12.0], [0.0, 0.0, 1.0]]))


def test_letterbox_pads_rows_when_the_frame_is_wider_than_the_window():
    image = np.full((4, 8, 3), 200, dtype=np.uint8)
    out = letterbox(image, 1.0, (10, 20, 30))
    assert out.shape == (8, 8, 3)
    assert (out[2:6] == 200).all()
    assert (out[:2] == [10, 20, 30]).all() and (out[6:] == [10, 20, 30]).all()


def test_letterbox_pads_columns_when_the_frame_is_narrower_than_the_window():
    image = np.full((8, 4, 3), 200, dtype=np.uint8)
    out = letterbox(image, 1.0, (10, 20, 30))
    assert out.shape == (8, 8, 3)
    assert (out[:, 2:6] == 200).all()
    assert (out[:, :2] == [10, 20, 30]).all() and (out[:, 6:] == [10, 20, 30]).all()


def test_letterbox_leaves_a_matching_frame_alone():
    image = np.full((4, 8, 3), 200, dtype=np.uint8)
    assert letterbox(image, 2.0, (0, 0, 0)).shape == (4, 8, 3)
    assert letterbox(image, 8 / 4.2, (0, 0, 0)).shape == (4, 8, 3), "within a pixel"


def _pose(wxyz, position):
    return SimpleNamespace(wxyz=tuple(wxyz), position=tuple(position))


def test_a_pose_matches_its_camera_but_not_a_moved_or_turned_one():
    q = (S, 0.0, S, 0.0)
    assert pose_matches(_pose(q, (1.0, 2.0, 3.0)), q, (1.0, 2.0, 3.0), tol=1e-3)
    assert not pose_matches(_pose(q, (1.0, 2.0, 3.002)), q, (1.0, 2.0, 3.0), tol=1e-3)
    one_degree = math.radians(1.0)
    turned = (math.cos(0.5 * (math.pi / 2 + one_degree)), 0.0,
              math.sin(0.5 * (math.pi / 2 + one_degree)), 0.0)
    assert not pose_matches(_pose(turned, (1.0, 2.0, 3.0)), q, (1.0, 2.0, 3.0), tol=1e-3)


def test_a_pose_matches_whatever_roll_the_browser_chose():
    """viser rebuilds a camera from position, look_at and its own up direction,
    so a rolled training camera comes back unrolled. Only where it is and where
    it looks decide whether the browser arrived."""
    roll = math.radians(30.0)
    rolled = torch.tensor(quat_to_matrix((S, 0.0, S, 0.0)).tolist()) @ torch.tensor(
        [[math.cos(roll), -math.sin(roll), 0.0], [math.sin(roll), math.cos(roll), 0.0],
         [0.0, 0.0, 1.0]])
    assert pose_matches(_pose(matrix_to_quat(rolled), (0.0, 0.0, 0.0)),
                        (S, 0.0, S, 0.0), (0.0, 0.0, 0.0), tol=1e-3)


def test_a_snap_survives_the_poses_on_the_way_in():
    """The browser may report in-between poses before it reaches the camera."""
    snap = SnapState()
    assert [snap.update(m) for m in (False, False, True, True)] == [True, True, True, True]
    assert snap.update(False) is False


def test_a_snap_ends_when_the_camera_leaves():
    snap = SnapState()
    assert snap.update(True) is True
    assert snap.update(False) is False


def test_the_frustum_a_browser_eye_sits_on_is_the_one_to_hide():
    """A frustum whose apex is the eye catches every click, at distance zero.

    Found in the browser: clicks on three different frustums all arrived as
    train/0, the camera the view had opened on, so nothing else could be
    snapped to. Hidden frustums are not pickable.
    """
    centres = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert frustums_at_eyes(centres, [], 0.2) == set()
    assert frustums_at_eyes(centres, [(0.0, 0.0, 0.0)], 0.2) == {0}
    assert frustums_at_eyes(centres, [(1.05, 0.0, 0.0), (0.0, 2.0, 0.1)], 0.2) == {1, 2}
    assert frustums_at_eyes(centres, [(0.5, 0.0, 0.0)], 0.2) == set()


def test_lazy_views_report_whether_they_have_been_decoded():
    """The viewer draws held-out cameras only once decoded; it must not decode."""
    decodes = []

    def materialise():
        decodes.append(1)
        return ["a", "b"]

    views = LazyViews(2, materialise)
    assert not views.materialised
    assert len(views) == 2 and not views.materialised and decodes == []
    assert views[1] == "b"
    assert views.materialised and decodes == [1]


def test_stats_survive_a_snap_ending_on_another_thread(monkeypatch):
    """Click and camera callbacks drop the 'view' line on viser's threads.

    Checking for the key and then reading it raced with that: the line could
    vanish between the two and raise KeyError on the training thread.
    """
    from types import SimpleNamespace as NS

    class SnapEndsMidRead(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if key == "view":
                self.pop("view", None)       # the callback thread wins the race
            return value

    live, _ = _fake_live(monkeypatch, "TrainingView")
    live.stats = NS(content="")
    live._stats = SnapEndsMidRead(step=10, active=5, ms_step=20.0, view="view r_0.png: 30 dB")
    live._show_stats()
    assert "step 10" in live.stats.content


def test_missing_viser_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "viser", None)
    with pytest.raises(SystemExit, match=r"metal-gauss\[viewer\]"):
        import_viser()


# ---------------------------------------------------------------- pixels

@mps
def test_the_viewer_renders_exactly_what_metal_gauss_render_does():
    from metal_gauss.io import Splats
    from metal_gauss.render_path import camera_path
    from metal_gauss.viewer import SplatBatch, render_view

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

    ours = render_view(SplatBatch.from_splats(sp), vm, K, W, H,
                       background=(1.0, 1.0, 1.0), far=100.0)
    theirs = next(iter(render_frames(sp, [vm], K, W, H, background=(1.0, 1.0, 1.0))))
    assert torch.equal(ours, theirs)
