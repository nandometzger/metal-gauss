"""Orbit a .ply in the browser, rendered on Metal.

    metal-gauss-view scene.ply --up +z

The rasteriser is fast enough to be interactive (bench/render_fps.py) and
nothing used that. This serves a viser page: drag to orbit, scroll to dolly,
and each view is rendered on this Mac's GPU and sent to the tab as a JPEG.
viser is an optional extra, `pip install "metal-gauss[viewer]"`, so the
package's own dependencies do not grow.

The render loop is written here rather than taken from nerfview, the viser
wrapper gsplat uses. nerfview 0.1.3 calls os._exit(1) on any exception in the
render function, interrupts renders through sys.settrace, and tells its two
render_fn signatures apart by catching TypeError, so a TypeError raised inside
a render silently switches API. It also has no way to refine a defocused frame
sample by sample, which is the one thing it would have had to do here.

LiveView owns no thread. viser's callbacks only record the newest pose and wake
whoever drives `pump()`: metal-gauss-view loops on it, and the trainer calls it
between optimisation steps on the training thread, so the GPU never sees two
threads. What to render next is decided by RenderScheduler, and how much of the
trainer's time the preview may take by PreviewBudget; both are pure so they can
be tested without a browser or a GPU. Copying frames off the GPU is not the
cost worth avoiding: memory is unified, a 768x768 uint8 frame is 1.7 MB, and
the render and the JPEG encode dominate.

The server binds to 127.0.0.1 unless told otherwise. viser's own default is
0.0.0.0, which would serve the scene to the whole network.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path

import torch

from metal_gauss.api import render
from metal_gauss.dataset import LazyViews
from metal_gauss.io import Splats, load_ply
from metal_gauss.keyframes import Keyframe, interpolate, preset
from metal_gauss.schedule import format_duration
from metal_gauss.render_path import (
    UP_AXES,
    Mp4Writer,
    _GL2CV,
    aperture_views,
    frame_cloud,
    world_to_camera,
)

_UP_VECTORS = {"-y": (0.0, -1.0, 0.0), "+y": (0.0, 1.0, 0.0),
               "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}


# --------------------------------------------------------------- camera maths

def quat_to_matrix(wxyz) -> torch.Tensor:
    """(3,3) rotation from a unit quaternion in viser's (w, x, y, z) order."""
    w, x, y, z = (float(v) for v in wxyz)
    return torch.tensor([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> tuple[float, float, float, float]:
    """(w, x, y, z) with w >= 0 for a (3,3) rotation; the inverse of quat_to_matrix.

    Branches on the largest of w, x, y, z so a half turn (w = 0) divides by
    something large rather than by nearly nothing.
    """
    m = [[float(R[i][j]) for j in range(3)] for i in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(1.0 + trace)
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, \
            (m[1][0] - m[0][1]) / s
    elif m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, \
            (m[0][2] + m[2][0]) / s
    elif m[1][1] >= m[2][2]:
        s = 2.0 * math.sqrt(1.0 - m[0][0] + m[1][1] - m[2][2])
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, \
            (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 - m[0][0] - m[1][1] + m[2][2])
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, \
            (m[1][2] + m[2][1]) / s, 0.25 * s
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return (w, x, y, z)


def view_frame_size(W0: int, H0: int, max_side: int, scale: float) -> tuple[int, int]:
    """Frame size for a snapped view: long side on the 16 grid, aspect kept.

    `frame_size` rounds both sides to 16, which stretches a view by up to a few
    percent once its intrinsics are scaled to that frame. Only the long side
    is rounded here; the short side follows the photograph.
    """
    long0, short0 = max(W0, H0), min(W0, H0)
    long = max(16, int(max_side * scale) // 16 * 16)
    short = max(1, round(long * short0 / long0))
    return (long, short) if W0 >= H0 else (short, long)


def vfov_from_K(K, H: int) -> float:
    """Vertical FOV in radians of a camera with intrinsics K and H rows."""
    return 2.0 * math.atan(0.5 * H / float(K[1][1]))


def scale_K(K: torch.Tensor, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> torch.Tensor:
    """Intrinsics for the same camera resampled from src_wh to dst_wh pixels."""
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    out = K.clone()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def letterbox(image, target_aspect: float, fill) -> "np.ndarray":
    """Centre an (H,W,3) uint8 frame in a canvas of `target_aspect` (W/H).

    The browser stretches whatever it is sent over the whole window, so a
    camera's frame has to be padded to the window's shape to keep its own:
    rows when the frame is wider than the window, columns when it is narrower.
    Within a pixel, unchanged.
    """
    import numpy as np

    H, W = image.shape[:2]
    rows = round(W / target_aspect) - H
    cols = round(H * target_aspect) - W
    if rows > 1:
        out = np.empty((H + rows, W, 3), dtype=image.dtype)
        out[:] = fill
        out[rows // 2:rows // 2 + H] = image
        return out
    if cols > 1:
        out = np.empty((H, W + cols, 3), dtype=image.dtype)
        out[:] = fill
        out[:, cols // 2:cols // 2 + W] = image
        return out
    return image


def pose_matches(pose, wxyz, position, tol: float) -> bool:
    """Whether a browser camera sits at `position` looking where `wxyz` looks.

    Roll is ignored on purpose: viser rebuilds the camera from position,
    look_at and its own up direction, so a rolled training camera always comes
    back unrolled and a full quaternion comparison would never match.
    """
    forward = quat_to_matrix(pose.wxyz)[:, 2]
    want = quat_to_matrix(wxyz)[:, 2]
    return (math.dist(pose.position, position) < tol
            and float(forward @ want) > 1.0 - 1e-5)


class SnapState:
    """Whether a client is still snapped to the camera it clicked.

    The browser may report poses on its way to the camera, so a snap only ends
    once the camera has been reached and then left.
    """

    def __init__(self) -> None:
        self.reached = False

    def update(self, matches: bool) -> bool:
        if matches:
            self.reached = True
            return True
        return not self.reached


def frustums_at_eyes(centres: torch.Tensor, eyes, radius: float) -> set[int]:
    """Indices of camera frustums with some browser's eye within `radius` of their apex.

    A frustum's lines start at its camera centre. With the eye on that centre,
    every click ray leaves from a point on those lines, so that frustum wins
    every pick and no other camera can be clicked. Those frustums are hidden,
    and hidden ones are not pickable.
    """
    if not eyes or centres.numel() == 0:
        return set()
    d = torch.cdist(torch.as_tensor(eyes, dtype=centres.dtype), centres)
    return set(torch.nonzero((d < radius).any(dim=0)).flatten().tolist())


def pose_to_viewmat(wxyz, position, world_flip: torch.Tensor | None = None) -> torch.Tensor:
    """World-to-camera matrix for a viser camera pose.

    viser reports camera-to-world in OpenCV axes (+Z forward, +Y down), the
    same axes `world_to_camera` takes, so no axis swap belongs here. An OpenGL
    file is framed in the flipped world, and the flip is applied last.
    """
    vm = world_to_camera(quat_to_matrix(wxyz), torch.as_tensor(position, dtype=torch.float32))
    return vm if world_flip is None else vm @ world_flip


def vertical_fov(fov_h_deg: float, aspect: float) -> float:
    """Horizontal FOV in degrees -> viser's vertical FOV in radians."""
    return 2.0 * math.atan(math.tan(0.5 * math.radians(fov_h_deg)) / aspect)


def intrinsics_vfov(W: int, H: int, fov_v: float) -> torch.Tensor:
    """Intrinsics from a vertical FOV in radians, square pixels, centred."""
    f = 0.5 * H / math.tan(0.5 * fov_v)
    return torch.tensor([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])


def frame_size(aspect: float, max_side: int, scale: float) -> tuple[int, int]:
    """(W, H) with the browser's aspect, the long side at max_side * scale.

    Both sides are multiples of 16. That keeps the set of frame sizes small, so
    the MPS allocator reuses buffers rather than allocating new ones every time
    the preview scale or the window changes, and it matches the tile size.
    """
    long = max(16, int(max_side * scale) // 16 * 16)
    short = max(16, round(long / max(aspect, 1.0 / aspect) / 16) * 16)
    return (long, short) if aspect >= 1.0 else (short, long)


# --------------------------------------------------------------- scheduling

@dataclasses.dataclass(frozen=True)
class Job:
    """One frame for one client.

    `lens` is the [start, stop) range of aperture samples to add to that
    client's running mean, or None for a pinhole frame. `gen` identifies the
    camera the job was issued for.
    """
    scale: float
    lens: tuple[int, int] | None
    preview: bool
    gen: int

    @property
    def quality(self) -> int:
        return 60 if self.preview else 90


class RenderScheduler:
    """What to render next for one client. Pure: the caller supplies the clock.

    While the camera moves, one reduced-resolution pinhole preview per new
    pose, at the largest scale step that fits the frame budget. Once it has
    been still for SETTLE_S: one full-resolution pinhole frame, or with an
    aperture, the lens refined at 8, 16, 32 and 64 samples and then all of
    them. After that, nothing, so an untouched viewer leaves the GPU alone.
    """

    SETTLE_S = 0.15
    FRAME_BUDGET_S = 1.0 / 30.0
    SCALES = (0.25, 0.5, 0.75, 1.0)
    CHECKPOINTS = (8, 16, 32, 64)

    def __init__(self) -> None:
        self._gen = 0
        self._changed_at = -math.inf
        self._dirty = False
        self._full_done = False
        self._lens_done = 0
        self._stalled = False
        self._scale = 1.0

    def touch(self, now: float) -> None:
        """The camera or a render setting changed."""
        self._gen += 1
        self._changed_at = now
        self._dirty = True
        self._full_done = False
        self._lens_done = 0
        self._stalled = False

    def invalidate(self) -> None:
        """The splats changed under an unchanged camera: training took a step.

        A new settled frame is due, but this is not motion, so no preview. A
        render that failed stays stopped: the model changes every step, and a
        persistent error would otherwise be retried every step.
        """
        self._gen += 1
        self._full_done = False
        self._lens_done = 0

    def _moving(self, now: float) -> bool:
        return now - self._changed_at < self.SETTLE_S

    def next_job(self, now: float, samples: int) -> Job | None:
        """`samples` is the aperture's sample count, 0 for a pinhole."""
        if self._moving(now):
            if not self._dirty:
                return None
            self._dirty = False
            return Job(self._scale, None, True, self._gen)
        if self._stalled:
            return None
        if samples <= 0:
            return None if self._full_done else Job(1.0, None, False, self._gen)
        if self._lens_done >= samples:
            return None
        stop = next(c for c in (*self.CHECKPOINTS, samples)
                    if self._lens_done < c <= samples)
        return Job(1.0, (self._lens_done, stop), False, self._gen)

    def wait_s(self, now: float) -> float | None:
        """Seconds until work appears without a new touch, or None if it will not."""
        if self._moving(now) and not self._dirty:
            return self.SETTLE_S - (now - self._changed_at)
        return None

    def done(self, job: Job, seconds: float) -> None:
        """A job finished in `seconds`, encode included."""
        if job.lens is None:
            # Timing holds whichever camera it was for: cost scales with pixels.
            per_full = seconds / (job.scale * job.scale)
            fits = [s for s in self.SCALES if per_full * s * s <= self.FRAME_BUDGET_S]
            self._scale = fits[-1] if fits else self.SCALES[0]
        if job.gen != self._gen:
            return
        if job.lens is not None:
            self._lens_done = job.lens[1]
        elif not job.preview:
            self._full_done = True

    def failed(self, job: Job) -> None:
        """Stop until the next change rather than retrying a failing render."""
        if job.gen == self._gen:
            self._stalled = True


class PreviewBudget:
    """How much wall-clock the preview may take from whoever calls pump().

    Preview seconds inside a sliding window must stay under `fraction` of the
    window. A window rather than a running total, so a burst of dragging early
    in a run is not paid for by an hour of no preview at all later.

    One second, not five: a 5 s window let a drag spend half a second of
    frames up front and then froze the preview for seconds while the window
    emptied, which read as a hang. A short window spreads the same share out.
    """

    def __init__(self, fraction: float, window_s: float = 1.0) -> None:
        self.fraction = fraction
        self.window_s = window_s
        self._frames: deque[tuple[float, float]] = deque()

    def allow(self, now: float) -> bool:
        if self.fraction <= 0.0:
            return False
        while self._frames and self._frames[0][0] < now - self.window_s:
            self._frames.popleft()
        return sum(s for _, s in self._frames) < self.fraction * self.window_s

    def spend(self, now: float, seconds: float) -> None:
        self._frames.append((now, seconds))


class TrainClock:
    """Training wall-clock that stops while the viewer has training paused.

    wall_s, ms/step and the stall detector all read this, so a pause neither
    inflates a reported time nor reads as the machine having gone to sleep.
    """

    def __init__(self, now: Callable[[], float] = time.perf_counter) -> None:
        self._now = now
        self._start = now()
        self._paused_at: float | None = None
        self.paused_s = 0.0

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = self._now()

    def resume(self) -> None:
        if self._paused_at is not None:
            self.paused_s += self._now() - self._paused_at
            self._paused_at = None

    def elapsed(self) -> float:
        end = self._paused_at if self._paused_at is not None else self._now()
        return end - self._start - self.paused_s


def infer_up(viewmats) -> torch.Tensor:
    """World up from a camera rig: the normalised mean of the cameras' up vectors.

    Captures hold the camera roughly level, so each camera's image-up is close
    to world up, and around a rig the individual tilts cancel. That is right
    for a Blender ring (Z up) and a hand-held COLMAP walk-around alike, so the
    trainer's viewer needs no --up. Row 1 of a world-to-camera matrix is the
    camera's down axis in world coordinates.
    """
    up = -torch.stack([vm[1, :3] for vm in viewmats]).mean(dim=0)
    return up / torch.linalg.norm(up)


# --------------------------------------------------------------- rendering

@dataclasses.dataclass(frozen=True)
class SplatBatch:
    """Activated splats, ready for the rasteriser: what one pump renders."""
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh: torch.Tensor
    sh_rest: torch.Tensor | None
    sh_degree: int
    antialias: bool = False

    @classmethod
    def from_splats(cls, sp: Splats) -> "SplatBatch":
        return cls(sp.means, sp.quats, sp.scales, sp.opacities, sp.sh, None, sp.sh_degree)

    def __len__(self) -> int:
        return int(self.means.shape[0])


def render_view(batch: SplatBatch, viewmat: torch.Tensor, K: torch.Tensor, W: int, H: int,
                background, far: float) -> torch.Tensor:
    """One (H,W,3) frame in [0,1] on the GPU, as `render_frames` draws it."""
    rgb, _, _ = render(batch.means, batch.quats, batch.scales, batch.opacities, batch.sh,
                       K, viewmat, W, H, sh_degree=batch.sh_degree, backend="metal",
                       background=background, far=far, sh_rest=batch.sh_rest,
                       antialias=batch.antialias)
    return rgb.detach().clamp(0.0, 1.0)


def readout_text(seconds: float, splats: int, W: int, H: int, *, samples: int | None = None,
                 fps: float | None = None, gb: float | None = None) -> str:
    """What the last frame cost, for the Render panel."""
    parts = [f"{1000.0 * seconds:.1f} ms/frame", f"{splats:,} splats", f"{W}×{H}"]
    if samples is not None:
        parts.append(f"{samples} samples")
    if fps is not None:
        parts.append(f"{fps:.0f} fps")
    if gb is not None:
        parts.append(f"{gb:.1f} GB")
    return " · ".join(parts)


def export_plan(keys: list[Keyframe], frames: int, resolution: int, aspect: float,
                world_flip: torch.Tensor | None = None,
                loop: bool = True) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One (view matrix, intrinsics) per frame of a video, planned up front.

    Planned before anything is rendered so the writer knows its frame size and
    the job can be handed out one frame at a time, however slowly they come.
    """
    W, H = frame_size(aspect, resolution, 1.0)
    return [(pose_to_viewmat(k.wxyz, k.position, world_flip), intrinsics_vfov(W, H, k.fov))
            for k in interpolate(keys, frames, loop)]


@dataclasses.dataclass
class _Export:
    """A video being written, a frame per pump."""
    plan: list[tuple[torch.Tensor, torch.Tensor]]
    W: int
    H: int
    writer: object
    out: str
    radius: float
    samples: int
    focus: float
    index: int = 0


def import_viser():
    try:
        import viser
    except ImportError as e:
        raise SystemExit(
            "metal-gauss-view needs viser, which is an optional extra: "
            'pip install "metal-gauss[viewer]"') from e
    return viser


def scene_far(means: torch.Tensor, eye: torch.Tensor, target: torch.Tensor,
              quantile: float = 0.98) -> float:
    """A far plane that does not cull the scene when the camera dollies out.

    The renderer's default of 100 is fine for a framed render and not for a
    viewer, where the camera goes wherever it is dragged.
    """
    lo, hi = means.quantile(1.0 - quantile, dim=0), means.quantile(quantile, dim=0)
    radius = float(torch.linalg.norm(hi - lo)) * 0.5
    return max(100.0, 10.0 * (float(torch.linalg.norm(target - eye)) + radius))


@dataclasses.dataclass(frozen=True)
class _Pose:
    wxyz: tuple[float, ...]
    position: tuple[float, ...]
    look_at: tuple[float, ...]
    fov: float
    aspect: float


def _pose_of(camera) -> _Pose:
    """The fields that change the image. Not image_width: that is device pixels,
    which the browser changes on its own and must not restart refinement."""
    return _Pose(tuple(float(v) for v in camera.wxyz),
                 tuple(float(v) for v in camera.position),
                 tuple(float(v) for v in camera.look_at),
                 float(camera.fov), float(camera.aspect))


class _Client:
    def __init__(self, handle) -> None:
        self.handle = handle
        self.lock = threading.Lock()
        self.scheduler = RenderScheduler()
        self.pose: _Pose | None = None
        self.lens_gen = -1
        self.views: list[torch.Tensor] = []
        self.acc: torch.Tensor | None = None
        self.snap: _Snap | None = None


@dataclasses.dataclass
class _Snap:
    """A client showing one training or held-out camera, clicked in the scene."""
    split: str
    index: int
    view: object
    wxyz: tuple[float, float, float, float]
    position: tuple[float, float, float]
    state: SnapState = dataclasses.field(default_factory=SnapState)
    psnr: float | None = None


def _centre_of(viewmat: torch.Tensor) -> torch.Tensor:
    return -viewmat[:3, :3].T @ viewmat[:3, 3]


class LiveView:
    """The viser server, its GUI and per-client scheduling, driven by `pump()`.

    `source` returns the SplatBatch to draw and is called once per pump that
    renders anything, so a trainer can hand in a closure over parameters that
    change every step. The FOV is either horizontal (`fov_h`, degrees, turned
    into each browser's vertical FOV through its aspect, as metal-gauss-render
    means it) or already vertical (`fov_v`, radians, e.g. from a camera's K).
    """

    def __init__(self, viser, source: Callable[[], SplatBatch], *, eye: torch.Tensor,
                 target: torch.Tensor, up, background, far: float, host: str, port: int,
                 max_resolution: int = 1024, fov_h: float | None = None,
                 fov_v: float | None = None, world_flip: torch.Tensor | None = None,
                 warm_up: bool = True) -> None:
        self.source = source
        self.fov_h, self.fov_v = fov_h, fov_v
        self.flip = world_flip
        self.background = background
        self.far = far
        self.clients: dict[int, _Client] = {}
        self.clients_lock = threading.Lock()
        self._keys: list[Keyframe] = []
        self._export: _Export | None = None
        self.path_cancel = None
        self.wake = threading.Event()
        self.preview_s = 0.0
        self._rr = 0

        # Build the Metal extension here, before any client can ask for a frame:
        # nothing in metal_backend._load guards a second caller. A trainer skips
        # this: it renders on the same thread and builds the extension anyway.
        if warm_up:
            with torch.no_grad():
                render_view(source(), world_to_camera(torch.eye(3), eye),
                            intrinsics_vfov(16, 16, 1.0), 16, 16, background, far)

        up_vector = _UP_VECTORS[up] if isinstance(up, str) else tuple(float(v) for v in up)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction(up if isinstance(up, str) else up_vector)
        # Set before any client connects: afterwards it only moves Reset View.
        self.server.initial_camera.position = eye.numpy()
        self.server.initial_camera.look_at = target.numpy()
        self.server.initial_camera.up = up_vector

        dist = float(torch.linalg.norm(target - eye))
        gui = self.server.gui
        with gui.add_folder("Render"):
            self.max_resolution = gui.add_slider(
                "Max resolution", min=256, max=2048, step=16, initial_value=max_resolution)
            self.readout = gui.add_markdown("waiting for the first frame")
        with gui.add_folder("Lens"):
            self.aperture = gui.add_slider(
                "Aperture", min=0.0, max=0.1 * dist, step=0.001 * dist, initial_value=0.0,
                hint="lens radius in world units; 0 is a pinhole")
            self.auto_focus = gui.add_checkbox("Focus on orbit centre", initial_value=True)
            self.focus = gui.add_slider(
                "Focus distance", min=0.05 * dist, max=4.0 * dist, step=0.005 * dist,
                initial_value=dist, disabled=True)
            self.samples = gui.add_slider("Samples", min=8, max=128, step=8, initial_value=96)
        with gui.add_folder("Path", order=0.5):
            self.path_count = gui.add_markdown("no keyframes")
            self.path_add = gui.add_button("Add keyframe")
            self.path_clear = gui.add_button("Clear")
            self.path_orbit = gui.add_button("Orbit preset")
            self.path_wiggle = gui.add_button("Wiggle preset")
            self.path_sweep = gui.add_slider("Preset sweep", min=2.0, max=180.0, step=1.0,
                                             initial_value=20.0)
            self.path_frames = gui.add_slider("Frames", min=12, max=600, step=6,
                                              initial_value=60)
            self.path_fps = gui.add_slider("Frames per second", min=6, max=60, step=6,
                                           initial_value=30)
            self.path_resolution = gui.add_slider("Video resolution", min=256, max=2048,
                                                  step=16, initial_value=1024)
            self.path_lens = gui.add_checkbox("Use the current aperture", initial_value=False)
            self.path_out = gui.add_text("Output file", initial_value="path.mp4")
            self.path_export = gui.add_button("Export")
            self.path_cancel = gui.add_button("Cancel", disabled=True)
            self.path_progress = gui.add_markdown("")
        self.status = gui.add_markdown("")

        @self.path_add.on_click
        def _(event) -> None:
            camera = event.client.camera
            self._keys.append(Keyframe(tuple(float(v) for v in camera.position),
                                       tuple(float(v) for v in camera.wxyz),
                                       float(camera.fov)))
            self._show_keys()

        @self.path_clear.on_click
        def _(_) -> None:
            self._keys.clear()
            self._show_keys()

        @self.path_orbit.on_click
        def _(event) -> None:
            self._preset("orbit", event.client)

        @self.path_wiggle.on_click
        def _(event) -> None:
            self._preset("wiggle", event.client)

        @self.path_export.on_click
        def _(event) -> None:
            self._start_export(event.client)

        @self.path_cancel.on_click
        def _(_) -> None:
            export, self._export = self._export, None
            if export is not None:
                export.writer.abort()
            self._finish_export(None, "cancelled")

        for handle in (self.max_resolution, self.aperture, self.focus, self.samples):
            handle.on_update(lambda _: self.touch_all())

        @self.auto_focus.on_update
        def _(_) -> None:
            self.focus.disabled = self.auto_focus.value
            self.touch_all()

        @self.server.on_client_connect
        def _(handle) -> None:
            client = _Client(handle)
            handle.camera.fov = (self.fov_v if self.fov_v is not None
                                 else vertical_fov(self.fov_h, handle.camera.aspect))

            @handle.camera.on_update
            async def _(camera) -> None:
                pose = _pose_of(camera)
                with client.lock:
                    if pose != client.pose:
                        client.pose = pose
                        client.scheduler.touch(time.monotonic())
                        self._moved(client, pose)
                self.wake.set()

            with client.lock:
                client.pose = _pose_of(handle.camera)
            with self.clients_lock:
                self.clients[handle.client_id] = client
            self._connected()
            self.wake.set()

        @self.server.on_client_disconnect
        def _(handle) -> None:
            with self.clients_lock:
                self.clients.pop(handle.client_id, None)
            self._connected()

    def _clients(self) -> list[_Client]:
        with self.clients_lock:
            return list(self.clients.values())

    def _moved(self, client: _Client, pose: _Pose) -> None:
        """Called with client.lock held whenever a client's camera changed."""

    def _connected(self) -> None:
        """Called after a client joined or left, with no lock held."""

    def touch_all(self) -> None:
        """A render setting changed: every client counts as moved."""
        now = time.monotonic()
        for client in self._clients():
            with client.lock:
                client.scheduler.touch(now)
        self.wake.set()

    def invalidate_all(self) -> None:
        """The splats changed: every client needs a new settled frame."""
        for client in self._clients():
            with client.lock:
                client.scheduler.invalidate()
        self.wake.set()

    def next_wait(self) -> float | None:
        """Seconds until some client has work without a new event, or None."""
        now = time.monotonic()
        wait = None
        for client in self._clients():
            with client.lock:
                until = client.scheduler.wait_s(now)
            if until is not None:
                wait = until if wait is None else min(wait, until)
        return wait

    def pump(self, max_jobs: int | None = None, budget: PreviewBudget | None = None,
             lens: bool = True) -> int:
        """Render the frames that are due, round-robin over clients; return how many.

        Nothing due means no GPU work at all, not even a sync. Otherwise the
        queue is synchronised BEFORE the frame clock starts, so work the caller
        left queued is charged to the caller and not to the preview.
        """
        clients = self._clients()
        export = self._export
        if not clients and export is None:
            return 0
        if clients:
            self._rr = (self._rr + 1) % len(clients)
            clients = clients[self._rr:] + clients[:self._rr]
        samples = int(self.samples.value) if lens and self.aperture.value > 0 else 0
        batch = None
        rendered = 0
        with torch.no_grad():                   # thread-local, so entered here
            # The video first: it was asked for explicitly, and a preview frame
            # is cheap to postpone.
            if export is not None and (budget is None or budget.allow(time.monotonic())):
                t_sync = time.perf_counter()
                torch.mps.synchronize()
                synced = time.perf_counter() - t_sync
                batch = self.source()
                t0 = time.perf_counter()
                if self._export_frame(export, batch):
                    seconds = time.perf_counter() - t0 + synced
                    if budget is not None:
                        budget.spend(time.monotonic(), seconds)
                        self.preview_s += seconds
                    rendered += 1
            for client in clients:
                if max_jobs is not None and rendered >= max_jobs:
                    break
                now = time.monotonic()
                if budget is not None and not budget.allow(now):
                    break
                with client.lock:
                    pose = client.pose
                    job = None if pose is None else client.scheduler.next_job(now, samples)
                if job is None:
                    continue
                synced = 0.0
                if batch is None:
                    t_sync = time.perf_counter()
                    torch.mps.synchronize()
                    synced = time.perf_counter() - t_sync
                    batch = self.source()
                t0 = time.perf_counter()
                try:
                    W, H, image = self._frame(client, job, pose, samples, batch)
                    client.handle.scene.set_background_image(
                        image, format="jpeg", jpeg_quality=job.quality)
                except Exception as e:
                    traceback.print_exc()
                    self.status.content = f"**render failed:** `{type(e).__name__}: {e}`"
                    with client.lock:
                        client.scheduler.failed(job)
                    continue
                seconds = time.perf_counter() - t0
                with client.lock:
                    client.scheduler.done(job, seconds)
                if budget is not None:
                    # Only budgeted frames took time from running training;
                    # paused and finished renders cost the trainer nothing. The
                    # flush before the frame is charged too: it stalls the
                    # trainer's CPU/GPU pipelining. Measured without it, frames
                    # were 9% of wall-clock while training ran 17% slower.
                    budget.spend(time.monotonic(), seconds + synced)
                    self.preview_s += seconds + synced
                rendered += 1
                self._report(job, W, H, seconds, len(batch))
        return rendered

    def _show_keys(self) -> None:
        n = len(self._keys)
        self.path_count.content = ("no keyframes" if n == 0 else
                                   "1 keyframe" if n == 1 else f"{n} keyframes")

    def _preset(self, kind: str, client) -> None:
        """Fill the path with an orbit or wiggle around what this browser sees.

        Keyframes stay in the browser's world, which is already the OpenCV one
        the paths are drawn in; an OpenGL file's flip belongs at the end, in
        `export_plan`, exactly where a single rendered frame applies it.
        """
        camera = client.camera
        self._keys = preset(kind,
                            torch.tensor(tuple(float(v) for v in camera.position)),
                            torch.tensor(tuple(float(v) for v in camera.look_at)),
                            frames=12, sweep_deg=float(self.path_sweep.value),
                            fov=float(camera.fov))
        self._show_keys()

    def _start_export(self, client) -> None:
        if self._export is not None:
            self.path_progress.content = "already exporting; cancel it first"
            return
        keys = list(self._keys)              # another browser may be adding to it
        if not keys:
            self.path_progress.content = "add a keyframe first, or pick a preset"
            return
        aspect = float(client.camera.aspect)
        resolution = int(self.path_resolution.value)
        W, H = frame_size(aspect, resolution, 1.0)
        out = str(self.path_out.value) or "path.mp4"
        try:
            writer = Mp4Writer(Path(out), W, H, int(self.path_fps.value))
        except RuntimeError as e:
            self.path_progress.content = f"**{e}**"
            return
        radius = float(self.aperture.value) if self.path_lens.value else 0.0
        focus = (math.dist(client.camera.look_at, client.camera.position)
                 if self.auto_focus.value else float(self.focus.value))
        self._export = _Export(
            plan=export_plan(keys, int(self.path_frames.value), resolution, aspect,
                             self.flip),
            W=W, H=H, writer=writer, out=out, radius=radius,
            samples=int(self.samples.value), focus=focus)
        self.path_export.disabled = True
        self.path_cancel.disabled = False
        self.path_progress.content = f"frame 0/{len(self._export.plan)}"
        self.wake.set()

    def _export_frame(self, export: _Export, batch: SplatBatch) -> bool:
        """Render and write the next frame of the video. True if one was written."""
        try:
            vm, K = export.plan[export.index]
            views = ([vm] if export.radius <= 0.0 else
                     self._lens_views(vm, export.focus, export.radius, export.samples))
            acc = None
            for view in views:
                frame = render_view(batch, view, K, export.W, export.H,
                                    self.background, self.far)
                acc = frame if acc is None else acc + frame
            export.writer.write((acc / len(views) * 255.0).round().to(torch.uint8).cpu())
        except Exception as e:
            # The Path panel, not the shared status line: a preview frame that
            # succeeds right after would clear that one and the failure would
            # never be read.
            traceback.print_exc()
            export.writer.abort()
            self._finish_export(export, f"**export failed:** `{type(e).__name__}: {e}`")
            return False

        export.index += 1
        self.path_progress.content = f"frame {export.index}/{len(export.plan)}"
        if export.index >= len(export.plan):
            try:
                export.writer.close()
            except Exception as e:
                self._finish_export(export, f"**export failed:** `{type(e).__name__}: {e}`")
                return True
            self._finish_export(export, f"wrote {export.out}")
        return True

    def _finish_export(self, export: "_Export | None", message: str | None) -> None:
        # A cancelled export can still fail a frame on the render thread after
        # the next one has started; its cleanup must not take the new one down.
        if export is not None and self._export is not export:
            return
        self._export = None
        if message is not None:
            self.path_progress.content = message
        if self.path_cancel is not None:
            self.path_cancel.disabled = True
            self.path_export.disabled = False

    def _lens_views(self, vm: torch.Tensor, focus: float, radius: float,
                    samples: int) -> list[torch.Tensor]:
        """The aperture's views around a rendered view matrix, roll and all.

        `render_path.lens_views` rebuilds each sample upright from an up axis,
        which would straighten a path camera that banks; the plan's own
        rotation is used instead.
        """
        cv = vm if self.flip is None else vm @ self.flip
        R = cv[:3, :3].T
        eye = -R @ cv[:3, 3]
        views = aperture_views(eye, eye + R[:, 2] * focus, radius, samples, R0=R)
        return views if self.flip is None else [v @ self.flip for v in views]

    def serve_forever(self) -> None:
        """Keep rendering on the calling thread until Ctrl-C, then stop the server."""
        try:
            while True:
                self.wake.clear()
                if self.pump() == 0:
                    self.wake.wait(timeout=self.next_wait() or 0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()

    def _frame(self, client: _Client, job: Job, pose: _Pose, samples: int,
               batch: SplatBatch):
        W, H = frame_size(pose.aspect, int(self.max_resolution.value), job.scale)
        K = intrinsics_vfov(W, H, pose.fov)
        if job.lens is None:
            vm = pose_to_viewmat(pose.wxyz, pose.position, self.flip)
            rgb = render_view(batch, vm, K, W, H, self.background, self.far)
        else:
            if client.lens_gen != job.gen:
                R = quat_to_matrix(pose.wxyz)
                eye = torch.tensor(pose.position, dtype=torch.float32)
                focus = (math.dist(pose.look_at, pose.position) if self.auto_focus.value
                         else float(self.focus.value))
                views = aperture_views(eye, eye + R[:, 2] * focus,
                                       float(self.aperture.value), samples, R0=R)
                client.views = views if self.flip is None else [v @ self.flip for v in views]
                client.acc = None
                client.lens_gen = job.gen
            start, stop = job.lens
            for vm in client.views[start:stop]:
                frame = render_view(batch, vm, K, W, H, self.background, self.far)
                client.acc = frame if client.acc is None else client.acc + frame
            rgb = client.acc / stop
        image = (rgb * 255.0).round().to(torch.uint8).cpu().numpy()
        return W, H, image

    def _report(self, job: Job, W: int, H: int, seconds: float, splats: int) -> None:
        self.readout.content = readout_text(
            seconds, splats, W, H,
            samples=job.lens[1] if job.lens is not None else None,
            fps=None if job.lens is not None else 1.0 / seconds,
            gb=torch.mps.driver_allocated_memory() / 1e9)
        if self.status.content:
            self.status.content = ""


class TrainingView(LiveView):
    """The live preview of a model while it trains.

    The trainer calls `after_step` once per step, on its own thread. With no
    browser connected that returns at once and costs nothing. Otherwise it
    marks every client's frame stale, since the splats just moved, and renders
    at most one frame, and only if PreviewBudget allows. `hold` blocks while
    the Pause button is down, rendering with the whole GPU, lens included.
    """

    STATS_EVERY_S = 0.25
    LOSS_EVERY_S = 0.5      # loss.item() synchronises the queue; not every step

    def __init__(self, viser, scene, source: Callable[[], SplatBatch], *, background,
                 host: str, port: int, budget: float) -> None:
        first = scene.train[0]
        c2w = first.viewmat[:3, :3].T
        eye = -c2w @ first.viewmat[:3, 3]
        points = torch.as_tensor(scene.points, dtype=torch.float32)
        depth = max(float((points.mean(dim=0) - eye) @ c2w[:, 2]), 1e-3)
        target = eye + c2w[:, 2] * depth
        fov_v = 2.0 * math.atan(0.5 * int(first.image.shape[0]) / float(first.K[1, 1]))
        super().__init__(viser, source, eye=eye, target=target,
                         up=infer_up([v.viewmat for v in scene.train]),
                         background=background, far=scene_far(points, eye, target),
                         host=host, port=port, fov_v=fov_v, warm_up=False)

        self.budget = PreviewBudget(budget)
        self.paused = False
        self._stats: dict = {}
        self._stats_at = -math.inf
        self._loss_at = -math.inf

        gui = self.server.gui
        with gui.add_folder("Training", order=-1.0):
            self.stats = gui.add_markdown("waiting for the first step")
            self.pause_button = gui.add_button("Pause")
            self.budget_slider = gui.add_slider(
                "Preview budget", min=0.0, max=0.5, step=0.01, initial_value=budget,
                hint="largest share of wall-clock the preview may take while training runs")

        @self.pause_button.on_click
        def _(_) -> None:
            self.paused = not self.paused
            self.pause_button.label = "Resume" if self.paused else "Pause"
            self.wake.set()

        @self.budget_slider.on_update
        def _(_) -> None:
            self.budget.fraction = float(self.budget_slider.value)

        self.scene = scene
        self._centre = points.mean(dim=0)
        centres = torch.stack([_centre_of(v.viewmat) for v in scene.train])
        spread = max(float(torch.linalg.norm(centres - centres.mean(dim=0), dim=1).max()), 1e-3)
        self._frustum_scale = 0.05 * spread
        self._snap_tol = 1e-3 * spread
        self._fill = tuple(int(round(255.0 * c)) for c in background)
        self._frustums: list = []
        self._frustum_centres: list[torch.Tensor] = []
        self._frustum_shown: list[bool] = []
        self._heldout_drawn = False
        with gui.add_folder("Cameras", order=-0.5):
            self.show_cameras = gui.add_checkbox("Show cameras", initial_value=True)
            self.show_photo = gui.add_checkbox(
                "Show photo", initial_value=False,
                hint="after clicking a camera, show its photograph in place of the render")

        @self.show_cameras.on_update
        def _(_) -> None:
            self._sync_frustums()

        @self.show_photo.on_update
        def _(_) -> None:
            self.touch_all()

        self._draw_cameras("train", scene.train)
        self.heldout_ready()

    def heldout_ready(self) -> None:
        """Draw the held-out cameras once their split is decoded. Never decodes it.

        A Blender scene decodes its 200 held-out images on first use, which was
        4.1 s of startup; drawing their frustums must not be what triggers it.
        """
        heldout = self.scene.heldout
        if self._heldout_drawn or (isinstance(heldout, LazyViews) and not heldout.materialised):
            return
        self._draw_cameras("heldout", heldout)
        self._heldout_drawn = True
        self._sync_frustums()

    def _draw_cameras(self, split: str, views) -> None:
        colour = (40, 110, 255) if split == "train" else (255, 140, 0)
        for i, view in enumerate(views):
            H, W = (int(s) for s in view.image.shape[:2])
            centre = _centre_of(view.viewmat)
            frustum = self.server.scene.add_camera_frustum(
                f"/cameras/{split}/{i}", fov=vfov_from_K(view.K, H), aspect=W / H,
                scale=self._frustum_scale, color=colour,
                # Pixels, not world units: world-width lines were too thin to
                # click far away and swelled into bars over the render up close.
                thickness=4.0, thickness_units="screen",
                wxyz=matrix_to_quat(view.viewmat[:3, :3].T),
                position=centre.numpy(),
                visible=self.show_cameras.value)
            frustum.on_click(self._snapper(split, i, view))
            self._frustums.append(frustum)
            self._frustum_centres.append(centre)
            self._frustum_shown.append(bool(self.show_cameras.value))

    def _sync_frustums(self) -> None:
        """Show every frustum except those a browser's eye sits on.

        Client poses are read without their locks: a pose is replaced whole,
        never mutated, and a stale read is corrected by the next camera update.
        """
        # A browser left open reconnects the moment the server starts, which is
        # before __init__ has drawn anything.
        if not getattr(self, "_frustums", None):
            return
        eyes = [c.pose.position for c in self._clients() if c.pose is not None]
        at_eye = frustums_at_eyes(torch.stack(self._frustum_centres), eyes,
                                  self._frustum_scale)
        show = bool(self.show_cameras.value)
        for i, frustum in enumerate(self._frustums):
            want = show and i not in at_eye
            if self._frustum_shown[i] != want:
                frustum.visible = want
                self._frustum_shown[i] = want

    def _connected(self) -> None:
        self._sync_frustums()

    def _snapper(self, split: str, index: int, view):
        def snap(event) -> None:
            self._snap(event.client, split, index, view)
        return snap

    def _snap(self, handle, split: str, index: int, view) -> None:
        """Move the clicking browser onto a camera, in one message."""
        with self.clients_lock:
            client = self.clients.get(handle.client_id)
        if client is None:
            return
        c2w = view.viewmat[:3, :3].T
        centre = _centre_of(view.viewmat)
        depth = max(float((self._centre - centre) @ c2w[:, 2]), 1e-3)
        snap = _Snap(split, index, view, matrix_to_quat(c2w), tuple(centre.tolist()))
        with client.lock:
            client.snap = snap
            # Already sitting on this camera: no pose update will arrive to say so.
            if client.pose is not None and pose_matches(client.pose, snap.wxyz,
                                                        snap.position, self._snap_tol):
                snap.state.update(True)
            client.scheduler.touch(time.monotonic())
        self._stats.pop("view", None)
        with handle.atomic():
            handle.camera.fov = vfov_from_K(view.K, int(view.image.shape[0]))
            handle.camera.position = centre.numpy()
            handle.camera.look_at = (centre + c2w[:, 2] * depth).numpy()
        self.wake.set()

    def _moved(self, client: _Client, pose: _Pose) -> None:
        snap = client.snap
        if snap is not None and not snap.state.update(
                pose_matches(pose, snap.wxyz, snap.position, self._snap_tol)):
            client.snap = None
            self._stats.pop("view", None)
            self._show_stats()
        self._sync_frustums()

    def _frame(self, client: _Client, job: Job, pose: _Pose, samples: int,
               batch: SplatBatch):
        """A snapped client sees its camera exactly: that view's own pose and K,
        at the view's aspect, letterboxed into the window, so render and photo
        share pixel coordinates. The lens path stays pose-based."""
        with client.lock:
            snap = client.snap
        if snap is None or job.lens is not None:
            return super()._frame(client, job, pose, samples, batch)
        view = snap.view
        H0, W0 = (int(s) for s in view.image.shape[:2])
        W, H = view_frame_size(W0, H0, int(self.max_resolution.value), job.scale)
        if self.show_photo.value:
            photo = view.image.permute(2, 0, 1)[None].float()
            image = torch.nn.functional.interpolate(photo, size=(H, W), mode="area")[0] \
                .permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
        else:
            rgb = render_view(batch, view.viewmat, scale_K(view.K, (W0, H0), (W, H)),
                              W, H, self.background, self.far)
            image = (rgb * 255.0).round().to(torch.uint8).cpu().numpy()
            if not job.preview and snap.psnr is None:
                snap.psnr = self._view_psnr(view, batch)
                if snap.split == "train":
                    label = "train, no appearance correction"
                else:
                    label = "held-out"
                self._stats["view"] = f"view {view.name} ({label}): {snap.psnr:.2f} dB"
                self._show_stats()
        image = letterbox(image, pose.aspect, self._fill)
        return image.shape[1], image.shape[0], image

    def _view_psnr(self, view, batch: SplatBatch) -> float:
        """PSNR of one view at its native resolution, computed as evaluate() does."""
        H, W = (int(s) for s in view.image.shape[:2])
        rgb = render_view(batch, view.viewmat, view.K, W, H, self.background, self.far)
        gt = view.image.to(rgb.device).float() / 255.0
        mse = float(((rgb - gt) ** 2).mean())
        return -10.0 * math.log10(max(mse, 1e-10))

    def after_step(self, step: int, active: int, loss: torch.Tensor, clock: TrainClock) -> None:
        if not self.clients:
            return
        now = time.monotonic()
        if now - self._loss_at >= self.LOSS_EVERY_S:
            t_loss = time.perf_counter()
            self._stats["loss"] = float(loss.item())
            waited = time.perf_counter() - t_loss
            # A sync that happens only because a browser is watching.
            self.budget.spend(now, waited)
            self.preview_s += waited
            self._loss_at = now
        if now - self._stats_at >= self.STATS_EVERY_S:
            elapsed = clock.elapsed()
            self._stats.update(step=step, active=active, ms_step=1000.0 * elapsed / step,
                               share=self.preview_s / max(elapsed, 1e-9))
            self._show_stats()
            self._stats_at = now
        self.invalidate_all()
        self.pump(max_jobs=1, budget=self.budget, lens=False)

    def hold(self, clock: TrainClock) -> None:
        """Block while paused. Paused time is not training time."""
        if not self.paused:
            return
        clock.pause()
        self.set_status("paused")
        try:
            while self.paused:
                self.wake.clear()
                if self.pump() == 0:
                    self.wake.wait(timeout=self.next_wait() or 0.5)
        finally:
            clock.resume()
            self.set_status(None)

    def set_status(self, text: str | None) -> None:
        self._stats["status"] = text
        self._show_stats()

    def set_psnr(self, psnr: float) -> None:
        self._stats["psnr"] = psnr
        self.set_status(None)

    def set_eta(self, eta_s: float | None) -> None:
        """Seconds the trainer thinks are left, or None while it cannot tell."""
        self._stats["eta"] = eta_s

    def finish(self) -> None:
        """Training is over: keep showing the final model until Ctrl-C."""
        # A pause clicked after the last step was never held; release it.
        self.paused = False
        self.pause_button.label = "Pause"
        self.pause_button.disabled = True
        self.set_status("training finished · Ctrl-C to exit")
        self.invalidate_all()
        self.serve_forever()

    def _show_stats(self) -> None:
        # A copy: click and camera callbacks drop "view" on viser's threads, and
        # a check-then-read on the live dict raced with that.
        s = dict(self._stats)
        lines = []
        if "step" in s:
            lines.append(f"step {s['step']:,} · {s['active']:,} splats · "
                         f"{s['ms_step']:.0f} ms/step")
        parts = []
        if "loss" in s:
            parts.append(f"loss {s['loss']:.4f}")
        if "psnr" in s:
            parts.append(f"held-out {s['psnr']:.2f} dB")
        if parts:
            lines.append(" · ".join(parts))
        if s.get("eta"):
            lines.append(f"eta {format_duration(s['eta'])}")
        if "share" in s:
            lines.append(f"preview {s['share']:.0%} of wall-clock")
        if s.get("view"):
            lines.append(s["view"])
        if s.get("status"):
            lines.append(f"**{s['status']}**")
        self.stats.content = "  \n".join(lines) or "waiting for the first step"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="metal-gauss-view",
        description="Orbit a .ply in the browser, rendered on Metal.")
    ap.add_argument("ply")
    ap.add_argument("--up", choices=UP_AXES, default="-y",
                    help="the scene's vertical axis. -y is the OpenCV world; a "
                         "scene trained with --blender is +z. Negative axes need "
                         "the = form: --up=-z.")
    ap.add_argument("--background", choices=("white", "black"), default="white")
    ap.add_argument("--convention", choices=("opencv", "opengl"), default="opencv",
                    help="frame the .ply is written in")
    ap.add_argument("--fov", type=float, default=None,
                    help="horizontal FOV in degrees, as for metal-gauss-render")
    ap.add_argument("--max-resolution", type=int, default=1024,
                    help="long side of a settled frame, in pixels")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to serve on; 0.0.0.0 serves the scene to the network")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)

    viser = import_viser()
    if not torch.backends.mps.is_available():
        raise SystemExit("metal-gauss-view renders on Metal and needs MPS.")

    sp = load_ply(a.ply, device="mps")
    print(f"{len(sp):,} splats, SH degree {sp.sh_degree}", file=sys.stderr)
    means = sp.means.detach().cpu()
    _, fov_h, eye, target = frame_cloud(means, "auto", a.up, a.convention, a.fov)
    batch = SplatBatch.from_splats(sp)
    LiveView(viser, lambda: batch, eye=eye, target=target, up=a.up,
             background=(1.0, 1.0, 1.0) if a.background == "white" else (0.0, 0.0, 0.0),
             far=scene_far(means, eye, target), host=a.host, port=a.port,
             max_resolution=a.max_resolution, fov_h=fov_h,
             world_flip=_GL2CV if a.convention == "opengl" else None).serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
