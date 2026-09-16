"""Camera keyframes and the path that runs through them.

The viewer's flythrough is decided here, before anything is rendered: a list of
camera poses is turned into one pose per frame. Pure arithmetic, no viser and
no GPU, so the path can be checked against hand-derived values rather than by
watching the video afterwards.

Positions follow a uniform Catmull-Rom spline, which passes through every
keyframe rather than merely near it -- a camera that misses the view you placed
it at is worse than useless. Rotations are interpolated as quaternions and
normalised, with each successive one flipped onto the shorter arc first: q and
-q are the same orientation, and lerping between them as written swings the
camera almost the whole way round.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from metal_gauss.render_path import camera_path

PRESETS = ("orbit", "wiggle")


@dataclasses.dataclass(frozen=True)
class Keyframe:
    """One camera pose: viser's own (position, wxyz, vertical FOV in radians)."""
    position: tuple[float, float, float]
    wxyz: tuple[float, float, float, float]
    fov: float


def _catmull_rom(p0, p1, p2, p3, t: float) -> tuple[float, float, float]:
    t2, t3 = t * t, t * t * t
    return tuple(
        0.5 * (2.0 * b + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
               + (-a + 3.0 * b - 3.0 * c + d) * t3)
        for a, b, c, d in zip(p0, p1, p2, p3))


def _shorter_arc(previous, q):
    """`q`, negated if that is the short way round from `previous`."""
    return tuple(-c for c in q) if sum(a * b for a, b in zip(previous, q)) < 0.0 else q


def _normalise(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q) if n > 1e-12 else (1.0, 0.0, 0.0, 0.0)


def interpolate(keys: list[Keyframe], frames: int, loop: bool = True) -> list[Keyframe]:
    """`frames` poses running through `keys`.

    A looped path wraps, so with one frame per key it is the keys themselves and
    the step from the last frame back to the first is an ordinary one. An open
    path starts on the first key and ends on the last, with its end tangents
    clamped.
    """
    if not keys:
        raise ValueError("a camera path needs at least one keyframe")
    if frames < 1:
        raise ValueError("a camera path needs at least one frame")
    if len(keys) == 1:
        return [keys[0]] * frames

    # Quaternions are put on a common hemisphere once, in key order, so the
    # choice does not depend on which segment is being drawn.
    turns = [keys[0].wxyz]
    for k in keys[1:]:
        turns.append(_shorter_arc(turns[-1], k.wxyz))
    if loop:
        turns.append(_shorter_arc(turns[-1], keys[0].wxyz))

    segments = len(keys) if loop else len(keys) - 1
    out = []
    for i in range(frames):
        u = (i / frames if loop else (i / (frames - 1) if frames > 1 else 0.0)) * segments
        s = min(int(u), segments - 1)
        t = u - s

        def at(j: int) -> int:
            return j % len(keys) if loop else max(0, min(len(keys) - 1, j))

        position = _catmull_rom(keys[at(s - 1)].position, keys[at(s)].position,
                                keys[at(s + 1)].position, keys[at(s + 2)].position, t)
        a, b = turns[s], turns[s + 1] if s + 1 < len(turns) else turns[0]
        wxyz = _normalise(tuple(x + (y - x) * t for x, y in zip(a, b)))
        fov = keys[at(s)].fov + (keys[at(s + 1)].fov - keys[at(s)].fov) * t
        out.append(Keyframe(position, wxyz, fov))
    return out


def preset(kind: str, eye, target, frames: int, sweep_deg: float, fov: float,
           up: str = "-y") -> list[Keyframe]:
    """Keyframes for the orbit and wiggle paths `metal-gauss-render` draws.

    Built from `camera_path` rather than rewritten, so a video made in the
    viewer and one made from the command line move the same way.
    """
    if kind not in PRESETS:
        raise ValueError(f"unknown path {kind!r}; expected one of {PRESETS}")
    from metal_gauss.viewer import matrix_to_quat

    out = []
    for vm in camera_path(eye, target, frames, sweep_deg, kind, up=up):
        R = vm[:3, :3].T
        centre = -R @ vm[:3, 3]
        out.append(Keyframe(tuple(float(v) for v in centre), matrix_to_quat(R), fov))
    return out
