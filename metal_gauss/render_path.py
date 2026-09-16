"""Render an existing .ply along a camera path generated here.

Training writes a .ply and that was the end of the road. Turning one back into
pixels needed a dataset to borrow cameras from: `bench/compare/score_ply.py`
reads `transforms_test.json`, `render_scene_reel.py` sweeps the official test
orbit. A .ply on its own -- a checkpoint from this trainer, a scene trained
elsewhere, a download, anything a feedforward model predicted -- carries no
cameras at all, so there is nothing to borrow and the camera has to be
synthesised.

So this module synthesises one: a still, a wiggle or an orbit, framed either on
the input view or on the cloud's bounding box, at a chosen resolution and field
of view, through a pinhole or a thin lens.

Framing on the input view is what a monocular prediction needs. Such a
predictor works in the input photograph's camera frame, which means the
identity world-to-camera matrix reproduces the original shot and the path can
move around it. The pivot comes from splats near the optical axis rather than
from all of them: a monocular prediction leaves a wall of background splats
behind the subject, that wall outnumbers the subject, and a pivot taken over
everything therefore lands behind it, turning a small wobble into a swing.

Keep that sweep small. A model that only ever saw one photograph starts showing
surfaces it had no evidence for past roughly eight degrees. A scene trained
from many views carries no such limit and orbits as far as it is asked to.

Named `render_path` and not `render` because `metal_gauss/__init__.py`
re-exports the rasteriser as `metal_gauss.render`; a sibling module of that
name shadows it and breaks every `from metal_gauss import render`.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import torch

from metal_gauss.api import render as _render
from metal_gauss.io import Splats, load_ply

# OpenGL camera-to-world (+Y up, -Z forward) to OpenCV (+Y down, +Z forward).
# Same flip `bench/compare/score_ply.py` applies to Blender cameras.
_GL2CV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

# The world's up axis, spelled the way viser spells it. "-y" is the OpenCV world
# every path here was first written for; a scene trained with --blender keeps
# Blender's world, which is "+z". Stored as the DOWN vector `look_at` builds from,
# written out rather than negated so the default stays bit-identical (-0.0).
_DOWN = {"-y": (0.0, 1.0, 0.0), "+y": (0.0, -1.0, 0.0),
         "+z": (0.0, 0.0, -1.0), "-z": (0.0, 0.0, 1.0)}
UP_AXES = tuple(_DOWN)


def _down(up: str) -> torch.Tensor:
    if up not in _DOWN:
        raise ValueError(f"unknown up axis {up!r}; expected one of {UP_AXES}")
    return torch.tensor(_DOWN[up])


# --------------------------------------------------------------- camera

def _rot_x(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Yaw turns about the DOWN vector, which for "-y" is exactly the _rot_y the
# paths were built on, so every up axis swings the same way on screen.
_YAW = {"-y": _rot_y, "+y": lambda a: _rot_y(-a),
        "+z": lambda a: _rot_z(-a), "-z": _rot_z}


def world_to_camera(R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """(3,3) camera orientation and (3,) camera centre -> (4,4) world->camera."""
    vm = torch.eye(4)
    vm[:3, :3] = R.T
    vm[:3, 3] = -R.T @ C
    return vm


def pivot_depth(means: torch.Tensor, fov_deg: float | None = None,
                cone_frac: float = 0.4, quantile: float = 0.5,
                min_splats: int = 256) -> float:
    """Depth of whatever the camera is pointed at, not of the scene as a whole.

    Taking the median over ALL splats is wrong, and wrong in a way that only
    shows up once you render it. A monocular portrait prediction is strongly
    BIMODAL in depth: the face sits at one distance and the wall behind it at
    another, and the wall carries more splats because it fills more of the
    frame. On the portrait this was written against, the face is at z~2.0, the
    background at z~9.5-12, and the median over everything lands at 9.5, deep
    in the wall. Orbiting about that point swung the head clean out of frame at
    four degrees.

    So the pivot comes from splats near the optical axis instead, which is a
    direct answer to "what is this camera looking at" and is unmoved by a
    background that merely outnumbers the subject. Within 2% of the axis that
    same portrait reads 2.06.
    """
    z = means[:, 2]
    tan_half = math.tan(0.5 * math.radians(fov_deg)) if fov_deg else 0.125
    r = (means[:, 0] ** 2 + means[:, 1] ** 2).sqrt() / z.clamp_min(1e-6)
    near_axis = z[r < cone_frac * tan_half]
    # Fall back to the whole cloud rather than an empty selection: a cloud with
    # nothing on the axis is not a portrait and has no subject to find.
    if near_axis.numel() < min_splats:
        near_axis = z
    return float(near_axis.quantile(quantile))


def look_at(eye: torch.Tensor, target: torch.Tensor, up: str = "-y") -> torch.Tensor:
    """Camera-to-world rotation looking from `eye` at `target`, OpenCV axes.

    +Z forward and +Y DOWN, so the basis is built from the world's down vector,
    which `up` names. A helper written for the +Y-up convention rolls the camera
    180 degrees here; a Z-up scene framed as if Y were vertical is seen from
    underneath.
    """
    down = _down(up)
    # Looking straight along the vertical, fall back to the axis the framing
    # itself looks along.
    fallback = (0.0, 0.0, 1.0) if up in ("-y", "+y") else (0.0, 1.0, 0.0)
    return _aim(eye, target, down, torch.tensor(fallback))


def _aim(eye, target, down: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    z = target - eye
    n = torch.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("camera and target coincide")
    z = z / n
    if torch.linalg.norm(torch.linalg.cross(down, z)) < 1e-6:
        down = fallback
    x = torch.linalg.cross(down, z)
    x = x / torch.linalg.norm(x)
    y = torch.linalg.cross(z, x)
    return torch.stack([x, y, z], dim=1)


def camera_path(eye, target, frames: int, sweep_deg: float,
                path: str = "wiggle", pitch_ratio: float = 0.5,
                up: str = "-y") -> list[torch.Tensor]:
    """World-to-camera matrices swinging the camera about `target`.

    Both paths are built on sin(t) over a full period, so frame 0 is exactly the
    starting view and the last frame joins back onto the first. That matters
    twice over: frame 0 is what you check the .ply's convention against, and the
    result has to loop cleanly when it plays on hover.

    `wiggle` adds a sin(2t) pitch, a figure-eight that reads as a head shifting
    rather than a turntable. The yaw turns about the vertical `up` names; the
    pitch stays about world X, which is horizontal for every framing here.
    """
    if path not in ("orbit", "wiggle"):
        raise ValueError(f"unknown path {path!r}; expected 'orbit' or 'wiggle'")
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    R0 = look_at(eye, target, up)
    yaw_about = _YAW[up]
    sweep = math.radians(sweep_deg)
    out = []
    for i in range(frames):
        t = 2.0 * math.pi * i / frames
        yaw = sweep * math.sin(t)
        pitch = sweep * pitch_ratio * math.sin(2.0 * t) if path == "wiggle" else 0.0
        R = yaw_about(yaw) @ _rot_x(pitch)
        out.append(world_to_camera(R @ R0, target + R @ (eye - target)))
    return out


GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def aperture_views(eye, target, radius: float, samples: int, up: str = "-y",
                   R0: torch.Tensor | None = None) -> list[torch.Tensor]:
    """World-to-camera matrices for one thin lens, all focused on one plane.

    A thin lens is many pinhole views spread over the lens area, averaged, with
    every view aimed at the same focal plane. Points ON that plane project to
    the same pixel from every sample and stay sharp; points off it disperse by
    an amount proportional to their distance from it, which is the blur. So
    defocus needs no new rasteriser, only more cameras.

    Positions follow a Fibonacci disc. Random sampling leaves visible clumps in
    the bokeh at sample counts anyone can afford, and a square grid leaves a
    lattice; measured on a portrait, 32 samples still showed structure at full
    blur and 96 did not. The sqrt gives equal-area spacing, without which the
    samples bunch toward the centre and the blur keeps a bright core.

    radius 0 returns exactly one view, which is the pinhole. That is deliberate:
    it makes the aperture path a strict superset of the ordinary one.

    Each sample is AIMED at the focal point rather than shifted. A shift lens
    holds the image planes parallel and moves the principal point instead,
    which registers the whole focal plane and not just the point on the axis;
    aiming leaves a second-order error in radius/focus_depth. Measured against
    a shift-lens accumulation on a portrait at 512px, 32 samples, focus 2.13:
    at radius 0.03 the two differ by at most 10.5 levels out of 255 at a single
    pixel and 0.10 on average, and at radius 0.15 by 23.9 and 0.65. So it does
    not matter at ordinary apertures and does grow with the radius.

    `R0` is the camera-to-world rotation of the pinhole being defocused, for a
    camera that did not come from `look_at` -- the viewer's, which orbits
    freely. Rebuilding each sample upright from `up` would roll such a camera;
    given R0 the samples keep its roll, and radius 0 returns it exactly.
    `target` is then the focal point on its axis.
    """
    if samples < 1:
        raise ValueError("need at least one aperture sample")
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    own_rotation = R0 is not None
    if R0 is None:
        R0 = look_at(eye, target, up)
    if radius <= 0.0:
        return [world_to_camera(R0, eye)]

    right, down = R0[:, 0], R0[:, 1]    # lens plane, perpendicular to the axis
    out = []
    for i in range(samples):
        r = radius * math.sqrt((i + 0.5) / samples)
        a = i * GOLDEN_ANGLE
        e = eye + right * (r * math.cos(a)) + down * (r * math.sin(a))
        R = _aim(e, target, down, R0[:, 2]) if own_rotation else look_at(e, target, up)
        out.append(world_to_camera(R, e))
    return out


def bbox_framing(means: torch.Tensor, fov_deg: float, margin: float = 1.25,
                 quantile: float = 0.98, up: str = "-y") -> tuple[torch.Tensor, torch.Tensor]:
    """(eye, target) for a .ply that did not come from a monocular predictor.

    A trained scene sits around its own origin with no input camera to anchor
    to, so the camera has to be placed rather than assumed. Robust percentiles
    again: a few stray splats a long way out would otherwise set the framing.

    The camera sits level with the target, back along -Z when Y is vertical and
    along -Y when Z is, which is Blender's front view. Always backing off along
    -Z put a Z-up scene's camera underneath it, looking up.
    """
    _down(up)
    lo = means.quantile(1.0 - quantile, dim=0)
    hi = means.quantile(quantile, dim=0)
    target = 0.5 * (lo + hi)
    radius = float(torch.linalg.norm(hi - lo)) * 0.5
    dist = margin * radius / max(math.tan(0.5 * math.radians(fov_deg)), 1e-6)
    back = [0.0, 0.0, dist] if up in ("-y", "+y") else [0.0, dist, 0.0]
    return target - torch.tensor(back), target


def intrinsics(W: int, H: int, fov_deg: float) -> torch.Tensor:
    f = 0.5 * W / math.tan(0.5 * math.radians(fov_deg))
    return torch.tensor([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])


def in_front_fraction(means: torch.Tensor, eps: float = 1e-3) -> float:
    """Fraction of splats ahead of the origin along +Z.

    This is what separates a monocular prediction from a trained scene, and
    the median depth is not: a trained scene centred on its own origin can
    easily have a positive median simply from how it is oriented, which made an
    earlier median test anchor lego to a camera sitting inside it.
    """
    return float((means[:, 2] > eps).float().mean())


def framing_fov(means: torch.Tensor, margin: float = 1.08, quantile: float = 0.98,
                max_fov: float = 120.0) -> float:
    """Horizontal FOV that happens to contain the cloud. NOT a focal length.

    This is a bounding operation on a point cloud: it never looks at image
    content and it recovers nothing about the camera that took the photograph.
    Where a real focal length is known -- and for a monocular prediction it IS
    known, see `fov_from_photo` -- use that instead. Rendering a SHARP
    prediction at a FOV it was not predicted under gives the right geometry in
    the wrong crop, which quietly breaks the one guarantee input-anchoring
    exists for: that frame 0 reproduces the original shot.

    Percentiles rather than extremes, because a monocular prediction throws a
    few splats a long way out and fitting to those zooms the subject to
    nothing. Splats close to the camera plane are dropped outright: x/z runs
    away there and drove the FOV to 180 degrees on the first scene this was
    pointed at. Note the near cutoff is derived from the same cloud it filters,
    so a large enough near population lowers its own threshold; the clamp is
    load-bearing, not belt-and-braces.
    """
    depth = means[:, 2]
    near = max(float(depth.quantile(0.5)) * 0.2, 1e-3)
    means = means[depth > near]
    if means.numel() == 0:
        return max_fov
    z = means[:, 2].clamp_min(1e-6)
    tan = torch.maximum((means[:, 0] / z).abs(), (means[:, 1] / z).abs())
    fov = 2.0 * math.degrees(math.atan(float(tan.quantile(quantile)) * margin))
    return min(fov, max_fov)




def fov_from_focal_35mm(f_35mm: float, width: int, height: int) -> float:
    """Horizontal FOV from a 35mm-equivalent focal length.

    SHARP's own conversion, from `sharp/utils/io.py`:
        f_px = f_35mm * diag(W, H) / diag(36, 24)
    Reproduced rather than approximated, because the point is to match what the
    predictor assumed, not to be independently correct about the lens.
    """
    f_px = f_35mm * math.sqrt(width ** 2 + height ** 2) / math.sqrt(36 ** 2 + 24 ** 2)
    return 2.0 * math.degrees(math.atan(width / (2.0 * f_px)))


def fov_from_photo(path: str | Path) -> tuple[float, float]:
    """(fov_deg, f_35mm) read from a photograph's EXIF, the way SHARP reads it.

    A monocular predictor works under the input camera's intrinsics, so those
    are the intrinsics the prediction has to be rendered back under. SHARP takes
    FocalLengthIn35mmFilm, falls back to FocalLength (scaled by 8.4 below 10mm,
    its own crude guess at a non-35mm figure), and defaults to 30mm when the
    file carries nothing.
    """
    try:
        from PIL import ExifTags, Image, TiffTags
    except ImportError as e:
        raise SystemExit(
            "--like-photo needs Pillow to read EXIF. Install it, or pass --fov "
            "directly. Refusing to guess the focal length the predictor used."
        ) from e

    im = Image.open(path)
    tags = {ExifTags.TAGS[k]: v for k, v in im.getexif().get_ifd(0x8769).items()
            if k in ExifTags.TAGS}
    tags.update({TiffTags.TAGS_V2[k].name: v for k, v in im.getexif().items()
                 if k in TiffTags.TAGS_V2})

    f35 = tags.get("FocalLengthIn35mmFilm", tags.get("FocalLenIn35mmFilm"))
    if f35 is None or f35 < 1:
        f35 = tags.get("FocalLength")
        if f35 is None:
            f35 = 30.0
        elif f35 < 10.0:
            f35 = f35 * 8.4
    W, H = im.size
    return fov_from_focal_35mm(float(f35), W, H), float(f35)


def frame_cloud(means: torch.Tensor, frame: str = "auto", up: str = "-y",
                convention: str = "opencv", fov: float | None = None,
                like_photo: str | None = None, depth: float | None = None):
    """(frame_mode, fov, eye, target): where the camera goes, in OpenCV world axes.

    With `convention="opengl"` the renderer applies _GL2CV to the world, so the
    cloud is flipped the same way before any of it is measured. Framing the
    file's own coordinates aimed the camera at a mirror image of the scene: a
    cloud centred at y=1, z=5 rendered with none of it in front of the camera,
    and an OpenGL monocular prediction was never recognised as one.
    """
    if convention == "opengl":
        means = means * torch.tensor([1.0, -1.0, -1.0])

    frame_mode = frame
    if frame_mode == "auto":
        ahead = in_front_fraction(means)
        frame_mode = "input" if ahead > 0.99 else "bbox"
        print(f"{100 * ahead:.1f}% of splats in front of the origin "
              f"-> --frame {frame_mode}", file=sys.stderr)

    if frame_mode == "input":
        if up != "-y":
            raise SystemExit(
                f"--up {up} does not apply to input framing: the input camera is "
                "OpenCV, so its up is -y. Use --frame bbox for a trained scene.")
        if fov is None:
            if like_photo:
                fov, f35 = fov_from_photo(like_photo)
                print(f"{like_photo}: {f35:g}mm (35mm equiv) -> {fov:.2f} deg",
                      file=sys.stderr)
            else:
                fov = framing_fov(means)
                print("WARNING: no focal length given, so the FOV is fitted to the "
                      "cloud. The geometry is right but the crop is not the one the "
                      "prediction was made under, so frame 0 will NOT reproduce the "
                      "source photograph. Pass --like-photo or --fov.", file=sys.stderr)
        # after the FOV, because the cone that finds the subject is sized by it
        if depth is None:
            depth = pivot_depth(means, fov)
        if not math.isfinite(depth) or depth <= 1e-3:
            raise SystemExit(
                f"subject depth is {depth:.4g}, so the cloud is not in front of "
                "the input camera. This .ply did not come from a monocular "
                "predictor; use --frame bbox.")
        eye, target = torch.zeros(3), torch.tensor([0.0, 0.0, float(depth)])
    else:
        fov = fov if fov is not None else 45.0
        eye, target = bbox_framing(means, fov, up=up)
    return frame_mode, fov, eye, target

# --------------------------------------------------------------- rendering

def render_frames(sp: Splats, views: list[torch.Tensor], K: torch.Tensor,
                  W: int, H: int, background=(1.0, 1.0, 1.0),
                  backend: str = "metal"):
    """Yield (H,W,3) float frames, one per view.

    `sh_degree` comes from the file. Hardcoding 3 (as score_ply does, where
    every input is a degree-3 trained model) would ask `eval_sh` for bases a
    feedforward predictor never wrote.
    """
    for vm in views:
        rgb, _, _ = _render(
            sp.means, sp.quats, sp.scales, sp.opacities, sp.sh,
            K, vm, W, H, sh_degree=sp.sh_degree, backend=backend,
            background=background,
        )
        yield rgb.detach().clamp(0.0, 1.0)


def lens_views(vm: torch.Tensor, focus: float, radius: float, samples: int,
               up: str = "-y", world_flip: torch.Tensor | None = None) -> list[torch.Tensor]:
    """The aperture's views around the pinhole view `vm`, focused `focus` along its axis.

    Each path view becomes its own little lens: recover that view's centre and
    axis, then spread the aperture around it. `world_flip` is the convention
    flip `vm` already carries. The lens is built in the OpenCV world and flipped
    afterwards; rebuilt from the flipped matrix, every sample was aimed with the
    other world's up and the defocused image came out upside down.
    """
    if world_flip is not None:
        vm = vm @ world_flip                 # a diagonal of +-1 is its own inverse
    c = -vm[:3, :3].T @ vm[:3, 3]
    t = c + vm[:3, :3][2] * focus
    views = aperture_views(c, t, radius, samples, up=up)
    return views if world_flip is None else [v @ world_flip for v in views]


def render_defocused(sp: Splats, views: list[torch.Tensor], K: torch.Tensor,
                     W: int, H: int, background=(1.0, 1.0, 1.0),
                     backend: str = "metal"):
    """One defocused frame: the mean of the aperture's views."""
    acc = None
    for frame in render_frames(sp, views, K, W, H, background=background, backend=backend):
        acc = frame if acc is None else acc + frame
    return acc / len(views)


class Mp4Writer:
    """Raw RGB streamed into ffmpeg, a frame at a time.

    An object rather than a generator sink because the viewer produces frames
    slowly -- one per pump, while training carries on between them -- and
    cannot hold the whole path open inside a generator.

    No intermediate files and no new dependency: the package needs only torch,
    numpy, plyfile and ninja, and this keeps it that way. Failures are ordinary
    exceptions, so a server can show them and carry on; `metal-gauss-render`
    turns them back into a clean exit.
    """

    def __init__(self, out: Path, W: int, H: int, fps: int,
                 tail: list[str] | None = None) -> None:
        self.out = Path(out)
        # ffmpeg exits immediately on a missing folder, and the first write then
        # fails with a bare broken pipe.
        self.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-framerate", str(fps), "-i", "-",
               *(tail if tail is not None else
                 ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                  "-movflags", "+faststart"]), str(self.out)]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg not found on PATH; needed to write the output.") from e

    def write(self, frame) -> None:
        """One (H,W,3) frame: a float tensor in [0,1], or uint8 already."""
        if torch.is_tensor(frame):
            if frame.dtype != torch.uint8:
                frame = (frame * 255.0).round().to(torch.uint8)
            frame = frame.cpu().numpy()
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            # ffmpeg is already gone; its own status says more than the pipe does.
            raise RuntimeError(
                f"ffmpeg stopped while writing {self.out} (exit {self._proc.wait()})") from e

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing the output.")

    def abort(self) -> None:
        """Give up and leave nothing behind: a partial mp4 is not a result."""
        self._proc.kill()
        self._proc.wait()
        self.out.unlink(missing_ok=True)


def _pipe_to_ffmpeg(frames, out: Path, W: int, H: int, tail: list[str],
                    fps: int) -> None:
    """Write every frame of `frames`, as the CLI does: fail means exit."""
    try:
        writer = Mp4Writer(out, W, H, fps, tail)
        for f in frames:
            writer.write(f)
        writer.close()
    except RuntimeError as e:
        raise SystemExit(str(e))


def write_mp4(frames, out: Path, W: int, H: int, fps: int) -> None:
    _pipe_to_ffmpeg(frames, out, W, H,
                    ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart"], fps)


def write_png(frame, out: Path, W: int, H: int) -> None:
    _pipe_to_ffmpeg([frame], out, W, H, ["-frames:v", "1"], 1)


# --------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="metal-gauss-render",
        description="Render a .ply along a generated camera path, on Metal.")
    ap.add_argument("ply")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", type=int, default=512, help="square output")
    ap.add_argument("--sweep-deg", type=float, default=5.0,
                    help="peak angle from the input view; small on purpose, see "
                         "the module docstring")
    ap.add_argument("--path", choices=("wiggle", "orbit"), default="wiggle")
    ap.add_argument("--pitch-ratio", type=float, default=0.5)
    ap.add_argument("--fov", type=float, default=None,
                    help="horizontal FOV in degrees. Overrides --like-photo.")
    ap.add_argument("--like-photo", default=None, metavar="IMAGE",
                    help="read the 35mm-equivalent focal length from this "
                         "photograph's EXIF so frame 0 reproduces it. The right "
                         "flag for a monocular prediction: the predictor worked "
                         "under that camera's intrinsics.")
    ap.add_argument("--depth", type=float, default=None,
                    help="pivot depth for --frame input; default is the median "
                         "depth of the splats near the optical axis")
    ap.add_argument("--frame", choices=("auto", "input", "bbox"), default="auto",
                    help="'input' anchors on the predicting camera, which is what "
                         "a monocular .ply wants; 'bbox' places the camera around "
                         "the cloud, which is what a trained scene wants. 'auto' "
                         "picks by whether the cloud sits in front of the origin.")
    ap.add_argument("--background", default="white", choices=("white", "black"))
    ap.add_argument("--convention", choices=("opencv", "opengl"), default="opencv",
                    help="frame the .ply is written in. If the first frame comes "
                         "out flipped, it is the other one.")
    ap.add_argument("--up", choices=UP_AXES, default="-y",
                    help="the scene's vertical axis, for --frame bbox. -y is the "
                         "OpenCV world; a scene trained with --blender is +z. "
                         "Negative axes need the = form: --up=-z.")
    ap.add_argument("--backend", choices=("metal", "torch_ref"), default="metal")
    ap.add_argument("--aperture", type=float, default=0.0,
                    help="lens radius in world units. 0 is a pinhole, which is "
                         "the default and leaves rendering unchanged.")
    ap.add_argument("--aperture-samples", type=int, default=96,
                    help="views averaged per frame. Below about 32 the sampling "
                         "disc shows as a lattice in the bokeh.")
    ap.add_argument("--focus", type=float, default=None,
                    help="focal plane depth; defaults to the pivot, i.e. the "
                         "subject the camera is already pointed at")
    ap.add_argument("--still", action="store_true",
                    help="write frame 0 only, as a .png, to check the convention "
                         "against the original photograph before rendering a video")
    a = ap.parse_args(argv)

    dev = "mps" if a.backend == "metal" else "cpu"
    if a.backend == "metal" and not torch.backends.mps.is_available():
        raise SystemExit("metal backend needs MPS; pass --backend torch_ref.")

    sp = load_ply(a.ply, device=dev)
    means_cpu = sp.means.detach().cpu()
    print(f"{len(sp)} splats, SH degree {sp.sh_degree}", file=sys.stderr)

    frame_mode, fov, eye, target = frame_cloud(
        means_cpu, a.frame, a.up, a.convention, a.fov, a.like_photo, a.depth)

    W = H = a.resolution
    K = intrinsics(W, H, fov)
    views = camera_path(eye, target, 1 if a.still else a.frames,
                        a.sweep_deg, a.path, a.pitch_ratio, up=a.up)
    flip = _GL2CV if a.convention == "opengl" else None
    if flip is not None:
        views = [vm @ flip for vm in views]

    print(f"framing {frame_mode}, target {[round(float(v), 3) for v in target]}, "
          f"fov {fov:.1f} deg, {len(views)} frames, sweep +/-{a.sweep_deg} deg",
          file=sys.stderr)

    bg = (1.0, 1.0, 1.0) if a.background == "white" else (0.0, 0.0, 0.0)
    if a.aperture > 0.0:
        focus = a.focus if a.focus is not None else float(
            torch.linalg.norm(torch.as_tensor(target, dtype=torch.float32)
                              - torch.as_tensor(eye, dtype=torch.float32)))
        print(f"aperture {a.aperture} over {a.aperture_samples} samples, "
              f"focus at {focus:.3f}", file=sys.stderr)

        def frames_gen():
            for vm in views:
                lens = lens_views(vm, focus, a.aperture, a.aperture_samples,
                                  up=a.up, world_flip=flip)
                yield render_defocused(sp, lens, K, W, H,
                                       background=bg, backend=a.backend)
        frames = frames_gen()
    else:
        frames = render_frames(sp, views, K, W, H, background=bg, backend=a.backend)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        out = out.with_suffix(".png")
        write_png(next(iter(frames)), out, W, H)
    else:
        write_mp4(frames, out, W, H, a.fps)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
