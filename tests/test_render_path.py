"""The synthesised camera path, checked without touching the rasteriser.

`render_path` is the only place in the repo that invents cameras instead of
reading them from a dataset, so nothing downstream can catch it being wrong: a
bad matrix here renders cleanly and just shows the wrong thing. The two
properties everything else rests on are that frame 0 reproduces the input view
EXACTLY -- that is how you check a .ply's coordinate convention against the
original photograph -- and that every matrix stays a rigid transform, since a
scaled or reflected "rotation" turns into a covariance the rasteriser will
happily draw.

The framing helpers get regression tests rather than value tests, because both
were written in response to a specific wrong answer: `framing_fov` fitted 180
degrees off splats sitting on the camera plane, and `--frame auto` used to
switch on median depth, which does not separate a monocular prediction from a
trained scene at all.

Rasteriser behaviour, MPS, and real .ply files are covered elsewhere; these
tests are pure geometry and run anywhere.
"""

from __future__ import annotations

import math

import pytest
import torch

from metal_gauss.render_path import (
    _GL2CV,
    aperture_views,
    bbox_framing,
    fov_from_focal_35mm,
    camera_path,
    frame_cloud,
    framing_fov,
    in_front_fraction,
    intrinsics,
    lens_views,
    look_at,
    pivot_depth,
    world_to_camera,
)

PATHS = ["orbit", "wiggle"]
# Divisible by 8, so the sweep extremes (t = pi/2 for yaw, t = pi/4 for pitch)
# land on actual frames and can be asserted exactly rather than approximately.
FRAMES = 64


def centre(vm):
    """Camera centre in world coordinates, recovered from a world->camera matrix."""
    return -vm[:3, :3].T @ vm[:3, 3]


def c2w(vm):
    """Camera-to-world rotation, i.e. the R that `world_to_camera` was given."""
    return vm[:3, :3].T


def yaw_pitch(vm):
    """Yaw/pitch in degrees from a view built as rot_y(yaw) @ rot_x(pitch).

    That product's third column is (sin y cos p, -sin p, cos y cos p), so the
    pitch reads straight off and the yaw comes from the ratio, where cos p
    cancels. Valid for any sweep small enough to be worth rendering.
    """
    R = c2w(vm)
    yaw = math.degrees(math.atan2(float(R[0, 2]), float(R[2, 2])))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(R[1, 2])))))
    return yaw, pitch


def scene(n=2000, seed=0, z_lo=1.0, z_hi=3.0, spread=0.3):
    """A deterministic cloud sitting in front of the origin, as a prediction does."""
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=g) * (z_hi - z_lo) + z_lo
    xy = torch.randn(n, 2, generator=g) * spread
    return torch.stack([xy[:, 0], xy[:, 1], z], dim=1)


def cube_cloud(k=10):
    """A k^3 grid filling [-1,1]^3, centred on the origin like a trained scene.

    Dense enough that the 0.98 percentile still lands on the box corner, so the
    framing this produces can be predicted exactly; a handful of points would
    make the percentiles interpolate and the expected numbers arbitrary.
    """
    lin = torch.linspace(-1.0, 1.0, k)
    return torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1).reshape(-1, 3)


# ----------------------------------------------------------------- the anchor

def test_frame_zero_is_exactly_the_identity_view():
    """Eye at the origin looking down +Z must give the identity, to the bit.

    This is the whole reason the path is built on sin(t) rather than a ramp:
    frame 0 renders the monocular predictor's own input view, so it can be held
    against the original photograph to tell a good .ply from one in the wrong
    convention. Approximately-identity would not do -- any drift shows up as
    the check disagreeing with the photograph for reasons that are the path's
    fault, not the file's. Asserted with torch.equal on purpose.
    """
    views = camera_path(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]), FRAMES, 5.0)
    assert torch.equal(views[0], torch.eye(4))


def test_still_render_is_one_identity_frame():
    """`--still` asks for frames=1; it must be the anchor view, not a mean pose."""
    views = camera_path(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]), 1, 8.0)
    assert len(views) == 1
    assert torch.equal(views[0], torch.eye(4))


@pytest.mark.parametrize("path", PATHS)
def test_frame_zero_is_the_starting_view_for_any_eye(path):
    """Off the +Z axis the anchor is no longer the identity, but it is still the
    unmoved camera -- the sweep starts at zero, it does not start half a step in."""
    eye = torch.tensor([0.3, -0.2, 0.0])
    target = torch.tensor([0.1, 0.4, 2.5])
    views = camera_path(eye, target, FRAMES, 8.0, path=path)
    assert torch.allclose(views[0], world_to_camera(look_at(eye, target), eye), atol=1e-6)


# ------------------------------------------------------------- rigid transforms

@pytest.mark.parametrize("path", PATHS)
def test_every_view_is_a_rigid_transform(path):
    """R R^T = I and det R = +1 for every frame.

    A rotation that has picked up a scale factor still renders -- it silently
    resizes the scene -- and a reflection (det -1) mirrors it. Neither raises,
    so it has to be asserted.
    """
    for i, vm in enumerate(camera_path(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]),
                                       FRAMES, 8.0, path=path)):
        R = vm[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), f"frame {i} not orthonormal"
        assert float(torch.linalg.det(R)) == pytest.approx(1.0, abs=1e-5), \
            f"frame {i} is a reflection"
        assert torch.equal(vm[3], torch.tensor([0.0, 0.0, 0.0, 1.0])), \
            f"frame {i} is not affine"


@pytest.mark.parametrize("path", PATHS)
def test_camera_holds_a_constant_distance_from_the_target(path):
    """The camera swings ABOUT the target; it must not dolly in and out.

    Both paths rotate the eye offset by a rotation matrix, so the radius is
    conserved by construction -- which is exactly the kind of property that
    survives until someone adds a translation term to the sweep.
    """
    eye = torch.tensor([0.3, -0.2, 0.0])
    target = torch.tensor([0.1, 0.4, 2.5])
    r0 = float(torch.linalg.norm(eye - target))
    radii = [float(torch.linalg.norm(centre(vm) - target))
             for vm in camera_path(eye, target, FRAMES, 8.0, path=path)]
    assert max(radii) - min(radii) < 1e-5 * r0
    assert radii[0] == pytest.approx(r0, rel=1e-5)


# ------------------------------------------------------------------- the loop

@pytest.mark.parametrize("path", PATHS)
def test_the_path_loops(path):
    """The result plays on hover, so the wrap from the last frame back to the
    first has to be an ordinary step, not a seam.

    Both paths are sin(t) over a full period, so the return is built in: the
    last frame sits one step short of the start, and there is no jump. The
    comparison for "on its way back" is against the EXTREME of the sweep at
    t = pi/2, not against the middle of the sequence: sin(pi) = sin(2pi) = 0,
    so frame FRAMES/2 is the starting view again. That midpoint crossing is
    asserted as well -- it is the half of the period where the sweep changes
    sides, and it is what makes the wiggle's sin(2t) pitch a figure-eight.
    """
    views = camera_path(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]),
                        FRAMES, 8.0, path=path)
    steps = [float((views[i + 1] - views[i]).norm()) for i in range(FRAMES - 1)]
    wrap = float((views[0] - views[-1]).norm())
    assert wrap <= 1.05 * max(steps), f"wrap {wrap:.4f} vs largest step {max(steps):.4f}"

    extreme = float((views[FRAMES // 4] - views[0]).norm())   # peak of the sweep
    assert wrap < 0.2 * extreme, "the last frame is not on its way back to the start"
    assert torch.allclose(views[FRAMES // 2], views[0], atol=1e-6)


def test_yaw_sweep_is_symmetric_and_peaks_at_the_requested_angle():
    """+/- sweep_deg, and no bias to one side.

    A path that swung further one way than the other would still look like a
    wiggle, and would still be showing more unseen geometry on one side than
    the caller asked for -- the sweep is small precisely because the predictor
    only ever saw one photograph.
    """
    sweep = 8.0
    views = camera_path(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]),
                        FRAMES, sweep, path="orbit")
    yaws = [yaw_pitch(vm)[0] for vm in views]

    assert max(yaws) == pytest.approx(sweep, abs=1e-4)
    assert min(yaws) == pytest.approx(-sweep, abs=1e-4)
    # Half a period apart, sin flips sign: yaw[i] must mirror yaw[i + N/2].
    for i in range(FRAMES // 2):
        assert yaws[i] == pytest.approx(-yaws[i + FRAMES // 2], abs=1e-4)


def test_orbit_has_no_pitch_and_wiggle_does():
    """`wiggle` adds the sin(2t) pitch that reads as a head shifting; `orbit`
    must be a pure turntable, with the camera staying in its own plane."""
    sweep, ratio = 8.0, 0.5
    kw = dict(frames=FRAMES, sweep_deg=sweep, pitch_ratio=ratio)
    eye, target = torch.zeros(3), torch.tensor([0.0, 0.0, 2.0])

    orbit = camera_path(eye, target, path="orbit", **kw)
    assert max(abs(yaw_pitch(vm)[1]) for vm in orbit) == pytest.approx(0.0, abs=1e-6)
    # No pitch means the eye never leaves the plane it started in.
    assert max(abs(float(centre(vm)[1])) for vm in orbit) < 1e-6

    wiggle = camera_path(eye, target, path="wiggle", **kw)
    pitches = [yaw_pitch(vm)[1] for vm in wiggle]
    assert max(pitches) == pytest.approx(sweep * ratio, abs=1e-4)
    assert min(pitches) == pytest.approx(-sweep * ratio, abs=1e-4)
    # Same yaw sweep as the orbit: pitch_ratio scales the pitch, nothing else.
    assert max(yaw_pitch(vm)[0] for vm in wiggle) == pytest.approx(sweep, abs=1e-4)


# --------------------------------------------------------------- bad arguments

def test_unknown_path_is_rejected():
    """Silently falling back to one of the two would render a video that does not
    match what was asked for, and nothing about the output would say so."""
    with pytest.raises(ValueError, match="unknown path"):
        camera_path(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]), 4, 5.0, path="dolly")


def test_look_at_rejects_a_coincident_target():
    """A zero view direction has no rotation; the alternative is a NaN matrix."""
    with pytest.raises(ValueError, match="coincide"):
        look_at(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))


def test_look_at_survives_looking_along_the_down_axis():
    """Straight down +Y makes `down x z` degenerate; the fallback axis must still
    produce a real rotation rather than a normalised zero vector."""
    R = look_at(torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    assert float(torch.linalg.det(R)) == pytest.approx(1.0, abs=1e-6)
    # +Z of the camera is the view direction, in OpenCV axes.
    assert torch.allclose(R[:, 2], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_look_at_uses_opencv_axes():
    """+Z forward and +Y DOWN. A +Y-up helper dropped in here rolls the camera
    180 degrees, which renders an upside-down scene that otherwise looks fine."""
    R = look_at(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)
    # A point below the camera in the world must land in the LOWER half of the
    # image, i.e. at positive camera-y.
    vm = world_to_camera(R, torch.zeros(3))
    below = torch.tensor([0.0, 0.5, 2.0, 1.0])
    assert float((vm @ below)[1]) > 0


# --------------------------------------------------------------- world->camera

def test_world_to_camera_inverts_the_camera_pose():
    """The camera centre maps to the origin, and the target lands on +Z at the
    distance between them. Getting the sign of `-R^T C` wrong puts the scene
    behind the camera, where the whole render culls to background."""
    eye = torch.tensor([0.3, -0.2, 0.5])
    target = torch.tensor([0.1, 0.4, 2.5])
    vm = world_to_camera(look_at(eye, target), eye)

    at_eye = vm @ torch.cat([eye, torch.ones(1)])
    assert torch.allclose(at_eye, torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5)

    at_target = vm @ torch.cat([target, torch.ones(1)])
    assert float(at_target[2]) == pytest.approx(float(torch.linalg.norm(target - eye)),
                                                rel=1e-5)
    assert abs(float(at_target[0])) < 1e-5 and abs(float(at_target[1])) < 1e-5


# ------------------------------------------------------------------ intrinsics

@pytest.mark.parametrize("fov", [30.0, 60.0, 90.0])
def test_intrinsics_project_the_fov_edge_to_the_image_edge(fov):
    """Horizontal FOV is a promise about where the frustum edge lands.

    A point at exactly fov/2 off-axis must project to u = W. If it does not,
    `framing_fov`'s framing is meaningless: it fits an angle whose relationship to
    the image is unspecified.
    """
    W, H, z = 640, 480, 3.0
    K = intrinsics(W, H, fov)
    x = z * math.tan(math.radians(0.5 * fov))
    u = float(K[0, 0] * x / z + K[0, 2])
    assert u == pytest.approx(W, rel=1e-5)

    assert float(K[0, 2]) == W / 2 and float(K[1, 2]) == H / 2   # centred
    assert float(K[0, 1]) == 0.0                                 # no skew
    assert float(K[0, 0]) == float(K[1, 1])                      # square pixels
    assert float(K[2, 2]) == 1.0


# ---------------------------------------------------------------- fov fitting

def test_framing_fov_frames_the_cloud_it_is_given():
    """Every splat at |x/z| = 0.5 means a half-angle of atan(0.5), so with no
    margin the fitted FOV is 2*atan(0.5) and nothing else."""
    means = torch.tensor([[0.5, 0.0, 1.0], [-0.5, 0.0, 1.0],
                          [0.0, 0.5, 1.0], [0.0, -0.5, 1.0]] * 10)
    assert framing_fov(means, margin=1.0) == pytest.approx(
        2.0 * math.degrees(math.atan(0.5)), rel=1e-6)


def test_framing_fov_is_clamped_at_max_fov():
    """Regression: splats near z=0 drove the fit to 180 degrees.

    x/z runs away on the camera plane, and the whole subject then rendered as a
    speck in the middle of a fisheye. The near-plane filter and the clamp are
    two separate defences and the filter alone does not bound the answer: the
    splats here sit just PAST the near cut (0.2 * median depth) with x = 50, so
    they survive the filter and would fit 179 degrees on their own.
    """
    means = scene()
    means[:400, 2] = 0.45      # near lands at ~0.35 once these drag the median down
    means[:400, 0] = 50.0
    assert framing_fov(means) == pytest.approx(120.0)         # the default max
    assert framing_fov(means, max_fov=60.0) == pytest.approx(60.0)
    # The clamp is a backstop, not the normal answer.
    assert framing_fov(scene()) < 120.0


def test_framing_fov_with_nothing_in_front_returns_max():
    """A cloud entirely behind the camera has no angle to fit; returning max_fov
    keeps the caller going rather than dividing by an empty quantile."""
    means = scene()
    means[:, 2] *= -1.0
    assert framing_fov(means) == pytest.approx(120.0)


# ----------------------------------------------------------- framing selection

def test_in_front_fraction_separates_monocular_from_trained():
    """Regression: `--frame auto` used to switch on median depth, which cannot.

    A monocular prediction lives entirely in front of its input camera, so the
    identity view reproduces the photograph. A trained scene sits around its own
    origin and the identity view is inside it. The straddling cloud below has a
    POSITIVE median depth -- the old test called it monocular and anchored the
    camera inside lego -- while the fraction ahead separates the two cleanly.
    """
    ahead = scene(z_lo=1.0, z_hi=3.0)
    g = torch.Generator().manual_seed(1)
    straddle = scene(seed=2)
    straddle[:, 2] = torch.where(torch.rand(len(straddle), generator=g) < 0.7,
                                 straddle[:, 2], -straddle[:, 2])

    assert in_front_fraction(ahead) == 1.0
    assert in_front_fraction(straddle) < 0.99          # the threshold --frame auto uses
    # The discriminator this replaced says both are in front:
    assert float(ahead[:, 2].median()) > 0 and float(straddle[:, 2].median()) > 0

    # Splats exactly on the camera plane are not "in front" -- eps is a floor,
    # not a tolerance, and z=0 is a division by zero waiting to happen.
    assert in_front_fraction(torch.zeros(10, 3)) == 0.0


def test_pivot_depth_is_the_median_not_the_mean():
    """A monocular prediction leaves background splats far behind the subject.

    The mean gets dragged back into them, which puts the pivot behind the head
    and turns a small sweep into a swing. The tail here is 5% of the splats at
    100x the subject depth, which moves the mean by more than 4x and the median
    not at all.
    """
    means = scene(n=1000, z_lo=1.9, z_hi=2.1)
    means[:50, 2] = 200.0
    assert pivot_depth(means) == pytest.approx(2.0, abs=0.05)
    assert float(means[:, 2].mean()) > 4.0 * pivot_depth(means)


# --------------------------------------------------------------- bbox framing

def test_bbox_framing_puts_the_whole_cloud_in_frame():
    """The trained-scene case: place the camera so the cloud fills the image.

    Checked end to end through `look_at` and `intrinsics`, since "framed" is a
    statement about pixels, not about the distance in isolation: every splat
    must be in front of the camera AND inside the image.
    """
    cube = cube_cloud()
    W = H = 256
    fov = 45.0
    eye, target = bbox_framing(cube, fov)

    assert torch.allclose(target, torch.zeros(3), atol=1e-6)     # centre of the box
    assert float(eye[2]) < float(target[2])                      # camera behind, looking +Z
    assert torch.allclose(eye[:2], target[:2], atol=1e-6)        # no lateral offset

    vm = world_to_camera(look_at(eye, target), eye)
    cam = (torch.cat([cube, torch.ones(len(cube), 1)], dim=1) @ vm.T)[:, :3]
    assert (cam[:, 2] > 0).all(), "part of the cloud is behind the camera"
    uv = cam @ intrinsics(W, H, fov).T
    uv = uv[:, :2] / uv[:, 2:3]
    assert (uv >= 0).all() and (uv[:, 0] <= W).all() and (uv[:, 1] <= H).all(), \
        f"cloud leaves the frame: u/v range {uv.min(0).values.tolist()} .. " \
        f"{uv.max(0).values.tolist()}"


def test_bbox_framing_distance_tracks_fov_and_margin():
    """Wider lens, closer camera; more margin, further back.

    The distance is margin * radius / tan(fov/2), so both are monotone. A sign
    slip in either makes the framing worse the harder you ask it to try.
    """
    cube = cube_cloud()

    def dist(means, **kw):
        eye, target = bbox_framing(means, **kw)
        return float(torch.linalg.norm(target - eye))

    assert dist(cube, fov_deg=90.0) < dist(cube, fov_deg=45.0) < dist(cube, fov_deg=20.0)
    assert dist(cube, fov_deg=45.0, margin=1.0) < dist(cube, fov_deg=45.0, margin=1.5)

    # Robust percentiles: a few splats a long way out must not set the framing.
    # Note the 0.98 quantile only excludes the top 2%, so this holds for a real
    # .ply (100k+ splats) and would NOT hold for a handful of points.
    strays = torch.cat([cube, torch.full((5, 3), 500.0)])
    assert dist(strays, fov_deg=45.0) == pytest.approx(dist(cube, fov_deg=45.0), rel=1e-5)


# ----------------------------------------------------- focal length, not framing

def test_fov_from_focal_35mm_matches_sharps_own_conversion():
    """Reproduces SHARP's formula, because the point is to match the predictor.

    From `sharp/utils/io.py`: f_px = f_35mm * diag(W, H) / diag(36, 24). The
    prediction was made under those intrinsics, so rendering it back under any
    other FOV gives the right geometry in the wrong crop. Being independently
    correct about the lens is not the goal; agreeing with SHARP is.
    """
    f35, W, H = 135.0, 2160, 2160
    f_px = f35 * math.sqrt(W ** 2 + H ** 2) / math.sqrt(36 ** 2 + 24 ** 2)
    expected = 2.0 * math.degrees(math.atan(W / (2.0 * f_px)))
    assert fov_from_focal_35mm(f35, W, H) == pytest.approx(expected, rel=1e-12)
    # a 135mm portrait lens is narrow; this is the number the easter egg needs
    assert fov_from_focal_35mm(135.0, 2160, 2160) == pytest.approx(12.93, abs=0.01)
    # SHARP's fallback when a file carries no EXIF, four times wider
    assert fov_from_focal_35mm(30.0, 2160, 2160) == pytest.approx(54.03, abs=0.01)


@pytest.mark.parametrize("side", [256, 640, 2160, 4096])
def test_fov_from_focal_35mm_is_resolution_independent_when_square(side):
    """Resizing the photograph must not change the field of view it was shot at.

    The portrait on the website was downscaled 2160 -> 640 during unrelated
    work. If the FOV moved with the pixel count, frame 0 would silently stop
    matching whichever copy SHARP was actually run on.
    """
    assert fov_from_focal_35mm(135.0, side, side) == pytest.approx(12.93, abs=0.01)


def test_fov_from_focal_35mm_widens_as_the_lens_shortens():
    fovs = [fov_from_focal_35mm(f, 1000, 1000) for f in (24.0, 50.0, 135.0, 200.0)]
    assert fovs == sorted(fovs, reverse=True)


# ----------------------------------------------------------------- the lens

def test_zero_radius_is_exactly_the_pinhole():
    """The aperture path must be a strict superset of the ordinary one.

    If radius 0 returned a ring of samples at distance 0, or a slightly
    different matrix, then adding defocus support would silently perturb every
    existing render. It returns one view and that view is the pinhole.
    """
    vs = aperture_views([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 0.0, 64)
    assert len(vs) == 1
    assert torch.allclose(vs[0], torch.eye(4), atol=1e-6)


@pytest.mark.parametrize("radius", [0.01, 0.05, 0.2])
def test_samples_lie_inside_the_aperture_disc(radius):
    """Every sample is on the lens plane and within the lens."""
    eye, target = torch.zeros(3), torch.tensor([0.0, 0.0, 3.0])
    centres = torch.stack([-(v[:3, :3].T @ v[:3, 3])
                           for v in aperture_views(eye, target, radius, 128)])
    assert torch.all(centres[:, :2].norm(dim=1) <= radius + 1e-6)
    assert torch.allclose(centres[:, 2], torch.zeros(len(centres)), atol=1e-6)


def test_every_aperture_view_aims_at_the_focal_plane():
    """This is what makes it a lens rather than a shake.

    Translating the camera without re-aiming would move the whole image, and
    averaging that gives uniform motion blur with nothing in focus anywhere.
    Re-aiming every sample at the focal plane is what holds that plane sharp
    while everything else disperses.
    """
    target = torch.tensor([0.0, 0.0, 2.5])
    for vm in aperture_views(torch.zeros(3), target, 0.08, 64):
        centre = -(vm[:3, :3].T @ vm[:3, 3])
        forward = vm[:3, :3][2]                     # camera +Z in world terms
        want = target - centre
        want = want / want.norm()
        assert torch.allclose(forward, want, atol=1e-5)


def test_samples_are_spread_by_area_not_by_radius():
    """sqrt spacing, or the blur keeps a bright core.

    Spacing linearly in r puts far more samples per unit AREA near the centre,
    so the averaged result is brightest on the axis and the bokeh looks lit
    from within. Half the samples should fall inside r/sqrt(2), which is the
    radius that halves the disc's area.
    """
    n, radius = 512, 0.1
    centres = torch.stack([-(v[:3, :3].T @ v[:3, 3])
                           for v in aperture_views(torch.zeros(3),
                                                   torch.tensor([0.0, 0.0, 2.0]),
                                                   radius, n)])
    inside = (centres[:, :2].norm(dim=1) < radius / math.sqrt(2)).float().mean()
    assert abs(float(inside) - 0.5) < 0.05


def test_aperture_rejects_a_zero_sample_count():
    with pytest.raises(ValueError, match="at least one"):
        aperture_views(torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]), 0.05, 0)


def test_a_given_camera_rotation_is_kept_including_its_roll():
    """The viewer's camera orbits freely and is not built by `look_at`.

    Rebuilding each lens sample upright from the up axis would roll a camera
    that was not upright, so the defocused frame would not line up with the
    pinhole frame it replaces. Given R0, radius 0 is that camera exactly and
    every sample keeps its roll while still aiming at the focal point.
    """
    eye = torch.tensor([0.2, -0.1, -2.0])
    roll = math.radians(30.0)
    c, s = math.cos(roll), math.sin(roll)
    R0 = look_at(eye, torch.zeros(3)) @ torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    focal = eye + R0[:, 2] * 2.0

    pinhole = aperture_views(eye, focal, 0.0, 16, R0=R0)
    assert len(pinhole) == 1
    assert torch.allclose(pinhole[0], world_to_camera(R0, eye), atol=1e-6)

    for vm in aperture_views(eye, focal, 0.05, 16, R0=R0):
        down = vm[:3, :3][1]                        # camera +Y in world terms
        assert float(down @ R0[:, 1]) > 0.999, "sample lost the camera's roll"
        forward = vm[:3, :3][2]
        want = focal - centre(vm)
        assert torch.allclose(forward, want / want.norm(), atol=1e-5)


# ------------------------------------------------------------------- up axis

UP_VECTORS = {"-y": (0.0, -1.0, 0.0), "+y": (0.0, 1.0, 0.0),
              "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}


def level_eye(up, dist=2.0):
    """An eye level with the origin, back along the axis the framing uses."""
    return torch.tensor([0.0, 0.0, -dist]) if up in ("-y", "+y") else torch.tensor([0.0, -dist, 0.0])


def test_default_up_is_exactly_the_existing_behaviour():
    """Every render made before --up existed must come out bit-identical."""
    eye, target = torch.tensor([0.3, -0.2, -2.0]), torch.tensor([0.1, 0.2, 0.4])
    assert torch.equal(look_at(eye, target, up="-y"), look_at(eye, target))
    for path in PATHS:
        assert all(torch.equal(a, b) for a, b in zip(
            camera_path(eye, target, 8, 5.0, path, up="-y"),
            camera_path(eye, target, 8, 5.0, path)))
    assert all(torch.equal(a, b) for a, b in zip(
        aperture_views(eye, target, 0.05, 8, up="-y"), aperture_views(eye, target, 0.05, 8)))
    cube = cube_cloud()
    assert all(torch.equal(a, b) for a, b in zip(
        bbox_framing(cube, 45.0, up="-y"), bbox_framing(cube, 45.0)))


def test_z_up_look_at_is_blenders_front_view():
    """Z up, looking along +Y: image right is +X and image down is -Z."""
    R = look_at(torch.tensor([0.0, -2.0, 0.0]), torch.zeros(3), up="+z")
    assert torch.allclose(R, torch.tensor([[1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0],
                                           [0.0, -1.0, 0.0]]), atol=1e-6)


@pytest.mark.parametrize("up", list(UP_VECTORS))
def test_what_is_above_the_target_renders_in_the_upper_half(up):
    """The property #18 is about: the scene's up is the image's up."""
    eye = level_eye(up)
    vm = world_to_camera(look_at(eye, torch.zeros(3), up=up), eye)
    above = torch.tensor([*[0.5 * v for v in UP_VECTORS[up]], 1.0])
    assert float((vm @ above)[1]) < 0
    right = vm[:3, :3][0]
    assert abs(float(right @ torch.tensor(UP_VECTORS[up]))) < 1e-6, "camera is rolled"


@pytest.mark.parametrize("up", ["+z", "-z"])
def test_z_up_look_at_survives_looking_along_the_up_axis(up):
    R = look_at(torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]), up=up)
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    assert float(torch.linalg.det(R)) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(R[:, 2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_unknown_up_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown up axis"):
        look_at(torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]), up="z")


def test_z_up_bbox_framing_sits_level_in_front_of_the_cloud():
    """Back along -Y at the target's height, the same distance, whole cloud in frame."""
    cube = cube_cloud()
    W = H = 256
    eye, target = bbox_framing(cube, 45.0, up="+z")
    default_eye, _ = bbox_framing(cube, 45.0)

    assert torch.allclose(target, torch.zeros(3), atol=1e-6)
    assert float(eye[1]) < float(target[1])
    assert float(eye[0]) == pytest.approx(float(target[0]), abs=1e-6)
    assert float(eye[2]) == pytest.approx(float(target[2]), abs=1e-6)
    assert float(torch.linalg.norm(target - eye)) == pytest.approx(
        float(torch.linalg.norm(target - default_eye)), rel=1e-6)

    vm = world_to_camera(look_at(eye, target, up="+z"), eye)
    cam = (torch.cat([cube, torch.ones(len(cube), 1)], dim=1) @ vm.T)[:, :3]
    assert (cam[:, 2] > 0).all()
    uv = cam @ intrinsics(W, H, 45.0).T
    uv = uv[:, :2] / uv[:, 2:3]
    assert (uv >= 0).all() and (uv[:, 0] <= W).all() and (uv[:, 1] <= H).all()


@pytest.mark.parametrize("up", ["+y", "+z", "-z"])
def test_an_orbit_circles_the_vertical_axis(up):
    """Level all the way round: constant height, constant distance, never rolled."""
    eye, target = level_eye(up), torch.zeros(3)
    upv = torch.tensor(UP_VECTORS[up])
    views = camera_path(eye, target, FRAMES, 20.0, "orbit", up=up)

    assert torch.allclose(views[0], world_to_camera(look_at(eye, target, up=up), eye),
                          atol=1e-6)
    for vm in views:
        R = c2w(vm)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert float(centre(vm) @ upv) == pytest.approx(0.0, abs=1e-5)
        assert float(torch.linalg.norm(centre(vm) - target)) == pytest.approx(2.0, rel=1e-5)
        assert abs(float(R[:, 0] @ upv)) < 1e-5
    swing = max(float(torch.linalg.norm(centre(vm) - eye)) for vm in views)
    assert swing > 0.5, "the orbit did not move"


@pytest.mark.parametrize("up", ["+y", "+z", "-z"])
def test_a_wiggle_under_another_up_is_rigid_and_loops(up):
    eye, target = level_eye(up), torch.zeros(3)
    views = camera_path(eye, target, FRAMES, 5.0, "wiggle", up=up)
    for vm in views:
        R = c2w(vm)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert float(torch.linalg.det(R)) == pytest.approx(1.0, abs=1e-5)
    steps = [float((views[i + 1] - views[i]).norm()) for i in range(FRAMES - 1)]
    wrap = float((views[0] - views[-1]).norm())
    assert 0 < wrap <= 1.05 * max(steps)


# --------------------------------------------------------- opengl convention

def test_opengl_bbox_framing_keeps_the_cloud_in_front():
    """The flip is applied to the world, so framing has to see the flipped cloud.

    Framing the file's own coordinates aimed the camera at a mirror image: this
    cloud, centred at y=1 z=5 in OpenGL axes, rendered with none of it in front
    of the camera.
    """
    cloud = cube_cloud() * 0.3 + torch.tensor([0.0, 1.0, 5.0])
    W = H = 256
    mode, fov, eye, target = frame_cloud(cloud, frame="bbox", convention="opengl")
    assert mode == "bbox"
    vm = camera_path(eye, target, 1, 0.0)[0] @ _GL2CV
    cam = (torch.cat([cloud, torch.ones(len(cloud), 1)], dim=1) @ vm.T)[:, :3]
    assert (cam[:, 2] > 0).all(), f"{float((cam[:, 2] > 0).float().mean()):.0%} in front"
    uv = cam @ intrinsics(W, H, fov).T
    uv = uv[:, :2] / uv[:, 2:3]
    assert (uv >= 0).all() and (uv[:, 0] <= W).all() and (uv[:, 1] <= H).all()


def test_opengl_auto_framing_recognises_a_monocular_prediction():
    """An OpenGL prediction sits at -Z; only the flipped cloud is 'in front'."""
    prediction_gl = scene() * torch.tensor([1.0, -1.0, -1.0])
    mode, _, eye, target = frame_cloud(prediction_gl, frame="auto", fov=40.0,
                                       convention="opengl")
    assert mode == "input"
    assert torch.equal(eye, torch.zeros(3)) and float(target[2]) > 0


def test_opengl_lens_at_zero_radius_is_the_opengl_pinhole():
    """The lens is built in the OpenCV world and flipped afterwards.

    Rebuilding it from the already-flipped matrix aimed every sample with the
    wrong world's up and rendered the defocused image upside down.
    """
    eye, target = torch.tensor([0.1, 0.0, -2.0]), torch.tensor([0.0, 0.2, 0.5])
    pinhole = camera_path(eye, target, 1, 0.0)[0] @ _GL2CV
    views = lens_views(pinhole, 2.0, 0.0, 16, world_flip=_GL2CV)
    assert len(views) == 1
    assert torch.allclose(views[0], pinhole, atol=1e-6)


def test_opencv_lens_views_match_the_aperture_around_the_view():
    """Without a flip, the lens is exactly `aperture_views` around the view's own axis."""
    eye, target = torch.tensor([0.1, 0.0, -2.0]), torch.tensor([0.0, 0.2, 0.5])
    vm = camera_path(eye, target, 1, 0.0)[0]
    focal = eye + c2w(vm)[:, 2] * 2.0
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(
        lens_views(vm, 2.0, 0.05, 8), aperture_views(eye, focal, 0.05, 8)))


def test_input_framing_refuses_another_up_axis():
    """An input camera is OpenCV by construction, so its up is -y and nothing else."""
    with pytest.raises(SystemExit, match="--up"):
        frame_cloud(scene(), frame="input", up="+z", fov=40.0)
