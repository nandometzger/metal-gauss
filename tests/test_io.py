"""`load_ply` de-parameterises, and six bench scripts trust it to.

The 3DGS .ply stores *parameterised* values: opacity as a logit, scale as a
log, and an unnormalised quaternion. Every one of those has a
plausible-looking numeric range, so a reader that drops a single activation
raises nothing -- it returns a scene that renders as fog. That failure mode is
invisible to any test that only checks shapes, so these tests build a file
whose de-parameterised values are known in advance and check that exactly
those come back.

The other half is layout. `f_dc_{0,1,2}` is the degree-0 term per channel and
`f_rest_` is stored CHANNEL-MAJOR, `f_rest_[c*per_channel + (b-1)]`. Reading
it channel-minor is silent too: the colours stay in range and the error shows
up only as drift away from the training views, which is easy to mistake for a
tuning problem. So the fixture writes the stored index into the coefficient
itself and the test asserts where each index lands.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metal_gauss.io import Splats, load_ply

# Fields whose values must survive a reordering of the file unchanged.
FIELDS = ("means", "quats", "scales", "opacities", "sh")


def write_ply(path, *, n=3, means=None, quats=None, log_scales=None,
              logit_opac=None, sh_dc=None, sh_rest=None, sort_properties=False):
    """Write an INRIA-convention .ply, in STORED (parameterised) space.

    Every argument is what lands on disk: log scales, logit opacities, an
    unnormalised quaternion. Defaults are benign, so each test sets only the
    column it is about.

    `sh_rest` is (n, per_channel, 3) in the (splat, band-1, channel) order the
    loader is expected to return; the writer is what flattens it channel-major,
    so the test never has to spell the stride out twice. `None` writes no
    `f_rest_` properties at all -- the degree-0 file a monocular predictor emits.

    `sort_properties` writes the header alphabetically instead
    (`f_rest_0, f_rest_1, f_rest_10, ...`), which is what at least one third-party
    exporter does.
    """
    import plyfile

    means = np.zeros((n, 3), np.float32) if means is None else np.asarray(means, np.float32)
    quats = (np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
             if quats is None else np.asarray(quats, np.float32))
    log_scales = (np.zeros((n, 3), np.float32) if log_scales is None
                  else np.asarray(log_scales, np.float32))
    logit_opac = (np.zeros(n, np.float32) if logit_opac is None
                  else np.asarray(logit_opac, np.float32))
    sh_dc = np.zeros((n, 3), np.float32) if sh_dc is None else np.asarray(sh_dc, np.float32)

    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate("xyz"):
        cols[name] = means[:, i]
    for c in range(3):
        cols[f"f_dc_{c}"] = sh_dc[:, c]
    if sh_rest is not None:
        sh_rest = np.asarray(sh_rest, np.float32)
        per_channel = sh_rest.shape[1]
        for c in range(3):
            for b in range(per_channel):
                cols[f"f_rest_{c * per_channel + b}"] = sh_rest[:, b, c]
    cols["opacity"] = logit_opac
    for i in range(3):
        cols[f"scale_{i}"] = log_scales[:, i]
    for i in range(4):
        cols[f"rot_{i}"] = quats[:, i]

    names = sorted(cols) if sort_properties else list(cols)
    data = np.zeros(n, dtype=[(nm, "f4") for nm in names])
    for nm in names:
        data[nm] = cols[nm]
    plyfile.PlyData([plyfile.PlyElement.describe(data, "vertex")]).write(str(path))
    return path


def full_sh(n, per_channel):
    """(n, per_channel, 3) where each entry encodes where it must land.

    Value `1000*splat + stored_index`, so a wrong channel stride, a transposed
    read and a broadcast of splat 0 across the batch all produce different
    numbers rather than plausible ones.
    """
    idx = np.arange(per_channel)[None, :, None] + per_channel * np.arange(3)[None, None, :]
    return (1000.0 * np.arange(n)[:, None, None] + idx).astype(np.float32)


# --------------------------------------------------------- de-parameterisation

def test_opacity_is_read_as_a_logit(tmp_path):
    """Stored logits, returned probabilities.

    Chosen backwards: pick the opacities we want, store their logits, and
    require those opacities back. A loader that skipped the sigmoid would
    return 2.197 for the 0.9 splat -- a number with no error attached to it.
    """
    want = np.array([0.5, 0.9, 0.1], np.float32)
    sp = load_ply(write_ply(tmp_path / "o.ply", logit_opac=np.log(want / (1.0 - want))))
    assert sp.opacities.tolist() == pytest.approx(want.tolist(), rel=1e-6)


def test_opacity_is_always_a_probability(tmp_path):
    """Whatever the file holds, what comes out has to be usable as alpha."""
    stored = np.array([-30.0, 0.0, 12.5], np.float32)
    sp = load_ply(write_ply(tmp_path / "o.ply", logit_opac=stored))
    assert (sp.opacities > 0).all() and (sp.opacities < 1).all()
    assert sp.opacities[1].item() == 0.5           # sigmoid(0), exactly
    assert torch.isfinite(sp.opacities).all()      # exp(30) must not overflow


def test_scale_is_read_as_a_log(tmp_path):
    """Same trick: store log(want), require want back, and no negatives."""
    want = np.array([[0.01, 0.02, 0.03], [1.0, 1.0, 1.0], [0.5, 0.25, 0.125]], np.float32)
    sp = load_ply(write_ply(tmp_path / "s.ply", log_scales=np.log(want)))
    assert sp.scales.numpy() == pytest.approx(want, rel=1e-6)
    # Stored log-scales are routinely negative; a linear scale never can be.
    assert (sp.scales > 0).all()


def test_quaternion_is_normalised(tmp_path):
    """The rasteriser builds R straight from wxyz, so the norm is not free.

    An unnormalised quaternion scales the rotation matrix by |q|^2, which the
    covariance then inherits: the two splats here differ in stored norm by 50x
    and must come back describing the same rotation.
    """
    stored = np.array([[5.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.1],
                       [1.0, 1.0, 1.0, 1.0]], np.float32)
    sp = load_ply(write_ply(tmp_path / "q.ply", quats=stored))
    assert sp.quats.norm(dim=1).tolist() == pytest.approx([1.0, 1.0, 1.0], rel=1e-6)
    # Direction is preserved, not just magnitude: q and -q are the same
    # rotation but a loader has no business flipping the sign.
    unit = stored / np.linalg.norm(stored, axis=1, keepdims=True)
    assert sp.quats.numpy() == pytest.approx(unit, rel=1e-6)


def test_zero_quaternion_does_not_become_nan(tmp_path):
    """A dead splat can carry an all-zero rotation; NaN there poisons the render.

    One NaN mean or covariance survives into the tile binning and takes out
    more than its own pixel, so the loader clamps the norm rather than dividing
    by zero. The result is not a rotation -- nothing can make it one -- but it
    must at least be finite.
    """
    stored = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], np.float32)
    sp = load_ply(write_ply(tmp_path / "q0.ply", n=2, quats=stored))
    assert torch.isfinite(sp.quats).all()


def test_positions_pass_through_unchanged(tmp_path):
    """x/y/z is the one block with no activation on it.

    Worth pinning in the same file as the others: the symmetric mistake to
    forgetting an activation is applying one where there is none.
    """
    means = np.array([[0.0, 1.0, 2.0], [-3.5, 4.25, 5.125], [1e3, -1e3, 0.0]], np.float32)
    sp = load_ply(write_ply(tmp_path / "m.ply", means=means))
    assert sp.means.numpy() == pytest.approx(means)


# ------------------------------------------------------------------ SH layout

def test_dc_band_is_the_degree_zero_term_per_channel(tmp_path):
    """f_dc_c belongs at sh[:, 0, c] -- band 0, not channel 0."""
    dc = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], np.float32)
    sp = load_ply(write_ply(tmp_path / "dc.ply", sh_dc=dc, sh_rest=full_sh(3, 15)))
    assert sp.sh[:, 0, :].numpy() == pytest.approx(dc)


def test_f_rest_is_channel_major(tmp_path):
    """sh[n, b, c] must come from f_rest_[c*per_channel + (b-1)].

    Both strides give a full, plausible-looking coefficient tensor, so this is
    asserted against the index arithmetic directly. The spot check at the end
    is the one that would fail loudest under a channel-minor read.
    """
    per_channel = 15
    sp = load_ply(write_ply(tmp_path / "r.ply", sh_rest=full_sh(3, per_channel)))
    assert sp.sh.shape == (3, 16, 3)
    for n in range(3):
        for c in range(3):
            for b in range(1, 16):
                assert sp.sh[n, b, c].item() == 1000.0 * n + c * per_channel + (b - 1), \
                    f"sh[{n},{b},{c}] came from the wrong f_rest column"
    # Band 1 of the green channel is f_rest_15. Read channel-minor
    # (f_rest_[(b-1)*3 + c]) it would be f_rest_1, i.e. 1.0.
    assert sp.sh[0, 1, 1].item() == 15.0


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_degree_follows_the_coefficient_count(tmp_path, degree):
    """The count in the file is the only record of the degree; nothing else says.

    `render_frames` passes `sp.sh_degree` to the rasteriser, so guessing 3 here
    would ask `eval_sh` for bases the file never wrote.
    """
    n_bases = (degree + 1) ** 2
    rest = None if degree == 0 else full_sh(3, n_bases - 1)
    sp = load_ply(write_ply(tmp_path / f"d{degree}.ply", sh_rest=rest))
    assert sp.sh_degree == degree
    assert sp.sh.shape == (3, n_bases, 3)


def test_degree_zero_ply_has_no_f_rest_at_all(tmp_path):
    """A monocular predictor emits exactly this: DC only, no f_rest_ properties.

    Called out separately from the parametrised case because it is the file
    that motivated `render_path`, and because it is the one input where the
    loader has to infer the degree from an ABSENCE rather than a count.
    """
    dc = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], np.float32)
    sp = load_ply(write_ply(tmp_path / "d0.ply", sh_dc=dc))
    assert sp.sh_degree == 0
    assert sp.sh.shape == (3, 1, 3)
    assert sp.sh[:, 0, :].numpy() == pytest.approx(dc)


def test_non_square_f_rest_count_is_rejected(tmp_path):
    """6 coefficients is 2 bands per channel, and no SH degree has 3 bases.

    Refusing beats guessing: the loader would otherwise hand the rasteriser a
    band count `eval_sh` has no basis functions for.
    """
    path = write_ply(tmp_path / "bad.ply", sh_rest=np.zeros((3, 2, 3), np.float32))
    with pytest.raises(ValueError, match="whole SH degree"):
        load_ply(path)


def test_property_order_in_the_file_does_not_matter(tmp_path):
    """At least one third-party exporter writes the header alphabetically.

    `f_rest_0, f_rest_1, f_rest_10, ...` with x/y/z last, where the INRIA
    convention puts x/y/z first and numbers f_rest in order. The loader indexes
    by property NAME, so the two files must be indistinguishable; a reader
    using fixed column offsets would produce garbage from the second one.
    """
    kwargs = dict(means=np.arange(9, dtype=np.float32).reshape(3, 3),
                  quats=np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0,
                  log_scales=np.linspace(-4, 0, 9, dtype=np.float32).reshape(3, 3),
                  logit_opac=np.array([-2.0, 0.0, 2.0], np.float32),
                  sh_dc=np.array([[0.1, 0.2, 0.3]] * 3, np.float32),
                  sh_rest=full_sh(3, 15))
    ordered = load_ply(write_ply(tmp_path / "ordered.ply", **kwargs))
    alpha = load_ply(write_ply(tmp_path / "alpha.ply", sort_properties=True, **kwargs))

    assert alpha.sh_degree == ordered.sh_degree
    for f in FIELDS:
        a, b = getattr(ordered, f), getattr(alpha, f)
        assert torch.equal(a, b), f"{f} depends on header order"


# ------------------------------------------------------------------ container

def test_subset_and_len_keep_the_degree(tmp_path):
    """`sh_degree` is not per-splat, and dropping it on a subset renders wrong.

    Nothing downstream re-derives it -- `render_frames` reads it off whatever
    Splats it is handed -- so a subset that reset it to 0 would quietly render
    a degree-3 model flat.
    """
    sp = load_ply(write_ply(tmp_path / "sub.ply", n=3, sh_rest=full_sh(3, 15)))
    assert len(sp) == 3

    sub = sp.subset(torch.tensor([2, 0]))
    assert len(sub) == 2
    assert sub.sh_degree == sp.sh_degree
    for f in FIELDS:
        assert torch.equal(getattr(sub, f), getattr(sp, f)[torch.tensor([2, 0])])

    same = sp.to("cpu")
    assert isinstance(same, Splats) and same.sh_degree == sp.sh_degree


def test_dtype_and_device_are_honoured(tmp_path):
    """Every field has to land in one dtype; a stray float64 breaks the kernels."""
    sp = load_ply(write_ply(tmp_path / "t.ply", sh_rest=full_sh(3, 15)), dtype=torch.float64)
    for f in FIELDS:
        t = getattr(sp, f)
        assert t.dtype is torch.float64, f"{f} is {t.dtype}"
        assert t.device.type == "cpu"


def test_f_rest_count_must_be_divisible_by_three(tmp_path):
    """A count that is not a whole number per channel is rejected, not truncated.

    `per_channel = n_rest // 3` silently drops a column otherwise, which shifts
    the channel-major layout underneath it and gives subtly wrong colour rather
    than a failure. 46 is exactly the case the whole-degree check does NOT catch
    on its own: 46 // 3 == 15, so it reads as a clean degree 3 and the 46th
    coefficient vanishes. Built by hand because `write_ply` takes an
    (n, per_channel, 3) array and so can only ever emit a multiple of three.
    """
    from plyfile import PlyData, PlyElement

    n, n_rest = 2, 46
    cols = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")]
    cols += [(f"scale_{i}", "f4") for i in range(3)]
    cols += [(f"rot_{i}", "f4") for i in range(4)]
    cols += [(f"f_dc_{i}", "f4") for i in range(3)]
    cols += [(f"f_rest_{i}", "f4") for i in range(n_rest)]
    arr = np.zeros(n, dtype=cols)
    arr["rot_0"] = 1.0
    path = tmp_path / "ragged.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))

    with pytest.raises(ValueError, match="whole number per channel"):
        load_ply(path)


# ------------------------------------------------------------------- writing

def _synthetic_splats(n=32, degree=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    bases = (degree + 1) ** 2
    return Splats(
        means=torch.randn(n, 3, generator=g),
        quats=torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1),
        scales=torch.rand(n, 3, generator=g) * 0.1 + 1e-3,
        opacities=torch.rand(n, generator=g),
        sh=torch.randn(n, bases, 3, generator=g),
        sh_degree=degree,
    )


@pytest.mark.parametrize("degree", [0, 3])
def test_a_written_file_reads_back_as_the_same_splats(tmp_path, degree):
    """The viewer holds activated values; the file stores log scales and logits.

    A round trip is the only check that both conversions agree -- writing a
    probability where the loader expects a logit gives a file that renders as
    fog rather than an error.
    """
    from metal_gauss.io import save_ply

    sp = _synthetic_splats(degree=degree)
    save_ply(sp, tmp_path / "out.ply")
    back = load_ply(tmp_path / "out.ply")

    assert back.sh_degree == degree
    assert torch.allclose(back.means, sp.means, atol=1e-6)
    assert torch.allclose(back.quats, sp.quats, atol=1e-6)
    assert torch.allclose(back.scales, sp.scales, rtol=1e-5, atol=1e-7)
    assert torch.allclose(back.opacities, sp.opacities, atol=1e-5)
    assert torch.allclose(back.sh, sp.sh, atol=1e-6)


def test_a_fully_opaque_splat_survives_the_logit(tmp_path):
    """logit(1) is not writable; clamping keeps 0 and 1 finite and close."""
    from metal_gauss.io import save_ply

    sp = _synthetic_splats(n=2)
    sp.opacities[:] = torch.tensor([0.0, 1.0])
    save_ply(sp, tmp_path / "edge.ply")
    back = load_ply(tmp_path / "edge.ply")

    assert torch.isfinite(back.opacities).all()
    assert float(back.opacities[0]) < 1e-4
    assert float(back.opacities[1]) > 1.0 - 1e-4


def test_a_written_file_carries_the_fields_the_renderer_reads(tmp_path):
    """Same names and the same channel-major SH as train.export_ply writes."""
    from plyfile import PlyData

    from metal_gauss.io import save_ply

    save_ply(_synthetic_splats(), tmp_path / "fields.ply")
    names = {p.name for p in PlyData.read(str(tmp_path / "fields.ply"))["vertex"].properties}
    assert {"x", "y", "z", "opacity"} <= names
    assert {f"scale_{i}" for i in range(3)} <= names
    assert {f"rot_{i}" for i in range(4)} <= names
    assert {f"f_dc_{i}" for i in range(3)} <= names
    assert {f"f_rest_{i}" for i in range(45)} <= names
