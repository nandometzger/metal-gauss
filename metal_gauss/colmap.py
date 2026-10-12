"""Read a COLMAP sparse model (cameras, images, points3D), binary or text.

Why not pycolmap: its macOS wheel bundles its own libomp.dylib, and so does
torch. Loading both into one process aborts with ``OMP: Error #15`` (see
scripts/fix_openmp.py), so `metal-gauss-train --colmap` died at import on a
plain `pip install`. Training only needs intrinsics, poses and the sparse
points, which is a few hundred lines of a documented file format -- no reason
to pull a second OpenMP runtime into the process for it.

Format: https://colmap.github.io/format.html
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# name -> (model id, number of params, one shared focal length)
# Single-focal models store (f, cx, cy, extra...), the others
# (fx, fy, cx, cy, extra...). Matches pycolmap 4.2's CameraModelId table.
MODELS: dict[str, tuple[int, int, bool]] = {
    "SIMPLE_PINHOLE": (0, 3, True),
    "PINHOLE": (1, 4, False),
    "SIMPLE_RADIAL": (2, 4, True),
    "RADIAL": (3, 5, True),
    "OPENCV": (4, 8, False),
    "OPENCV_FISHEYE": (5, 8, False),
    "FULL_OPENCV": (6, 12, False),
    "FOV": (7, 5, False),
    "SIMPLE_RADIAL_FISHEYE": (8, 4, True),
    "RADIAL_FISHEYE": (9, 5, True),
    "THIN_PRISM_FISHEYE": (10, 12, False),
    "RAD_TAN_THIN_PRISM_FISHEYE": (11, 16, False),
    "SIMPLE_DIVISION": (12, 4, True),
    "DIVISION": (13, 5, False),
    "SIMPLE_FISHEYE": (14, 3, True),
    "FISHEYE": (15, 4, False),
    "EUCM": (16, 6, False),
}
_BY_ID = {mid: name for name, (mid, _, _) in MODELS.items()}

# Models whose projection is not a pinhole even with all extra params at zero.
NON_PINHOLE = frozenset(n for n in MODELS if "FISHEYE" in n) | {"FOV", "EUCM"}


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray       # float64, laid out per model

    @property
    def _single(self) -> bool:
        return MODELS[self.model][2]

    @property
    def fx(self) -> float:
        return float(self.params[0])

    @property
    def fy(self) -> float:
        return float(self.params[0] if self._single else self.params[1])

    @property
    def cx(self) -> float:
        return float(self.params[1] if self._single else self.params[2])

    @property
    def cy(self) -> float:
        return float(self.params[2] if self._single else self.params[3])

    @property
    def extra(self) -> np.ndarray:
        """Distortion (or other non-pinhole) parameters after f/fx, fy, cx, cy."""
        return self.params[3 if self._single else 4:]


@dataclass(frozen=True)
class Image:
    id: int
    camera_id: int
    name: str
    R: np.ndarray            # (3,3) world -> camera rotation
    t: np.ndarray            # (3,) world -> camera translation


@dataclass(frozen=True)
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: np.ndarray       # (P,3) float64
    colors: np.ndarray       # (P,3) uint8


def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _camera(cid: int, model: str, width: int, height: int, params) -> Camera:
    if model not in MODELS:
        raise ValueError(f"camera {cid}: unsupported COLMAP camera model {model}")
    n = MODELS[model][1]
    params = np.asarray(params, np.float64)
    if params.size != n:
        raise ValueError(f"camera {cid}: {model} has {n} params, got {params.size}")
    return Camera(cid, model, int(width), int(height), params)


# --- binary -----------------------------------------------------------------

class _Reader:
    def __init__(self, path: Path):
        self.buf = memoryview(path.read_bytes())
        self.off = 0

    def take(self, fmt: str):
        out = struct.unpack_from("<" + fmt, self.buf, self.off)
        self.off += struct.calcsize("<" + fmt)
        return out

    def skip(self, n: int):
        self.off += n

    def cstr(self) -> str:
        end = bytes(self.buf[self.off:]).index(b"\0")
        s = bytes(self.buf[self.off:self.off + end]).decode("utf-8")
        self.off += end + 1
        return s


def _cameras_bin(path: Path) -> dict[int, Camera]:
    r, cams = _Reader(path), {}
    for _ in range(r.take("Q")[0]):
        cid, mid, w, h = r.take("iiQQ")
        if mid not in _BY_ID:
            raise ValueError(f"camera {cid}: unsupported COLMAP camera model id {mid}")
        model = _BY_ID[mid]
        cams[cid] = _camera(cid, model, w, h, r.take(f"{MODELS[model][1]}d"))
    return cams


def _images_bin(path: Path) -> dict[int, Image]:
    r, ims = _Reader(path), {}
    for _ in range(r.take("Q")[0]):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = r.take("i7di")
        name = r.cstr()
        r.skip(r.take("Q")[0] * 24)      # points2D: x, y (double), point3D_id (int64)
        ims[iid] = Image(iid, cid, name, quat_to_R(qw, qx, qy, qz), np.array([tx, ty, tz]))
    return ims


def _points_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    r = _Reader(path)
    n = r.take("Q")[0]
    xyz, rgb = np.empty((n, 3), np.float64), np.empty((n, 3), np.uint8)
    for i in range(n):
        _, x, y, z, cr, cg, cb, _, track = r.take("Q3d3BdQ")
        xyz[i], rgb[i] = (x, y, z), (cr, cg, cb)
        r.skip(track * 8)                # track: image_id, point2D_idx (int32 each)
    return xyz, rgb


# --- text -------------------------------------------------------------------

def _rows(path: Path):
    """Non-comment lines, blank ones kept: images.txt pairs each image line
    with a points2D line that is empty when the image has no observations."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            yield line


def _cameras_txt(path: Path) -> dict[int, Camera]:
    cams = {}
    for line in _rows(path):
        f = line.split()
        if f:
            cams[int(f[0])] = _camera(int(f[0]), f[1], f[2], f[3], [float(v) for v in f[4:]])
    return cams


def _images_txt(path: Path) -> dict[int, Image]:
    ims, rows = {}, _rows(path)
    for line in rows:
        f = line.split(maxsplit=9)
        if not f:
            continue
        iid, cid = int(f[0]), int(f[8])
        qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
        ims[iid] = Image(iid, cid, f[9], quat_to_R(qw, qx, qy, qz), np.array([tx, ty, tz]))
        next(rows, None)                 # the points2D line
    return ims


def _points_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    for line in _rows(path):
        f = line.split()
        if f:
            xyz.append([float(v) for v in f[1:4]])
            rgb.append([int(v) for v in f[4:7]])
    return (np.array(xyz, np.float64).reshape(-1, 3),
            np.array(rgb, np.uint8).reshape(-1, 3))


def read_model(path: str | Path) -> Model:
    """Read a sparse model directory, preferring .bin like COLMAP does."""
    path = Path(path)
    if (path / "cameras.bin").exists():
        cams, ims = _cameras_bin(path / "cameras.bin"), _images_bin(path / "images.bin")
        pts, cols = _points_bin(path / "points3D.bin")
    elif (path / "cameras.txt").exists():
        cams, ims = _cameras_txt(path / "cameras.txt"), _images_txt(path / "images.txt")
        pts, cols = _points_txt(path / "points3D.txt")
    else:
        raise FileNotFoundError(f"no COLMAP model (cameras.bin or cameras.txt) in {path}")
    return Model(cams, ims, pts, cols)
