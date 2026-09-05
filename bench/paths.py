"""Where the competitor binaries and the private capture live.

These were absolute paths under one developer's home directory, hardcoded across
five files. That breaks for every other user and publishes a username, so they
are resolved here: environment variable first, then a documented default, then a
clear error naming the variable.

The room1 scene is a private capture and is deliberately NOT vendored, so its
resolver returns None when unset and callers skip rather than fail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_THIRD_PARTY = Path.home() / "third_party"


def third_party() -> Path:
    return Path(os.environ.get("METAL_GAUSS_THIRD_PARTY", _DEFAULT_THIRD_PARTY))


def _resolve(env: str, *rel: str) -> str:
    return os.environ.get(env) or str(third_party().joinpath(*rel))


def brush_bin() -> str:
    return _resolve("METAL_GAUSS_BRUSH", "brush",
                    "brush-app-aarch64-apple-darwin", "brush_app")


def spirula_bin() -> str:
    return _resolve("METAL_GAUSS_SPIRULA", "spirula-studio", "build", "spirula")


def msplat_bin() -> str:
    # Was /tmp/cmp_msplat/bin/msplat-train, alone among these resolvers in
    # pointing somewhere a reboot deletes. Every msplat row in the README was
    # produced by a binary that no longer existed by the time anyone tried to
    # reproduce it. bench/compare/setup_competitors.py installs it here.
    return _resolve("METAL_GAUSS_MSPLAT", "msplat", "bin", "msplat-train")


def room1(kind: str):
    """Private real-capture scene. None when unset, so callers can skip."""
    root = os.environ.get("METAL_GAUSS_ROOM1")
    if not root:
        return None
    return {"colmap": f"{root}/02_poses/sparse/1",
            "images": f"{root}/01_frames/images",
            "ply": f"{root}/03_splats/exports/splat_30000.ply"}[kind]


def competitor_versions() -> dict:
    """What bench/compare/setup_competitors.py installed, or {} if nothing did.

    A row that names its competitor's build can be re-run; one that does not
    can only be trusted. Returns {} rather than raising so a sweep on a machine
    that installed its binaries by hand still runs -- it just cannot say which
    build it used, which is exactly what the empty dict means.
    """
    p = third_party() / "versions.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
