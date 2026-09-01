"""One way to run the trainer from a benchmark, and one way to read its result.

Every wrong number this project has published came from a harness, never from a
kernel. The pattern repeated twice with the same shape:

  * `--steps-scaler` was overridden by the harness that was measuring it;
  * `--budget` was declared `default=300_000` in nerf_synthetic_sweep.py and
    forwarded to every child unconditionally, so `auto_budget()` never ran in a
    single 8-scene sweep and a table labelled "old defaults vs new defaults"
    actually held budget fixed in both arms.

Neither was detectable from the committed JSON, because the harness recorded
its OWN argparse namespace as "the protocol". The trainer knew the truth and
was never asked.

So: harnesses state what they want, the trainer states what it did, and this
module refuses to let those two disagree silently.

    from bench.runner import run
    rep = run({"blender": scene, "steps": 7000})       # budget unspecified
    rep["resolved"]["budget"]                          # -> 100000, recorded

A knob the caller does not pass is left to the trainer AND still recorded, so
"unspecified" stops meaning "unknown". A knob the caller does pass is verified
against what ran, and a mismatch raises rather than quietly producing a number
attributed to the wrong settings.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunFailed(RuntimeError):
    """The trainer did not produce a usable report."""


class RunDiverged(RuntimeError):
    """The trainer ran with settings other than the ones requested.

    This is the exception that the two historical bugs above would have raised
    on their first run instead of after a week of published numbers.
    """


def _transform_allowed(key: str, want, got, resolved: dict) -> str | None:
    """Is this divergence a documented rewrite by main(), or a harness bug?

    An unconditional allow-list is not good enough here. Listing `steps` as
    "the steps-scaler may rewrite it" would wave through a run that used 3500
    steps when 7000 were requested and no scaler was set -- which is precisely
    the class of bug this module exists to catch. Each allowance therefore has
    to check that its triggering condition actually held.
    """
    scaler = resolved.get("steps_scaler", 1.0)
    if key in ("steps", "relocate_every", "eval_every", "sh_warmup"):
        if scaler != 1.0:
            return f"--steps-scaler {scaler} rewrites it"
        return None
    if key == "start_active":
        # main() clamps start_active to budget//2 when it exceeds the budget,
        # because the parameter tensors are preallocated at `budget`.
        if want > resolved.get("budget", 0) and got == max(1000, resolved["budget"] // 2):
            return f"clamped to budget//2 ({got:,}); requested {want:,} exceeds budget"
        return None
    return None


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_cmd(spec: dict, report: Path) -> list[str]:
    """Turn a spec into argv. Only what the caller asked for is passed.

    This is the whole fix in one line: there is no default here. A knob absent
    from `spec` produces no flag, so the trainer's own default applies and gets
    recorded. Harness defaults are what caused the 300k mixup, so this module
    has none.
    """
    cmd = [sys.executable, "-m", "metal_gauss.train", "--report", str(report)]
    for k, v in spec.items():
        if v is None:
            continue
        if isinstance(v, bool):
            cmd.append(_flag(k) if v else _flag("no_" + k))
        else:
            cmd += [_flag(k), str(v)]
    return cmd


def check(spec: dict, report: dict) -> list[str]:
    """Compare requested against resolved. Returns the unexpected divergences."""
    resolved = report.get("resolved")
    if resolved is None:
        raise RunFailed("report has no 'resolved' block -- trainer too old? "
                        "Every benchmarked number needs one.")
    bad = []
    for k, want in spec.items():
        if want is None or k not in resolved:
            continue
        got = resolved[k]
        if got == want:
            continue
        # str() both sides: paths and numbers arrive as strings on argv.
        if str(got) == str(want):
            continue
        msg = f"{k}: requested {want!r}, ran with {got!r}"
        why = _transform_allowed(k, want, got, resolved)
        if why:
            print(f"    [transform] {msg}  ({why})", flush=True)
        else:
            bad.append(msg)
    return bad


def run(spec: dict, *, report: Path | None = None, cwd: Path = ROOT,
        timeout: float | None = None, strict: bool = True) -> dict:
    """Train once and return the trainer's own report.

    Never parses stdout. Scraping is how "278,571 splats" became 571, and how a
    zero exit code from splat-apple counted as a successful run.
    """
    tmp = None
    if report is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".report.json", delete=False)
        tmp.close()
        report = Path(tmp.name)
    report = Path(report)
    if report.exists():
        report.unlink()          # never read a stale report from a failed run

    cmd = build_cmd(spec, report)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd),
                       timeout=timeout)
    wall = time.perf_counter() - t0

    if not report.exists():
        tail = ((p.stderr or p.stdout).strip().splitlines() or ["no output"])[-4:]
        raise RunFailed(f"no report written (exit {p.returncode})\n  cmd: "
                        f"{' '.join(cmd)}\n  " + "\n  ".join(tail))

    rep = json.loads(report.read_text())
    rep["harness_wall_s"] = round(wall, 1)
    rep["cmd"] = cmd

    bad = check(spec, rep)
    if bad:
        detail = "\n  ".join(bad)
        rep["divergence"] = bad
        if strict:
            raise RunDiverged(
                "the trainer ran with settings other than the ones requested. "
                "Any number from this run would be attributed to the wrong "
                f"protocol.\n  {detail}\n  cmd: {' '.join(cmd)}")
        print(f"    [DIVERGENCE, recorded not raised]\n  {detail}", flush=True)

    if tmp is not None:
        Path(tmp.name).unlink(missing_ok=True)
    return rep


def gpu_competitors(exclude_pids=()) -> list[str]:
    """Other trainer processes currently running, for timing hygiene.

    Written after a naive guard reported "4 competing GPU procs" during a run
    that had exactly one: `pgrep -f "spirula|msplat|brush"` also matched the
    wrapper shell whose command line CONTAINED that pattern, and the python
    process running the benchmark. A guard that can never read zero is worse
    than no guard, because it trains you to ignore it.

    So: match the executable name only (ps `comm`), never the full argv, and
    drop our own process tree.
    """
    import os
    import subprocess as sp
    names = ("spirula", "brush_app", "brush", "msplat-train")
    skip = set(exclude_pids) | {os.getpid(), os.getppid()}
    out = sp.run(["ps", "-eo", "pid=,comm="], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        pid, _, comm = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) in skip:
            continue
        base = comm.strip().rsplit("/", 1)[-1]
        if base in names:
            hits.append(f"{base} (pid {pid})")
    return hits


def require_gpu_exclusive() -> None:
    """Raise unless this process has the GPU to itself.

    Every wall-clock number in this repo assumes exclusive use; a contended run
    is not slightly wrong, it is meaningless.
    """
    busy = gpu_competitors()
    if busy:
        raise RunFailed("GPU is not exclusive -- timings would be invalid. "
                        "Running: " + ", ".join(busy))


def psnr(rep: dict):
    return (rep.get("metrics") or {}).get("psnr")


def wall_s(rep: dict):
    return (rep.get("metrics") or {}).get("wall_s")
