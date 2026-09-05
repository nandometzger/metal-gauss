"""Run spirula-studio across the Pareto iteration ladder.

spirula-studio was the last unbenchmarked Apple-native trainer and was recorded
in STATUS.md as a known gap. It builds on macOS via MoltenVK (CMake fetches it),
exposes a real `spirula train` CLI, and reads the Nerfstudio layout, so it can
take the SAME seed point cloud as metal-gauss and msplat.

Three traps, all found before they produced a number:

  * `--keep-viewer-alive` defaults to 1, so the process serves an HTTP viewer
    forever after training ends. Timing that measures the viewer, not the run.
    `--disable-viewer 1` is mandatory for any wall-clock figure.
  * its Nerfstudio parser honours `ply_file_path` from transforms.json BEFORE
    falling back to sparse_pc.ply / pointcloud.ply, so the seed cloud has to be
    named to match or the run silently starts from a different initialisation
    than every other row. The scene dir is staged so the cloud is byte-identical.
  * the `synthetic` preset is the right one for NeRF-synthetic (it disables the
    bilateral grid and per-pixel ISP, which exist for real captures). Running
    the generic preset here would benchmark a misconfiguration, the same
    reasoning applied to msplat's schedules.

Unlike msplat, its schedules already scale to --num-iterations (it reaches
~100k splats by step 500), so nothing is hand-scaled: this is stock.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from bench.paths import competitor_versions, spirula_bin  # noqa: E402

BIN = Path(spirula_bin())
RESULTS = ROOT / "bench" / "results"


def score(ply: Path, scene: Path, resolution: int) -> dict:
    js = ply.with_suffix(".score.json")
    p = subprocess.run(
        [sys.executable, str(ROOT / "bench/compare/score_ply.py"), str(ply),
         # --ssim inline, matching pareto.py. Its absence here meant the
         # 8-scene sweep produced SSIM for every implementation EXCEPT spirula,
         # which was only noticed when a two-metric domination count divided by
         # zero. A scorer that silently omits a metric is the same failure shape
         # as a filename that silently omits a knob.
         "--scene", str(scene), "--resolution", str(resolution), "--ssim",
         "--out", str(js)],
        capture_output=True, text=True, cwd=ROOT)
    if js.exists():
        d = json.loads(js.read_text())
        return {"psnr": d.get("psnr"), "ssim": d.get("ssim"),
                "lpips": d.get("lpips"), "n_splats": d.get("n_splats")}
    return {"psnr": None,
            "error": ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1][:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(ROOT / "data/nerf_synthetic/lego"))
    ap.add_argument("--data", default="/tmp/cmp_data/lego_spirula")
    ap.add_argument("--iters", nargs="*", type=int,
                    default=[500, 1000, 2000, 4000, 7000, 15000])
    ap.add_argument("--preset", default="synthetic")
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--tag", default="",
                    help="suffix for the output dir, so repeated runs of the "
                         "same (scene, iters) do not overwrite each other -- "
                         "the collision this repo has now hit three times")
    ap.add_argument("--out", default=str(RESULTS / "pareto_lego_spirula.json"))
    a = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"spirula binary not built: {BIN}")
    scene, rows = Path(a.scene), []

    for n in a.iters:
        outdir = Path("/tmp/spirula_out")
        # The SCENE must be in the directory name. It was hardcoded to "lego"
        # while this only ever ran on lego; the 8-scene sweep then had drums and
        # ficus overwrite each other's exports. Scores were unaffected -- each
        # run is scored immediately, against its own --scene, before the next
        # overwrites -- but the plys were lost, so they could not be re-scored
        # for LPIPS afterwards. Third instance of this bug class in this repo,
        # after pareto.py's scene and msplat-variant collisions.
        name = f"{scene.name}_{n}{a.tag}"
        shutil.rmtree(outdir / name, ignore_errors=True)
        cmd = [str(BIN), "train", a.preset, "--data", a.data,
               "--data-format", "nerfstudio", "--num-iterations", str(n),
               "--sh-degree", "3", "--disable-viewer", "1",
               "--output-dir-prefix", str(outdir), "--output-dir-name", name]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0

        plys = sorted((outdir / name).glob("**/splat.ply"))
        if p.returncode != 0 or not plys:
            tail = ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1]
            print(f"  spirula {n:>6}  FAILED  {tail[:110]}", flush=True)
            rows.append({"impl": "spirula", "iters": n, "ok": False,
                         "tail": tail[:400]})
            continue
        ply = plys[-1]
        # Verify the run actually did the requested steps rather than trusting
        # the flag -- the checkpoint name carries the resolved step count.
        got = int(ply.parent.name.split("-")[-1].replace(".ckpt", ""))
        cfg = {}
        cfgp = outdir / name / "config.json"
        if cfgp.exists():
            cfg = json.loads(cfgp.read_text())
        s = score(ply, scene, a.resolution)
        row = {"impl": "spirula", "iters": n, "resolved_steps": got,
               "preset": a.preset, "ok": s.get("psnr") is not None,
               "wall_s": round(wall, 1), **s}
        # Same stamp pareto.py puts on the other competitors. Missing it here
        # meant spirula rows stayed unattributable while msplat and brush
        # became reproducible, which is a worse state than either.
        build = competitor_versions().get("spirula")
        if build:
            row["build"] = build
        if got != n:
            row["divergence"] = f"requested {n} steps, checkpoint says {got}"
            print(f"    [DIVERGENCE] {row['divergence']}", flush=True)
        rows.append(row)
        q = f"{s['psnr']:.3f}" if s.get("psnr") else "score FAILED"
        print(f"  spirula {n:>6}  {wall/60:6.2f} min  {q:>12} dB  "
              f"{s.get('n_splats', 0):>9,} splats", flush=True)
        Path(a.out).write_text(json.dumps(
            {"scene": str(scene), "resolution": a.resolution,
             "preset": a.preset, "binary": str(BIN),
             "spirula_version": cfg.get("version", "2026.8.28"),
             "rows": rows}, indent=2, default=str))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
