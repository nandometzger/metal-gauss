"""PSNR vs wall-clock Pareto front on a NeRF-synthetic scene.

Comparing trainers at matched ITERATIONS is meaningless when per-iteration cost
differs several-fold -- the slower-per-step implementation simply looks better
because it did more work. What a user actually wants to know is: given N
minutes, which implementation gives the best reconstruction? That is a Pareto
front in (wall-clock, PSNR) space, and it is what this measures.

Fairness rules, all enforced here rather than asserted in prose:
  * every implementation trains on the OFFICIAL train split and is scored on
    the OFFICIAL test split by one evaluator (bench/compare/score_ply.py),
    so neither the holdout nor the metric differs between rows;
  * every implementation starts from the SAME random point cloud (written by
    blender_to_nerfstudio.py with the seed and extent rule metal_gauss uses);
  * runs are strictly sequential -- GPU contention invalidates timings.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from bench.paths import brush_bin, competitor_versions, msplat_bin  # noqa: E402
sys.path.insert(0, str(ROOT))
from bench.runner import RunDiverged, RunFailed  # noqa: E402
from bench.runner import run as run_trainer      # noqa: E402
RESULTS = ROOT / "bench" / "results"
OUT_PLY = Path("/tmp/cmp_out")


def export_tag(scene_name: str, impl: str, iters: int, *, msplat_stock: bool,
               bud: int = 0, sweeping_budgets: bool = False) -> str:
    """Filename stem for one run's .ply.

    EVERY knob that distinguishes two runs has to appear here, because the
    front is re-scored later FROM THESE FILES: a collision does not error, it
    silently attaches one run's metrics to another run's row.

    Both additions were made after the failure, not before it:
      * `scene`, after a drums run overwrote lego's exports;
      * the msplat variant, after running the stock arm of a stock-vs-scaled
        A/B destroyed the scaled arm's plys on drums and ficus. Only PSNR had
        been written to JSON by then, so SSIM and LPIPS for that arm were
        unrecoverable without re-running.
    """
    var = ("_stock" if msplat_stock else "_scaled") if impl == "msplat" else ""
    suffix = f"_b{bud // 1000}k" if sweeping_budgets else "_auto"
    return f"{scene_name}_{impl}{var}_{iters}{suffix}"


def score(ply: Path, scene: Path, resolution: int) -> dict:
    """Read the scorer's JSON, never its printed line.

    Scraping stdout got the splat count wrong: "278,571 splats" split on commas
    gives "571". The scorer already writes structured output -- use it.
    """
    js = ply.with_suffix(".score.json")
    p = subprocess.run(
        # SSIM inline: it is our own Metal kernel and costs little. LPIPS is
        # NOT inline -- an alexnet forward per view over 240 exports adds hours
        # to a sweep whose wall-clock rows are the point. It is added later for
        # the points that actually reach the front.
        [sys.executable, str(ROOT / "bench" / "compare" / "score_ply.py"), str(ply),
         "--scene", str(scene), "--resolution", str(resolution), "--ssim",
         "--out", str(js)],
        capture_output=True, text=True, cwd=ROOT)
    if js.exists():
        d = json.loads(js.read_text())
        return {"psnr": d.get("psnr"), "ssim": d.get("ssim"),
                "n_splats": d.get("n_splats")}
    return {"psnr": None,
            "error": ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1][:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(ROOT / "data" / "nerf_synthetic" / "lego"))
    ap.add_argument("--scene-ns", default="/tmp/cmp_data/lego_ns",
                    help="same scene in Nerfstudio layout (for msplat)")
    ap.add_argument("--iters", nargs="*", type=int,
                    default=[500, 1000, 2000, 4000, 7000, 15000])
    ap.add_argument("--budget", type=int, default=0,
                    help="0 = let the trainer pick (auto_budget)")
    ap.add_argument("--budgets", nargs="*", type=int, default=None,
                    help="sweep capacity too (metal-gauss only; msplat's ADC "
                         "chooses its own count)")
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--impls", nargs="*",
                    default=["metal-gauss", "msplat", "brush"])
    ap.add_argument("--msplat-stock", action="store_true",
                    help="run msplat with its shipped defaults instead of "
                         "schedules scaled to the run length (for A/B)")
    ap.add_argument("--tag", default="",
                    help="suffix for exported plys, so repeated runs of an "
                         "identical config do not overwrite each other")
    ap.add_argument("--out", default=str(RESULTS / "pareto_lego.json"))
    a = ap.parse_args()

    scene, OUTP = Path(a.scene), OUT_PLY
    OUTP.mkdir(exist_ok=True)
    rows = []

    budgets = a.budgets or [a.budget]
    for impl in a.impls:
      for bud in (budgets if impl == "metal-gauss" else [a.budget]):
        for n in a.iters:
            tag = export_tag(scene.name, impl, n, msplat_stock=a.msplat_stock,
                             bud=bud, sweeping_budgets=bool(a.budgets)) + a.tag
            ply = OUTP / f"pareto_{tag}.ply"
            if impl == "metal-gauss":
                spec = {"blender": str(scene), "steps": n,
                        "max_resolution": a.resolution,
                        "eval_every": n * 10,            # skip internal eval
                        "export": str(ply)}
                # bud=0 means "let the trainer choose" -- passing --budget at
                # all defeats auto_budget(), which is the thing under test.
                if bud:
                    # start-active must not exceed the budget: the parameter
                    # tensors are preallocated at `budget`, and an `active`
                    # above that reads past the end. Half the budget mirrors
                    # the shipped default's ratio (150k of 300k).
                    spec["budget"] = bud
                    spec["start_active"] = max(1000, min(150_000, bud // 2))
                t0 = time.perf_counter()
                try:
                    rep = run_trainer(spec)
                except (RunFailed, RunDiverged) as e:
                    print(f"  {impl:<12} {n:>6}  FAILED  {str(e)[:110]}", flush=True)
                    rows.append({"impl": impl, "iters": n, "ok": False,
                                 "tail": str(e)[:400]})
                    continue
                wall = time.perf_counter() - t0
                s_ = score(ply, scene, a.resolution)
                row = {"impl": impl, "iters": n,
                       "requested_budget": bud or "auto",
                       "ok": s_.get("psnr") is not None,
                       "wall_s": round(wall, 1),
                       "resolved": rep["resolved"], "env": rep["env"], **s_}
                rows.append(row)
                q = f"{s_['psnr']:.3f}" if s_.get("psnr") else "score FAILED"
                print(f"  {impl:<12} {n:>6} b={rep['resolved']['budget']//1000}k  "
                      f"{wall/60:6.2f} min  {q:>12} dB  "
                      f"{s_.get('n_splats', 0):>9,} splats", flush=True)
                Path(a.out).write_text(json.dumps(
                    {"scene": str(scene), "resolution": a.resolution,
                     "msplat_variant": "stock" if a.msplat_stock else "scaled",
                     "rows": rows}, indent=2, default=str))
                continue
            elif impl == "brush":
                # Brush needs no coordinate-frame flag: its exported ply lands
                # in the same world frame as transforms_test.json, verified by
                # geometry (every splat above alpha 0.1 sits within r < 1.5 of
                # the origin with DC colour [0.74, 0.63, 0.26], which is lego).
                # Its "comment Vertical axis: y" header is a hardcoded string
                # with no z variant in the binary and carries no information.
                #
                # Stock defaults, as with msplat. Note they are tuned for its
                # own 30000-step default: at 1000 steps growth has barely
                # fired and it exports ~10.5k splats that are mostly a
                # near-transparent veil, scoring below a constant image. That
                # is a real property of running it short, not a harness fault,
                # and it is reported rather than tuned away.
                cmd = [brush_bin(), str(scene), "--total-steps", str(n),
                       "--max-resolution", str(a.resolution),
                       "--sh-degree", "3",
                       "--export-every", str(n),
                       "--export-path", str(ply.parent),
                       "--export-name", ply.name.replace(".ply", "_{iter}.ply")]
                cwd = str(ply.parent)
            elif impl == "msplat":
                # --keep-crs is REQUIRED for cross-implementation scoring.
                # Without it msplat applies Nerfstudio's scene transform
                # (auto-orient + auto-scale), so its .ply lands in a rotated,
                # rescaled world frame. Its own eval stays self-consistent and
                # looks fine, but any external evaluator reads ~9 dB instead of
                # ~18. Cost us an afternoon.
                cmd = [msplat_bin(), "--input", a.scene_ns,
                       "--num-iters", str(n), "--keep-crs", "--output", str(ply)]
                if not a.msplat_stock:
                    # Scale msplat's schedules to the run length, exactly as we
                    # scale ours. Its defaults target its own default of 30000
                    # iterations, so at 500 iters stock settings mean
                    # densification NEVER starts (warmup 500), resolution never
                    # leaves 1/4 (doubles every 3000), and SH never passes
                    # degree 0 (interval 1000). Benchmarking that would be
                    # measuring a misconfiguration, not the implementation.
                    k = n / 30_000.0
                    for flag, default in (("--resolution-schedule", 3000),
                                          ("--sh-degree-interval", 1000),
                                          ("--refine-every", 100),
                                          ("--warmup-length", 500),
                                          ("--stop-screen-size-at", 4000)):
                        cmd += [flag, str(max(1, int(round(default * k))))]
                cwd = "/tmp"
            else:
                print(f"  {impl}: no runner, skipped", flush=True)
                continue

            t0 = time.perf_counter()
            p = subprocess.run([str(c) for c in cmd], capture_output=True,
                               text=True, cwd=cwd)
            wall = time.perf_counter() - t0
            if impl == "brush" and not ply.exists():
                # --export-name expands {iter}; adopt whatever it wrote
                cand = sorted(ply.parent.glob(
                    ply.name.replace(".ply", "_*.ply")))
                if cand:
                    cand[-1].rename(ply)
            if p.returncode != 0 or not ply.exists():
                tail = ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1]
                print(f"  {impl:<12} {n:>6}  FAILED  {tail[:110]}", flush=True)
                rows.append({"impl": impl, "iters": n, "ok": False, "tail": tail[:400]})
                continue

            s = score(ply, scene, a.resolution)
            # `budget` is a metal-gauss knob. pareto_lego_msplat.json stamped
            # `budget: 300000` on msplat rows -- a number that had no effect on
            # those runs and misleads any later reader. What matters for an
            # msplat row is which schedule variant produced it.
            row = {"impl": impl, "iters": n,
                   "msplat_variant": "stock" if a.msplat_stock else "scaled",
                   "ok": s.get("psnr") is not None,
                   "wall_s": round(wall, 1), **s}
            # Which competitor build produced this. metal-gauss rows already
            # carry `env` from bench.provenance; the competitors carried
            # nothing, so a row could not say what it had raced against.
            build = competitor_versions().get(
                {"msplat": "msplat", "brush": "brush"}.get(impl, ""))
            if build:
                row["build"] = build
            rows.append(row)
            q = f"{s['psnr']:.3f}" if s.get("psnr") else "score FAILED"
            print(f"  {impl:<12} {n:>6} b={(str(bud//1000)+'k') if bud else 'auto':>5}  {wall/60:6.2f} min  "
                  f"{q:>12} dB  {s.get('n_splats', 0):>9,} splats", flush=True)
            Path(a.out).write_text(json.dumps(
                {"scene": str(scene), "resolution": a.resolution,
                 "msplat_variant": "stock" if a.msplat_stock else "scaled",
                 "rows": rows}, indent=2, default=str))

    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
