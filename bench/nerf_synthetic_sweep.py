"""NeRF-synthetic sweep: the table every 3DGS paper reports.

Runs all 8 Blender scenes at a fixed iteration count and writes one JSON.
Defaults to 7k iterations, which is the protocol rayanht/msplat publishes
against, so the head-to-head is exact. Published 30k numbers are a DIFFERENT
protocol and must not be mixed into the same column -- lego at 30k is kept
separately as the calibration anchor.

Every row carries the trainer's own `resolved` config. That is not decoration:
this script previously declared `--budget default=300_000` and forwarded it to
every child, so `auto_budget()` never ran in any sweep it produced, and a table
committed as "old defaults vs new defaults" actually held budget fixed in both
arms. The harness recorded its own arguments as the protocol and nothing
disagreed. Now the trainer states what it did and bench/runner.py refuses to
let that differ silently from what was asked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.runner import RunDiverged, RunFailed, run  # noqa: E402

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def _write(out: Path, args, rows: list, t_all: float) -> tuple:
    """Write after every scene, not only at the end.

    A sweep that only writes on completion loses eight scenes to a crash in the
    eighth. The log is not a substitute: it has no resolved config.
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    by_scene = {}
    for r in ok:
        by_scene.setdefault(r["scene"], []).append(r["psnr"])
    per_scene = {k: sum(v) / len(v) for k, v in by_scene.items()}
    mean = sum(per_scene.values()) / len(per_scene) if per_scene else None
    total = time.perf_counter() - t_all
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": {"steps": args.steps,
                     "budget": args.budget or "auto",
                     "num_downscales": ("default" if args.num_downscales is None
                                        else args.num_downscales),
                     "resolution": args.resolution,
                     "repeats": args.repeats,
                     "antialias": bool(args.antialias),
                     "selective_adam": ("default" if args.selective_adam is None
                                        else args.selective_adam),
                     "selective_adam_groups": args.selective_adam_groups or "default",
                     "selective_adam_steps": args.selective_adam_steps or "default",
                     "densify_weight": args.densify_weight or "default",
                     "split": "official Blender train/test",
                     "background": "white"},
        "complete": len(per_scene) == len(args.scenes),
        "mean_psnr": mean, "per_scene_psnr": per_scene,
        "total_wall_s": round(total, 1), "scenes": rows, "rows": rows,
    }, indent=2, default=str))
    return per_scene, by_scene, mean, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "nerf_synthetic"))
    ap.add_argument("--steps", type=int, default=7000)
    # 0 = do not override; let auto_budget() choose and let the report say what
    # it chose. A harness-side numeric default here is the original bug.
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--num-downscales", type=int, default=None,
                    help="0 disables coarse-to-fine; unset = trainer default")
    ap.add_argument("--antialias", action="store_true",
                    help="pass --antialias to the trainer")
    ap.add_argument("--filter-3d", action="store_true",
                    help="pass --filter-3d to the trainer")
    ap.add_argument("--selective-adam", action=argparse.BooleanOptionalAction, default=None,
                    help="pass --[no-]selective-adam to the trainer; unset = its default")
    ap.add_argument("--selective-adam-groups", default=None,
                    help="pass --selective-adam-groups to the trainer")
    ap.add_argument("--selective-adam-steps", default=None,
                    help="pass --selective-adam-steps to the trainer")
    ap.add_argument("--densify-weight", default=None,
                    help="pass --densify-weight to the trainer")
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per scene; >1 for scenes near the noise floor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else RESULTS / f"nerf_synthetic_{args.steps}.json"
    rows, t_all = [], time.perf_counter()

    for scene in args.scenes:
        d = Path(args.data) / scene
        if not (d / "transforms_train.json").exists():
            print(f"  {scene}: MISSING, skipped", flush=True)
            rows.append({"scene": scene, "status": "missing"})
            continue

        spec = {"blender": str(d), "steps": args.steps,
                "max_resolution": args.resolution, "eval_every": args.steps}
        if args.budget:
            spec["budget"] = args.budget
        if args.num_downscales is not None:
            spec["num_downscales"] = args.num_downscales
        if args.antialias:
            spec["antialias"] = True
        if args.filter_3d:
            spec["filter_3d"] = True
        if args.selective_adam is not None:
            spec["selective_adam"] = args.selective_adam
        if args.selective_adam_groups is not None:
            spec["selective_adam_groups"] = args.selective_adam_groups
        if args.selective_adam_steps is not None:
            spec["selective_adam_steps"] = args.selective_adam_steps
        if args.densify_weight is not None:
            spec["densify_weight"] = args.densify_weight

        for rep_i in range(args.repeats):
            try:
                rep = run(spec)
            except (RunFailed, RunDiverged) as e:
                print(f"  {scene:<10} FAILED: {e}", flush=True)
                rows.append({"scene": scene, "status": "failed",
                             "error": str(e)[:500]})
                continue
            m, res = rep["metrics"], rep["resolved"]
            tag = "" if args.repeats == 1 else f" [{rep_i + 1}/{args.repeats}]"
            print(f"  {scene:<10} {m['psnr']:6.2f} dB   "
                  f"{rep['harness_wall_s'] / 60:5.1f} min   "
                  f"{m['n_splats']:>7,} splats  "
                  f"budget={res['budget']:,} nd={res['num_downscales']}{tag}",
                  flush=True)
            # Two clocks, and they are not interchangeable. metrics.wall_s is
            # the trainer's own timer, started AFTER scene loading, parameter
            # init and runtime Metal kernel compilation -- roughly 50 s of work
            # on an 800px Blender scene. harness_wall_s is the whole process,
            # which is what a user waits and what every previously published
            # figure here measured. Publishing one against the other would
            # invent a speedup out of a definition.
            rows.append({"scene": scene, "status": "ok", "repeat": rep_i,
                         "psnr": m["psnr"],
                         "wall_s": rep["harness_wall_s"],
                         "train_only_wall_s": m["wall_s"],
                         "n_splats": m["n_splats"],
                         "resolved": res, "env": rep["env"]})

            _write(out, args, rows, t_all)

    per_scene, by_scene, mean, total = _write(out, args, rows, t_all)

    if mean is not None:
        print(f"\n  MEAN over {len(per_scene)} scenes: {mean:.2f} dB")
        spread = {k: round(max(v) - min(v), 2) for k, v in by_scene.items()
                  if len(v) > 1}
        if spread:
            print(f"  within-scene spread: {spread}")
    else:
        print("\n  no scenes completed")
    print(f"  total {total / 60:.1f} min")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
