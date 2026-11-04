"""Mip-NeRF 360 sweep: the real-scene table 3DGS papers report.

NeRF-synthetic is objects on white; the only real capture here, room1, is
private. Mip-NeRF 360 is public, unbounded and real, and it is the scene set
whose numbers other implementations publish.

The protocol is 3DGS's: outdoor scenes train on images_4, indoor scenes on
images_2, and every 8th image is held out (the trainer's default split). The
COLMAP models ship undistorted (PINHOLE) at full resolution; the loader scales
the intrinsics to whichever image folder it reads.

Stage the data under data/mipnerf360/<scene>/ (see docs/BENCHMARKS.md):

    curl -LO http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
    curl -LO https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
    unzip 360_v2.zip -d data/mipnerf360; unzip 360_extra_scenes.zip -d data/mipnerf360

As in nerf_synthetic_sweep.py, the harness sets no training defaults of its
own: an unset knob is the trainer's, and each row records what the trainer
resolved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.runner import RunDiverged, RunFailed, run  # noqa: E402

OUTDOOR = ["bicycle", "flowers", "garden", "stump", "treehill"]
INDOOR = ["room", "counter", "kitchen", "bonsai"]
SCENES = OUTDOOR + INDOOR
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def image_dir(scene: str) -> str:
    """3DGS's resolution per scene: a quarter outdoors, half indoors."""
    if scene in OUTDOOR:
        return "images_4"
    if scene in INDOOR:
        return "images_2"
    raise ValueError(f"unknown Mip-NeRF 360 scene {scene!r}")


def _write(out: Path, args, rows: list, t_all: float) -> tuple:
    """Write after every scene, so a crash in the ninth keeps the other eight."""
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
                     "resolution": args.resolution,
                     "repeats": args.repeats,
                     "images": {s: image_dir(s) for s in args.scenes},
                     "selective_adam": ("default" if args.selective_adam is None
                                        else args.selective_adam),
                     "split": "every 8th image held out (trainer default)"},
        "complete": len(per_scene) == len(args.scenes),
        "mean_psnr": mean, "per_scene_psnr": per_scene,
        "total_wall_s": round(total, 1), "rows": rows,
    }, indent=2, default=str))
    return per_scene, mean, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "mipnerf360"))
    ap.add_argument("--steps", type=int, default=7000)
    # 0 = let the trainer choose. auto_budget() was tuned on lego and gives
    # 100k at 7k steps, far below what an unbounded scene needs: pass --budget.
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=1600,
                    help="max side; images_4 and images_2 are all below 1600")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--selective-adam", action=argparse.BooleanOptionalAction, default=None,
                    help="pass --[no-]selective-adam to the trainer; unset = its default")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for s in args.scenes:
        image_dir(s)                    # fail on a typo before hours of training

    out = Path(args.out) if args.out else RESULTS / f"mipnerf360_{args.steps}.json"
    rows, t_all = [], time.perf_counter()

    for scene in args.scenes:
        d = Path(args.data) / scene
        images = d / image_dir(scene)
        if not (d / "sparse" / "0").is_dir() or not images.is_dir():
            print(f"  {scene}: MISSING, skipped", flush=True)
            rows.append({"scene": scene, "status": "missing"})
            continue

        spec = {"colmap": str(d / "sparse" / "0"), "images": str(images),
                "steps": args.steps, "max_resolution": args.resolution,
                "eval_every": args.steps}
        if args.budget:
            spec["budget"] = args.budget
        if args.selective_adam is not None:
            spec["selective_adam"] = args.selective_adam

        for rep_i in range(args.repeats):
            try:
                rep = run(spec)
            except (RunFailed, RunDiverged) as e:
                print(f"  {scene:<10} FAILED: {e}", flush=True)
                rows.append({"scene": scene, "status": "failed", "error": str(e)[:500]})
                continue
            m, res = rep["metrics"], rep["resolved"]
            tag = "" if args.repeats == 1 else f" [{rep_i + 1}/{args.repeats}]"
            print(f"  {scene:<10} {m['psnr']:6.2f} dB   "
                  f"{rep['harness_wall_s'] / 60:5.1f} min   "
                  f"{m['n_splats']:>9,} splats  budget={res['budget']:,}{tag}",
                  flush=True)
            # Same two clocks as nerf_synthetic_sweep.py: wall_s is the whole
            # process, train_only_wall_s the trainer's own timer.
            rows.append({"scene": scene, "status": "ok", "repeat": rep_i,
                         "psnr": m["psnr"],
                         "wall_s": rep["harness_wall_s"],
                         "train_only_wall_s": m["wall_s"],
                         "n_splats": m["n_splats"],
                         "resolved": res, "env": rep["env"]})
            _write(out, args, rows, t_all)

    per_scene, mean, total = _write(out, args, rows, t_all)
    if mean is not None:
        print(f"\n  MEAN over {len(per_scene)} scenes: {mean:.2f} dB")
    else:
        print("\n  no scenes completed")
    print(f"  total {total / 60:.1f} min")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
