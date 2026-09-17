"""Generate README tables from committed result JSONs.

Every headline number in the README has, at some point in this project, been
typed by hand. Three of them went stale without anyone noticing, and one
("materials is an unexplained outlier") survived two plan revisions after the
data that explained it had already been collected. Hand-typed numbers drift.

So the tables are generated. Each one lives between

    <!-- BEGIN:<name> -->  ...  <!-- END:<name> -->

markers in README.md and is rendered here from a JSON under bench/results/.
`--check` regenerates and diffs without writing; it exits non-zero if the
README disagrees with the data, which is what CI runs.

Adding a claim to the README means adding a renderer here. That is the point:
a number with no renderer has no provenance, and a number with no provenance
does not go in the README.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
README = ROOT / "README.md"

# Scene order is fixed so the table is stable across runs; alphabetical would
# reshuffle whenever a scene is added.
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

# 3DGS as published, 30k iterations, for orientation only. Our column is a 7k
# protocol, so the honest reading of the gap is against the measured cost of
# the short protocol (lego, run at both), never against zero.
PUBLISHED_30K = {"chair": 35.8, "drums": 26.1, "ficus": 34.8, "hotdog": 37.7,
                 "lego": 35.8, "materials": 30.0, "mic": 35.4, "ship": 30.9}


def load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"MISSING RESULT: {p}\n"
                         f"  A README table cites data that is not committed.")
    d = json.loads(p.read_text())
    if d.get("provenance") == "incomplete":
        raise SystemExit(
            f"QUARANTINED: {p.name}\n"
            f"  This file has no `resolved` block, so it cannot say which\n"
            f"  settings produced its numbers. Two published tables were wrong\n"
            f"  for exactly that reason. Re-run it through bench/runner.py.")
    return d


def _rows_by_scene(d: dict) -> dict:
    """scene -> list of rows (a scene may be repeated for noise-floor work)."""
    out = {}
    for r in d.get("rows", d.get("scenes", [])):
        if r.get("psnr") is not None:
            out.setdefault(r["scene"], []).append(r)
    return out


def _fmt(x, nd=2, dash="—"):
    return dash if x is None else f"{x:.{nd}f}"


def table_nerf_synthetic(fn: str = "nerf_synthetic_7000_auto_v2.json") -> str:
    """Per-scene PSNR and wall-clock at the shipped defaults.

    The resolved budget is printed per row, not just in a caption. When a
    harness silently forced 300k on every scene, a per-row budget column would
    have shown it on the first line of the first table.
    """
    d = load(fn)
    by = _rows_by_scene(d)
    lines = ["| scene | PSNR | wall | splats | budget | 3DGS @30k | gap |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    psnrs, walls = [], []
    for sc in SCENES:
        rs = by.get(sc)
        if not rs:
            lines.append(f"| {sc} | — | — | — | — | — | — |")
            continue
        pm = sum(r["psnr"] for r in rs) / len(rs)
        wm = sum(r["wall_s"] for r in rs) / len(rs) / 60.0
        res = rs[0].get("resolved", {})
        n = rs[0].get("n_splats")
        rep = f" ×{len(rs)}" if len(rs) > 1 else ""
        pub = PUBLISHED_30K.get(sc)
        lines.append(f"| {sc}{rep} | {_fmt(pm)} | {_fmt(wm, 1)} min | "
                     f"{n:,} | {res.get('budget', '?'):,} | "
                     f"{_fmt(pub, 1)} | {_fmt(pub - pm, 1) if pub else '—'} |")
        psnrs.append(pm); walls.append(wm)
    if psnrs:
        pubs = [PUBLISHED_30K[k] for k in by if k in PUBLISHED_30K]
        mp = sum(pubs) / len(pubs) if pubs else None
        mm = sum(psnrs) / len(psnrs)
        lines.append(f"| **mean** | **{_fmt(mm)}** | "
                     f"**{_fmt(sum(walls), 1)} min total** | | | "
                     f"{_fmt(mp, 1)} | **{_fmt(mp - mm, 1) if mp else '—'}** |")
    env = (d.get("rows") or [{}])[0].get("env", {})
    p_ = d.get("protocol", {})
    lines += ["", f"*{p_.get('steps','?')} steps @ {p_.get('resolution','?')}px, "
                  f"official Blender train/test split, white background, "
                  f"budget `{p_.get('budget','?')}`, "
                  f"num_downscales `{p_.get('num_downscales','?')}`. "
                  f"Built from `{env.get('git','?')}`.*"]
    return "\n".join(lines)


def table_pareto(fn: str = "pareto_all.json") -> str:
    d = load(fn)
    rows = [r for r in d["rows"] if r.get("ok") and r.get("psnr") is not None]
    rows.sort(key=lambda r: r["wall_s"])
    var = d.get("msplat_variant", "?")
    lines = ["| implementation | iters | wall | PSNR | splats |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['impl']} | {r['iters']:,} | {r['wall_s']/60:.2f} min | "
                     f"{r['psnr']:.2f} | {r.get('n_splats') or 0:,} |")
    lines += ["", f"*One evaluator (`bench/compare/score_ply.py`) on the official "
                  f"200-view test split, identical random init, strictly "
                  f"sequential. msplat variant: **{var}**.*"]
    return "\n".join(lines)


def table_margin(fn: str = "sweep8") -> str:
    """Paired per-scene margin, metal-gauss minus each competitor.

    Paired is the point. Each implementation ran the SAME eight scenes, so the
    right uncertainty is the spread of the per-scene difference, not the spread
    of either mean. Scene difficulty dominates the latter (SD 2.4-5.4 dB) and
    would make every series overlap, implying no difference where there is a
    large one.

    95% CI is Student-t with 7 degrees of freedom (t = 2.365).
    """
    import json as _json
    root = RESULTS / fn
    scenes = ["lego", "drums", "ficus", "chair", "hotdog", "materials", "mic", "ship"]

    def best(scene, key, iters):
        keys = ["msplat-stock", "msplat-scaled"] if key == "msplat" else [key]
        vals = []
        for k in keys:
            f = root / f"{scene}_{k}.json"
            if not f.exists():
                continue
            vals += [r["psnr"] for r in _json.loads(f.read_text())["rows"]
                     if r.get("ok") and r.get("psnr") is not None and r["iters"] == iters]
        return max(vals) if vals else None

    out = ["| vs | iters | mean Δ | 95% CI | scenes won |", "|---|---:|---:|---|---:|"]
    for key, label in (("spirula", "spirula-studio"), ("brush", "Brush"),
                       ("msplat", "msplat")):
        for n in (500, 7000, 15000):
            d = []
            for sc in scenes:
                a, b = best(sc, "metal-gauss", n), best(sc, key, n)
                if a is not None and b is not None:
                    d.append(a - b)
            if len(d) != len(scenes):
                continue
            m = sum(d) / len(d)
            var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
            sem = (var / len(d)) ** 0.5
            lo, hi = m - 2.365 * sem, m + 2.365 * sem
            ci = f"[{lo:+.2f}, {hi:+.2f}]"
            if lo <= 0 <= hi:
                ci = f"**{ci}** — includes 0"
            out.append(f"| {label} | {n:,} | {m:+.2f} dB | {ci} | {sum(1 for x in d if x > 0)}/8 |")
    out += ["", "*Per-scene difference across all 8 scenes, 95% CI from Student-t "
                "(df 7). A CI containing zero means the quality margin at that "
                "point is not resolved by 8 scenes, whatever the mean says.*"]
    return "\n".join(out)


def table_budget(fn: str = "sweep8") -> str:
    """Best 8-scene-mean PSNR reachable inside a given wall-clock budget.

    The full ladder (table_pareto_scenes) is 30 interleaved rows, which is
    correct but does not answer the question the section asks -- "given N
    minutes, what is the best reconstruction I can get?" This answers it
    directly: for each budget, the best each implementation manages without
    exceeding it, taking msplat's better variant at each point.
    """
    import json as _json
    root = RESULTS / fn
    scenes = ["lego", "drums", "ficus", "chair", "hotdog", "materials", "mic", "ship"]
    cols = [("metal-gauss", ["metal-gauss"]), ("msplat", ["msplat-stock", "msplat-scaled"]),
            ("Brush", ["brush"]), ("spirula", ["spirula"])]
    budgets = [0.5, 1, 3, 6, 15, 30]

    def rows_for(scene, impl):
        f = root / f"{scene}_{impl}.json"
        if not f.exists():
            return []
        return [r for r in _json.loads(f.read_text())["rows"]
                if r.get("ok") and r.get("psnr") is not None]

    out = ["| you have | " + " | ".join(c for c, _ in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    for b in budgets:
        cells = []
        best_val = None
        vals = []
        for _, impls in cols:
            # Mean over scenes of the best each scene reaches within budget.
            per = []
            for sc in scenes:
                cand = [r["psnr"] for i in impls for r in rows_for(sc, i)
                        if r["wall_s"] <= b * 60]
                if cand:
                    per.append(max(cand))
            v = sum(per) / len(per) if len(per) == len(scenes) else None
            vals.append(v)
        best_val = max([v for v in vals if v is not None], default=None)
        for v in vals:
            cells.append("—" if v is None
                         else (f"**{v:.1f}**" if v == best_val else f"{v:.1f}"))
        label = f"{b:g} min" if b >= 1 else f"{b*60:g} s"
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += ["", "*Best 8-scene-mean PSNR reachable without exceeding each budget; "
                "msplat takes its better variant at each point. Em dash means the "
                "implementation produces nothing within that budget on all 8 scenes. "
                "Full per-rung ladder in [docs/BENCHMARKS.md]"
                "(https://github.com/nandometzger/metal-gauss/blob/main/"
                "docs/BENCHMARKS.md).*"]
    return "\n".join(out)


def table_pareto_scenes(fn: str = "sweep8") -> str:
    """The competitor front averaged over all 8 NeRF-synthetic scenes.

    This replaces a lego-only front. A single scene cannot separate an
    implementation's behaviour from one scene's quirks, and that is exactly what
    made a lego-only front look like a statement about a method.

    msplat's two variants are listed separately rather than merged into a best-of.
    They are genuinely different operating points -- stock stays at quarter
    resolution far longer, so it is 3-6x faster and much weaker -- and collapsing
    them would hide the only region either of them wins.
    """
    import json as _json
    root = RESULTS / fn
    scenes = ["lego", "drums", "ficus", "chair", "hotdog", "materials", "mic", "ship"]
    impls = [("metal-gauss", "metal-gauss"), ("spirula", "spirula-studio"),
             ("brush", "brush"), ("msplat-scaled", "msplat (scaled)"),
             ("msplat-stock", "msplat (stock)")]
    rungs = [500, 1000, 2000, 4000, 7000, 15000]

    def rows_for(scene, impl):
        f = root / f"{scene}_{impl}.json"
        if not f.exists():
            return []
        return [r for r in _json.loads(f.read_text())["rows"]
                if r.get("ok") and r.get("psnr") is not None]

    out = ["| implementation | iters | wall (8-scene mean) | PSNR | SSIM |",
           "|---|---:|---:|---:|---:|"]
    body = []
    for impl, label in impls:
        for n in rungs:
            vals = [(r["wall_s"], r["psnr"], r["ssim"]) for sc in scenes
                    for r in rows_for(sc, impl) if r["iters"] == n
                    and r.get("ssim") is not None]
            if len(vals) < len(scenes):
                continue          # never average a partial set across scenes
            w = sum(v[0] for v in vals) / len(vals)
            q = sum(v[1] for v in vals) / len(vals)
            m = sum(v[2] for v in vals) / len(vals)
            body.append((w, label, n, q, m))
    for w, label, n, q, m in sorted(body):
        out.append(f"| {label} | {n:,} | {w/60:.2f} min | {q:.2f} | {m:.4f} |")

    # Domination is the actual claim; the caption carries it.
    def dom(scene, impl, two_metric):
        """Domination on wall-clock plus PSNR, optionally plus SSIM.

        Both counts are reported. Adding an axis can only make domination
        harder, so quoting the PSNR-only figure alone would overstate it --
        msplat goes from 69% to 52% once SSIM has to hold too.
        """
        mine = rows_for(scene, "metal-gauss")
        theirs = rows_for(scene, impl)
        d = sum(1 for p in theirs
                if any(m["wall_s"] <= p["wall_s"] and m["psnr"] >= p["psnr"]
                       and (not two_metric or m["ssim"] >= p["ssim"])
                       and (m["wall_s"] < p["wall_s"] or m["psnr"] > p["psnr"])
                       for m in mine))
        return d, len(theirs)
    tot = {}
    for two in (False, True):
        for key, label in (("brush", "Brush"), ("spirula", "spirula-studio")):
            a = b = 0
            for sc in scenes:
                x, y = dom(sc, key, two)
                a += x
                b += y
            tot.setdefault(label, {})["ps" if two else "p"] = (a, b)
        ms_a = ms_b = 0
        for sc in scenes:
            for v in ("msplat-stock", "msplat-scaled"):
                x, y = dom(sc, v, two)
                ms_a += x
                ms_b += y
        tot.setdefault("msplat (both variants)", {})["ps" if two else "p"] = (ms_a, ms_b)

    out += ["", "*All 8 NeRF-synthetic scenes, official 200-view test split, one "
                "evaluator, identical seed point cloud per scene, strictly "
                "sequential. metal-gauss dominates (faster **and** better on "
                "PSNR / on PSNR **and** SSIM): "
            + ", ".join(f"**{v['p'][0]}/{v['p'][1]}** / **{v['ps'][0]}/{v['ps'][1]}** of {k}"
                        for k, v in tot.items())
            + ". Single runs per cell; noise floors differ per implementation "
              "AND per scene (see NEGATIVE_RESULTS.md): metal-gauss 0.22 dB, "
              "spirula 0.15-1.27, Brush 0.74, msplat up to 3.35 -- msplat's "
              "counts are indicative only.*"]
    return "\n".join(out)


def table_spirula(fn: str = "pareto_lego_spirula.json") -> str:
    """spirula-studio against our front on lego, on all three metrics.

    Reported as domination rather than a bare ladder, because spirula runs its
    shipped 1M-splat cap while we run 100k: matched ITERATIONS would say
    nothing, and final quality alone would hide that it takes five times as long.

    Both counts are printed on purpose. On PSNR alone we dominate every point.
    On PSNR+SSIM+LPIPS together we do not: at its longest run spirula wins SSIM
    and wins LPIPS clearly, losing only PSNR. Quoting the PSNR-only figure as
    "dominated at 6/6" would be the same single-metric shortcut this repo
    refuses everywhere else, and Brush's 18/18 is a three-metric claim.
    """
    d = load(fn)
    ours = [r for r in load("pareto_lego_metrics.json")["rows"]
            if r.get("impl") == "metal-gauss" and r.get("psnr") is not None]
    rows = sorted([r for r in d["rows"] if r.get("ok") and r.get("psnr") is not None],
                  key=lambda r: r["wall_s"])
    lines = ["| iters | wall | PSNR | SSIM | LPIPS | beaten on all three by |",
             "|---:|---:|---:|---:|---:|---|"]
    dom3 = dom1 = 0
    for r in rows:
        w = r["wall_s"]
        win3 = [o for o in ours if o["wall_s"] <= w and o["psnr"] >= r["psnr"]
                and o["ssim"] >= r["ssim"] and o["lpips"] <= r["lpips"]
                and (o["wall_s"] < w or o["psnr"] > r["psnr"])]
        dom1 += any(o["wall_s"] <= w and o["psnr"] >= r["psnr"]
                    and (o["wall_s"] < w or o["psnr"] > r["psnr"]) for o in ours)
        b = min(win3, key=lambda o: o["wall_s"]) if win3 else None
        dom3 += bool(b)
        cell = (f"{b['iters']:,} it — {b['wall_s']/60:.2f} min"
                if b else "**holds the front**")
        lines.append(f"| {r['iters']:,} | {w/60:.2f} min | {r['psnr']:.2f} | "
                     f"{r['ssim']:.4f} | {r['lpips']:.4f} | {cell} |")
    n = len(rows)
    lines += ["", f"*spirula-studio {d.get('spirula_version','?')}, `{d.get('preset','?')}` preset, "
                  f"shipped defaults (1M splat cap), same seed point cloud, one evaluator on the "
                  f"official 200-view test split. **Single runs on one scene, superseded by "
                  f"the 8-scene front in the README.** Kept because it is the only table here "
                  f"carrying LPIPS. Read its 'holds the front' rows with the correction above in "
                  f"mind: repeating lego gives spirula 33.02 and 31.74 (1.27 dB spread), and the "
                  f"remaining gap is our 100k capacity default rather than its method. Dominated "
                  f"at **{dom1}/{n}** on PSNR, **{dom3}/{n}** on PSNR+SSIM+LPIPS.*"]
    return "\n".join(lines)


def table_raster(fn: str = "stages_latest.json") -> str:
    """Forward and forward+backward raster cost, from bench/quick.py stages."""
    d = load(fn)
    rows = d.get("data", d if isinstance(d, list) else [])
    lines = ["| gaussians | resolution | forward | forward + backward |",
             "|---|---|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["gaussians"], r["res"])):
        lines.append(f"| {r['gaussians']:,} | {r['res']} | "
                     f"{r['fwd_ms']:.1f} ms | **{r['fwd_bwd_ms']:.1f} ms** |")
    env = d.get("env", {})
    lines += ["", f"*tile 16, after a 2 s sustained-load clock ramp. "
                  f"Built from `{env.get('git', '?')}`.*"]
    return "\n".join(lines)


def table_step_profile(fn: str = "step_profile_600000_270.json") -> str:
    """Per-phase cost of one training step."""
    d = load(fn)
    med, total = d["median_ms"], d["total_ms"]
    lines = ["| phase | ms | share |", "|---|---:|---:|"]
    for k, v in sorted(med.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:.1f} | {100 * v / total:.0f} % |")
    lines.append(f"| **total** | **{total:.1f}** | 283 ms when this work started |")
    env = d.get("env", {})
    lines += ["", f"*{d.get('budget', '?'):,} gaussians @ {d.get('W','?')}×"
                  f"{d.get('H','?')}"
                  + (f", {d['scene'].split('/')[-1]}" if d.get('scene') else "")
                  + f". Phases separated by `mps.synchronize()`, so "
                  f"the total is an upper bound: compare phases with it, do not "
                  f"predict wall-clock. **`backward` is the whole autograd pass** "
                  f"— raster, projection/SH and loss backward together — not the "
                  f"raster backward alone. Built from `{env.get('git', '?')}`.*"]
    return "\n".join(lines)


def table_calibration(fn: str = "lego_30k_v2.json",
                      score_fn: str = "lego_30k_v2.score.json") -> str:
    """lego at 30k against published numbers for the algorithm we implement.

    A private scene has no ground truth to be below, so a systematic deficit
    there is invisible. This is the one number that can catch it.
    """
    run, sc = load(fn), load(score_fn)
    ours, res, env = sc["psnr"], run["resolved"], run["env"]
    rows = [("3DGS-MCMC (paper, random init)", 36.01, False),
            ("**metal-gauss**", ours, True),
            ("3DGS baseline (paper, random init)", 35.84, False)]
    rows.sort(key=lambda r: -r[1])
    lines = ["| lego, 30 k steps | PSNR |", "|---|---:|"]
    for name, v, mine in rows:
        lines.append(f"| {name} | {'**' if mine else ''}{v:.2f}{'**' if mine else ''} |")
    lines += ["", f"*{res['budget']:,} splats, {run['metrics']['wall_s'] / 60:.1f} min, "
                  f"scored by `score_ply.py` over {sc['views']} official test views. "
                  f"Built from `{env.get('git', '?')}`.*"]
    return "\n".join(lines)


# Which file each table lives in. The README was split for public release:
# it is a landing page, and the full tables moved to docs/ rather than being
# deleted. A table's marker and its generator have to agree on the file, or
# --check silently passes by finding nothing to compare.
TARGETS = {
    # The lego-only ladder moved to the detailed doc when the headline became
    # the 8-scene aggregate: one scene is an example, not a claim about a method.
    "pareto": "docs/BENCHMARKS.md",
    "nerf-synthetic": "docs/BENCHMARKS.md",
    "raster": "docs/BENCHMARKS.md",
    "step-profile": "docs/BENCHMARKS.md",
    "calibration": "docs/BENCHMARKS.md",
    "spirula": "docs/BENCHMARKS.md",
    "pareto-scenes": "docs/BENCHMARKS.md",
    "budget": "README.md",
    # README shows the forest plot; the exact CIs live in the detailed doc.
    "margin": "docs/BENCHMARKS.md",
}

TABLES = {
    "nerf-synthetic": table_nerf_synthetic,
    "pareto": table_pareto,
    "raster": table_raster,
    "step-profile": table_step_profile,
    "calibration": table_calibration,
    "spirula": table_spirula,
    "pareto-scenes": table_pareto_scenes,
    "budget": table_budget,
    "margin": table_margin,
}


def splice(text: str, name: str, body: str) -> str:
    b, e = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    pat = re.compile(re.escape(b) + r".*?" + re.escape(e), re.S)
    if not pat.search(text):
        raise SystemExit(f"README has no {b} ... {e} block")
    return pat.sub(lambda _: f"{b}\n{body}\n{e}", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="diff only; non-zero exit if a target file is stale")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    names = a.only or list(TABLES)
    stale = []
    for n in names:
        if n not in TABLES:
            raise SystemExit(f"unknown table {n!r}; known: {', '.join(TABLES)}")
        target = ROOT / TARGETS.get(n, "README.md")
        if not target.exists():
            raise SystemExit(f"{n}: target {target} does not exist")
        text = target.read_text()
        new = splice(text, n, TABLES[n]())
        if a.check:
            if new != text:
                stale.append((n, target, text, new))
        else:
            target.write_text(new)

    if a.check:
        if stale:
            import difflib
            for n, target, text, new in stale:
                print(f"STALE: {n} in {target.relative_to(ROOT)}")
                sys.stdout.writelines(difflib.unified_diff(
                    text.splitlines(True), new.splitlines(True),
                    str(target), "regenerated", n=2))
            raise SystemExit(1)
        print(f"tables match their JSONs ({', '.join(names)})")
        return
    print(f"regenerated: {', '.join(names)}")


if __name__ == "__main__":
    main()
