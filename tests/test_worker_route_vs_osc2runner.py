"""Does the sensor worker hand a policy the same route osc2runner would?

    /scratch/veerk41/venvs/tfv6/bin/python tests/test_worker_route_vs_osc2runner.py

The reference is osc2runner's own observation path, imported the way osc2runner
imports itself: `osc2carla.backend.route.RoutePlan.ahead` for the slice and
`scenario_orchestration/carla_state_obs.py`'s `EgoFrame.to_ego` and ROUTE_*
constants for the frame and sampling. The worker loads route.py by path and has
its own frame transform and its own resampling of ProSim's lane-graph route, so
the two are compared point for point over pose sequences -- one fresh plan each,
called in the same order, because progress is stateful.

Poses: the PlanT 2.0 left_turn run of 2026-09-14 that over-rotated to -106 deg
inside the junction and was then handed the approach it had already driven
(fixtures/left_turn_overrotation.json), a synthetic straight-through run, and,
when the harness results are present, every recorded prosim_carla cell. The
control is the worker's previous route function (OLD below): it must DISAGREE
with osc2runner on the over-rotation, or this test could not have caught the bug.
No CARLA server, no GPU; needs the `carla` client package (osc2runner imports it).
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import glob
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

import sensor_worker as W

O = Path(W.DEFAULT_OSC2RUNNER)
FIXTURE = Path("tests/fixtures/left_turn_overrotation.json")
RESULTS = Path("/scratch/veerk41/scenario_orchestration/results/raw_prosim_eval")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def banner(t):
    print(f"\n=== {t} ===")


def load_reference():
    """osc2runner's carla_state_obs, with osc2runner's own package on the path.
    Loaded by file: this repo has a `scenario_orchestration` package too."""
    sys.path.insert(0, str(O))
    path = O / "scenario_orchestration" / "carla_state_obs.py"
    spec = importlib.util.spec_from_file_location("_ref_carla_state_obs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def OLD(route_world, x, y, yaw_rad):
    """sensor_worker.route_in_ego_frame before 2026-09-14: every point in front
    of the car, from the whole frozen route, arc length measured from the ego."""
    if route_world is None or len(route_world) < 2:
        return []
    d = np.asarray(route_world, dtype=float)[:, :2] - np.array([x, y])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    local = np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], axis=1)
    ahead = local[local[:, 0] > 0.0]
    if len(ahead) < 2:
        return []
    pts = np.concatenate([np.zeros((1, 2)), ahead])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    want = np.clip(2.5 + 1.0 * np.arange(20), 0.0, arc[-1])
    return np.stack([np.interp(want, arc, pts[:, 0]), np.interp(want, arc, pts[:, 1])], axis=1).tolist()


def to_world(route, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + fx * c - fy * s, y + fx * s + fy * c) for fx, fy in route]


def arc_of(route_world, px, py):
    """Arc length along the ORIGINAL polyline of its point nearest (px, py)."""
    pts = np.asarray(route_world, dtype=float)[:, :2]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return float(s[int(np.argmin(np.hypot(pts[:, 0] - px, pts[:, 1] - py)))])


def main():
    REF = load_reference()
    RM = W.load_route_module(str(O))

    def ref_route(plan, x, y, yaw):
        frame = REF.EgoFrame(x=x, y=y, z=0.0, yaw_rad=yaw)
        return [list(frame.to_ego(px, py, None)[:2])
                for px, py, _h in plan.ahead(x, y, first_m=REF.ROUTE_FIRST_M,
                                             step_m=REF.ROUTE_STEP_M, count=REF.ROUTE_POINTS)]

    def run_both(route_world, poses):
        """(worst |worker - reference|, worker routes) over one pose sequence.
        The plan starts at the first pose, as the worker's does at the spawn."""
        wplan = W.plan_from_route_world(RM, route_world, start_xy=poses[0][:2])
        rplan = REF.route_plan.RoutePlan(points=list(wplan.points), step_m=wplan.step_m)
        worst, routes = 0.0, []
        for x, y, yaw in poses:
            a = np.asarray(W.route_in_ego_frame(wplan, x, y, yaw), dtype=float).reshape(-1, 2)
            b = np.asarray(ref_route(rplan, x, y, yaw), dtype=float).reshape(-1, 2)
            if a.shape != b.shape:
                return math.inf, routes
            if len(a):
                worst = max(worst, float(np.abs(a - b).max()))
            routes.append(a.tolist())
        return worst, routes

    # ---------------------------------------------------------------- 1
    banner("1. same sampling, same plan geometry")
    check("worker's ROUTE_FIRST_M/STEP_M/POINTS are carla_state_obs's",
          (W.ROUTE_FIRST_M, W.ROUTE_STEP_M, W.ROUTE_POINTS)
          == (REF.ROUTE_FIRST_M, REF.ROUTE_STEP_M, REF.ROUTE_POINTS),
          f"{(W.ROUTE_FIRST_M, W.ROUTE_STEP_M, W.ROUTE_POINTS)} vs "
          f"{(REF.ROUTE_FIRST_M, REF.ROUTE_STEP_M, REF.ROUTE_POINTS)}")
    check("the worker loads the same route.py osc2runner imports",
          Path(RM.__file__).resolve() == Path(REF.route_plan.__file__).resolve(), RM.__file__)
    fx = json.loads(FIXTURE.read_text())
    route_world = np.asarray(fx["route_world"], dtype=float)
    plan = W.plan_from_route_world(RM, route_world)
    xy = np.asarray([(p[0], p[1]) for p in plan.points])
    spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    seg_a, seg_b = route_world[:-1], route_world[1:]
    def dist_to_polyline(p):
        ab = seg_b - seg_a
        t = np.clip(((p - seg_a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        return float(np.min(np.linalg.norm(seg_a + t[:, None] * ab - p, axis=1)))
    off = max(dist_to_polyline(p) for p in xy)
    check("resampled plan: consecutive points <= 1 m apart, >= 0.95 m on straights",
          spacing.max() <= 1.0 + 1e-9 and np.median(spacing) > 0.95,
          f"spacing min {spacing.min():.3f} median {np.median(spacing):.3f} max {spacing.max():.3f}")
    check("resampled plan: every point lies ON ProSim's polyline",
          off < 1e-6, f"max distance {off:.2e} m")
    check("resampled plan: same length as ProSim's route (within one step)",
          abs((len(xy) - 1) * plan.step_m - arc_of(route_world, *route_world[-1])) <= plan.step_m,
          f"{(len(xy) - 1) * plan.step_m:.1f} m vs {arc_of(route_world, *route_world[-1]):.1f} m")

    # ---------------------------------------------------------------- 2
    banner("2. the PlanT 2.0 left_turn over-rotation (real route, real poses)")
    poses_t = [(t, x, y, math.radians(h)) for t, x, y, h in fx["poses_t_x_y_hdeg"]]
    poses = [(x, y, h) for _t, x, y, h in poses_t]
    worst, routes = run_both(route_world, poses)
    check("worker == osc2runner at every pose", worst < 1e-9,
          f"max |worker - osc2runner| {worst:.2e} m over {len(poses)} poses")
    old_worst = 0.0
    for (x, y, h), r in zip(poses, routes):
        a = np.asarray(OLD(route_world, x, y, h)).reshape(-1, 2)
        b = np.asarray(r).reshape(-1, 2)
        old_worst = math.inf if a.shape != b.shape else max(old_worst, float(np.abs(a - b).max()) if len(a) else 0.0)
    check("CONTROL: the OLD worker route disagrees with osc2runner here",
          old_worst > 1.0, f"max |old - osc2runner| {old_worst:.1f} m")
    forward_ok, old_back = True, []
    for (t, x, y, h), r in zip(poses_t, routes):
        if t < 4.8 or not r:
            continue
        ego_s = arc_of(route_world, x, y)
        far = to_world([r[-1]], x, y, h)[0]
        forward_ok &= arc_of(route_world, *far) > ego_s
        old = OLD(route_world, x, y, h)
        if old:
            old_back.append(arc_of(route_world, *to_world([old[-1]], x, y, h)[0]) < ego_s)
    check("from 4.8 s on, the far route point is always FURTHER along the plan than the ego",
          forward_ok)
    check("CONTROL: the OLD route pointed BACK along the plan at those poses",
          old_back and all(old_back), f"{sum(old_back)}/{len(old_back)} poses")

    # ---------------------------------------------------------------- 3
    banner("3. straight through a junction the plan turns at (synthetic)")
    arc = np.linspace(math.pi / 2, 0.0, 32)
    L_PLAN = np.concatenate([
        np.stack([np.arange(0.0, 40.0, 0.5), np.zeros(80)], axis=1),
        np.stack([40.0 + 10.0 * np.cos(arc), -10.0 + 10.0 * np.sin(arc)], axis=1),
        np.stack([np.full(100, 50.0), -10.0 - 0.5 * np.arange(1, 101)], axis=1)])
    thru = [(float(x), 0.0, 0.0) for x in np.arange(0.0, 71.0, 1.0)]
    worst, routes = run_both(L_PLAN, thru)
    check("worker == osc2runner driving straight past the turn", worst < 1e-9,
          f"max {worst:.2e} m, final route {len(routes[-1])} points")
    check("CONTROL: the OLD worker route was empty there",
          OLD(L_PLAN, 70.0, 0.0, 0.0) == [], str(len(OLD(L_PLAN, 70.0, 0.0, 0.0))))

    banner("3b. a ProSim route that starts far behind the ego (overtake's is 72 m)")
    far = np.stack([np.arange(-100.0, 100.0, 0.5), np.zeros(400)], axis=1)
    ego = (0.0, 0.3, 0.0)
    cut = W.plan_from_route_world(RM, far, start_xy=ego[:2])
    check("the plan is cut at the spawn's projection, where osc2runner's begins",
          abs(cut.points[0][0]) < 1e-9 and abs(cut.points[0][1]) < 1e-9,
          f"plan starts at {np.round(cut.points[0][:2], 3).tolist()}")
    spawn_plan = REF.route_plan.RoutePlan(
        points=[(float(x), 0.0, 0.0) for x in np.arange(0.0, 99.6, 1.0)], step_m=1.0)
    a = np.asarray(W.route_in_ego_frame(cut, *ego))
    b = np.asarray(ref_route(spawn_plan, *ego))
    check("worker == osc2runner's own plan walked from the spawn",
          a.shape == b.shape and float(np.abs(a - b).max()) < 1e-9,
          f"first {np.round(a[0], 3).tolist()} vs {np.round(b[0], 3).tolist()}")
    uncut = W.route_in_ego_frame(W.plan_from_route_world(RM, far), *ego)
    check("CONTROL: uncut, the 60-point search window stops short and the route starts BEHIND the ego",
          bool(uncut) and uncut[0][0] < 0.0, f"first point {np.round(uncut[0], 2).tolist()}")

    # ---------------------------------------------------------------- 4
    banner("4. every recorded prosim_carla cell (skipped without results)")
    cells = sorted(glob.glob(str(RESULTS / "*__prosim_carla__*_s000")))
    if not cells:
        print("  (no results directory; skipped)")
    worst_all, n_poses, changed = 0.0, 0, []
    for cell in cells:
        meta = json.loads(Path(cell, "rollout.meta.json").read_text())
        rw = np.asarray(meta.get("ego_route_polyline") or [], dtype=float)
        rows = [json.loads(l) for l in open(Path(cell, "states.jsonl"))]
        ego = min(rows[0]["a"], key=int)
        ps = [(r["a"][ego][0], r["a"][ego][1], math.radians(r["a"][ego][2])) for r in rows]
        worst, routes = run_both(rw, ps)
        worst_all, n_poses = max(worst_all, worst), n_poses + len(ps)
        d_old, n_len = 0.0, 0
        for p, r in zip(ps, routes):
            o = OLD(rw, *p)
            if len(o) != len(r):
                n_len += 1
            elif r:
                d_old = max(d_old, float(np.abs(np.asarray(o) - np.asarray(r)).max()))
        changed.append((Path(cell).name, d_old, n_len))
    if cells:
        check("worker == osc2runner on every recorded pose", worst_all < 1e-9,
              f"max {worst_all:.2e} m over {n_poses} poses in {len(cells)} cells")
        print("  informative -- how far the NEW route is from what those runs were given:")
        for name, d, n_len in changed:
            print(f"    {name:44s} max |new - old| {d:.3f} m, point count differs at {n_len} poses")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
