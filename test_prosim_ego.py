"""Tests for prosim_ego.py -- the trajdata/ProSim adapter.

Everything except the ProSim model call itself is exercised:
  1. frame conversion round-trips
  2. VecMapLaneGraph against the REAL cached Town10HD map
  3. heading-aware lane snapping (must not pick the oncoming lane)
  4. a full IDM + pure-pursuit rollout on the real map from a real recorded
     start pose, scored with the same lane-distance metric used for ProSim

(4) is the one that matters: it answers "does a rule-based ego stay on the road
in Town10HD?" without needing the checkpoint, Llama, or a GPU.

    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 test_prosim_ego.py"
"""

import sys

import numpy as np

sys.path.insert(0, "/scratch/veerk41/ProSim")

from ego_control import IDM, VehicleState, make_policy, wrap_angle
from prosim_ego import VecMapLaneGraph, local_to_world, world_to_local

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def section(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


# ---------------------------------------------------------------------------
section("1. frame conversion")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(2000):
    init_pos = rng.uniform(-200, 200, 2)
    init_h = rng.uniform(-np.pi, np.pi)
    x, y, h = rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(-np.pi, np.pi)
    lx, ly, lh = world_to_local(x, y, h, init_pos, init_h)
    x2, y2, h2 = local_to_world((lx, ly), lh, init_pos, init_h)
    worst = max(worst, abs(x2 - x), abs(y2 - y), abs(wrap_angle(h2 - h)))
check("world -> local -> world round-trips exactly", worst < 1e-9, f"worst err={worst:.2e}")

# a known case: 90-degree frame, unit step forward
lx, ly, lh = world_to_local(0.0, 1.0, np.pi / 2, np.array([0.0, 0.0]), np.pi / 2)
check("90-degree frame: +y world is +x local",
      abs(lx - 1.0) < 1e-9 and abs(ly) < 1e-9 and abs(lh) < 1e-9,
      f"local=({lx:.3f}, {ly:.3f}, {lh:.3f})")


# ---------------------------------------------------------------------------
section("2. VecMapLaneGraph on the real Town10HD map")
# ---------------------------------------------------------------------------
from pathlib import Path

from trajdata.maps.map_api import MapAPI

api = MapAPI(Path("demo_dataset/trajdata_cache"))
vm = api.get_map("carla_town10hd:carla_town10hd_0", None, incl_road_lanes=True)
g = VecMapLaneGraph(vm)
print(f"  loaded {vm.map_id}: {len(vm.lanes)} lanes")

lane0 = vm.lanes[0]
c = g.centerline(lane0.id)
check("centerline returns an (N,2) array", c.ndim == 2 and c.shape[1] == 2, f"{c.shape}")
check("centerline is cached (same object on re-query)",
      g.centerline(lane0.id) is c)

n_succ = sum(len(g.successors(l.id)) for l in vm.lanes)
n_adj = sum(len(g.adjacent(l.id)) for l in vm.lanes)
check("successors are exposed", n_succ > 0, f"{n_succ} total")
check("adjacent lanes are exposed", n_adj > 0, f"{n_adj} total")
check("all successor ids resolve",
      all(vm.get_road_lane(s) is not None for l in vm.lanes for s in g.successors(l.id)))


# ---------------------------------------------------------------------------
section("3. heading-aware lane snapping")
# ---------------------------------------------------------------------------
# Take a real lane, stand on its centre, and query with the correct heading and
# with the reverse. Purely spatial snapping would return the same lane for both.
probe = None
for lane in vm.lanes:
    c = g.centerline(lane.id)
    if len(c) >= 6:
        probe = (lane, c)
        break
lane, c = probe
i = len(c) // 2
h = float(np.arctan2(c[i + 1, 1] - c[i - 1, 1], c[i + 1, 0] - c[i - 1, 0]))

got = g.closest_lane(float(c[i, 0]), float(c[i, 1]), h)
check("snaps to the lane we are standing on, with matching heading",
      got == lane.id, f"got {got}, expected {lane.id}")

got_rev = g.closest_lane(float(c[i, 0]), float(c[i, 1]), wrap_angle(h + np.pi))
check("reversed heading does NOT return the same lane",
      got_rev != lane.id, f"got {got_rev}")

n_ok = 0
n_try = 0
for lane in vm.lanes[:60]:
    c = g.centerline(lane.id)
    if len(c) < 6:
        continue
    i = len(c) // 2
    hh = float(np.arctan2(c[i + 1, 1] - c[i - 1, 1], c[i + 1, 0] - c[i - 1, 0]))
    n_try += 1
    if g.closest_lane(float(c[i, 0]), float(c[i, 1]), hh) == lane.id:
        n_ok += 1
check("snapping is reliable across many lanes", n_ok / max(n_try, 1) > 0.9,
      f"{n_ok}/{n_try} correct")


# ---------------------------------------------------------------------------
section("4. rule-based ego driving on the real Town10HD map")
# ---------------------------------------------------------------------------
import pandas as pd

lane_pts = np.concatenate([g.centerline(l.id) for l in vm.lanes], axis=0)


def lane_dist(x, y):
    return float(np.linalg.norm(lane_pts - np.array([x, y]), axis=1).min())


src = pd.read_csv("/scratch/veerk41/history_20hz.csv")
starts = []
for aid, grp in src.groupby("id"):
    grp = grp.sort_values("frame")
    r0, r1 = grp.iloc[10], grp.iloc[11]
    speed = float(np.hypot(r0["vx"], r0["vy"]))
    if speed < 2.0:
        continue
    starts.append((int(aid), float(r0["x"]), float(r0["y"]), float(r0["yaw"]),
                   speed, float(r0["length"]), float(r0["width"])))
print(f"  {len(starts)} recorded agents moving >2 m/s at scene_ts=10")

results = []
for aid, x, y, yaw, speed, L, W in starts:
    pol = make_policy("idm_pursuit", dt=0.1, lane_graph=g, v0=max(speed, 6.0))
    st = VehicleState(x=x, y=y, heading=yaw, speed=speed, length=L, width=W)
    traj = pol.rollout(st, 80)                       # 8 s, same horizon as ProSim
    d = [lane_dist(p.x, p.y) for p in traj]
    results.append((aid, d[0], max(d), np.mean(d),
                    float(np.hypot(traj[-1].x - x, traj[-1].y - y))))

d0 = np.array([r[1] for r in results])
dmax = np.array([r[2] for r in results])
dmean = np.array([r[3] for r in results])
travel = np.array([r[4] for r in results])

print(f"  start offset : median={np.median(d0):.2f} m  max={d0.max():.2f} m")
print(f"  max dev      : median={np.median(dmax):.2f} m  p95={np.percentile(dmax,95):.2f} m"
      f"  worst={dmax.max():.2f} m")
print(f"  mean dev     : median={np.median(dmean):.2f} m")
print(f"  displacement : median={np.median(travel):.1f} m over 8 s")
print(f"  ever >3 m off: {100*np.mean(dmax>3):.0f}%      ever >6 m off: {100*np.mean(dmax>6):.0f}%")

check("every rule-ego rollout stays finite",
      all(np.isfinite([r[2] for r in results])))
check("rule ego actually moves (median displacement > 15 m)",
      np.median(travel) > 15.0, f"{np.median(travel):.1f} m")
check("rule ego stays on the road: median max-deviation < 2 m",
      np.median(dmax) < 2.0, f"median={np.median(dmax):.2f} m")
check("rule ego: no agent ever more than 6 m off-lane",
      np.mean(dmax > 6) == 0.0, f"{100*np.mean(dmax>6):.0f}%")
check("rule ego beats ProSim's measured 20% >3 m off-lane",
      np.mean(dmax > 3) < 0.20, f"{100*np.mean(dmax>3):.0f}%")

# collision-avoidance in the real scene: drive one agent with the others as
# static obstacles at their recorded t=10 poses, and check it never overlaps.
others = [VehicleState(x=s[1], y=s[2], heading=s[3], speed=s[4],
                       length=s[5], width=s[6]) for s in starts[1:]]
aid, x, y, yaw, speed, L, W = starts[0]
pol = make_policy("idm_pursuit", dt=0.1, lane_graph=g, v0=max(speed, 6.0))
traj = pol.rollout(VehicleState(x, y, yaw, speed, L, W), 80,
                   neighbors_fn=lambda i: others)
min_sep = min(min(np.hypot(p.x - o.x, p.y - o.y) for o in others) for p in traj)
check("keeps a positive separation from other agents", min_sep > 1.0,
      f"min centre-to-centre={min_sep:.2f} m")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("   FAILED:", f)
print("=" * 74)
sys.exit(1 if FAIL else 0)
