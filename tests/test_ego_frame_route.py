"""Regression tests for the two rule-ego bugs found 2026-09-07.

Needs the cached CARLA batch (for the real Town10HD lane graph) but NOT the
model, a GPU, or CARLA.

  1. FRAME. a_traj is in the ego-CENTRED frame; the VectorMap is in WORLD
     coordinates. prosim_ego.py applied only local->centred and then queried the
     world map, so it drove by lanes ~85 m from where the car actually was.

  2. ROUTE. LaneGraphRoute judged each junction branch by its first two points
     -- 0.5 m of lane, where every branch of a junction looks identical -- and
     then re-snapped with closest_lane() every step, discarding the plan inside
     the junction box.

    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 tests/test_ego_frame_route.py"
"""

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import json
import math
import sys

import numpy as np


from carla_dataset import register

register()

from torch.utils.data import DataLoader

from ego_control import LaneGraphRoute, VehicleState, make_policy, wrap_angle
from prosim.config.default import get_config
from prosim.core.registry import registry
from prosim_ego import (VecMapLaneGraph, centred_to_world_pose,
                        world_to_centred_pose)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def section(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


_DATA_DIR = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))  # recordings live one level above the repo
cfg = get_config("prosim_demo/cfg/waymo_demo.yaml", cluster="local")
cfg.defrost()
cfg.DATASET.SOURCE.TRAIN = ["carla_town10hd"]   # the yaml ships waymo_train
# data path keyed by whatever source is set, so switching town is one line
cfg.DATASET.DATA_PATHS[cfg.DATASET.SOURCE.TRAIN[0].upper()] = _DATA_DIR
cfg.PROMPT.CONDITION.TYPES = ["goal", "llm_text_OneText"]
cfg.freeze()
ds = registry.get_dataset(cfg.DATASET.TYPE)(cfg, "train")
ds._data_index = [ds._data_index[8]]
ds._data_len = 1
for batch in DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=ds.get_collate_fn(), num_workers=0):
    break

lg = VecMapLaneGraph(batch.vector_maps[0])
LANES = json.load(open(__import__("os").path.join(_DATA_DIR, "town10hd_lanes.json")))["lanes"]
PTS = np.concatenate([np.asarray(l["center"], float)[:, :2] for l in LANES.values()])


def lane_dist(p):
    return float(np.linalg.norm(PTS - np.asarray(p, float)[:2], axis=1).min())


# ---------------------------------------------------------------------------
section("1. centred <-> world round-trips")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(3000):
    centre = (rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(-np.pi, np.pi))
    p = (rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(-np.pi, np.pi))
    back = centred_to_world_pose(*world_to_centred_pose(*p, centre), centre)
    worst = max(worst, abs(back[0] - p[0]), abs(back[1] - p[1]),
                abs(wrap_angle(back[2] - p[2])))
check("world -> centred -> world is exact", worst < 1e-9, f"worst={worst:.2e}")


# ---------------------------------------------------------------------------
section("2. DECISIVE: the two frames are far apart, and only one is the map's")
# ---------------------------------------------------------------------------
centre_arr = batch.centered_agent_state.as_format("x,y,z,h").cpu().numpy()[0]
centre = (float(centre_arr[0]), float(centre_arr[1]), float(centre_arr[3]))
iop = batch.extras["io_pairs_batch"]
names = list(batch.extras["prompt"]["motion_pred"]["agent_ids"][0])
print(f"  centre (world): ({centre[0]:.2f}, {centre[1]:.2f}) h={centre[2]:+.4f} rad")

seps, d_world, d_centred = [], [], []
for i, nm in enumerate(names):
    p = iop["position"][0, 0, i].numpy()
    h = float(iop["heading"][0, 0, i].reshape(-1)[0])
    wx, wy, _ = centred_to_world_pose(float(p[0]), float(p[1]), h, centre)
    seps.append(math.hypot(wx - p[0], wy - p[1]))
    d_world.append(lane_dist((wx, wy)))
    d_centred.append(lane_dist(p))
    print(f"  {nm!r:6}: centred ({p[0]:8.2f},{p[1]:8.2f}) -> world "
          f"({wx:8.2f},{wy:8.2f})   lane dist  world {d_world[-1]:6.2f} m | "
          f"centred {d_centred[-1]:6.2f} m")

check("the two frames really are far apart for this scene",
      min(seps) > 10.0, f"min separation {min(seps):.2f} m")
check("agents sit ON a lane in WORLD coords", max(d_world) < 1.0,
      f"max {max(d_world):.2f} m")
# The honest control is NOT "the centred reading is off-road". Town10HD has road
# almost everywhere, so a centred coordinate can land near SOME lane by pure
# coincidence -- agent '25' does, at 0.78 m. The defect is that it lands near a
# DIFFERENT lane, i.e. the car would be driven by the wrong piece of road.
snap_world, snap_centred = [], []
for i, nm in enumerate(names):
    p = iop["position"][0, 0, i].numpy()
    h = float(iop["heading"][0, 0, i].reshape(-1)[0])
    wx, wy, wh = centred_to_world_pose(float(p[0]), float(p[1]), h, centre)
    snap_world.append(lg.closest_lane(wx, wy, wh))
    snap_centred.append(lg.closest_lane(float(p[0]), float(p[1]), h))
print(f"  lane snapped in WORLD  : {snap_world}")
print(f"  lane snapped in CENTRED: {snap_centred}")
check("CONTROL: the centred reading snaps to a DIFFERENT lane for every agent",
      all(a != b for a, b in zip(snap_world, snap_centred)),
      f"{sum(a != b for a, b in zip(snap_world, snap_centred))}/{len(names)} differ")
check("and the ego in particular is far off-lane in the centred frame",
      d_centred[0] > 5.0, f"{d_centred[0]:.2f} m")


# ---------------------------------------------------------------------------
section("3. a map without a centre pose is refused, not driven")
# ---------------------------------------------------------------------------
import prosim_ego


class _FakeBase:
    def step_agent_traj(self, a_traj, model_output, policy_agent_ids, t, mode):
        return a_traj


Ego = prosim_ego.make_rule_ego_class(_FakeBase)


class _Cfg:
    class DATASET:
        class MOTION:
            DT = 0.1


m = Ego.__new__(Ego)
m.config = _Cfg()
m.tasks = ["motion_pred"]
m.set_ego_policy("ego", vec_map=batch.vector_maps[0], kind="idm_pursuit")
check("set_ego_policy without centre leaves ego_centre unset",
      m.ego_centre is None)
try:
    m.step_agent_traj({}, None, {"motion_pred": []}, 0, "val")
    check("stepping with a map but no centre raises", False, "it ran")
except SystemExit as e:
    check("stepping with a map but no centre raises", "NO CENTRE POSE" in str(e))

m2 = Ego.__new__(Ego)
m2.config = _Cfg()
m2.tasks = ["motion_pred"]
m2.set_ego_policy("ego", vec_map=batch.vector_maps[0], kind="idm_pursuit",
                  centre=centre)
check("CONTROL: with a centre it is accepted and stored",
      m2.ego_centre is not None
      and abs(m2.ego_centre[0] - centre[0]) < 1e-9
      and abs(m2.ego_centre[2] - centre[2]) < 1e-9,
      f"{tuple(round(v, 3) for v in m2.ego_centre)}")


# ---------------------------------------------------------------------------
section("4. the junction: branches judged by where they GO")
# ---------------------------------------------------------------------------
APPROACH = "19049"
prev = np.asarray(lg.centerline(APPROACH), float)
h = math.atan2(prev[-1, 1] - prev[-2, 1], prev[-1, 0] - prev[-2, 0])
succ = list(lg.successors(APPROACH))
print(f"  lane {APPROACH} ends heading {math.degrees(h):+.1f} deg, "
      f"{len(succ)} successors:")
start_errs, end_turns = [], []
for c in succ:
    seg = np.asarray(lg.centerline(c), float)
    sh = math.atan2(seg[1, 1] - seg[0, 1], seg[1, 0] - seg[0, 0])
    eh = math.atan2(seg[-1, 1] - seg[-2, 1], seg[-1, 0] - seg[-2, 0])
    start_errs.append(abs(wrap_angle(sh - h)))
    end_turns.append(abs(wrap_angle(eh - h)))
    print(f"    {c:>8}  start-hdg err {math.degrees(start_errs[-1]):5.1f} deg   "
          f"end turn {math.degrees(end_turns[-1]):6.1f} deg")

check("CONTROL: the OLD rule (first two points) cannot tell them apart",
      len(succ) > 2 and (max(start_errs) - min(start_errs)) < math.radians(1.0),
      f"start-heading errors span only "
      f"{math.degrees(max(start_errs) - min(start_errs)):.2f} deg")
check("the NEW rule separates them",
      (max(end_turns) - min(end_turns)) > math.radians(45.0),
      f"end turns span {math.degrees(max(end_turns) - min(end_turns)):.1f} deg")

picked = LaneGraphRoute(lg)._best_successor(APPROACH, prev, {APPROACH})
seg = np.asarray(lg.centerline(picked), float)
turn = abs(wrap_angle(math.atan2(seg[-1, 1] - seg[-2, 1], seg[-1, 0] - seg[-2, 0]) - h))
check("with no goal it picks the branch that actually goes straight",
      math.degrees(turn) < 15.0, f"picked {picked}, turns {math.degrees(turn):.1f} deg")

turning = [c for c in succ
           if abs(wrap_angle(math.atan2(*(np.asarray(lg.centerline(c), float)[-1, ::-1]
                                          - np.asarray(lg.centerline(c), float)[-2, ::-1]))
                             - h)) > math.radians(45)]
goal_pt = np.asarray(lg.centerline(turning[0]), float)[-1]
picked_g = LaneGraphRoute(lg, goal=goal_pt)._best_successor(APPROACH, prev, {APPROACH})
check("a goal on a turning branch flips the choice",
      picked_g != picked, f"no goal -> {picked}, goal -> {picked_g}")


# ---------------------------------------------------------------------------
section("5. end to end: driving the real policy from the ego's start")
# ---------------------------------------------------------------------------
START = dict(x=-81.14, y=24.47, heading=math.radians(2.3), speed=6.0,
             length=4.5, width=2.0)


def drive(goal, steps=80, v0=8.0):
    pol = make_policy("idm_pursuit", dt=0.1, lane_graph=lg, v0=v0, route_goal=goal)
    st = VehicleState(**START)
    p = [(st.x, st.y)]
    for _ in range(steps):
        st = pol.step(st, [])
        p.append((st.x, st.y))
    p = np.array(p)
    return p, np.linalg.norm(PTS[None] - p[:, None], axis=2).min(axis=1)

p_straight, d_straight = drive(None)
print(f"  no goal : ends ({p_straight[-1,0]:7.2f},{p_straight[-1,1]:7.2f})  "
      f"max off-lane {d_straight.max():.2f} m")
check("it stays on the road", d_straight.max() < 1.5, f"max {d_straight.max():.2f} m")
check("it goes STRAIGHT through the junction (y barely changes)",
      abs(p_straight[-1, 1] - START["y"]) < 5.0 and p_straight[-1, 0] > -30.0,
      f"ends ({p_straight[-1,0]:.1f},{p_straight[-1,1]:.1f})")

p_turn, d_turn = drive((-45.0, -20.0))
print(f"  goal (-45,-20): ends ({p_turn[-1,0]:7.2f},{p_turn[-1,1]:7.2f})  "
      f"max off-lane {d_turn.max():.2f} m")
check("a goal down the cross road produces a DIFFERENT route",
      float(np.linalg.norm(p_turn[-1] - p_straight[-1])) > 20.0,
      f"endpoints {np.linalg.norm(p_turn[-1]-p_straight[-1]):.1f} m apart")
check("and that route is also on the road", d_turn.max() < 1.5,
      f"max {d_turn.max():.2f} m")

g = np.array([-20.0, 24.5])
p_g, _ = drive(tuple(g))
check("a reachable straight-on goal is actually reached",
      float(np.linalg.norm(p_g[-1] - g)) < 3.0,
      f"ends {np.linalg.norm(p_g[-1]-g):.2f} m from it")

print()
print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("   FAILED:", f)
print("=" * 78)
sys.exit(1 if FAIL else 0)
