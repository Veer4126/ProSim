"""Tests for goal_control.py. Needs the CARLA batch but NOT the model.

The important one is test 2: rather than trusting my reading of
format_utils.py, it reconstructs ProSim's OWN populated goal values from the
recorded trajectory using my frame maths. If the two agree, the frame
convention is right; if they don't, a goal written through this module would
steer the agent somewhere silently wrong.

    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 test_goal_control.py"
"""

import sys

import numpy as np

sys.path.insert(0, "/scratch/veerk41/ProSim")

from carla_dataset import register

register()

import torch
from torch.utils.data import DataLoader

from goal_control import (LEFT_SIGN, agent_centred_pose, agent_world_pose,
                          body_to_world, set_goal_condition, side_name,
                          turn_goal_from_lane_graph, turn_name, world_to_body,
                          wrap_angle)
from prosim.config.default import get_config
from prosim.core.registry import registry
from prosim_ego import VecMapLaneGraph

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def section(t):
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


# ---------------------------------------------------------------------------
section("1. body <-> world frame maths")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(3000):
    axy = rng.uniform(-200, 200, 2)
    ah = rng.uniform(-np.pi, np.pi)
    gw = rng.uniform(-200, 200, 2)
    back = body_to_world(world_to_body(gw, axy, ah), axy, ah)
    worst = max(worst, float(np.abs(back - gw).max()))
check("world -> body -> world round-trips", worst < 1e-9, f"worst={worst:.2e}")

# sign convention. The ROTATION must put a +y world offset at +y body; which
# SIDE of the driver that is depends on the frame's handedness, and CARLA's is
# left-handed, so it is their RIGHT. Both facts are asserted separately -- the
# rotation was always right, the label was not.
b = world_to_body([0.0, 10.0], [0.0, 0.0], 0.0)     # agent facing +x
check("+y world maps to +y body when facing +x", b[1] > 0, f"body={np.round(b,2)}")
check("+y body is named RIGHT (CARLA is left-handed)",
      side_name(b[1]) == "RIGHT" and side_name(-b[1]) == "LEFT",
      f"side_name({b[1]:+.0f})={side_name(b[1])}")
b = world_to_body([10.0, 0.0], [0.0, 0.0], 0.0)
check("straight ahead is +x body, zero lateral",
      abs(b[0] - 10) < 1e-9 and abs(b[1]) < 1e-9, f"body={np.round(b,2)}")
b = world_to_body([0.0, 10.0], [0.0, 0.0], np.pi / 2)  # agent facing north
check("facing north, a point due north is straight ahead",
      abs(b[0] - 10) < 1e-9 and abs(b[1]) < 1e-9, f"body={np.round(b,2)}")


# ---------------------------------------------------------------------------
section("2. DECISIVE: reproduce ProSim's own goal values")
# ---------------------------------------------------------------------------
_DATA_DIR = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))  # recordings live one level above the repo
cfg = get_config("prosim_demo/cfg/waymo_demo.yaml", cluster="local")
cfg.defrost()
cfg.DATASET.SOURCE.TRAIN = ["carla_town10hd"]   # the yaml ships waymo_train
# data path keyed by whatever source is set, so switching town is one line
cfg.DATASET.DATA_PATHS[cfg.DATASET.SOURCE.TRAIN[0].upper()] = _DATA_DIR
cfg.PROMPT.CONDITION.TYPES = ["goal", "llm_text_OneText"]
cfg.freeze()
ds = registry.get_dataset(cfg.DATASET.TYPE)(cfg, "train")
ds._data_index = ds._data_index[:1]
ds._data_len = 1
dl = DataLoader(ds, batch_size=1, shuffle=False,
                collate_fn=ds.get_collate_fn(), num_workers=0)
for batch in dl:
    break

names = list(batch.extras["prompt"]["motion_pred"]["agent_ids"][0])
g = batch.extras["condition"]["goal"]
prompt_idx = g["prompt_idx"][0, :, 0].numpy()
print(f"  {len(names)} agents, goal condition slots: {g['input'].shape}")

# There are TWO agent orderings; the goal condition is built from io_pairs_batch.
iop = batch.extras["io_pairs_batch"]
tgt = batch.tgt_agent_idxs[0]
tgt = tgt.tolist() if hasattr(tgt, "tolist") else list(tgt)
print(f"  orderings differ: agent_names[:5]={list(batch.agent_names[0])[:5]}")
print(f"                    io_pairs   [:5]={list(iop['agent_names'][0])[:5]}")
print(f"  cond prompt_idx[:8] = {prompt_idx[:8].tolist()}  (shuffled, not arange)")

# (a) slot c must equal io_pairs['goal'] at io-pairs index prompt_idx[c]
slot_err = []
for c in range(len(prompt_idx)):
    n = int(prompt_idx[c])
    if n < 0:
        continue
    slot_err.append(float(np.abs(g["input"][0, c, :2].numpy()
                                 - iop["goal"][0, 0, n].numpy()).max()))
slot_err = np.array(slot_err)
check("cond slot c maps to io_pairs index prompt_idx[c]", slot_err.max() < 1e-5,
      f"max err={slot_err.max():.2e}")

# (b) reconstruct that goal from the recorded future with MY frame maths
errs = []
for c in range(len(prompt_idx)):
    n = int(prompt_idx[c])
    if n < 0 or n >= len(tgt):
        continue
    o = int(tgt[n])                        # io-pairs index -> agent_names index
    fut_len = int(batch.agent_fut_len[0, o])
    if fut_len < 1:
        continue
    fut_end = batch.agent_fut[0, o, fut_len - 1].numpy()[:2]   # centred frame
    p, h = agent_centred_pose(batch, n)                        # io-pairs ordering
    mine = world_to_body(fut_end, p, h)
    errs.append(float(np.abs(mine - g["input"][0, c, :2].numpy()).max()))

errs = np.array(errs)
print(f"  compared {len(errs)} agents")
print(f"  |my conversion - ProSim's goal|: median={np.median(errs):.2e}  "
      f"max={errs.max():.2e} m")
check("frame convention matches ProSim's own goal values", errs.max() < 1e-3,
      f"max err={errs.max():.2e} m")

# control 1: translation only, no rotation
wrong = []
# control 2: the WRONG agent ordering (init_obs instead of io_pairs)
wrong_order = []
for c in range(len(prompt_idx)):
    n = int(prompt_idx[c])
    if n < 0 or n >= len(tgt):
        continue
    o = int(tgt[n])
    fut_len = int(batch.agent_fut_len[0, o])
    if fut_len < 1:
        continue
    fut_end = batch.agent_fut[0, o, fut_len - 1].numpy()[:2]
    p, h = agent_centred_pose(batch, n)
    wrong.append(float(np.abs((fut_end - p) - g["input"][0, c, :2].numpy()).max()))
    pi = batch.extras["init_obs"]["position"][0, n].numpy()
    hi = float(batch.extras["init_obs"]["heading"][0, n].reshape(-1)[0])
    wrong_order.append(float(np.abs(world_to_body(fut_end, pi, hi)
                                    - g["input"][0, c, :2].numpy()).max()))
check("CONTROL: translation-only (no rotation) does NOT match",
      np.median(wrong) > 1.0, f"median err={np.median(wrong):.2f} m")
# This control only has teeth for agents whose position in the TWO ORDERINGS
# actually differs -- for the rest, reading init_obs with an io-pairs index
# happens to hit the right agent and the error is legitimately 0. Taking the
# MEDIAN over all agents therefore hides the bug whenever only a few agents are
# swapped: in the 6-agent scene agent_names is ['ego','87','89','88',...] vs
# io_pairs ['ego','87','88','89',...] -- exactly two swapped -- so the median is
# 0.00 m while the swapped pair is metres out. Score the swapped agents only.
_names_io = list(iop["agent_names"][0])
_names_an = list(batch.agent_names[0])
_swapped = [n for n in range(len(_names_io))
            if n < len(_names_an) and _names_io[n] != _names_an[n]]
print(f"  agents whose two orderings disagree: {_swapped} "
      f"(io_pairs {[_names_io[n] for n in _swapped]} vs "
      f"agent_names {[_names_an[n] for n in _swapped]})")
if _swapped:
    _sw_err = [wrong_order[i] for i, c in enumerate(range(len(prompt_idx)))
               if int(prompt_idx[c]) in _swapped][:len(wrong_order)]
    _sw_err = [e for e in _sw_err if e is not None]
    check("CONTROL: for a SWAPPED agent the init_obs ordering does NOT match",
          len(_sw_err) > 0 and max(_sw_err) > 1.0,
          f"max err over swapped agents={max(_sw_err):.2f} m; "
          f"median over ALL agents={np.median(wrong_order):.2f} m (which is why "
          "the median is the wrong statistic here)")
else:
    print("  [SKIP] CONTROL: init_obs ordering -- the two orderings are identical "
          "in this scene, so the control cannot discriminate")


# ---------------------------------------------------------------------------
section("2b. DECISIVE: which side is +y, from CARLA's OWN edge labels")
# ---------------------------------------------------------------------------
# export_lane_graph.py builds left_edge/right_edge from CARLA's own waypoint
# transforms. Projecting them into the body frame therefore reports the
# handedness as a MEASUREMENT rather than as a convention I asserted. This is
# the same class of bug as the original Y-flip, so it gets a hard test.
import json as _json
_lanes = _json.load(open(__import__("os").path.join(_DATA_DIR, "town10hd_lanes.json")))["lanes"]
_L, _R = [], []
for _lane in _lanes.values():
    _c = np.asarray(_lane["center"], float)
    _le = np.asarray(_lane["left_edge"], float)
    _re = np.asarray(_lane["right_edge"], float)
    if len(_c) < 3:
        continue
    for _k in range(1, len(_c) - 1):
        _h = float(np.arctan2(_c[_k+1, 1] - _c[_k-1, 1], _c[_k+1, 0] - _c[_k-1, 0]))
        _L.append(world_to_body(_le[_k, :2], _c[_k, :2], _h)[1])
        _R.append(world_to_body(_re[_k, :2], _c[_k, :2], _h)[1])
_L, _R = np.array(_L), np.array(_R)
print(f"  {len(_L)} lane points sampled across {len(_lanes)} lanes")
print(f"  CARLA's LEFT  edge -> body y median {np.median(_L):+.3f}  "
      f"({100*(_L < 0).mean():.1f}% negative)")
print(f"  CARLA's RIGHT edge -> body y median {np.median(_R):+.3f}  "
      f"({100*(_R > 0).mean():.1f}% positive)")
check("CARLA's own left edge lands at NEGATIVE body y", (_L < 0).mean() > 0.99)
check("CARLA's own right edge lands at POSITIVE body y", (_R > 0).mean() > 0.99)
check("side_name agrees with CARLA's edge labels",
      side_name(float(np.median(_L))) == "LEFT"
      and side_name(float(np.median(_R))) == "RIGHT")
check("LEFT_SIGN encodes that", LEFT_SIGN == -1.0, f"LEFT_SIGN={LEFT_SIGN}")


# ---------------------------------------------------------------------------
section("3. lane-graph turn goals")
# ---------------------------------------------------------------------------
vm = batch.vector_maps[0]
lg = VecMapLaneGraph(vm)
print(f"  map {vm.map_id}: {len(vm.lanes)} lanes")

found = {"left": 0, "right": 0, "straight": 0}
strict_ok = strict_tested = 0
mislabelled = []       # a branch returned for a direction it does not go
dup_claims = []        # two directions returning the SAME point
junction_agents = 0
for nidx in range(len(names)):
    xy, h = agent_world_pose(batch, nidx)
    res = {}
    for d in ("left", "straight", "right"):
        out = turn_goal_from_lane_graph(lg, xy, h, d)
        if out is not None:
            res[d] = out
            found[d] += 1
            # THE REGRESSION FOR THE 2026-09-05 BUG: the selector used to score
            # branches by the goal point's lateral offset, so at a junction whose
            # only two exits both turned ~-90 deg it returned one of them for
            # 'straight' AND the same one again for 'left'. Asking for 'straight'
            # then produced a 90-degree turn.
            actual = turn_name(out[1])
            if actual != d:
                mislabelled.append((nidx, d, actual, np.degrees(out[1]) * -1))
    if "left" in res and "right" in res:
        n_opts = res["left"][2]
        if n_opts >= 2:
            junction_agents += 1
            strict_tested += 1
            # the guarantee, restated for turn-angle scoring: the branch chosen
            # for 'left' never turns further right than the one chosen for 'right'
            tl = res["left"][1] * LEFT_SIGN
            tr = res["right"][1] * LEFT_SIGN
            if tl >= tr - 1e-9:
                strict_ok += 1
            else:
                print(f"      agent {nidx}: left turn {np.degrees(tl):+.1f} deg "
                      f"< right turn {np.degrees(tr):+.1f} deg")
    for d1 in res:
        for d2 in res:
            if d1 < d2 and float(np.linalg.norm(res[d1][0] - res[d2][0])) < 1e-6:
                dup_claims.append((nidx, d1, d2))

print(f"  branches found: {found}")
print(f"  agents at a real junction (>1 option): {junction_agents}/{len(names)}")
check("turn goals are produced for real agents", found["left"] > 0)
if strict_tested:
    check("selector picks the most-left / most-right available branch",
          strict_ok == strict_tested, f"{strict_ok}/{strict_tested}")
else:
    # Now that a direction with no matching branch returns None, no agent in
    # THIS scene has both a left and a right option -- Town10HD's junctions here
    # are all single-turn. The guarantee is still worth testing, so it is tested
    # below on a synthetic junction that does offer all three.
    print("  [SKIP] no agent in this scene has BOTH a left and a right branch, "
          "so the real map cannot exercise the ordering guarantee")


# --- synthetic junction: the one place all three directions genuinely exist ---
class _StubGraph:
    """Approach lane along +x, then three branches: left, straight, right.

    CARLA is left-handed, so the LEFT branch is the one curving toward -y.
    """
    def __init__(self):
        ap = np.stack([np.arange(0, 40.5, 0.5), np.zeros(81)], axis=1)
        t = np.arange(0, 30.5, 0.5)
        self.lanes = {
            "approach": ap,
            "left":     np.stack([40 + t * 0.0 + t, -t * 0.0], axis=1) * 0,  # placeholder
        }
        # build the branches as quarter-turns / a straight, all starting at (40, 0)
        ang = np.linspace(0, np.pi / 2, 61)
        R = 20.0
        self.lanes["left"] = np.stack([40 + R * np.sin(ang), -R * (1 - np.cos(ang))], axis=1)
        self.lanes["right"] = np.stack([40 + R * np.sin(ang), R * (1 - np.cos(ang))], axis=1)
        self.lanes["straight"] = np.stack([40 + t, np.zeros_like(t)], axis=1)

    def closest_lane(self, x, y, h):
        return "approach"

    def centerline(self, lid):
        return self.lanes[lid]

    def successors(self, lid):
        return ["left", "straight", "right"] if lid == "approach" else []


_stub = _StubGraph()
_axy, _ah = np.array([5.0, 0.0]), 0.0        # on the approach, facing +x
print()
print("  synthetic junction (approach +x, three real branches):")
_picked = {}
for _d in ("left", "straight", "right"):
    _out = turn_goal_from_lane_graph(_stub, _axy, _ah, _d, lookahead=45.0)
    if _out is None:
        print(f"    {_d:8s}: none")
        continue
    _picked[_d] = _out
    _b = world_to_body(_out[0], _axy, _ah)
    print(f"    {_d:8s}: world [{_out[0][0]:6.1f},{_out[0][1]:6.1f}]  "
          f"fwd {_b[0]:+6.1f}  lat {_b[1]:+6.1f} ({side_name(_b[1])})  "
          f"branch turn {np.degrees(_out[1]) * LEFT_SIGN:+6.1f} deg (+ve = left)")

check("all three directions exist at a junction that really offers all three",
      set(_picked) == {"left", "straight", "right"}, f"got {sorted(_picked)}")
if set(_picked) == {"left", "straight", "right"}:
    _lat = {d: world_to_body(o[0], _axy, _ah)[1] for d, o in _picked.items()}
    check("'left' lands on the driver's LEFT", side_name(_lat["left"]) == "LEFT",
          f"lat={_lat['left']:+.1f}")
    check("'right' lands on the driver's RIGHT", side_name(_lat["right"]) == "RIGHT",
          f"lat={_lat['right']:+.1f}")
    check("'straight' is laterally between the two",
          _lat["left"] < _lat["straight"] < _lat["right"],
          f"{_lat['left']:+.1f} < {_lat['straight']:+.1f} < {_lat['right']:+.1f}")
    check("the three goals are three DIFFERENT points",
          len({tuple(np.round(o[0], 3)) for o in _picked.values()}) == 3)
    check("each branch is classified as what it is",
          all(turn_name(_picked[d][1]) == d for d in _picked))

    # CONTROL: mirror the whole junction; left and right must swap.
    class _Mirror(_StubGraph):
        def __init__(self):
            super().__init__()
            for k in self.lanes:
                self.lanes[k] = self.lanes[k] * np.array([1.0, -1.0])
    _m = turn_goal_from_lane_graph(_Mirror(), _axy, _ah, "left", lookahead=45.0)
    _ml = world_to_body(_m[0], _axy, _ah)[1]
    # Mirroring maps the ORIGINAL RIGHT branch onto the left side, so asking for
    # 'left' must now return the y-mirror of the original 'right' goal point --
    # the same branch, on the other side. If the selector ignored geometry and
    # just returned a fixed successor, this would come back unchanged instead.
    _expect = _picked["right"][0] * np.array([1.0, -1.0])
    check("CONTROL: mirroring the junction swaps which branch is 'left'",
          float(np.abs(_m[0] - _expect).max()) < 1e-6 and side_name(_ml) == "LEFT",
          f"mirrored 'left' goal {np.round(_m[0], 2)} == mirror of original "
          f"'right' {np.round(_expect, 2)}; lat {_ml:+.1f} ({side_name(_ml)})")

if mislabelled:
    for nidx, asked, actual, deg in mislabelled:
        print(f"      agent {nidx}: asked {asked!r}, branch actually goes "
              f"{actual!r} ({deg:+.1f} deg)")
check("a branch returned for direction D actually goes D", not mislabelled,
      f"{len(mislabelled)} mislabelled")

if dup_claims:
    for nidx, d1, d2 in dup_claims:
        print(f"      agent {nidx}: {d1!r} and {d2!r} return the SAME point")
check("two different directions never return the same goal point", not dup_claims,
      f"{len(dup_claims)} duplicate claims")

# CONTROL: the old lateral-offset scoring must FAIL this, or the test is vacuous.
# Re-score the ego's junction the old way and show it mislabels.
_bad = 0
for nidx in range(len(names)):
    xy, h = agent_world_pose(batch, nidx)
    out = turn_goal_from_lane_graph(lg, xy, h, "straight")
    if out is None:
        continue
    # what the OLD rule would have picked: smallest |lateral| of the goal point
    cands = {d: turn_goal_from_lane_graph(lg, xy, h, d) for d in ("left", "right")}
    cands = {d: o for d, o in cands.items() if o is not None}
    if not cands:
        continue
    old_pick = min(cands.values(),
                   key=lambda o: abs(world_to_body(o[0], xy, h)[1]))
    if turn_name(old_pick[1]) != "straight":
        _bad += 1
print(f"  CONTROL: the OLD lateral-offset rule would call a non-straight branch "
      f"'straight' for {_bad}/{len(names)} agents")
check("CONTROL: the old rule really was broken (so this test can fail)", _bad > 0,
      f"{_bad} agents mislabelled by the old rule")

# goals must land on the drivable map, not in a building
lane_pts = np.concatenate([lg.centerline(l.id) for l in vm.lanes], axis=0)
dists = []
for nidx in range(len(names)):
    xy, h = agent_world_pose(batch, nidx)
    out = turn_goal_from_lane_graph(lg, xy, h, "left")
    if out is not None:
        dists.append(float(np.linalg.norm(lane_pts - out[0], axis=1).min()))
check("turn goals lie on a lane centreline", max(dists) < 1.0,
      f"max dist to nearest lane={max(dists):.3f} m")


# ---------------------------------------------------------------------------
section("4. writing the condition into the batch")
# ---------------------------------------------------------------------------
xy, h = agent_world_pose(batch, 0)
goal_world = body_to_world([30.0, 15.0], xy, h)      # 30 m ahead, 15 m LEFT
info = set_goal_condition(batch, 0, goal_world)
print(f"  agent 0 at world {np.round(info['agent_world_xy'],2)} "
      f"heading {info['agent_heading']:.3f}")
print(f"  goal world {np.round(info['goal_world'],2)} -> body "
      f"{np.round(info['goal_body'],2)}  (slot {info['cond_idx']})")

check("goal written into the requested body offset",
      abs(info["goal_body"][0] - 30) < 1e-6 and abs(info["goal_body"][1] - 15) < 1e-6,
      f"body={np.round(info['goal_body'],3)}")

g = batch.extras["condition"]["goal"]
check("exactly one goal condition is enabled",
      int(g["mask"][0].sum()) == 1, f"{int(g['mask'][0].sum())} enabled")
check("prompt_mask marks exactly the target agent",
      int(g["prompt_mask"][0].sum()) == 1 and bool(g["prompt_mask"][0, 0]),
      f"{int(g['prompt_mask'][0].sum())} marked")
check("the written value survives in the tensor",
      abs(float(g["input"][0, info["cond_idx"], 1]) - 15.0) < 1e-6)

# a bad agent index must fail loudly, not silently no-op
try:
    set_goal_condition(batch, 999, goal_world)
    check("out-of-range agent index raises", False)
except SystemExit:
    check("out-of-range agent index raises", True)


print()
print("=" * 76)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("   FAILED:", f)
print("=" * 76)
sys.exit(1 if FAIL else 0)
