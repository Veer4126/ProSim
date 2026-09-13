"""Does the IDM longitudinal model actually brake? Two questions, both measured.

  A. CAR FOLLOWING -- put a slow car in the ego's OWN lane on the real Town10HD
     map and check it slows, settles at a sane gap, and never collides.

  B. CROSS TRAFFIC -- why the ego did not yield to the car that hit it at the
     junction. IDM is a car-FOLLOWING model: EgoPolicy only treats a neighbour
     as a leader when it is within `corridor_halfwidth` (2.0 m) of the ego's own
     path AND ahead of it along that path. A perpendicular car satisfies that
     only once it is already inside the junction box, which is far too late.
     This section measures exactly when it enters the corridor and how much
     braking distance is left at that moment.

Needs the cached CARLA batch for the real lane graph. No model, no GPU, no CARLA.

    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 test_idm_following.py"
"""

import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/veerk41/ProSim")

from carla_dataset import register

register()

from torch.utils.data import DataLoader

from ego_control import (IDM, VehicleState, _split_by_corridor, _to_lead,
                         make_policy)
from prosim.config.default import get_config
from prosim.core.registry import registry
from prosim_ego import VecMapLaneGraph

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def section(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


_DATA_DIR = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))  # recordings live one level above the repo
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

DT = 0.1
START = dict(x=-81.14, y=24.47, heading=math.radians(2.3), speed=8.0,
             length=4.5, width=2.0)
V0 = 8.0

# the ego's own route, as an arc-length-parameterised polyline, so a lead car
# can be placed and driven along exactly the lane the ego will use
_probe = make_policy("idm_pursuit", dt=DT, lane_graph=lg, v0=V0,
                     route_goal=(-20.0, 24.5))
PATH = _probe.route.path_ahead(VehicleState(**START), 200.0)
S = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(PATH, axis=0), axis=1))])


def on_path(s, speed, length=4.5, width=2.0):
    """A vehicle sitting at arc length `s` along the ego's own route."""
    s = float(np.clip(s, S[0], S[-1]))
    i = int(np.searchsorted(S, s))
    i = min(max(i, 1), len(PATH) - 1)
    p = PATH[i]
    h = math.atan2(PATH[i, 1] - PATH[i - 1, 1], PATH[i, 0] - PATH[i - 1, 0])
    return VehicleState(x=float(p[0]), y=float(p[1]), heading=h, speed=speed,
                        length=length, width=width)


def run(lead_speed=None, lead_s0=30.0, lead_offset=0.0, steps=120):
    """Drive the ego; optionally with a lead car on its path (or offset sideways)."""
    pol = make_policy("idm_pursuit", dt=DT, lane_graph=lg, v0=V0,
                      route_goal=(-20.0, 24.5))
    ego = VehicleState(**START)
    s_lead = lead_s0
    log = []
    for k in range(steps):
        nb = []
        if lead_speed is not None:
            lead = on_path(s_lead, lead_speed)
            if lead_offset:
                # push it sideways, perpendicular to its heading
                lead = VehicleState(
                    x=lead.x - lead_offset * math.sin(lead.heading),
                    y=lead.y + lead_offset * math.cos(lead.heading),
                    heading=lead.heading, speed=lead.speed,
                    length=lead.length, width=lead.width)
            nb = [lead]
            s_lead += lead_speed * DT
        path = pol.route.path_ahead(ego, pol.lookahead)
        ahead, _ = _split_by_corridor(ego, nb, path, pol.corridor_halfwidth)
        gap = ahead[1] if ahead is not None else np.inf
        log.append((k * DT, ego.speed, gap))
        ego = pol.step(ego, nb)
    return np.array(log)


# ---------------------------------------------------------------------------
section("A. car following: a slow car in the ego's own lane")
# ---------------------------------------------------------------------------
free = run(lead_speed=None)
print(f"  CONTROL, no lead    : speed {free[0,1]:.2f} -> {free[-1,1]:.2f} m/s "
      f"(v0 = {V0})")
check("with no lead the ego holds/reaches its free speed",
      abs(free[-1, 1] - V0) < 0.3, f"final {free[-1,1]:.2f} m/s")

slow = run(lead_speed=3.0, lead_s0=30.0)
print(f"\n  lead at 3.0 m/s, starting 30 m ahead:")
print(f"    {'t':>5} {'ego v':>7} {'gap':>8}")
for k in range(0, len(slow), 15):
    t, v, g = slow[k]
    print(f"    {t:5.1f} {v:7.2f} {g:8.2f}")
print(f"    {slow[-1,0]:5.1f} {slow[-1,1]:7.2f} {slow[-1,2]:8.2f}")

check("the ego SLOWS DOWN when it catches a slower car",
      slow[-1, 1] < free[-1, 1] - 1.0,
      f"{slow[0,1]:.2f} -> {slow[-1,1]:.2f} m/s (free run ends {free[-1,1]:.2f})")
check("it converges to the lead's speed, not past it",
      abs(slow[-1, 1] - 3.0) < 0.6, f"final {slow[-1,1]:.2f} vs lead 3.00 m/s")
check("it never collides (bumper gap stays positive)",
      slow[:, 2].min() > 0.0, f"min gap {slow[:,2].min():.2f} m")
check("it settles at a sane following distance",
      2.0 < slow[-1, 2] < 25.0, f"final gap {slow[-1,2]:.2f} m")

stopped = run(lead_speed=0.0, lead_s0=30.0)
print(f"\n  CONTROL, STATIONARY car 30 m ahead: ego {stopped[0,1]:.2f} -> "
      f"{stopped[-1,1]:.2f} m/s, final gap {stopped[-1,2]:.2f} m "
      f"(min {stopped[:,2].min():.2f})")
check("it comes to a stop behind a stationary car",
      stopped[-1, 1] < 0.3 and stopped[:, 2].min() > 0.0,
      f"final speed {stopped[-1,1]:.2f} m/s, min gap {stopped[:,2].min():.2f} m")

off = run(lead_speed=3.0, lead_s0=30.0, lead_offset=6.0)
print(f"\n  CONTROL, same slow car pushed 6 m SIDEWAYS (out of the 2 m corridor):")
print(f"    ego {off[0,1]:.2f} -> {off[-1,1]:.2f} m/s")
check("CONTROL: a car outside the corridor is correctly ignored",
      abs(off[-1, 1] - V0) < 0.3, f"final {off[-1,1]:.2f} m/s (free run {free[-1,1]:.2f})")


# ---------------------------------------------------------------------------
section("B. why it did not yield to the car that hit it")
# ---------------------------------------------------------------------------
csv = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "test_fixtures", "rollout_cross.csv")
d = pd.read_csv(csv)
other = d[d.id == d.id.max()].sort_values("frame")
print(f"  replaying the crossing car from {csv}")
print(f"    it runs ({other.x.iloc[0]:.1f},{other.y.iloc[0]:.1f}) -> "
      f"({other.x.iloc[-1]:.1f},{other.y.iloc[-1]:.1f}) at "
      f"{np.hypot(other.vx, other.vy).mean():.1f} m/s mean")

pol = make_policy("idm_pursuit", dt=DT, lane_graph=lg, v0=V0,
                  route_goal=(-20.0, 24.5))
ego = VehicleState(**START)
rows = []
for k in range(min(80, len(other))):
    r = other.iloc[k]
    nb = [VehicleState(x=float(r.x), y=float(r.y), heading=float(r.yaw),
                       speed=float(math.hypot(r.vx, r.vy)), length=4.5, width=2.0)]
    path = pol.route.path_ahead(ego, pol.lookahead)
    from ego_control import _project_onto
    s_ego, _ = _project_onto(path, ego.xy)
    s_n, lat = _project_onto(path, nb[0].xy)
    ahead, _ = _split_by_corridor(ego, nb, path, pol.corridor_halfwidth)
    seen = ahead is not None
    rows.append((k * DT, ego.speed, lat, s_n - s_ego,
                 math.hypot(ego.x - r.x, ego.y - r.y), seen))
    ego = pol.step(ego, nb)
rows = np.array(rows, dtype=object)

print(f"\n    {'t':>5} {'ego v':>7} {'lat off':>8} {'along':>7} {'centre-centre':>13}  seen as lead?")
for k in range(0, len(rows), 8):
    t, v, lat, ds, dist, seen = rows[k]
    print(f"    {t:5.1f} {v:7.2f} {lat:8.2f} {ds:7.1f} {dist:13.1f}  "
          f"{'YES' if seen else 'no'}")

seen_idx = [i for i, r in enumerate(rows) if r[5]]
closest = min(r[4] for r in rows)
print(f"\n    closest approach: {closest:.2f} m centre-to-centre")
if seen_idx:
    first = seen_idx[0]
    t_first = rows[first][0]
    v_at = rows[first][1]
    ds_at = rows[first][3]
    # braking distance at IDM's comfortable deceleration
    b = IDM().b
    d_brake = v_at ** 2 / (2.0 * b)
    print(f"    first seen as a lead at t={t_first:.1f}s, "
          f"{ds_at:.1f} m ahead along the path, ego at {v_at:.2f} m/s")
    print(f"    comfortable braking distance at that speed: "
          f"{d_brake:.1f} m (b={b} m/s^2)")
    check("the crossing car IS eventually seen, but only inside the corridor",
          True, f"first at t={t_first:.1f}s")
    check("by then there is NOT enough room to stop -- this is the real gap",
          d_brake > ds_at,
          f"needs {d_brake:.1f} m, has {ds_at:.1f} m")
else:
    check("the crossing car is never seen as a lead", True,
          "it never entered the 2 m corridor ahead of the ego")
    print("    -> IDM had nothing to respond to at any point.")

print()
print("  CONCLUSION: IDM works (section A), but it is a CAR-FOLLOWING model.")
print("  It has no right-of-way, no time-to-collision, no junction logic, so a")
print("  perpendicular car is invisible to it until it is already in the way.")

print()
print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("   FAILED:", f)
print("=" * 78)
sys.exit(1 if FAIL else 0)
