"""Unit tests for ego_control.py. Pure numpy -- no ProSim, no torch, no GPU.

Each test states the physical property being checked, so a failure says what is
physically wrong rather than just which assert tripped.

    python3 test_ego_control.py
"""

import sys

import numpy as np

sys.path.insert(0, "/scratch/veerk41/ProSim")

from ego_control import (
    IDM, MOBIL, ConstantHeading, ConstantSpeed, EgoPolicy, KeepLane,
    LaneGraphRoute, Lead, PurePursuit, StraightRoute, VehicleState,
    _project_onto, _split_by_corridor, make_policy, wrap_angle,
)

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
section("1. IDM longitudinal")
# ---------------------------------------------------------------------------
idm = IDM(v0=10.0, T=1.5, s0=2.0, a_max=2.0, b=1.5, delta=4.0)

a = idm.accel(0.0, None)
check("from rest on a free road, accelerate at a_max", abs(a - 2.0) < 1e-9, f"a={a:.3f}")

a = idm.accel(10.0, None)
check("at desired speed on a free road, accel ~ 0", abs(a) < 1e-9, f"a={a:.3f}")

a = idm.accel(12.0, None)
check("above desired speed, decelerate", a < 0, f"a={a:.3f}")

a = idm.accel(8.0, Lead(gap=100.0, speed=8.0))
free = idm.accel(8.0, None)
check("a very distant lead behaves like free road", abs(a - free) < 0.05,
      f"lead={a:.3f} free={free:.3f}")

a = idm.accel(8.0, Lead(gap=2.0, speed=0.0))
check("closing fast on a stopped car -> hard braking", a < -1.0, f"a={a:.3f}")

# THE SIGN BUG: approaching a slower lead must brake harder than trailing a
# faster one at the same gap. The old code had dv negated, inverting this.
same_gap = 15.0
a_closing = idm.accel(10.0, Lead(gap=same_gap, speed=2.0))   # we are faster
a_opening = idm.accel(10.0, Lead(gap=same_gap, speed=18.0))  # lead is faster
check("approach-rate sign: closing brakes harder than opening",
      a_closing < a_opening, f"closing={a_closing:.3f} opening={a_opening:.3f}")

# s* must never fall below the jam distance s0
a_far_faster = idm.accel(1.0, Lead(gap=idm.s0, speed=50.0))
check("s* floors at s0 even with a much faster lead", a_far_faster <= idm.a_max + 1e-9,
      f"a={a_far_faster:.3f}")

a = idm.accel(30.0, Lead(gap=0.5, speed=0.0))
check("braking is clipped at b_emergency", a >= -idm.b_emergency - 1e-9, f"a={a:.3f}")

# equilibrium: at the IDM equilibrium gap with a same-speed lead, accel ~ 0
v = 6.0
s_eq = (idm.s0 + v * idm.T) / np.sqrt(1 - (v / idm.v0) ** idm.delta)
a = idm.accel(v, Lead(gap=s_eq, speed=v))
check("equilibrium gap gives ~zero acceleration", abs(a) < 1e-6, f"a={a:.2e} s_eq={s_eq:.2f}m")


# ---------------------------------------------------------------------------
section("2. IDM car-following, closed loop")
# ---------------------------------------------------------------------------
pol = EgoPolicy(IDM(v0=10.0), ConstantHeading(), KeepLane(), StraightRoute(), dt=0.1)

ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.0)
traj = pol.rollout(ego, 200)
check("free road: reaches ~v0 after 20 s", abs(traj[-1].speed - 10.0) < 0.3,
      f"v={traj[-1].speed:.2f}")
check("free road: travels straight (|y| < 1e-9)", abs(traj[-1].y) < 1e-9,
      f"y={traj[-1].y:.2e}")

# a stopped car 30 m ahead: the ego must stop behind it, not through it
wall = VehicleState(x=30.0, y=0.0, heading=0.0, speed=0.0)
ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed=8.0)
traj = pol.rollout(ego, 300, neighbors_fn=lambda i: [wall])
final_gap = wall.x - traj[-1].x - 0.5 * (wall.length + ego.length)
check("stops behind a stationary car (never overlaps)", final_gap > 0,
      f"final gap={final_gap:.2f} m")
check("stops close to the jam distance s0", 0.5 < final_gap < 6.0,
      f"final gap={final_gap:.2f} m (s0={pol.longitudinal.s0})")
check("comes to rest", traj[-1].speed < 0.1, f"v={traj[-1].speed:.3f}")

# following a lead moving at constant speed: converge without collision
lead_v = 6.0
def moving_lead(i):
    return [VehicleState(x=40.0 + lead_v * i * 0.1, y=0.0, heading=0.0, speed=lead_v)]
ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed=2.0)
traj = pol.rollout(ego, 400, neighbors_fn=moving_lead)
check("matches a constant-speed lead", abs(traj[-1].speed - lead_v) < 0.3,
      f"v={traj[-1].speed:.2f} vs lead {lead_v}")
gaps = [40.0 + lead_v * i * 0.1 - traj[i].x for i in range(len(traj) - 1)]
check("never collides while following", min(gaps) > 0, f"min gap={min(gaps):.2f} m")


# ---------------------------------------------------------------------------
section("3. Pure pursuit lateral")
# ---------------------------------------------------------------------------
pp = PurePursuit()

straight = np.stack([np.arange(0, 100, 1.0), np.zeros(100)], axis=1)
s = VehicleState(0.0, 0.0, 0.0, 8.0)
check("on a straight path, heading is unchanged",
      abs(wrap_angle(pp.heading(s, straight, 0.1) - 0.0)) < 1e-9)

# offset laterally: must steer back toward the path
s_off = VehicleState(0.0, 2.0, 0.0, 8.0)
h = pp.heading(s_off, straight, 0.1)
check("offset left of path -> steers right (negative heading)", h < 0, f"h={h:.4f} rad")

s_off = VehicleState(0.0, -2.0, 0.0, 8.0)
h = pp.heading(s_off, straight, 0.1)
check("offset right of path -> steers left (positive heading)", h > 0, f"h={h:.4f} rad")

# a left turn: quarter circle of radius 20 m
class FixedPath(StraightRoute):
    def __init__(self, path):
        self.path = path
    def path_ahead(self, state, distance):
        return self.path


def quarter_arc(radius, left=True, runout=40.0):
    """Quarter circle starting at the origin heading +x, plus a straight run-out
    so a test vehicle never drives off the end of the path (which would look
    like a tracking failure but is really an exhausted route)."""
    th = np.linspace(0, np.pi / 2, 300)
    sgn = 1.0 if left else -1.0
    arc = np.stack([radius * np.sin(th), sgn * (radius - radius * np.cos(th))], axis=1)
    tail = np.stack([np.full(int(runout), radius),
                     sgn * (radius + np.arange(int(runout), dtype=float))], axis=1)
    return np.concatenate([arc, tail], axis=0)


arc = quarter_arc(20.0, left=True)


pol_turn = EgoPolicy(ConstantSpeed(), PurePursuit(), KeepLane(), FixedPath(arc), dt=0.1)
ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed=6.0)
traj = pol_turn.rollout(ego, 60)
dists = [np.linalg.norm(arc - np.array([p.x, p.y]), axis=1).min() for p in traj]
check("tracks a 20 m-radius left turn (max error < 1.0 m)", max(dists) < 1.0,
      f"max cross-track={max(dists):.3f} m")
check("converges to +pi/2 after the left turn",
      abs(wrap_angle(traj[-1].heading - np.pi / 2)) < 0.15,
      f"final heading={traj[-1].heading:.3f} rad")

# right turn: mirror
arc_r = quarter_arc(20.0, left=False)
pol_turn_r = EgoPolicy(ConstantSpeed(), PurePursuit(), KeepLane(), FixedPath(arc_r), dt=0.1)
traj_r = pol_turn_r.rollout(VehicleState(0.0, 0.0, 0.0, 6.0), 60)
d_r = [np.linalg.norm(arc_r - np.array([p.x, p.y]), axis=1).min() for p in traj_r]
check("tracks a 20 m-radius right turn", max(d_r) < 1.0, f"max cross-track={max(d_r):.3f} m")
check("converges to -pi/2 after the right turn",
      abs(wrap_angle(traj_r[-1].heading + np.pi / 2)) < 0.15,
      f"final heading={traj_r[-1].heading:.3f} rad")

# curvature clip must hold
pp_clip = PurePursuit(max_yaw_rate=0.3)
h = pp_clip.heading(VehicleState(0, 5, 0, 10.0), straight, 1.0)
check("yaw rate is clipped", abs(wrap_angle(h)) <= 0.3 + 1e-9, f"dh={h:.3f} rad in 1 s")


# ---------------------------------------------------------------------------
section("4. Corridor geometry (gap finding)")
# ---------------------------------------------------------------------------
s_arc, lat = _project_onto(straight, np.array([10.0, 0.0]))
check("projection onto a straight path: arc length", abs(s_arc - 10.0) < 1e-6, f"s={s_arc:.3f}")
check("projection: zero lateral on the path", abs(lat) < 1e-6, f"lat={lat:.2e}")

_, lat_left = _project_onto(straight, np.array([10.0, 3.0]))
check("left of path is positive lateral", lat_left > 0, f"lat={lat_left:.2f}")

ego = VehicleState(0.0, 0.0, 0.0, 5.0)
nb = [VehicleState(20.0, 0.0, 0.0, 3.0),     # ahead, in lane
      VehicleState(10.0, 8.0, 0.0, 3.0),     # ahead, other lane -> excluded
      VehicleState(-15.0, 0.0, 0.0, 3.0)]    # behind, in lane
ahead, behind = _split_by_corridor(ego, nb, straight, 2.0)
check("picks the nearest in-corridor vehicle ahead", ahead is not None and ahead[0].x == 20.0)
check("ignores a vehicle outside the corridor",
      ahead is not None and abs(ahead[0].y) < 1e-9)
check("finds the vehicle behind", behind is not None and behind[0].x == -15.0)
check("gap is bumper-to-bumper, not centre-to-centre",
      ahead is not None and abs(ahead[1] - (20.0 - 4.5)) < 1e-6, f"gap={ahead[1]:.2f}")

# curved-corridor test: a straight-ahead test would MISS this lead
lead_on_arc = VehicleState(x=float(arc[120, 0]), y=float(arc[120, 1]), heading=0.8, speed=3.0)
ahead_c, _ = _split_by_corridor(VehicleState(0, 0, 0, 6.0), [lead_on_arc], arc, 2.0)
check("finds a lead around a curve (arc-length, not straight-line)", ahead_c is not None)


# ---------------------------------------------------------------------------
section("5. MOBIL lane selection")
# ---------------------------------------------------------------------------
class TwoLaneGraph:
    """Two parallel straight lanes 3.5 m apart, no successors."""
    def __init__(self):
        x = np.arange(0, 300, 1.0)
        self._c = {"L0": np.stack([x, np.zeros_like(x)], 1),
                   "L1": np.stack([x, np.full_like(x, 3.5)], 1)}
    def centerline(self, lid): return self._c[lid]
    def successors(self, lid): return []
    def adjacent(self, lid): return ["L1"] if lid == "L0" else ["L0"]
    def closest_lane(self, x, y, h): return "L0" if abs(y) < abs(y - 3.5) else "L1"


g = TwoLaneGraph()
mob = MOBIL(longitudinal=IDM(v0=12.0), politeness=0.3, a_threshold=0.2)

ego = VehicleState(0.0, 0.0, 0.0, 10.0)
# blocked in L0 by a slow car, L1 empty -> should change
slow = VehicleState(20.0, 0.0, 0.0, 2.0)
choice = mob.select(ego, [slow], g, "L0")
check("changes lane to overtake a slow blocker", choice == "L1", f"chose {choice}")

# both lanes clear -> no reason to change
check("stays put when there is no incentive",
      mob.select(ego, [], g, "L0") == "L0")

# target lane also blocked, and worse -> stay
slow_l1 = VehicleState(12.0, 3.5, 0.0, 1.0)
check("does not change into a worse lane",
      mob.select(ego, [slow, slow_l1], g, "L0") == "L0")

# safety veto: a fast follower right behind us in the target lane
fast_follower = VehicleState(-6.0, 3.5, 0.0, 14.0)
choice = mob.select(ego, [slow, fast_follower], g, "L0")
check("safety criterion vetoes cutting off a close fast follower",
      choice == "L0", f"chose {choice}")

# politeness: a fully selfish driver should be at least as willing to change
selfish = MOBIL(longitudinal=IDM(v0=12.0), politeness=0.0, a_threshold=0.2)
polite = MOBIL(longitudinal=IDM(v0=12.0), politeness=1.0, a_threshold=0.2)
mild_follower = VehicleState(-25.0, 3.5, 0.0, 11.0)
sel_choice = selfish.select(ego, [slow, mild_follower], g, "L0")
pol_choice = polite.select(ego, [slow, mild_follower], g, "L0")
check("politeness is wired in (selfish changes at least as readily)",
      not (pol_choice == "L1" and sel_choice == "L0"),
      f"selfish={sel_choice} polite={pol_choice}")


# ---------------------------------------------------------------------------
section("6. LaneGraphRoute traversal")
# ---------------------------------------------------------------------------
class ChainGraph:
    """A -> B -> C, each 20 m, B turning left 90 degrees."""
    def __init__(self):
        a = np.stack([np.arange(0, 20, 0.5), np.zeros(40)], 1)
        th = np.linspace(0, np.pi / 2, 40)
        b = np.stack([20 + 12 * np.sin(th), 12 - 12 * np.cos(th)], 1)
        c = np.stack([np.full(80, 32.0), 12 + np.arange(0, 40, 0.5)], 1)
        self._c = {"A": a, "B": b, "C": c}
    def centerline(self, lid): return self._c[lid]
    def successors(self, lid): return {"A": ["B"], "B": ["C"], "C": []}[lid]
    def adjacent(self, lid): return []
    def closest_lane(self, x, y, h):
        best, bd = None, np.inf
        for lid, c in self._c.items():
            d = np.linalg.norm(c - np.array([x, y]), axis=1).min()
            if d < bd:
                best, bd = lid, d
        return best


cg = ChainGraph()
route = LaneGraphRoute(cg)
p = route.path_ahead(VehicleState(1.0, 0.0, 0.0, 5.0), 60.0)
check("route chains successors across lanes", len(p) > 40, f"{len(p)} points")
check("route starts near the vehicle",
      np.linalg.norm(p[0] - np.array([1.0, 0.0])) < 2.0)
check("route reaches into the third lane", p[-1][1] > 12.0, f"end={p[-1]}")
check("route records the current lane", route.current_lane_id == "A")

# drive the full policy through the turn
pol_full = EgoPolicy(IDM(v0=6.0), PurePursuit(), KeepLane(), LaneGraphRoute(cg),
                     dt=0.1, lane_graph=cg)
traj = pol_full.rollout(VehicleState(0.5, 0.0, 0.0, 5.0), 100)
allpts = np.concatenate([cg.centerline(l) for l in ("A", "B", "C")], axis=0)
errs = [np.linalg.norm(allpts - np.array([p_.x, p_.y]), axis=1).min() for p_ in traj]
check("IDM + pure pursuit follows a 90-degree turn on the lane graph",
      max(errs) < 1.5, f"max off-centreline={max(errs):.3f} m")
check("ends up heading north after the left turn",
      abs(wrap_angle(traj[-1].heading - np.pi / 2)) < 0.4,
      f"final heading={traj[-1].heading:.3f} rad (pi/2={np.pi/2:.3f})")


# ---------------------------------------------------------------------------
section("7. make_policy factory")
# ---------------------------------------------------------------------------
for kind in ("idm", "idm_pursuit", "idm_mobil"):
    p_ = make_policy(kind, dt=0.1, lane_graph=cg)
    t_ = p_.rollout(VehicleState(0.5, 0.0, 0.0, 5.0), 30)
    finite = all(bool(np.all(np.isfinite([float(s.x), float(s.y),
                                                float(s.heading), float(s.speed)])))
                 for s in t_)
    check(f"make_policy({kind!r}) runs and stays finite", finite)

p_straight = make_policy("idm", dt=0.1)
t_ = p_straight.rollout(VehicleState(0.0, 0.0, 0.0, 0.0), 50)
check("make_policy('idm') with no lane graph falls back to a straight route",
      t_[-1].x > 0 and abs(t_[-1].y) < 1e-9)


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("   FAILED:", f)
print("=" * 74)
sys.exit(1 if FAIL else 0)
