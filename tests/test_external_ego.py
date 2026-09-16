"""Does external_ego.py drive a harness ego_policy_v1 policy correctly?

No ProSim, no model, no CARLA, no map: a straight synthetic route and the REAL
`third_party/idm` policy objects, so what is tested is the bridge.

    python3 tests/test_external_ego.py

The behavioural section is a three-arm comparison, because "the ego changed
lane" on its own proves nothing about which part did the work:

    idm_mobil + a car seen in the next lane  -> SHOULD change lane
    idm_mobil + no car seen there            -> must NOT (require_known_lanes:
                                                a lane is a candidate only once
                                                a vehicle has been observed in
                                                it -- idm/scene.py)
    idm      + a car seen in the next lane   -> must NOT (allows_lane_change
                                                is False; isolates the lateral
                                                law from the observation)
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import math
import sys
from pathlib import Path

import numpy as np

from ego_control import VehicleState
from external_ego import (DELTA_MAX_RAD, WHEELBASE_M, ExternalEgoPolicy,
                          make_external_ego)

IDM_REPO = Path("/home/veerk41/scratch/scenario_orchestration/third_party/idm")
POLICY_PY = IDM_REPO / "scenario_orchestration" / "policy.py"

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def banner(t):
    print(f"\n=== {t} ===")


class StraightRoute:
    """A fixed straight plan from a pose, sampled every metre."""

    def __init__(self, x, y, heading):
        self.x, self.y, self.heading = x, y, heading
        self.current_lane_id = None

    def path_ahead(self, state, distance):
        s = np.arange(0.0, float(distance), 1.0)
        return np.stack([self.x + s * math.cos(self.heading),
                         self.y + s * math.sin(self.heading)], axis=1)


def request(name, implementation, **params):
    p = {"desired_speed_mps": 14.0, "time_headway_s": 1.5, "min_gap_m": 2.0,
         "max_accel_mps2": 1.5, "comfort_decel_mps2": 2.0,
         "lane_width_m": 3.5, "max_lane_offset": 1}
    p.update(params)
    return {"name": name, "implementation": implementation,
            "interface": "ego_policy_v1", "observation_space": "state",
            "action_space": "control", "seed": 0, "parameters": p}


def main():
    # ---------------------------------------------------------------- 1
    banner("1. world -> ego frame (+x forward, +y RIGHT, CARLA left-handed)")
    ego = VehicleState(x=10.0, y=20.0, heading=0.0, speed=5.0)
    br = ExternalEgoPolicy(policy=None, route=StraightRoute(10.0, 20.0, 0.0))
    ahead = ExternalEgoPolicy._to_ego(np.array([[30.0, 20.0]]), ego)[0]
    check("a car straight ahead is at +x, lat 0",
          abs(ahead[0] - 20.0) < 1e-9 and abs(ahead[1]) < 1e-9, str(ahead))
    right = ExternalEgoPolicy._to_ego(np.array([[10.0, 23.5]]), ego)[0]
    check("world +y maps to POSITIVE lat (the driver's right)",
          abs(right[1] - 3.5) < 1e-9, f"lat {right[1]:+.2f}")
    left = ExternalEgoPolicy._to_ego(np.array([[10.0, 16.5]]), ego)[0]
    check("CONTROL: world -y maps to NEGATIVE lat",
          abs(left[1] + 3.5) < 1e-9, f"lat {left[1]:+.2f}")
    turned = VehicleState(x=10.0, y=20.0, heading=math.pi / 2, speed=5.0)
    t = ExternalEgoPolicy._to_ego(np.array([[10.0, 30.0]]), turned)[0]
    check("heading +90 deg: world +y is now straight AHEAD",
          abs(t[0] - 10.0) < 1e-9 and abs(t[1]) < 1e-9, str(t))

    # ---------------------------------------------------------------- 2
    banner("2. integrating (acceleration, steer) with idm/lateral.py's model")
    st = VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.0)
    s = st
    for _ in range(10):
        s = br._integrate(s, accel=2.0, steer=0.0)
    check("accel 2.0 for 1.0 s -> 2.0 m/s", abs(s.speed - 2.0) < 1e-9, f"{s.speed:.3f}")
    check("straight: heading unchanged, y unchanged",
          abs(s.heading) < 1e-12 and abs(s.y) < 1e-12, f"h={s.heading:.2e} y={s.y:.2e}")
    check("distance is the midpoint integral (1.00 m), not v*t (2.00 m)",
          abs(s.x - 1.0) < 1e-9, f"{s.x:.3f} m")

    s0 = VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0)
    s_r = br._integrate(s0, accel=0.0, steer=1.0)
    expected = 10.0 * math.tan(DELTA_MAX_RAD) / WHEELBASE_M * 0.1
    check("steer +1 turns RIGHT by v*tan(DELTA_MAX)/WHEELBASE*dt",
          abs(s_r.heading - expected) < 1e-9,
          f"{s_r.heading:+.4f} vs {expected:+.4f} rad")
    s_l = br._integrate(s0, accel=0.0, steer=-1.0)
    check("CONTROL: steer -1 turns the other way by the same amount",
          abs(s_l.heading + expected) < 1e-9, f"{s_l.heading:+.4f} rad")
    # -200 m/s^2 for 0.1 s is -20 m/s applied to 10 m/s: the clamp is what
    # stops it going negative. (-50 would only reach 5 m/s and test nothing.)
    check("braking cannot push speed below zero",
          br._integrate(s0, accel=-200.0, steer=0.0).speed == 0.0,
          f"{br._integrate(s0, accel=-200.0, steer=0.0).speed:.2f} m/s")

    # ---------------------------------------------------------------- 3
    banner("3. the REAL policy, through the bridge")
    check("third_party/idm is checked out", POLICY_PY.is_file(), str(POLICY_PY))
    if not POLICY_PY.is_file():
        return 1

    def drive(impl, name, adjacent_car, steps=80):
        """Ego blocked by a stopped car; is there a lane to go round it?"""
        route = StraightRoute(0.0, 0.0, 0.0)
        pol = make_external_ego(POLICY_PY, request(name, impl), route, dt=0.1)
        st = VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0,
                          length=4.5, width=2.0)
        lead = VehicleState(x=60.0, y=0.0, heading=0.0, speed=0.0,
                            length=4.5, width=2.0)
        lat, speeds = [], []
        for k in range(steps):
            nbrs = [lead]
            if adjacent_car:
                # 60 m ahead in the RIGHT lane, pulling away: enough for the
                # lane to count as observed, not enough to block it.
                nbrs.append(VehicleState(x=60.0 + 10.0 * k * 0.1, y=3.5,
                                         heading=0.0, speed=10.0,
                                         length=4.5, width=2.0))
            st = pol.step(st, nbrs)
            lat.append(st.y)
            speeds.append(st.speed)
        return pol, np.array(lat), np.array(speeds)

    pol_m, lat_m, spd_m = drive("idm.policy.IDMMobilPolicy", "idm_mobil", True)
    meta_m = pol_m.metadata()
    print(f"   idm_mobil + seen lane : lateral {lat_m.min():+.2f}..{lat_m.max():+.2f} m, "
          f"lane_changes={meta_m.get('lane_changes')}, speed {spd_m.min():.1f}-{spd_m.max():.1f} m/s")
    check("it changes lane (~a lane width sideways)", lat_m.max() > 2.5,
          f"max lateral {lat_m.max():+.2f} m")
    check("and the policy itself reports the change",
          (meta_m.get("lane_changes") or 0) >= 1, str(meta_m.get("lane_changes")))

    pol_u, lat_u, _ = drive("idm.policy.IDMMobilPolicy", "idm_mobil", False)
    print(f"   idm_mobil + unseen    : lateral {lat_u.min():+.2f}..{lat_u.max():+.2f} m, "
          f"lane_changes={pol_u.metadata().get('lane_changes')}")
    check("CONTROL: with no car ever seen in the next lane it holds",
          abs(lat_u).max() < 1.0, f"max |lateral| {abs(lat_u).max():.2f} m")

    pol_i, lat_i, spd_i = drive("idm.policy.IDMPolicy", "idm", True)
    print(f"   idm + seen lane       : lateral {lat_i.min():+.2f}..{lat_i.max():+.2f} m, "
          f"speed {spd_i.min():.1f}-{spd_i.max():.1f} m/s")
    check("CONTROL: plain idm holds its lane even with the lane available",
          abs(lat_i).max() < 1.0, f"max |lateral| {abs(lat_i).max():.2f} m")
    check("plain idm brakes for the stopped car instead",
          spd_i.min() < 1.0, f"min speed {spd_i.min():.2f} m/s")

    banner("4. the route is FROZEN, not re-planned")
    route = StraightRoute(0.0, 0.0, 0.0)
    pol = make_external_ego(POLICY_PY, request("idm_mobil", "idm.policy.IDMMobilPolicy"),
                            route, dt=0.1)
    st = VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0)
    pol.step(st, [])
    first = pol._plan.copy()
    for _ in range(20):
        st = pol.step(st, [])
    check("the plan sampled at t=0 is still the plan 2 s later",
          np.array_equal(first, pol._plan), "re-planned mid-episode")
    obs = pol._observation(st, [])
    check("and it is re-expressed in the CURRENT ego frame (starts ahead)",
          obs["route"] and obs["route"][0][0] > 0.0,
          f"first route point {obs['route'][0] if obs['route'] else None}")

    banner("4b. route progress: an over-rotated ego is not sent back down its approach")
    # East 40 m, a left turn (CARLA: toward -y) of radius 10, then south.
    arc = np.linspace(math.pi / 2, 0.0, 32)
    L_PLAN = np.concatenate([
        np.stack([np.arange(0.0, 40.0, 0.5), np.zeros(80)], axis=1),
        np.stack([40.0 + 10.0 * np.cos(arc), -10.0 + 10.0 * np.sin(arc)], axis=1),
        np.stack([np.full(100, 50.0), -10.0 - 0.5 * np.arange(1, 101)], axis=1)])

    class LRoute:
        current_lane_id = None
        def path_ahead(self, state, distance):
            return L_PLAN

    def world_first(obs_route, st):
        c, s = math.cos(st.heading), math.sin(st.heading)
        fx, fy = obs_route[0]
        return st.x + fx * c - fy * s, st.y + fx * s + fy * c

    def old_first_idx(st):
        loc = ExternalEgoPolicy._to_ego(L_PLAN, st)
        idx = np.flatnonzero(loc[:, 0] > 0.0)
        return int(idx[0]) if len(idx) else None

    lt = ExternalEgoPolicy(type("W", (), {"action_space": "waypoints"})(), LRoute(), dt=0.1)
    for x in np.arange(0.0, 44.0, 1.0):                 # drive the approach
        lt._observation(VehicleState(x=float(x), y=0.0, heading=0.0, speed=8.0), [])
    over = VehicleState(x=48.0, y=-6.0, heading=math.radians(-106.0), speed=8.0)
    obs = lt._observation(over, [])
    wx, wy = world_first(obs["route"], over)
    check("over-rotated at the junction exit: the route continues SOUTH",
          wy < -5.0 and abs(wx - 50.0) < 6.0, f"first route point world ({wx:.1f}, {wy:.1f})")
    check("CONTROL: the old 'in front of the car' filter picked the approach start",
          old_first_idx(over) == 0, f"old first index {old_first_idx(over)}")
    idx_before = lt._progress_idx
    lt._observation(VehicleState(x=10.0, y=0.0, heading=0.0, speed=8.0), [])
    check("progress never moves backwards", lt._progress_idx >= idx_before,
          f"{idx_before} -> {lt._progress_idx}")

    thru = ExternalEgoPolicy(type("W", (), {"action_space": "waypoints"})(), LRoute(), dt=0.1)
    for x in np.arange(0.0, 71.0, 1.0):                 # straight through the turn
        st_thru = VehicleState(x=float(x), y=0.0, heading=0.0, speed=8.0)
        obs = thru._observation(st_thru, [])
    check("straight through the junction: the route is not empty",
          len(obs["route"]) == 20, f"{len(obs['route'])} points")
    check("CONTROL: the old filter left nothing in front",
          old_first_idx(st_thru) is None, f"old first index {old_first_idx(st_thru)}")
    from external_ego import _resample
    sr = ExternalEgoPolicy(type("W", (), {"action_space": "waypoints"})(),
                           StraightRoute(0.0, 0.0, 0.0), dt=0.1)
    worst = 0.0
    for x in np.arange(0.0, 30.0, 0.7):                 # 0.3 m off the line
        st_sr = VehicleState(x=float(x), y=0.3, heading=0.0, speed=8.0)
        obs = sr._observation(st_sr, [])
        loc = ExternalEgoPolicy._to_ego(sr._plan, st_sr)
        old = _resample(loc[loc[:, 0] > 0.0], 2.5, 1.0, 20)
        worst = max(worst, float(np.abs(np.asarray(obs["route"]) - old).max()))
    check("CONTROL: in normal driving the route is exactly the old filter's",
          worst < 1e-9, f"max |new - old| {worst:.2e} m over 43 offset poses")

    # ---------------------------------------------------------------- 5
    banner("5. waypoint policies (PlanT 2.0's surface)")

    class FakeWaypointPolicy:
        """Declares `waypoints` and answers like PlanT 2.0's policy.py."""
        interface, observation_space, action_space = "ego_policy_v1", "state", "waypoints"
        def __init__(self, control=True):
            self.control, self.seen = control, []
        def act(self, obs):
            self.seen.append(obs)
            out = {"waypoints": [[2.0, 0.0], [4.0, 0.0]],
                   "target_speed_mps": 6.0, "meta": {"step": len(self.seen)}}
            if self.control:
                out["control"] = {"steer": 0.25, "throttle": 0.5, "brake": 0.0}
            return out

    pol = ExternalEgoPolicy(FakeWaypointPolicy(), StraightRoute(0.0, 0.0, 0.0), dt=0.1)
    check("the action space is read off the policy, not configured",
          pol.action_space == "waypoints", pol.action_space)
    st = VehicleState(x=0.0, y=0.0, heading=0.0, speed=3.0, length=4.5, width=2.0)
    nxt = pol.step(st, [VehicleState(x=20.0, y=0.0, heading=0.0, speed=4.0)])
    obs = pol.policy.seen[-1]
    check("route resampled to 20 points, 1 m apart from 2.5 m ahead",
          len(obs["route"]) == 20
          and abs(obs["route"][0][0] - 2.5) < 1e-6
          and abs(obs["route"][1][0] - 3.5) < 1e-6,
          f"{len(obs['route'])} pts, first {obs['route'][0] if obs['route'] else None}")
    check("a speed limit is declared for the waypoint policy",
          obs.get("speed_limit_kph") == 50.0, str(obs.get("speed_limit_kph")))
    check("objects carry CARLA HALF-extents, as both repos specify",
          obs["objects"][0]["extent"][:2] == [2.25, 1.0], str(obs["objects"][0]["extent"]))
    check("it steers by the policy's OWN control.steer",
          nxt.heading > 0.0, f"heading {nxt.heading:+.4f} rad")
    # 3.0 -> 6.0 m/s cannot happen in one 0.1 s tick: the tracker is limited to
    # ACCEL_LIMITS, so it approaches over several steps.
    check("it accelerates toward the policy's own target_speed_mps",
          abs(nxt.speed - 3.4) < 1e-9, f"{nxt.speed:.2f} m/s after one tick")
    conv = ExternalEgoPolicy(FakeWaypointPolicy(), StraightRoute(0.0, 0.0, 0.0), dt=0.1)
    s_conv = VehicleState(x=0.0, y=0.0, heading=0.0, speed=3.0)
    for _ in range(20):
        s_conv = conv.step(s_conv, [])
    check("and reaches it", abs(s_conv.speed - 6.0) < 1e-6, f"{s_conv.speed:.3f} m/s")
    st_fast = VehicleState(x=0.0, y=0.0, heading=0.0, speed=30.0)
    check("CONTROL: the target-speed tracker is acceleration-limited",
          ExternalEgoPolicy(FakeWaypointPolicy(), StraightRoute(0.0, 0.0, 0.0),
                            dt=0.1).step(st_fast, []).speed > 29.0,
          "a 24 m/s drop in 0.1 s would be unphysical")
    try:
        ExternalEgoPolicy(FakeWaypointPolicy(control=False),
                          StraightRoute(0.0, 0.0, 0.0), dt=0.1).step(st, [])
        check("CONTROL: a waypoint action with no control is refused", False,
              "a steering law was invented")
    except SystemExit as exc:
        check("CONTROL: a waypoint action with no control is refused",
              "re-derive" in str(exc), str(exc)[:60])

    banner("6. the REAL plant2 policy object")
    P2 = Path("/home/veerk41/scratch/scenario_orchestration/third_party/plant2"
              "/scenario_orchestration/policy.py")
    check("third_party/plant2 is checked out", P2.is_file())
    if P2.is_file():
        req = {"name": "plant2", "implementation": "plant2.agent.PlanTAgent",
               "interface": "ego_policy_v1", "observation_space": "state",
               "action_space": "waypoints", "seed": 0,
               "checkpoint": "/scratch/veerk41/av_checkpoints/plant2",
               "parameters": {"num_waypoints": 4}}
        try:
            p2 = make_external_ego(P2, req, StraightRoute(0.0, 0.0, 0.0), dt=0.1)
            check("the action space comes from the REQUEST, which PlanT2Policy "
                  "does not expose as an attribute",
                  getattr(p2.policy, "action_space", None) is None)
            check("build_policy accepts the harness's policy request", True)
            check("and the bridge drives it as a waypoint policy",
                  p2.action_space == "waypoints", p2.action_space)
        except SystemExit as exc:
            check("build_policy accepts the harness's policy request", False,
                  str(exc)[:90])

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
