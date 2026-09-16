"""Check 5 (live): what happens in CARLA when a ProSim-driven car drives into the ego.

    CARLA_ROOT=<dist> /scratch/veerk41/venvs/tfv6/bin/python tests/live_check_contact.py --host <node>

The ego is an MKZ with physics, held on its brake, with CARLA's own collision
sensor attached (what osc2runner records collisions with,
osc2carla/backend/metrics.py). An agent is moved the way sensor_worker moves
ProSim's cars (place_actor: pose + target velocity, physics on, from each
tick's start pose) straight at the ego at 5 m/s, as if ProSim's trajectory ran
through it.

Reported, per run: collision events CARLA raised, how far the agent ended from
where ProSim put it (physics refusing to overlap), how far the ego was shoved,
and the deepest box overlap the harness's trajectory-only contact test would
see (metrics/geometry/contact.py counts a collision only past 0.08 m).

  CARLA reports the hit on the ego (>= 1 collision event).
  CONTROL: the same drive 5 m to the side raises none and the agent stays on
  its path (< 0.1 m), so the numbers above are the contact's, not the method's.
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import argparse
import math

import numpy as np

import sensor_worker as W

PASS, FAIL = [], []
EGO = (-84.2, 24.45)


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def overlap_depth(a_xy, a_yaw, a_half, b_xy, b_yaw, b_half):
    """Separating-axis penetration depth of two oriented boxes (0 when apart)."""
    def corners(xy, yaw, half):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[xy[0] + sx * half[0] * c - sy * half[1] * s,
                          xy[1] + sx * half[0] * s + sy * half[1] * c]
                         for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    A, B = corners(a_xy, a_yaw, a_half), corners(b_xy, b_yaw, b_half)
    depth = float("inf")
    for yaw in (a_yaw, b_yaw):
        for ax in (np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])):
            pa, pb = A @ ax, B @ ax
            o = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if o <= 0:
                return 0.0
            depth = min(depth, o)
    return depth


def drive(world, carla, lib, lateral, ticks=80, speed=5.0):
    z = world.get_map().get_waypoint(carla.Location(EGO[0], EGO[1], 0.0)).transform.location.z
    made, events = [], []
    try:
        ego = world.spawn_actor(lib.find(W.EGO_BLUEPRINT),
                                carla.Transform(carla.Location(EGO[0], EGO[1], z + 0.1), carla.Rotation(yaw=0.0)))
        made.append(ego)
        ego.apply_control(carla.VehicleControl(brake=1.0))
        sensor = world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        made.append(sensor)
        sensor.listen(lambda e: events.append((e.other_actor.type_id, math.hypot(e.normal_impulse.x, e.normal_impulse.y))))
        for _ in range(20):
            world.tick()
        start = ego.get_location()
        x0 = EGO[0] + 15.0
        agent = world.spawn_actor(lib.find(W.ACTOR_BLUEPRINT), carla.Transform(
            carla.Location(x0, EGO[1] + lateral, z + 0.1), carla.Rotation(yaw=180.0)))
        made.append(agent)
        agent.set_simulate_physics(True)
        off_path, shove, depth = 0.0, 0.0, 0.0
        eh = ego.bounding_box.extent
        ah = agent.bounding_box.extent
        dt = 0.05
        for k in range(ticks):
            pose = {"x": x0 - speed * dt * k, "y": EGO[1] + lateral, "yaw_rad": math.pi, "speed": speed}
            W.place_actor(carla, agent, carla.Transform(carla.Location(pose["x"], pose["y"], z + W.ACTOR_Z_OFFSET),
                                                        carla.Rotation(yaw=180.0)), pose)
            world.tick()
            want_x = x0 - speed * dt * (k + 1)
            al, el = agent.get_location(), ego.get_location()
            off_path = max(off_path, math.hypot(al.x - want_x, al.y - (EGO[1] + lateral)))
            shove = max(shove, math.hypot(el.x - start.x, el.y - start.y))
            depth = max(depth, overlap_depth((el.x, el.y), math.radians(ego.get_transform().rotation.yaw), (eh.x, eh.y),
                                             (al.x, al.y), math.radians(agent.get_transform().rotation.yaw), (ah.x, ah.y)))
        return {"collision_events": len(events), "max_impulse": max((i for _, i in events), default=0.0),
                "agent_off_path_m": off_path, "ego_shoved_m": shove, "max_box_overlap_m": depth}
    finally:
        for a in reversed(made):
            try:
                a.destroy()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if not W.same_town(world.get_map().name, "Town10HD"):
        world = client.load_world("Town10HD_Opt")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    lib = world.get_blueprint_library()
    try:
        hit = drive(world, carla, lib, lateral=0.0)
        miss = drive(world, carla, lib, lateral=5.0)
    finally:
        world.apply_settings(original)
    print(f"head-on: {hit}")
    print(f"5 m aside: {miss}")
    check("CARLA reports the hit on the ego (collision sensor)", hit["collision_events"] >= 1,
          f"{hit['collision_events']} events, peak impulse {hit['max_impulse']:.0f}")
    check("CONTROL: 5 m to the side, no collision event and the agent stays on ProSim's path",
          miss["collision_events"] == 0 and miss["agent_off_path_m"] < 0.1,
          f"{miss['collision_events']} events, off path {miss['agent_off_path_m']:.3f} m")
    print(f"  (for scoring) deepest box overlap in the hit: {hit['max_box_overlap_m']:.3f} m; "
          f"the harness counts a collision past 0.08 m. Agent pushed {hit['agent_off_path_m']:.2f} m off "
          f"ProSim's path, ego shoved {hit['ego_shoved_m']:.2f} m.")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
