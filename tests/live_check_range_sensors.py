"""Check 2: do tfv6's LiDAR and radars see a car where it is?

    CARLA_ROOT=<dist> PYTHONPATH=<dist>/PythonAPI/carla /scratch/veerk41/venvs/tfv6/bin/python \
        tests/live_check_range_sensors.py --host <node>

tfv6's own declared rig (its sensors(), through osc2runner's specs_from and
SensorRig -- what the worker attaches) on an MKZ on Town10HD. Captures are taken
with the road empty and then with a car 12 m ahead and 3 m to the right.

  LiDAR (Nx4 in the SENSOR frame, osc2carla/backend/sensors.py _to_points): taken
  to the ego frame through its mount, the car adds points inside its box.
  Radar (Nx4 already in the EGO frame, _to_radar): detections appear in the box.
  CONTROL, both: the mirrored box (3 m to the LEFT) gains nothing -- a sensor
  frame read with the wrong handedness would fill that box instead.
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import argparse
import importlib.util
import math
import sys

import numpy as np

import sensor_worker as W
from tests._live_policy import policy_py, policy_request

PASS, FAIL = [], []
FWD, RIGHT = 12.0, 3.0
EGO = (-84.2, 24.45, 0.0)


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def in_box(points_ego, right, half_x=2.8, half_y=1.4):
    p = np.asarray(points_ego)
    if p.size == 0:
        return 0
    return int(np.sum((np.abs(p[:, 0] - FWD) <= half_x) & (np.abs(p[:, 1] - right) <= half_y)))


def lidar_to_ego(points, spec):
    yaw = math.radians(spec.yaw)
    x, y = points[:, 0], points[:, 1]
    return np.stack([spec.x + x * math.cos(yaw) - y * math.sin(yaw),
                     spec.y + x * math.sin(yaw) + y * math.cos(yaw),
                     spec.z + points[:, 2]], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    import carla

    rig_mod = W.load_rig_module(W.DEFAULT_OSC2RUNNER)
    py = policy_py("tfv6")
    sys.path.insert(0, str(py.parent.parent))
    spec = importlib.util.spec_from_file_location("_sensor_policy", py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sensor_policy"] = mod
    spec.loader.exec_module(mod)
    policy = mod.build_policy(W.resolve_request_paths(policy_request("tfv6"), py))
    specs = [s for s in rig_mod.specs_from(policy.sensors()) if s.kind != "sensor.camera.rgb"]
    print("range sensors: " + "; ".join(f"{s.name} {s.kind.split('.')[-1]} at ({s.x}, {s.y}, {s.z}) yaw {s.yaw}" for s in specs))

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
    made = []

    def gather(rig, n=6):
        out = {s.name: [] for s in specs}
        for _ in range(n):
            cap = rig.capture(world.tick())
            for s in specs:
                if s.name in cap:
                    pts = np.asarray(cap[s.name])
                    out[s.name].append(lidar_to_ego(pts, s) if s.kind == "sensor.lidar.ray_cast" else pts)
        return out

    try:
        z = world.get_map().get_waypoint(carla.Location(EGO[0], EGO[1], 0.0)).transform.location.z
        ego = world.spawn_actor(lib.find(W.EGO_BLUEPRINT),
                                carla.Transform(carla.Location(EGO[0], EGO[1], z + 0.1), carla.Rotation(yaw=EGO[2])))
        made.append(ego)
        ego.set_simulate_physics(False)
        rig = rig_mod.SensorRig(world, ego, specs).spawn()
        made.append(rig)
        gather(rig, 6)
        before = gather(rig, 6)
        car = world.spawn_actor(lib.find(W.ACTOR_BLUEPRINT), carla.Transform(
            carla.Location(EGO[0] + FWD, EGO[1] + RIGHT, z + 0.1), carla.Rotation(yaw=EGO[2])))
        made.append(car)
        car.set_simulate_physics(False)
        gather(rig, 3)
        after = gather(rig, 6)

        lidar = [s for s in specs if s.kind == "sensor.lidar.ray_cast"]
        radars = [s for s in specs if s.kind == "sensor.other.radar"]
        for s in lidar:
            gain = np.mean([in_box(a, RIGHT) for a in after[s.name]]) - np.mean([in_box(b, RIGHT) for b in before[s.name]])
            mirror = np.mean([in_box(a, -RIGHT) for a in after[s.name]]) - np.mean([in_box(b, -RIGHT) for b in before[s.name]])
            check(f"{s.name}: the car adds points inside its box", gain >= 50, f"+{gain:.0f} points per sweep")
            check(f"{s.name}: CONTROL: the mirrored box gains nothing", mirror < 10, f"{mirror:+.0f}")
        total = total_mirror = 0.0
        for s in radars:
            g = np.mean([in_box(a, RIGHT, 3.5, 2.0) for a in after[s.name]]) - np.mean([in_box(b, RIGHT, 3.5, 2.0) for b in before[s.name]])
            m = np.mean([in_box(a, -RIGHT, 3.5, 2.0) for a in after[s.name]]) - np.mean([in_box(b, -RIGHT, 3.5, 2.0) for b in before[s.name]])
            print(f"    {s.name} (yaw {s.yaw}): +{g:.1f} detections in the car's box, {m:+.1f} in the mirrored box")
            total, total_mirror = total + g, total_mirror + m
        check("radars: detections appear in the car's box", total >= 3, f"+{total:.1f} per capture across radars")
        check("radars: CONTROL: the mirrored box gains nothing", total_mirror < 1.5, f"{total_mirror:+.1f}")
    finally:
        for a in reversed(made):
            try:
                a.destroy()
            except Exception:
                pass
        world.apply_settings(original)

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
