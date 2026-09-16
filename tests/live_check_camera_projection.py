"""B2: do the cameras look where their mounting says they look?

    CARLA_ROOT=<dist> <policy venv>/bin/python tests/live_check_camera_projection.py \
        --host <node> --policy tfv6|simlingo

The policy's own declared rig (its sensors(), through osc2runner's specs_from and
SensorRig, as the worker attaches it) on an MKZ on Town10HD. A frame is taken
with the road empty, a car is placed 12 m ahead and 3 m to the right, and a
second frame is taken. The car is found as the largest changed region, and its
centre is compared with the pixel predicted from the camera's own mounting
(x, y, z, yaw, pitch) and field of view:

    ego frame +x forward, +y right, +z up; f = (W/2) / tan(fov/2)
    u = W/2 + f * y_cam / x_cam,  v = H/2 - f * z_cam / x_cam

PASS: horizontal error within 6% of the image width on every camera that should
see the car. CONTROL: the mirrored prediction (the car on the LEFT) must miss by
more than that -- the Y-flip this project once shipped would pass the first
check only if the second were absent.
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
FWD, RIGHT, UP = 12.0, 3.0, 0.75            # the car's centre in the ego frame
EGO = (-84.2, 24.45, 0.0)                    # Town10HD, the red_light approach, heading +x


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def predict(spec, right):
    dx, dy, dz = FWD - spec.x, right - spec.y, UP - spec.z
    yaw, pitch = math.radians(spec.yaw), math.radians(spec.pitch)
    xc = dx * math.cos(yaw) + dy * math.sin(yaw)
    yc = -dx * math.sin(yaw) + dy * math.cos(yaw)
    xc, zc = xc * math.cos(pitch) + dz * math.sin(pitch), -xc * math.sin(pitch) + dz * math.cos(pitch)
    if xc <= 1.0:
        return None
    f = (spec.width / 2.0) / math.tan(math.radians(spec.fov) / 2.0)
    return spec.width / 2.0 + f * yc / xc, spec.height / 2.0 - f * zc / xc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--policy", required=True, choices=["tfv6", "simlingo"])
    args = ap.parse_args()
    import carla
    import cv2

    rig_mod = W.load_rig_module(W.DEFAULT_OSC2RUNNER)
    py = policy_py(args.policy)
    sys.path.insert(0, str(py.parent.parent))
    spec = importlib.util.spec_from_file_location("_sensor_policy", py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sensor_policy"] = mod
    spec.loader.exec_module(mod)
    policy = mod.build_policy(W.resolve_request_paths(policy_request(args.policy), py))
    cams = [s for s in rig_mod.specs_from(policy.sensors()) if s.kind == "sensor.camera.rgb"]
    print(f"{args.policy} cameras: " + "; ".join(
        f"{c.name} {c.width}x{c.height} fov {c.fov} at ({c.x}, {c.y}, {c.z}) yaw {c.yaw} pitch {c.pitch}" for c in cams))

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
    try:
        z = world.get_map().get_waypoint(carla.Location(EGO[0], EGO[1], 0.0)).transform.location.z
        ego = world.spawn_actor(lib.find(W.EGO_BLUEPRINT),
                                carla.Transform(carla.Location(EGO[0], EGO[1], z + 0.1), carla.Rotation(yaw=EGO[2])))
        made.append(ego)
        ego.set_simulate_physics(False)
        rig = rig_mod.SensorRig(world, ego, cams).spawn()
        made.append(rig)
        for _ in range(10):
            rig.capture(world.tick())
        before = rig.capture(world.tick())
        car = world.spawn_actor(lib.find(W.ACTOR_BLUEPRINT), carla.Transform(
            carla.Location(EGO[0] + FWD, EGO[1] + RIGHT, z + 0.1), carla.Rotation(yaw=EGO[2])))
        made.append(car)
        car.set_simulate_physics(False)
        for _ in range(3):
            rig.capture(world.tick())
        after = rig.capture(world.tick())

        seen = 0
        for cam in cams:
            p, m = predict(cam, RIGHT), predict(cam, -RIGHT)
            if p is None or not (0 <= p[0] < cam.width) or cam.name not in before or cam.name not in after:
                continue
            diff = np.abs(after[cam.name].astype(int) - before[cam.name].astype(int)).sum(axis=2) > 60
            n, labels, stats, cents = cv2.connectedComponentsWithStats(diff.astype(np.uint8))
            if n <= 1:
                check(f"{cam.name}: the car shows up in the frame", False, "no changed pixels")
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            u, v = cents[k]
            seen += 1
            tol = 0.06 * cam.width
            check(f"{cam.name}: car found where the mounting predicts (within {tol:.0f} px)",
                  abs(u - p[0]) <= tol,
                  f"found u {u:.0f}, v {v:.0f} ({stats[k, cv2.CC_STAT_AREA]} px); predicted u {p[0]:.0f}, v {p[1]:.0f}")
            if m is not None:
                check(f"{cam.name}: CONTROL: the mirrored prediction misses",
                      abs(u - m[0]) > tol, f"mirrored u {m[0]:.0f} is {abs(u - m[0]):.0f} px away")
        check("at least one camera was expected to see the car", seen >= 1, f"{seen} camera(s)")
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
