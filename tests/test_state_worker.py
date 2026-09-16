"""A `state` policy driven inside CARLA by the sensor worker, as osc2runner drives it.

    /scratch/veerk41/venvs/tfv6/bin/python tests/test_state_worker.py

No CARLA server, no GPU: the fake world from test_sensor_loop.py, extended with
what osc2runner's observation builder reads (actors by type, bounding boxes, a
speed limit). The code under test is REAL: sensor_worker.Session, osc2runner's
carla_state_obs.StateObservationBuilder, its policy bridge's `_to_command`, and
its actuation.SpawnGear / AccelerationTracker. The policy is a stand-in that
records what it was given.
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import importlib.util
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

import sensor_worker as W

spec = importlib.util.spec_from_file_location("_loop_fakes", "tests/test_sensor_loop.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


class Actor(F.FakeActor):
    """A FakeActor that answers what carla_state_obs reads."""
    def __init__(self, world, bp_id, tf):
        super().__init__(world, bp_id, tf)
        self.type_id, self.id = str(bp_id), 100 + len(world.actors)
        self.attributes = {}
        half = (2.45, 1.06, 0.75) if "lincoln" in self.type_id else (2.35, 0.92, 0.72)
        self.bounding_box = types.SimpleNamespace(extent=F.Vector3D(*half))

    def get_speed_limit(self):
        return 30.0

    def get_traffic_light(self):
        return None

    def get_location(self):
        return self.tf.location


class World(F.FakeWorld):
    def try_spawn_actor(self, bp_id, tf):
        actor = Actor(self, bp_id, tf)
        self.actors.append(actor)
        return actor

    def get_actors(self):
        actors = list(self.actors)
        return types.SimpleNamespace(filter=lambda pattern: [
            a for a in actors if pattern.strip("*") in a.type_id])


def policy_file(src):
    path = Path(tempfile.mkdtemp()) / "statepolicy" / "scenario_orchestration" / "policy.py"
    path.parent.mkdir(parents=True)
    path.write_text(src)
    return path


ACCEL_POLICY = '''
SEEN = []
class P:
    def load(self): pass
    def reset(self): pass
    def act(self, obs):
        SEEN.append(obs)
        return {"acceleration_mps2": 2.0, "steer": 0.1}
def build_policy(request):
    return P()
'''

PEDAL_POLICY = '''
SEEN = []
class P:
    def act(self, obs):
        SEEN.append(obs)
        return {"control": {"throttle": 0.3, "brake": 0.0, "steer": -0.2}}
def build_policy(request):
    return P()
'''


def session(src, action_space, frames_dir=None):
    world = World("Town04")
    s = W.Session(F.make_fake_carla(world), F.fake_rig_mod, "127.0.0.1", 2000)
    route = np.stack([np.arange(-10.0, 200.0, 0.5), np.zeros(420)], axis=1)
    init = {"op": "init", "town": "Town04", "dt": 0.1, "policy_hz": 20.0,
            "ego": {"x": 0.0, "y": 0.0, "yaw_rad": 0.0, "speed": 0.0, "length": 4.8, "width": 2.1},
            "actors": [{"x": 20.0, "y": 3.5, "yaw_rad": 0.0, "speed": 5.0, "length": 4.5, "width": 2.0}],
            "route_world": route.tolist(), "policy_py": str(policy_file(src)),
            "policy_request": {"name": "stand-in", "interface": "ego_policy_v1",
                               "observation_space": "state", "action_space": action_space},
            "frames_dir": frames_dir}
    return world, s, s.init(init)


def step(s, k):
    return s.step({"op": "step", "actors": [{"x": 20.5 + 0.5 * k, "y": 3.5, "yaw_rad": 0.0,
                                             "speed": 5.0, "length": 4.5, "width": 2.0}]})


def main():
    print("\n=== 1. the observation is osc2runner's, built from the CARLA world ===")
    world, s, reply = session(ACCEL_POLICY, "control")
    check("init reports a state policy, and the BEV renderer's absence (this repo has none)",
          reply.get("observation_space") == "state" and str(reply.get("bev")).startswith("unavailable"),
          f"{reply.get('observation_space')}, bev {str(reply.get('bev'))[:60]}")
    step(s, 0)
    obs = sys.modules["_sensor_policy"].SEEN[-1]
    cars = [o for o in obs.get("objects", []) if o.get("type") == "car"]
    # Expected from the poses the world holds now (the ego has already moved and
    # turned a little under the fake physics), with the rotation done here, not
    # by the builder: +x forward, +y right.
    e_tf, a_tf = s.ego.get_transform(), s.actors[0].get_transform()
    yaw = math.radians(e_tf.rotation.yaw)
    dx, dy = a_tf.location.x - e_tf.location.x, a_tf.location.y - e_tf.location.y
    want = (dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))
    check("the policy sees the other car, in the ego frame, at its CARLA speed",
          len(cars) == 1 and abs(cars[0]["position"][0] - want[0]) < 1e-6
          and abs(cars[0]["position"][1] - want[1]) < 1e-6 and abs(cars[0]["speed_mps"] - 5.0) < 1e-9,
          f"got {np.round(cars[0]['position'][:2], 4).tolist() if cars else None}, "
          f"expected {np.round(want, 4).tolist()}")
    check("CONTROL: the same car without the rotation would be off (the frame is really applied)",
          abs(yaw) > 1e-4 and abs(cars[0]["position"][1] - dy) > 1e-4, f"ego yaw {yaw:.5f} rad")
    check("CONTROL: the ego itself is not among the objects",
          all(o.get("type_id") != W.EGO_BLUEPRINT for o in obs.get("objects", [])),
          str([o.get("type_id") for o in obs.get("objects", [])]))
    check("CARLA half-extents, as osc2runner reports them",
          cars and cars[0]["extent"][:2] == [2.35, 0.92], str(cars[0]["extent"] if cars else None))
    check("route is ProSim's frozen plan, osc2runner-sliced: 20 points from 2.5 m ahead",
          len(obs["route"]) == 20 and abs(obs["route"][0][0] - 2.5) < 0.2, str(obs["route"][:2]))
    check("speed limit read off CARLA for the ego", obs.get("speed_limit_kph") == 30.0,
          str(obs.get("speed_limit_kph")))

    print("\n=== 2. actuation is osc2runner's: an acceleration realised closed-loop ===")
    throttles = []
    for k in range(1, 15):
        out = step(s, k)
        throttles.append(out["control"]["throttle"])
    open_loop = 2.0 / 3.0
    check("an acceleration demand is realised by AccelerationTracker, not the a/3 pedal map",
          any(abs(t - open_loop) > 0.05 for t in throttles[3:]),
          f"throttles {np.round(throttles, 3).tolist()} vs open-loop {open_loop:.3f}")
    check("steer is passed through", abs(out["control"]["steer"] - 0.1) < 1e-9, str(out["control"]))
    s.close()

    print("\n=== 3. the per-decision log, for a policy with no cameras ===")
    run_dir = Path(tempfile.mkdtemp())
    world, s3, _ = session(ACCEL_POLICY, "control", frames_dir=str(run_dir / "ego_sensor_frames"))
    for k in range(3):
        step(s3, k)
    s3.close()
    lines = [__import__("json").loads(l) for l in (run_dir / "ego_policy_log.jsonl").read_text().splitlines()]
    check("a state policy's log: one line per decision, with objects seen and its acceleration",
          len(lines) == 6 and lines[-1]["objects_seen"] == 1
          and lines[-1]["action"].get("acceleration_mps2") == 2.0, f"{len(lines)} lines, last {str(lines[-1])[:160]}")
    check("CONTROL: no empty camera-frame folder for a policy without cameras",
          not (run_dir / "ego_sensor_frames").exists())

    world, s2, _ = session(PEDAL_POLICY, "control")
    for k in range(6):
        out = step(s2, k)
    check("CONTROL: a pedal policy's throttle is applied as given (no tracker)",
          abs(out["control"]["throttle"] - 0.3) < 1e-9 and abs(out["control"]["steer"] + 0.2) < 1e-9,
          str(out["control"]))
    s2.close()

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
