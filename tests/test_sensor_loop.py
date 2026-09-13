"""The ProSim <-> CARLA sensor loop, exercised with no simulator and no GPU.

    python3 tests/test_sensor_loop.py        (host python with numpy)

What can go wrong in this loop goes wrong silently: a control applied after the
tick instead of before it, an actor placed at the target pose on every substep
instead of the interpolated one, a camera frame from an earlier tick, a route on
the wrong side of the car, or an ego pose echoed back from ProSim instead of
read from the simulator. Every one of those still "runs". So each is checked
against an event log kept by a fake world, not against the code's own
arithmetic.

The fakes: a minimal `carla` module (types plus a world whose tick integrates
the ego's applied control), a fake rig standing in for osc2runner's SensorRig,
and a fake policy file loaded exactly the way a real policy.py is.
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import json
import math
import sys
import tempfile
import threading
import types
from pathlib import Path

import numpy as np

import sensor_worker as W
from ego_control import VehicleState
from remote_ego import RemoteSensorEgoPolicy
from sensor_protocol import Client, ProtocolError

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def banner(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------
# a fake carla
# --------------------------------------------------------------------------

class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Rotation:
    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0):
        self.yaw, self.pitch, self.roll = yaw, pitch, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle, self.steer, self.brake = throttle, steer, brake


class Settings:
    def __init__(self, synchronous_mode=False, fixed_delta_seconds=None):
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds


class FakeActor:
    def __init__(self, world, bp_id, transform):
        self.world, self.bp_id, self.tf = world, bp_id, transform
        self.physics = True
        self.v = Vector3D()
        self.control = None
        self.destroyed = False

    def set_simulate_physics(self, on):
        self.physics = bool(on)

    def set_transform(self, tf):
        self.tf = tf
        self.world.log.append(("set_transform", self.bp_id, tf.location.x, tf.location.y))

    def set_target_velocity(self, v):
        self.v = v

    def apply_control(self, control):
        self.control = control
        self.world.log.append(("apply_control", control.throttle, control.steer))

    def get_transform(self):
        return self.tf

    def get_velocity(self):
        return self.v

    def destroy(self):
        self.destroyed = True


class FakeWorld:
    def __init__(self, town="Town04"):
        self.map_name = f"Carla/Maps/{town}"
        self.settings = Settings(False, None)
        self.frame = 100
        self.actors = []
        self.log = []

    def get_map(self):
        world = self
        wp = types.SimpleNamespace(transform=Transform(Location(z=0.2)))
        return types.SimpleNamespace(name=world.map_name,
                                     get_waypoint=lambda loc, project_to_road=True: wp)

    def get_settings(self):
        return Settings(self.settings.synchronous_mode, self.settings.fixed_delta_seconds)

    def apply_settings(self, s):
        self.settings = Settings(s.synchronous_mode, s.fixed_delta_seconds)

    def get_blueprint_library(self):
        return types.SimpleNamespace(find=lambda bp_id: bp_id)

    def try_spawn_actor(self, bp_id, tf):
        actor = FakeActor(self, bp_id, tf)
        self.actors.append(actor)
        return actor

    def tick(self):
        """Integrate every physics actor's held control: throttle pushes it
        forward, steer turns it. So the ego's pose is the WORLD's, and it only
        moves if a control reached it before the tick."""
        dt = self.settings.fixed_delta_seconds or 0.05
        for a in self.actors:
            if not a.physics or a.control is None:
                continue
            yaw = math.radians(a.tf.rotation.yaw)
            speed = math.hypot(a.v.x, a.v.y) + (4.0 * a.control.throttle - 8.0 * a.control.brake) * dt
            speed = max(0.0, speed)
            yaw += a.control.steer * 0.5 * dt
            a.v = Vector3D(speed * math.cos(yaw), speed * math.sin(yaw))
            a.tf = Transform(Location(a.tf.location.x + a.v.x * dt,
                                      a.tf.location.y + a.v.y * dt, a.tf.location.z),
                             Rotation(yaw=math.degrees(yaw)))
        self.frame += 1
        self.log.append(("tick", self.frame))
        return self.frame


def make_fake_carla(world):
    class Client:
        def __init__(self, host, port):
            pass

        def set_timeout(self, t):
            pass

        def get_world(self):
            return world

        def load_world(self, town):
            world.map_name = f"Carla/Maps/{town}"
            world.log.append(("load_world", town))
            return world

    return types.SimpleNamespace(Client=Client, Location=Location, Rotation=Rotation,
                                 Transform=Transform, Vector3D=Vector3D,
                                 VehicleControl=VehicleControl)


class FakeRig:
    def __init__(self, world, ego, specs):
        self.world, self.ego, self.specs = world, ego, list(specs)
        self.failed, self.dropped, self.sensor_hz = [], {}, 20.0
        self.frames_asked = []
        self.destroyed = False

    def spawn(self):
        return self

    @property
    def active(self):
        return bool(self.specs)

    @property
    def names(self):
        return [s["name"] for s in self.specs]

    def capture(self, frame=None):
        self.frames_asked.append(frame)
        self.world.log.append(("capture", frame))
        return {s["name"]: np.zeros((4, 6, 3), dtype=np.uint8) for s in self.specs}

    def destroy(self):
        self.destroyed = True


fake_rig_mod = types.SimpleNamespace(SensorRig=FakeRig, specs_from=lambda d: list(d))

POLICY_SRC = '''
SEEN = []
class P:
    def sensors(self):
        return [{"name": "PCAM_F0", "width": 6, "height": 4, "fov": 60.0}]
    def load(self): pass
    def reset(self): pass
    def close(self): pass
    def act(self, obs):
        SEEN.append(obs)
        return {"control": {"throttle": 1.7, "steer": 0.1, "brake": 0.0},
                "meta": {"command": "LANEFOLLOW"}}
def build_policy(request):
    return P()
'''


class StraightRoute:
    def __init__(self, x, y, h):
        self.x, self.y, self.h = x, y, h
        self.current_lane_id = None

    def path_ahead(self, state, distance):
        s = np.arange(0.0, float(distance), 1.0)
        return np.stack([self.x + s * math.cos(self.h), self.y + s * math.sin(self.h)], axis=1)


def main():
    tmp = Path(tempfile.mkdtemp())
    policy_py = tmp / "fakepolicy" / "scenario_orchestration" / "policy.py"
    policy_py.parent.mkdir(parents=True)
    policy_py.write_text(POLICY_SRC)
    request = {"name": "fake", "interface": "ego_policy_v1",
               "observation_space": "sensor", "action_space": "control"}

    # ---------------------------------------------------------------- 1
    banner("1. geometry")
    mid = W.lerp_pose({"x": 0, "y": 0, "yaw_rad": math.radians(179), "speed": 0},
                      {"x": 2, "y": 0, "yaw_rad": math.radians(-179), "speed": 4}, 0.5)
    check("heading interpolates along the SHORT arc across +-180 deg",
          abs(abs(math.degrees(mid["yaw_rad"])) - 180.0) < 1e-6,
          f"{math.degrees(mid['yaw_rad']):+.1f} deg (a naive lerp gives 0)")
    route = np.stack([np.arange(0, 60.0, 0.5), np.zeros(120)], axis=1)
    r = W.route_in_ego_frame(route, 0.0, 0.0, 0.0)
    check("route: 20 points, first 2.5 m ahead, 1 m apart",
          len(r) == 20 and abs(r[0][0] - 2.5) < 1e-9 and abs(r[1][0] - r[0][0] - 1.0) < 1e-9,
          f"{len(r)} pts, first {np.round(r[0], 3).tolist()}")
    r_side = W.route_in_ego_frame(np.array([[0, 0], [5, 3], [10, 6]], float), 0.0, 0.0, 0.0)
    r_mirror = W.route_in_ego_frame(np.array([[0, 0], [5, -3], [10, -6]], float), 0.0, 0.0, 0.0)
    check("a route bending toward world +y has POSITIVE lateral (the right)",
          r_side[-1][1] > 0 and r_mirror[-1][1] < 0,
          f"+y -> {r_side[-1][1]:+.2f}, control -y -> {r_mirror[-1][1]:+.2f}")
    r_turned = W.route_in_ego_frame(np.stack([np.zeros(60), np.arange(60.0)], axis=1),
                                    0.0, 0.0, math.pi / 2)
    check("with the ego facing world +y, that route is straight ahead",
          abs(r_turned[5][1]) < 1e-9 and r_turned[5][0] > 0, str(np.round(r_turned[5], 3).tolist()))
    c = W.to_control_fields({"control": {"throttle": 1.7, "steer": -3.0, "brake": -1}})
    check("controls are clamped to CARLA's ranges",
          c == {"throttle": 1.0, "brake": 0.0, "steer": -1.0}, str(c))
    check("Town10HD matches the server's Carla/Maps/Town10HD_Opt",
          W.same_town("Carla/Maps/Town10HD_Opt", "Town10HD")
          and W.same_town("Town04", "Carla/Maps/Town04"))
    check("CONTROL: Town04 does not match Town10HD_Opt",
          not W.same_town("Carla/Maps/Town10HD_Opt", "Town04"))
    fake_py = Path("/h/root/third_party/tfv6/scenario_orchestration/policy.py")
    rel = W.resolve_request_paths(
        {"checkpoint": "third_party/checkpoints/tfv6_cvpr2026/tfv6_resnet34"}, fake_py)
    check("a harness-relative checkpoint resolves against the harness root",
          rel["checkpoint"] == "/h/root/third_party/checkpoints/tfv6_cvpr2026/tfv6_resnet34",
          rel["checkpoint"])
    absolute = W.resolve_request_paths({"checkpoint": "/abs/ckpt"}, fake_py)
    check("CONTROL: an absolute checkpoint is left alone",
          absolute["checkpoint"] == "/abs/ckpt")
    try:
        W.to_control_fields({"waypoints": [[1, 0]]})
        check("CONTROL: an action with no control block is refused", False)
    except RuntimeError:
        check("CONTROL: an action with no control block is refused", True)

    # ---------------------------------------------------------------- 2
    banner("2. a session against a fake world")
    world = FakeWorld("Town04")
    fc = make_fake_carla(world)
    init = {"op": "init", "town": "Town10HD_Opt", "dt": 0.1, "policy_hz": 20.0,
            "ego": {"x": 0.0, "y": 0.0, "yaw_rad": 0.0, "speed": 2.0, "length": 4.8, "width": 2.1},
            "actors": [{"x": 20.0, "y": 0.0, "yaw_rad": 0.0, "speed": 5.0, "length": 4.5, "width": 2.0}],
            "route_world": route.tolist(), "policy_py": str(policy_py),
            "policy_request": request, "frames_dir": None}
    s = W.Session(fc, fake_rig_mod, "127.0.0.1", 2000, allow_load_town=False)
    try:
        s.init(init)
        check("CONTROL: a town mismatch is refused, not silently loaded", False)
    except RuntimeError as exc:
        check("CONTROL: a town mismatch is refused, not silently loaded",
              "--allow-load-town" in str(exc) and not any(e[0] == "load_world" for e in world.log))

    world = FakeWorld("Town04")
    fc = make_fake_carla(world)
    s = W.Session(fc, fake_rig_mod, "127.0.0.1", 2000)
    init["town"] = "Town04"
    reply = s.init(init)
    check("world set synchronous at the policy's 20 Hz, 2 ticks per ProSim step",
          world.settings.synchronous_mode and abs(world.settings.fixed_delta_seconds - 0.05) < 1e-12
          and reply["substeps"] == 2, str(reply))
    check("ego spawned WITH physics as the MKZ, agents WITHOUT",
          s.ego.physics and s.ego.bp_id == W.EGO_BLUEPRINT and not s.actors[0].physics)
    check("the policy's own sensors() became the rig", reply["sensors"] == ["PCAM_F0"])

    world.log.clear()
    policy_mod = sys.modules["_sensor_policy"]
    n_before = len(policy_mod.SEEN)
    step = {"op": "step", "actors": [{"x": 21.0, "y": 0.0, "yaw_rad": 0.0, "speed": 5.0,
                                      "length": 4.5, "width": 2.0}]}
    out = s.step(step)
    kinds = [e[0] for e in world.log]
    check("the policy decides once per tick: 2 decisions for one ProSim step",
          len(policy_mod.SEEN) - n_before == 2, f"{len(policy_mod.SEEN) - n_before}")
    placed = [e for e in world.log if e[0] == "set_transform"]
    check("the agent is placed on the INTERPOLATED pose, then the target",
          len(placed) == 2 and abs(placed[0][2] - 20.5) < 1e-9 and abs(placed[1][2] - 21.0) < 1e-9,
          f"x = {[round(p[2], 3) for p in placed]}")
    ticks = [e[1] for e in world.log if e[0] == "tick"]
    caps = [e[1] for e in world.log if e[0] == "capture"]
    check("each capture asks for exactly the frame that tick produced",
          caps == ticks, f"ticks {ticks}, captures {caps}")
    first_apply = kinds.index("apply_control") if "apply_control" in kinds else -1
    second_tick = [i for i, k in enumerate(kinds) if k == "tick"][1]
    check("a held control is applied BEFORE the tick it is meant to drive",
          0 <= first_apply < second_tick, f"events {kinds}")
    obs = policy_mod.SEEN[-1]
    check("observation carries the rig, ego speed and a 20-point route",
          set(obs["sensor"]["cameras"]) == {"PCAM_F0"} and "speed_mps" in obs["ego"]
          and len(obs["route"]) == 20)
    check("the ego pose returned is the WORLD's, after physics -- not ProSim's",
          out["ego"]["x"] > 0.0 and out["ego"]["x"] == s.ego.get_transform().location.x,
          f"ego x {out['ego']['x']:.4f}")
    check("clamped control reaches the car (throttle 1.7 -> 1.0)",
          abs(out["control"]["throttle"] - 1.0) < 1e-9)

    summary = s.close()
    check("close destroys ego and agents and restores the world's settings",
          s.ego.destroyed and all(a.destroyed for a in s.actors)
          and not world.settings.synchronous_mode and s.rig.destroyed, str(summary))

    # ---------------------------------------------------------------- 3
    banner("3. the real socket: RemoteSensorEgoPolicy <-> serve()")
    world = FakeWorld("Town04")
    fc = make_fake_carla(world)
    port_box, ready = {}, threading.Event()

    def on_ready(port):
        port_box["port"] = port
        ready.set()

    args = types.SimpleNamespace(host="127.0.0.1", port=0, carla_host="127.0.0.1",
                                 carla_port=2000, osc2runner="", allow_load_town=False, once=True)
    th = threading.Thread(target=W.serve, args=(args, fc, fake_rig_mod, on_ready), daemon=True)
    th.start()
    check("worker came up", ready.wait(10), f"port {port_box.get('port')}")

    req_path = tmp / "policy.json"
    req_path.write_text(json.dumps(request))
    ego = RemoteSensorEgoPolicy(f"127.0.0.1:{port_box['port']}",
                                StraightRoute(0.0, 0.0, 0.0), policy_py=str(policy_py),
                                policy_request_path=str(req_path), town="Town04")
    st = VehicleState(x=0.0, y=0.0, heading=0.0, speed=2.0, length=4.8, width=2.1)
    nb = [VehicleState(x=20.0, y=0.0, heading=0.0, speed=5.0)]
    xs = []
    for k in range(3):
        nb = [VehicleState(x=20.0 + 0.5 * (k + 1), y=0.0, heading=0.0, speed=5.0)]
        st = ego.step(st, nb)
        xs.append(st.x)
    check("three ProSim steps round-trip, and the ego moves under CARLA's physics",
          len(xs) == 3 and xs[0] < xs[1] < xs[2], f"x {np.round(xs, 3).tolist()}")
    check("each reply reports 2 decisions per step",
          ego.last_reply.get("decisions") == 6, str(ego.last_reply.get("decisions")))
    check("init told ProSim which sensors actually attached",
          ego.init_reply.get("sensors") == ["PCAM_F0"])
    try:
        ego.step(st, nb + nb)
        check("CONTROL: a scene that changes size mid-rollout is refused", False)
    except RuntimeError:
        check("CONTROL: a scene that changes size mid-rollout is refused", True)
    ego.close()
    th.join(10)
    check("worker exits cleanly after the episode (--once)", not th.is_alive())

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
