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
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False,
                 manual_gear_shift=False, gear=0):
        self.throttle, self.steer, self.brake = throttle, steer, brake
        self.manual_gear_shift, self.gear = manual_gear_shift, gear


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

    def set_target_angular_velocity(self, w):
        self.w = w

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
            if not a.physics:
                continue
            if a.control is None:
                # A physics body with no pedals coasts on its target velocity for
                # the tick, as CARLA carries a placed agent.
                a.tf = Transform(Location(a.tf.location.x + a.v.x * dt,
                                          a.tf.location.y + a.v.y * dt, a.tf.location.z),
                                 a.tf.rotation)
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
    RM = W.load_route_module(W.DEFAULT_OSC2RUNNER)
    def route_of(points, yaw=0.0):     # a fresh plan each: progress is stateful
        return W.route_in_ego_frame(W.plan_from_route_world(RM, np.asarray(points, float)),
                                    0.0, 0.0, yaw)
    route = np.stack([np.arange(0, 60.0, 0.5), np.zeros(120)], axis=1)
    r = route_of(route)
    check("route: 20 points, first 2.5 m ahead, 1 m apart",
          len(r) == 20 and abs(r[0][0] - 2.5) < 1e-9 and abs(r[1][0] - r[0][0] - 1.0) < 1e-9,
          f"{len(r)} pts, first {np.round(r[0], 3).tolist()}")
    r_side = route_of([[0, 0], [5, 3], [10, 6]])
    r_mirror = route_of([[0, 0], [5, -3], [10, -6]])
    check("a route bending toward world +y has POSITIVE lateral (the right)",
          r_side[-1][1] > 0 and r_mirror[-1][1] < 0,
          f"+y -> {r_side[-1][1]:+.2f}, control -y -> {r_mirror[-1][1]:+.2f}")
    r_turned = route_of(np.stack([np.zeros(60), np.arange(60.0)], axis=1), math.pi / 2)
    check("with the ego facing world +y, that route is straight ahead",
          abs(r_turned[5][1]) < 1e-9 and r_turned[5][0] > 0, str(np.round(r_turned[5], 3).tolist()))
    mag = np.zeros((8, 8, 3), np.uint8); mag[..., 0] = 255; mag[..., 2] = 255
    tree = np.zeros((8, 8, 3), np.uint8); tree[..., 1] = 160
    check("magenta_fraction: magenta image 1.0; black, green and a lidar array 0",
          W.magenta_fraction(mag) == 1.0 and W.magenta_fraction(np.zeros((8, 8, 3), np.uint8)) == 0.0
          and W.magenta_fraction(tree) == 0.0 and W.magenta_fraction(np.zeros((50, 4))) == 0.0)
    try:
        import cv2
        real = {n: W.magenta_fraction(cv2.imread(f"tests/fixtures/frame_{n}.jpg")[..., ::-1])
                for n in ("pink_simlingo", "clean_simlingo", "clean_tfv6")}
        check("real frames: the broken SimLingo frame is over the limit, clean ones far under",
              real["pink_simlingo"] > W.PINK_FRACTION
              and max(real["clean_simlingo"], real["clean_tfv6"]) < W.PINK_FRACTION / 3,
              str({k: round(v, 3) for k, v in real.items()}))
    except ImportError:
        print("  (cv2 not installed: real-frame magenta check skipped)")
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
    # Path PARAMETERS (SimLingo's `weights`) are resolved against the harness
    # root too, but only when the resolved file exists.
    harness = Path(tempfile.mkdtemp())
    (harness / "third_party/simlingo/scenario_orchestration").mkdir(parents=True)
    real_py = harness / "third_party/simlingo/scenario_orchestration/policy.py"
    real_py.write_text("")
    (harness / "ckpts/sim").mkdir(parents=True)
    (harness / "ckpts/sim/model.pt").write_text("x")
    resolved = W.resolve_request_paths(
        {"checkpoint": "ckpts/sim", "parameters": {"weights": "ckpts/sim/model.pt",
                                                   "config_path": "ckpts/sim/missing.yaml",
                                                   "device": "cuda:0"}}, real_py)
    rp = resolved["parameters"]
    check("an existing relative `weights` resolves against the harness root",
          rp["weights"] == str(harness.resolve() / "ckpts/sim/model.pt"), rp["weights"])
    check("CONTROL: a relative path parameter that does not exist is left alone",
          rp["config_path"] == "ckpts/sim/missing.yaml", rp["config_path"])
    check("CONTROL: non-path parameters are untouched",
          rp["device"] == "cuda:0")
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
    check("ego AND agents spawned with physics on (so CARLA knows the agents' speed)",
          s.ego.physics and s.ego.bp_id == W.EGO_BLUEPRINT and s.actors[0].physics)
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
    check("the agent is placed at each tick's START pose (the previous target, then halfway)",
          len(placed) == 2 and abs(placed[0][2] - 20.0) < 1e-9 and abs(placed[1][2] - 20.5) < 1e-9,
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
    a0 = s.actors[0]
    check("each placed agent gets ProSim's speed as target velocity, no spin, 5 cm lift",
          abs(math.hypot(a0.v.x, a0.v.y) - 5.0) < 1e-9 and (a0.w.x, a0.w.y, a0.w.z) == (0.0, 0.0, 0.0)
          and abs(a0.tf.location.z - (0.2 + W.ACTOR_Z_OFFSET)) < 1e-9,
          f"speed {math.hypot(a0.v.x, a0.v.y):.2f}, z {a0.tf.location.z:.3f}")
    check("the step reply carries the guard numbers", set(out.get("checks", {})) ==
          {"actor_speed_err_max_mps", "magenta_max"}, str(out.get("checks")))

    logged = W.Session(make_fake_carla(FakeWorld("Town04")), fake_rig_mod, "127.0.0.1", 2000)
    run_dir = Path(tempfile.mkdtemp())
    logged.init(dict(init, frames_dir=str(run_dir / "ego_sensor_frames")))
    logged.step(step)
    logged.close()
    lines = [json.loads(l) for l in (run_dir / "ego_policy_log.jsonl").read_text().splitlines()]
    check("the per-decision log has one line per decision (2 for one ProSim step)",
          len(lines) == 2 and [l["decision"] for l in lines] == [0, 1], f"{len(lines)} lines")
    na = lines[-1]["nearest_agent"] if lines else None
    check("it names the nearest other car in the ego frame (where the world has it, 5 m/s)",
          na is not None and abs(na["forward_m"] - (logged.actors[0].tf.location.x - lines[-1]["ego"]["x"])) < 0.05
          and abs(na["speed_mps"] - 5.0) < 1e-6, str(na))

    # ---------------------------------------------------------------- 2c
    banner("2c. timing: at every capture the car is where ProSim has it at that instant")

    def captured_x(targets):
        """The car's x at each capture, driving it 1 m per 0.1 s step at 10 m/s."""
        sess = W.Session(make_fake_carla(FakeWorld("Town04")), fake_rig_mod, "127.0.0.1", 2000)
        sess.init(dict(init, actors=[{"x": 20.0, "y": 0.0, "yaw_rad": 0.0, "speed": 10.0,
                                      "length": 4.5, "width": 2.0}]))
        seen, original = [], sess.rig.capture

        def capture(frame=None):
            seen.append(round(sess.actors[0].get_transform().location.x, 6))
            return original(frame)
        sess.rig.capture = capture
        for x in targets:
            sess.step({"op": "step", "actors": [{"x": x, "y": 0.0, "yaw_rad": 0.0, "speed": 10.0,
                                                 "length": 4.5, "width": 2.0}]})
        sess.close()
        return seen

    got = captured_x([21.0, 22.0, 23.0])                 # end-of-step poses, what prosim_ego now hands over
    check("with end-of-step targets, each capture sees the car at ProSim's pose for that tick",
          np.allclose(got, [20.5, 21.0, 21.5, 22.0, 22.5, 23.0], atol=1e-6), str(got))
    late = captured_x([20.0, 21.0, 22.0])                # start-of-step poses, the old handoff
    # Its first step targets the spawn pose itself, so the gap only settles from
    # the second step's captures on.
    check("CONTROL: start-of-step targets (the old handoff) put the car a whole step (1 m) behind",
          np.allclose(np.array(got[2:]) - np.array(late[2:]), 1.0, atol=1e-6), f"old handoff {late}")
    check("and the policy's own report of the decision (its command, and a meta block)",
          lines and lines[-1].get("command") == "LANEFOLLOW" and isinstance(lines[-1].get("meta"), dict),
          f"command {lines[-1].get('command') if lines else None}, meta {lines[-1].get('meta') if lines else None}")
    check("and what was applied: route size, action, pedals",
          lines and lines[-1]["route"]["points"] == 20 and "control" in lines[-1]["action"]
          and lines[-1]["control"]["throttle"] == 1.0, str(lines[-1] if lines else None)[:200])

    summary = s.close()
    check("close destroys ego and agents and restores the world's settings",
          s.ego.destroyed and all(a.destroyed for a in s.actors)
          and not world.settings.synchronous_mode and s.rig.destroyed, str(summary))

    # ---------------------------------------------------------------- 2b
    banner("2b. the guards: a moving car's speed, and broken rendering")

    class StuckActor(FakeActor):
        def get_velocity(self):          # what CARLA reports for a physics-off car
            return Vector3D()

    class StuckWorld(FakeWorld):
        def try_spawn_actor(self, bp_id, tf):
            actor = (StuckActor if bp_id == W.ACTOR_BLUEPRINT else FakeActor)(self, bp_id, tf)
            self.actors.append(actor)
            return actor

    class PinkRig(FakeRig):
        def capture(self, frame=None):
            super().capture(frame)
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[..., 0] = 255
            img[..., 2] = 255
            return {sp["name"]: img for sp in self.specs}

    pink_rig_mod = types.SimpleNamespace(SensorRig=PinkRig, specs_from=lambda d: list(d))

    def drive(world, rig_mod, n_steps):
        """Step until the worker stops the episode: (decisions made, error or None, session)."""
        sess = W.Session(make_fake_carla(world), rig_mod, "127.0.0.1", 2000)
        sess.init(dict(init))
        try:
            for k in range(n_steps):
                sess.step({"op": "step", "actors": [{"x": 21.0 + k, "y": 0.0, "yaw_rad": 0.0,
                                                     "speed": 5.0, "length": 4.5, "width": 2.0}]})
        except RuntimeError as exc:
            return sess.decisions, str(exc), sess
        return sess.decisions, None, sess

    n, err, good = drive(FakeWorld("Town04"), fake_rig_mod, 15)
    check("a normal episode runs 30 decisions with neither guard firing",
          err is None and n == 30, f"{n} decisions, error {err}")
    check("and its checks show the agent's reported speed matching ProSim's",
          good.checks["actor_speed_err_max_mps"] < 1e-9 and good.checks["magenta_max"] == 0.0,
          str(good.checks))
    n, err, _ = drive(StuckWorld("Town04"), fake_rig_mod, 15)
    check("CONTROL: an agent reported at 0 m/s while moving stops the episode on tick 20",
          err is not None and "reports its speed" in err and n == W.STOPPED_TICKS - 1,
          f"{n} decisions before stopping: {(err or '')[:60]}")
    n, err, _ = drive(FakeWorld("Town04"), pink_rig_mod, 15)
    check("CONTROL: magenta camera frames stop the episode on decision 10",
          err is not None and "magenta" in err and n == W.PINK_TICKS - 1,
          f"{n} decisions before stopping: {(err or '')[:60]}")

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
    log = ego.metadata().get("light_log") or []
    check("the light readout of init and every step reaches ProSim's metadata, in step order",
          [e.get("step") for e in log] == [0, 1, 2, 3], str([e.get("step") for e in log]))
    try:
        ego.step(st, nb + nb)
        check("CONTROL: a scene that changes size mid-rollout is refused", False)
    except RuntimeError:
        check("CONTROL: a scene that changes size mid-rollout is refused", True)
    ego.close()
    th.join(10)
    check("worker exits cleanly after the episode (--once)", not th.is_alive())

    # ---------------------------------------------------------------- 4
    banner("4. the ego's junction phase: osc2runner's real signals.apply")

    class TLS:
        Green, Yellow, Red = "Green", "Yellow", "Red"

    sys.modules.setdefault("carla", types.SimpleNamespace(TrafficLightState=TLS))
    events = []

    class FakeLight:
        def __init__(self, lid, approach_yaw):
            self.id, self.yaw, self.state = lid, approach_yaw, None
            self.group = [self]
        def get_stop_waypoints(self):
            return [types.SimpleNamespace(transform=Transform(rotation=Rotation(yaw=self.yaw)))]
        def get_group_traffic_lights(self):
            return self.group
        def set_state(self, state):
            self.state = state; events.append(("set_state", self.id, state))
        def freeze(self, flag):
            events.append(("freeze", self.id, flag))
        def get_state(self):
            return self.state

    ego_light, opposite = FakeLight(1, 0.0), FakeLight(2, 180.0)
    cross_a, cross_b = FakeLight(3, 90.0), FakeLight(4, -90.0)
    decoy = FakeLight(9, 90.0)                       # nearer, but faces across the ego's road
    group = [ego_light, opposite, cross_a, cross_b]
    for light in group:
        light.group = group
    marks = {10.0: decoy, 30.0: ego_light}

    class SigWaypoint:
        transform = Transform(rotation=Rotation(yaw=0.0))
        def get_landmarks_of_type(self, dist, kind, stop_at_junction):
            return [types.SimpleNamespace(distance=d) for d in sorted(marks)]
    sig_map = types.SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: SigWaypoint())
    sig_world = types.SimpleNamespace(get_traffic_light=lambda mark: marks[mark.distance])
    sig_actor = types.SimpleNamespace(get_location=lambda: Location(0.0, 0.0, 0.0))

    signals = W.load_signals_module(W.DEFAULT_OSC2RUNNER)
    note = signals.apply(sig_world, sig_map, sig_actor, "green")
    states = {l.id: l.state for l in group}
    check("the ego's light is the one FACING its approach, not the nearer decoy",
          note.get("ego_light_id") == 1 and decoy.state is None, str(note.get("ego_light_id")))
    check("ego and opposite approach green, crossing approaches red",
          states == {1: "Green", 2: "Green", 3: "Red", 4: "Red"}, str(states))
    set_idx = [i for i, e in enumerate(events) if e[0] == "set_state"]
    freeze_idx = [i for i, e in enumerate(events) if e[0] == "freeze"]
    check("frozen once, after every state was set",
          note.get("frozen") is True and len(freeze_idx) == 1 and freeze_idx[0] > max(set_idx),
          f"events {events}")
    events.clear()
    for light in group:
        light.state = None
    none_note = signals.apply(sig_world, sig_map, sig_actor, None)
    check("CONTROL: no declared phase touches nothing",
          not events and none_note.get("requested") is None, str(none_note))

    calls = []

    class RecordingSignals:
        def apply(self, world, carla_map, actor, phase):
            calls.append((actor, phase, world.frame))
            return {"requested": phase, "applied": phase, "ego_light_id": 1, "frozen": True}

    for light_msg, expect_call in (("green", True), (None, False)):
        world = FakeWorld("Town10HD_Opt")
        s2 = W.Session(make_fake_carla(world), fake_rig_mod, "127.0.0.1", 2000)
        s2.signals = RecordingSignals()
        calls.clear()
        msg = dict(init, town="Town10HD_Opt", ego_light=light_msg)
        reply2 = s2.init(msg)
        ticks_after = sum(1 for e in world.log if e[0] == "tick" and calls and e[1] > calls[0][2])
        if expect_call:
            check("Session.init applies the declared phase to the spawned EGO, then ticks",
                  len(calls) == 1 and calls[0][0] is s2.ego and calls[0][1] == "green"
                  and ticks_after >= 1 and reply2["ego_light"]["applied"] == "green",
                  f"calls {[(c[1]) for c in calls]}, ticks after {ticks_after}, reply {reply2['ego_light']}")
        else:
            check("CONTROL: no ego_light in the init message -> signals.apply is never called",
                  not calls and reply2["ego_light"]["requested"] is None, str(reply2["ego_light"]))
        s2.close()

    class JunctionSignals(RecordingSignals):
        def junction_lights(self, world, carla_map, actor):
            return [(ego_light, "ego"), (opposite, "opposing"),
                    (cross_a, "crossing"), (cross_b, "crossing")]

    for light, state in zip(group, ("Green", "Green", "Red", "Red")):
        light.state = state
    one_agent = {"op": "step", "actors": [{"x": 21.0, "y": 0.0, "yaw_rad": 0.0, "speed": 5.0,
                                           "length": 4.5, "width": 2.0}]}
    s3 = W.Session(make_fake_carla(FakeWorld("Town10HD_Opt")), fake_rig_mod, "127.0.0.1", 2000)
    s3.signals = JunctionSignals()
    reply3 = s3.init(dict(init, town="Town10HD_Opt", ego_light="green"))
    check("init names each junction light's role for the trace",
          reply3.get("signal_roles") == {"1": "ego", "2": "opposing", "3": "crossing", "4": "crossing"},
          str(reply3.get("signal_roles")))
    out3 = s3.step(one_agent)
    check("each step reads the lights back: ego and opposing green, crossing red",
          out3.get("lights") == {"ego": ["Green"], "opposing": ["Green"], "crossing": ["Red", "Red"],
                                 "lights": {"1": "Green", "2": "Green", "3": "Red", "4": "Red"}},
          str(out3.get("lights")))
    ego_light.state = "Red"
    out4 = s3.step(one_agent)
    check("a light that changes is read as changed, not cached from init",
          out4["lights"]["ego"] == ["Red"] and out4["lights"]["lights"]["1"] == "Red",
          str(out4["lights"]))
    s3.close()
    s4 = W.Session(make_fake_carla(FakeWorld("Town10HD_Opt")), fake_rig_mod, "127.0.0.1", 2000)
    s4.signals = RecordingSignals()
    reply4 = s4.init(dict(init, town="Town10HD_Opt", ego_light="green"))
    check("CONTROL: a signals module without junction_lights reports no lights at all",
          reply4.get("signal_roles") == {} and reply4["lights"]["ego"] == []
          and s4.step(one_agent)["lights"]["lights"] == {}, str(reply4.get("lights")))
    s4.close()

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
