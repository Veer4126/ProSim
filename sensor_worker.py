"""CARLA's side of the sensor loop: the world, the ego's physics, the policy's rig.

Runs NATIVELY on the node that hosts the rendering CARLA server, in a Python 3.10
environment that carries both a `carla` client matching the server and the
policy's own inference stack (for TFv6: torch 2.5). ProSim connects to it over a
local socket (`sensor_protocol.py`); see `remote_ego.py` for the other half.

    source /scratch/veerk41/venvs/tfv6/bin/activate
    python sensor_worker.py --port 2100 --carla-port 2000

WHAT IS REUSED, AND FROM WHERE
------------------------------
The rig is `osc2runner`'s own `osc2carla/backend/sensors.py`, loaded by path:
the harness's worked example of the method side of the sensor contract, already
validated end to end with TFv6 (reports/sensor_rig_validation.md). It attaches
exactly what the policy's `sensors()` declares and captures frame-stamped, so a
policy is never handed an image from an earlier tick. Loading it by path rather
than as `osc2carla.backend.sensors` matters: the package `__init__` imports the
whole scenario engine.

The route slice is osc2runner's too: `osc2carla/backend/route.py`'s `RoutePlan`,
also loaded by path, so a policy here reads the route exactly as it does on the
OSC2 arm -- including monotone progress along the frozen plan, which keeps an
ego that over-rotates in a junction from being handed the approach it already
drove (tests/test_worker_route_vs_osc2runner.py).

The control loop follows the orchestration port's `PolicyEgoDriver`: apply the
held control, tick, capture the rig for that frame, ask the policy, hold the
answer for the next tick.

WHAT EACH PROSIM STEP DOES
-------------------------
ProSim steps at 0.1 s; the world ticks at the policy's own rate (20 Hz for
TFv6, two ticks per step). The other agents are placed on interpolated poses
between ProSim's, with physics on and a target velocity, so they appear where
the rollout put them AND CARLA (and radar) knows how fast they move; the ego
keeps CARLA physics, so its motion is what the
policy's pedals and steering actually produce. The ego's pose after the last
tick goes back to ProSim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_protocol import PROTOCOL_VERSION, receive, send  # noqa: E402

DEFAULT_OSC2RUNNER = "/scratch/veerk41/scenario_orchestration/third_party/osc2runner"

#: The vehicle the CARLA Leaderboard -- and so TFv6's training data -- drives.
#: The rig's mountings (cameras at z = 2.25 m) are relative to this body.
EGO_BLUEPRINT = "vehicle.lincoln.mkz_2020"
ACTOR_BLUEPRINT = "vehicle.tesla.model3"

#: ProSim's cars are moved onto their poses with physics ON plus a target
#: velocity, as orchestration's carla_sync.py PHYSICS mode does. With physics
#: off CARLA reports their speed as 0 -- measured 2026-09-14: a car moved at
#: 8 m/s read 0.00 m/s, and so did a radar aimed at it -- so a radar-reading
#: policy (TFv6) saw a moving hero as a parked car. Lifted by orchestration's 5 cm.
ACTOR_Z_OFFSET = 0.05
#: Guard: a car ProSim moves faster than MOVING_MPS that CARLA reports under
#: STOPPED_MPS for STOPPED_TICKS ticks in a row stops the episode.
MOVING_MPS, STOPPED_MPS, STOPPED_TICKS = 3.0, 0.5, 20
#: Guard: after a town reload CARLA can render missing materials as magenta
#: (SimLingo red_light/left_turn/right_turn, 2026-09-14: 10-18% of pixels, clean
#: frames 0%). PINK_TICKS decisions in a row above PINK_FRACTION stop the episode.
PINK_FRACTION, PINK_TICKS = 0.03, 10

#: The route a waypoint- or command-conditioned policy reads: one point per
#: metre starting 2.5 m ahead, 20 points (osc2runner's ego-frame convention).
ROUTE_FIRST_M, ROUTE_STEP_M, ROUTE_POINTS = 2.5, 1.0, 20


# --------------------------------------------------------------------------
# the reused rig
# --------------------------------------------------------------------------

def load_rig_module(osc2runner: str):
    """osc2runner's simapi + sensors, as a two-module package, CARLA bound."""
    backend = Path(osc2runner) / "osc2carla" / "backend"
    name = "_osc2_rig"
    pkg = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None, is_package=True))
    pkg.__path__ = [str(backend)]
    sys.modules[name] = pkg
    mods = {}
    for mod in ("simapi", "sensors"):
        spec = importlib.util.spec_from_file_location(f"{name}.{mod}",
                                                      backend / f"{mod}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.{mod}"] = module
        spec.loader.exec_module(module)
        mods[mod] = module
        if mod == "simapi":
            module.bind("carla")
    return mods["sensors"]


#: Request parameters that name a file relative to the harness root.
PATH_PARAMETERS = ("weights", "checkpoint", "config", "config_path", "model_path", "weights_path")


def load_signals_module(osc2runner: str):
    """osc2runner's `osc2carla/backend/signals.py`, by path. It imports nothing
    at module level beyond `typing` (and `carla` only inside `apply`), so it
    loads standalone. Reused rather than re-implemented so the ego's junction
    phase is set exactly the way the OSC2 arm sets it."""
    path = Path(osc2runner) / "osc2carla" / "backend" / "signals.py"
    spec = importlib.util.spec_from_file_location("_osc2_signals", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_osc2_signals"] = module
    spec.loader.exec_module(module)
    return module


def load_route_module(osc2runner: str):
    """osc2runner's `osc2carla/backend/route.py`, by path. Stdlib only at module
    level, so it loads standalone."""
    path = Path(osc2runner) / "osc2carla" / "backend" / "route.py"
    spec = importlib.util.spec_from_file_location("_osc2_route", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_osc2_route"] = module
    spec.loader.exec_module(module)
    return module


def load_actuation_module(osc2runner: str):
    """osc2runner's `osc2carla/backend/actuation.py` (SpawnGear, AccelerationTracker),
    by path: stdlib only, so it loads without the scenario engine. Every
    osc2runner policy is actuated through it (osc2carla/backend/policy.py tick)."""
    path = Path(osc2runner) / "osc2carla" / "backend" / "actuation.py"
    spec = importlib.util.spec_from_file_location("_osc2_actuation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_osc2_actuation"] = module
    spec.loader.exec_module(module)
    return module


def load_state_modules(osc2runner: str) -> Dict[str, Any]:
    """What osc2runner's policy bridge uses for a `state` policy: its observation
    builder (carla_state_obs.py), the policy repository's BEV renderer (bev.py)
    and the bridge itself (for `_to_command`). Each file is loaded by path --
    this repository has a `scenario_orchestration` package of its own -- with
    osc2runner's root appended to sys.path so their `osc2carla` imports resolve."""
    root = str(Path(osc2runner))
    if root not in sys.path:
        sys.path.append(root)
    mods = {}
    for name in ("carla_state_obs", "bev", "osc2carla_policy_bridge"):
        path = Path(root) / "scenario_orchestration" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_osc2_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        mods[name] = module
    return mods


def same_town(loaded: str, wanted: str) -> bool:
    """'Carla/Maps/Town10HD_Opt' is Town10HD: the layered build has the same
    roads, and the harness names it either way."""
    def norm(name):
        name = str(name).split("/")[-1].lower()
        return name[:-4] if name.endswith("_opt") else name
    return norm(loaded) == norm(wanted)


def resolve_request_paths(request: Dict[str, Any], policy_py: Path) -> Dict[str, Any]:
    """The harness writes `checkpoint` (and conventional path parameters such as
    SimLingo's `weights`) relative to ITS root (`third_party/checkpoints/...`),
    and this worker's cwd is anywhere. The policy lives at
    <root>/third_party/<repo>/scenario_orchestration/policy.py, so the root is
    three directories above its folder. `checkpoint` is always resolved; the
    parameters only when the resolved path exists, so an unrelated string that
    merely looks relative is left alone."""
    request = dict(request)
    root = Path(policy_py).resolve().parents[3]
    checkpoint = request.get("checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        request["checkpoint"] = str(root / checkpoint)
    parameters = dict(request.get("parameters") or {})
    for key in PATH_PARAMETERS:
        value = parameters.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute() \
                and (root / value).exists():
            parameters[key] = str(root / value)
    if parameters:
        request["parameters"] = parameters
    return request


def load_policy(policy_py: str, request: Dict[str, Any]):
    policy_py = Path(policy_py)
    request = resolve_request_paths(request, policy_py)
    root = str(policy_py.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("_sensor_policy", policy_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sensor_policy"] = module
    spec.loader.exec_module(module)
    policy = module.build_policy(dict(request))
    for hook in ("load", "reset"):
        fn = getattr(policy, hook, None)
        if callable(fn):
            fn()
    return policy


# --------------------------------------------------------------------------
# geometry -- pure functions, tested without a simulator
# --------------------------------------------------------------------------

def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def lerp_pose(a: Dict[str, float], b: Dict[str, float], f: float) -> Dict[str, float]:
    """Interpolate a pose; heading along the shorter arc."""
    return {"x": a["x"] + (b["x"] - a["x"]) * f,
            "y": a["y"] + (b["y"] - a["y"]) * f,
            "yaw_rad": a["yaw_rad"] + wrap(b["yaw_rad"] - a["yaw_rad"]) * f,
            "speed": a["speed"] + (b["speed"] - a["speed"]) * f}


def plan_from_route_world(route_mod, route_world, start_xy=None):
    """ProSim's frozen world route as an osc2runner `RoutePlan`, or None.

    `RoutePlan` indexes by arc length at a fixed `step_m`, so the lane-graph
    polyline is resampled to `route.DEFAULT_STEP_M` first. Headings are the
    segment directions; osc2runner reads them only to interpolate.

    `start_xy` is the ego's spawn. osc2runner walks its plan from there, and
    `RoutePlan.locate` searches only 60 points past a cursor that starts at 0;
    ProSim's route can begin far behind the ego (72 m on overtake), which left
    the first slice 10.8 m behind the car. So the route is cut at the spawn's
    projection onto it."""
    if route_world is None or len(route_world) < 2:
        return None
    pts = np.asarray(route_world, dtype=float)[:, :2]
    if start_xy is not None:
        p = np.asarray(start_xy, dtype=float)[:2]
        a, ab = pts[:-1], np.diff(pts, axis=0)
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        foot = a + t[:, None] * ab
        k = int(np.argmin(np.linalg.norm(foot - p, axis=1)))
        pts = np.concatenate([foot[k:k + 1], pts[k + 1:]])
        if len(pts) < 2:
            return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], seg > 1e-9])]
    if len(pts) < 2:
        return None
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    step = float(route_mod.DEFAULT_STEP_M)
    want = np.arange(0.0, s[-1] + 1e-9, step)
    x, y = np.interp(want, s, pts[:, 0]), np.interp(want, s, pts[:, 1])
    if len(x) < 2:
        return None
    h = np.arctan2(np.diff(y), np.diff(x))
    h = np.append(h, h[-1])
    return route_mod.RoutePlan(points=[(float(a), float(b), float(c)) for a, b, c in zip(x, y, h)],
                               step_m=step)


def route_in_ego_frame(plan, x: float, y: float, yaw_rad: float) -> List[List[float]]:
    """The route as the policy reads it: osc2runner's slice of the frozen plan
    (`RoutePlan.ahead` -- arc length from the ego's monotone progress point, 20
    points 1 m apart from 2.5 m ahead, short when the plan runs out), in the ego
    frame (+x forward, +y right, CARLA's left-handed convention), the transform
    of `carla_state_obs.EgoFrame.to_ego`. `plan` is stateful: one per episode."""
    if plan is None:
        return []
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    out = []
    for px, py, _heading in plan.ahead(x, y, first_m=ROUTE_FIRST_M,
                                       step_m=ROUTE_STEP_M, count=ROUTE_POINTS):
        dx, dy = px - x, py - y
        out.append([dx * c + dy * s, -dx * s + dy * c])
    return out


def magenta_fraction(image) -> float:
    """Share of pixels in the magenta CARLA paints for a missing material: red
    and blue both bright, green well below red. `image` is RGB, HxWx3 (or 4);
    anything else (lidar, radar) is 0."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3 or img.size == 0:
        return 0.0
    img = img[::4, ::4, :3].astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return float(((r > 150) & (b > 150) & (g < 0.6 * r)).mean())


def place_actor(carla_mod, actor, transform, pose: Dict[str, float]) -> None:
    """Put a ProSim-driven car on its pose so CARLA also knows how fast it is
    going: the transform, a target velocity along its heading, and no spin
    (orchestration's carla_sync.apply_one). Needs the actor's physics ON."""
    actor.set_transform(transform)
    actor.set_target_velocity(carla_mod.Vector3D(
        x=pose["speed"] * math.cos(pose["yaw_rad"]),
        y=pose["speed"] * math.sin(pose["yaw_rad"]), z=0.0))
    actor.set_target_angular_velocity(carla_mod.Vector3D(x=0.0, y=0.0, z=0.0))


def to_control_fields(action: Dict[str, Any]) -> Dict[str, float]:
    """The action's `control` block, clamped. No sign flip: `steer` is already
    CARLA's own convention (+ = right)."""
    control = action.get("control") if isinstance(action, dict) else None
    if not isinstance(control, dict):
        raise RuntimeError(
            "sensor policy returned no `control` block; this loop actuates "
            f"pedals and steering directly (action keys: {sorted(action or {})})")
    clamp01 = lambda v: max(0.0, min(1.0, float(v or 0.0)))
    return {"throttle": clamp01(control.get("throttle")),
            "brake": clamp01(control.get("brake")),
            "steer": max(-1.0, min(1.0, float(control.get("steer") or 0.0)))}


# --------------------------------------------------------------------------
# one episode
# --------------------------------------------------------------------------

class Session:
    def __init__(self, carla_mod, rig_mod, carla_host: str, carla_port: int,
                 allow_load_town: bool = False):
        self.carla = carla_mod
        self.rig_mod = rig_mod
        self.carla_host, self.carla_port = carla_host, int(carla_port)
        self.allow_load_town = allow_load_town
        self.world = self.ego = self.rig = self.policy = None
        self.actors: List[Any] = []
        self.prev: List[Dict[str, float]] = []
        self.route_world: Optional[np.ndarray] = None
        self.route_mod = None                       # osc2runner's route.py
        self.route_plan = None
        self.original_settings = None
        self.control = None
        self.substeps = 1
        self.frames_dir: Optional[Path] = None
        self.decisions = 0
        self.last_meta: Dict[str, Any] = {}
        self.stopped_ticks: List[int] = []
        self.pink_ticks = 0
        self.checks: Dict[str, float] = {"actor_speed_err_max_mps": 0.0, "magenta_max": 0.0}
        self.lights: List[Any] = []                 # [(CARLA light, role)] read back each step
        self.observation_space = "sensor"
        self.action_space = "control"
        self.observer = None                        # osc2runner StateObservationBuilder
        self.bev = None
        self.bev_note: Optional[str] = None
        self.state_mods: Optional[Dict[str, Any]] = None
        self.actuation = None
        self.spawn_gear = None
        self.tracker = None
        self.log_file = None                        # ego_policy_log.jsonl, one line per decision

    # -- init -----------------------------------------------------------
    def _ground_z(self, x: float, y: float) -> float:
        loc = self.carla.Location(x=x, y=y, z=0.0)
        wp = self.world.get_map().get_waypoint(loc, project_to_road=True)
        return float(wp.transform.location.z) if wp is not None else 0.0

    def _transform(self, pose: Dict[str, float], lift: float):
        return self.carla.Transform(
            self.carla.Location(x=pose["x"], y=pose["y"],
                                z=self._ground_z(pose["x"], pose["y"]) + lift),
            self.carla.Rotation(yaw=math.degrees(pose["yaw_rad"])))

    def init(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        carla = self.carla
        client = carla.Client(self.carla_host, self.carla_port)
        client.set_timeout(120.0)
        self.world = client.get_world()
        town = str(msg["town"])
        loaded = self.world.get_map().name.split("/")[-1]
        if not same_town(loaded, town):
            if not self.allow_load_town:
                raise RuntimeError(
                    f"server has {loaded} loaded, episode needs {town}. Loading a "
                    "town under someone else's live run corrupts both, so this "
                    "worker does not do it unless started with --allow-load-town")
            self.world = client.load_world(town)

        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        policy_hz = float(msg.get("policy_hz") or 20.0)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / policy_hz
        self.world.apply_settings(settings)
        self.substeps = max(1, int(round(float(msg["dt"]) * policy_hz)))

        library = self.world.get_blueprint_library()
        ego_pose = msg["ego"]
        self.ego = self.world.try_spawn_actor(library.find(EGO_BLUEPRINT),
                                              self._transform(ego_pose, 0.3))
        if self.ego is None:
            raise RuntimeError(f"could not spawn the ego at {ego_pose}")
        self.ego.set_simulate_physics(True)

        self.prev = [dict(p) for p in msg.get("actors") or []]
        for pose in self.prev:
            actor = self.world.try_spawn_actor(library.find(ACTOR_BLUEPRINT),
                                               self._transform(pose, 0.3))
            if actor is None:
                raise RuntimeError(f"could not spawn an agent at {pose}")
            actor.set_simulate_physics(True)        # so its speed reaches CARLA
            self.actors.append(actor)

        self.route_world = np.asarray(msg.get("route_world") or [], dtype=float)
        if self.route_mod is None:
            self.route_mod = load_route_module(DEFAULT_OSC2RUNNER)
        self.route_plan = plan_from_route_world(self.route_mod, self.route_world,
                                                start_xy=(ego_pose["x"], ego_pose["y"]))
        self.policy = load_policy(msg["policy_py"], msg["policy_request"])
        request = msg["policy_request"] or {}
        self.observation_space = str(request.get("observation_space") or "sensor")
        self.action_space = str(request.get("action_space") or "control")
        # Every policy is actuated the way osc2runner actuates it: a spawn phase,
        # then an acceleration demand realised closed-loop.
        if self.actuation is None:
            self.actuation = load_actuation_module(DEFAULT_OSC2RUNNER)
        self.spawn_gear = self.actuation.SpawnGear()
        self.tracker = self.actuation.AccelerationTracker()
        if self.observation_space == "state":
            # osc2runner's object-centric observation of this CARLA world, with
            # the policy repository's own BEV renderer (osc2carla_policy_bridge.py
            # _build_observer), on ProSim's frozen route rather than a fresh walk.
            if self.state_mods is None:
                self.state_mods = load_state_modules(DEFAULT_OSC2RUNNER)
            repository = str(Path(msg["policy_py"]).resolve().parent.parent)
            try:
                self.bev = self.state_mods["bev"].build(repository).prepare(self.ego)
                self.bev_note = "prepared"
            except Exception as exc:                # reported: a policy needing it refuses
                self.bev, self.bev_note = None, f"unavailable: {type(exc).__name__}: {exc}"
            self.observer = self.state_mods["carla_state_obs"].StateObservationBuilder(
                world=self.world, carla_map=self.world.get_map(), bev=self.bev)
            self.observer.plan = self.route_plan

        sensors = getattr(self.policy, "sensors", None)
        declared = sensors() if callable(sensors) else []
        self.rig = self.rig_mod.SensorRig(
            self.world, self.ego, self.rig_mod.specs_from(declared)).spawn()
        if declared and not self.rig.active:
            raise RuntimeError(f"no sensor of the policy's rig attached: {self.rig.failed}")

        frame = self.world.tick()                   # settle; drain first frames
        self.ego.set_target_velocity(self._forward_velocity(ego_pose))
        if self.rig.active:
            self.rig.capture(frame)

        # The ego's junction phase, before the policy's first decision: the
        # light governing the ego's approach (and its group) is set and frozen
        # by osc2runner's own signals.apply. No declared phase leaves CARLA's
        # cycle alone.
        declared = msg.get("ego_light")
        signals = getattr(self, "signals", None)
        if declared and signals is not None:
            self.signal_note = signals.apply(self.world, self.world.get_map(),
                                             self.ego, declared)
            frame = self.world.tick()
            if self.rig.active:
                self.rig.capture(frame)
        elif declared:
            self.signal_note = {"requested": declared, "applied": None,
                                "note": "this worker has no signals module"}
        else:
            self.signal_note = {"requested": None,
                                "note": "no phase declared; CARLA's own cycle applies"}
        # The group governing the ego, read back after every step so the trace
        # records what the lights showed -- osc2runner's trace.py does the same
        # (_declare_signals / _record_signals). Empty when no light faces the ego.
        junction = getattr(signals, "junction_lights", None)
        self.lights = (list(junction(self.world, self.world.get_map(), self.ego))
                       if callable(junction) else [])

        if msg.get("frames_dir"):
            # Camera frames only for a policy with cameras; the per-decision log
            # beside them for every policy, so a run can be read in numbers.
            if self.rig.active:
                self.frames_dir = Path(msg["frames_dir"])
                self.frames_dir.mkdir(parents=True, exist_ok=True)
            log_path = Path(msg["frames_dir"]).parent / "ego_policy_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, "w")
        return {"ok": True, "map": self.world.get_map().name,
                "tick_hz": policy_hz, "substeps": self.substeps,
                "sensors": self.rig.names, "failed": self.rig.failed,
                "sensor_hz": self.rig.sensor_hz,
                "ego_light": getattr(self, "signal_note", None),
                "signal_roles": {str(light.id): role for light, role in self.lights},
                "observation_space": self.observation_space, "bev": self.bev_note,
                "lights": self._read_lights()}

    def _forward_velocity(self, pose: Dict[str, float]):
        return self.carla.Vector3D(x=pose["speed"] * math.cos(pose["yaw_rad"]),
                                   y=pose["speed"] * math.sin(pose["yaw_rad"]), z=0.0)

    # -- step -----------------------------------------------------------
    def step(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        carla = self.carla
        target = [dict(p) for p in msg.get("actors") or []]
        if len(target) != len(self.actors):
            raise RuntimeError(f"expected {len(self.actors)} agents, got {len(target)}")
        frame = None
        for k in range(1, self.substeps + 1):
            f = k / self.substeps
            poses = []
            for actor, a, b in zip(self.actors, self.prev, target):
                # Placed where it is at the START of this tick, moving at that
                # speed: the tick's physics carries it to its pose at the tick's
                # END, the instant the ego and the cameras are read. Placed at
                # the end pose, it overshot by one tick of travel.
                start = lerp_pose(a, b, (k - 1) / self.substeps)
                place_actor(carla, actor, self._transform(start, ACTOR_Z_OFFSET), start)
                poses.append(lerp_pose(a, b, f))
            if self.control is not None:
                self.ego.apply_control(self.control)
            frame = self.world.tick()
            cameras = self.rig.capture(frame) if self.rig.active else {}
            self._check_tick(poses, cameras)
            tf = self.ego.get_transform()
            v = self.ego.get_velocity()
            yaw = math.radians(tf.rotation.yaw)
            dt = float(self.world.get_settings().fixed_delta_seconds or 0.05)
            observation = {
                "t": self.decisions * dt,
                "ego": {"speed_mps": math.hypot(v.x, v.y)},
                "route": route_in_ego_frame(self.route_plan, tf.location.x,
                                            tf.location.y, yaw),
            }
            if self.observer is not None:
                observation.update(self.observer.build(self.ego, math.hypot(v.x, v.y)))
            if self.rig.active:
                observation["sensor"] = {"cameras": cameras}
            action = self.policy.act(observation)
            self.control = self._actuate(action, dt)
            self._log_decision(observation, action, self.control, tf, v)
            self.last_meta = dict(action.get("meta") or {}) if isinstance(action, dict) else {}
            self.decisions += 1
            self._save_frame(cameras, frame)
        self.prev = target

        tf = self.ego.get_transform()
        v = self.ego.get_velocity()
        return {"ok": True, "frame": frame, "decisions": self.decisions,
                "ego": {"x": tf.location.x, "y": tf.location.y,
                        "yaw_rad": math.radians(tf.rotation.yaw),
                        "speed": math.hypot(v.x, v.y), "vx": v.x, "vy": v.y},
                "control": {"throttle": self.control.throttle,
                            "brake": self.control.brake,
                            "steer": self.control.steer},
                "meta": self.last_meta, "dropped": dict(self.rig.dropped),
                "checks": dict(self.checks), "lights": self._read_lights()}

    def _log_decision(self, observation: Dict[str, Any], action: Any, control, tf, v) -> None:
        """One line of ego_policy_log.jsonl: what the policy was given and what
        was applied, in numbers -- the nearest other car in the ego frame, the
        route it was handed, its action, the pedals and gear, the lights."""
        if self.log_file is None:
            return
        yaw = math.radians(tf.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        nearest = None
        for i, actor in enumerate(self.actors):
            at, av = actor.get_transform(), actor.get_velocity()
            dx, dy = at.location.x - tf.location.x, at.location.y - tf.location.y
            d = math.hypot(dx, dy)
            if nearest is None or d < nearest["distance_m"]:
                nearest = {"agent": i, "distance_m": round(d, 2),
                           "forward_m": round(dx * c + dy * s, 2),
                           "right_m": round(-dx * s + dy * c, 2),
                           "speed_mps": round(math.hypot(av.x, av.y), 2)}
        route = observation.get("route") or []
        act = action if isinstance(action, dict) else {}
        meta = act.get("meta") if isinstance(act.get("meta"), dict) else {}
        line = {
            "t": round(float(observation.get("t") or 0.0), 3), "decision": self.decisions,
            "ego": {"x": round(tf.location.x, 2), "y": round(tf.location.y, 2),
                    "yaw_deg": round(tf.rotation.yaw, 1), "speed_mps": round(math.hypot(v.x, v.y), 2)},
            "route": {"points": len(route),
                      "first": [round(float(p), 2) for p in route[0]] if route else None,
                      "last": [round(float(p), 2) for p in route[-1]] if route else None},
            "nearest_agent": nearest,
            "objects_seen": len(observation["objects"]) if "objects" in observation else None,
            "speed_limit_kph": observation.get("speed_limit_kph"),
            "action": {k: act[k] for k in ("acceleration_mps2", "target_speed_mps", "steer", "control")
                       if k in act},
            "command": meta.get("command"),
            # What the policy says about its own decision: tfv6's next command and
            # the target points it was conditioned on, SimLingo's commentary,
            # PlanT 2.0's settling flag.
            "meta": {k: meta[k] for k in ("next_command", "target_points", "language", "settling")
                     if k in meta},
            "control": {"throttle": round(float(control.throttle), 3), "brake": round(float(control.brake), 3),
                        "steer": round(float(control.steer), 3), "gear": int(getattr(control, "gear", 0) or 0)},
            "lights": self._read_lights() if self.lights else None,
            "checks": dict(self.checks),
        }
        self.log_file.write(json.dumps(line, default=str) + "\n")

    def _actuate(self, action: Any, dt: float):
        """The policy's action as the VehicleControl for the next tick, realised
        the way osc2runner's ExternalEgoController.tick does
        (osc2carla/backend/policy.py): the spawn phase first, then an
        acceleration demand through AccelerationTracker, pedals as given."""
        if self.observation_space == "state":
            cmd = self.state_mods["osc2carla_policy_bridge"]._to_command(action, self.action_space)
            throttle, brake, steer, accel = cmd.throttle, cmd.brake, cmd.steer, cmd.accel
        else:
            fields = to_control_fields(action)
            throttle, brake, steer, accel = fields["throttle"], fields["brake"], fields["steer"], None
        speed = self.actuation.horizontal_speed(self.ego)
        pedals, gear = self.spawn_gear.step(self.ego, speed, dt, accel=accel, brake=brake)
        if pedals is not None:                      # the spawn phase drives the car
            self.tracker.reset(prime=True)
            throttle, brake = pedals
        elif accel is not None:
            throttle, brake = self.tracker.step(accel, speed, dt)
        return self.carla.VehicleControl(throttle=float(throttle), steer=float(steer),
                                         brake=float(brake), **gear)

    def _read_lights(self) -> Dict[str, Any]:
        """The ego's junction lights as CARLA shows them now: each role's phases
        (CARLA state names) and every light by id."""
        by_role: Dict[str, List[str]] = {"ego": [], "opposing": [], "crossing": []}
        phases: Dict[str, str] = {}
        for light, role in self.lights:
            try:
                state = str(light.get_state()).split(".")[-1]
            except RuntimeError:
                continue
            phases[str(light.id)] = state
            by_role.setdefault(role, []).append(state)
        return {"ego": by_role["ego"], "opposing": by_role["opposing"],
                "crossing": by_role["crossing"], "lights": phases}

    def _check_tick(self, poses: List[Dict[str, float]], cameras: Dict[str, Any]) -> None:
        """Two guards, after every tick. Either failing stops the episode, so a
        broken run cannot be exported as a result: a moving car whose speed
        does not reach CARLA, and cameras rendering missing materials."""
        if len(self.stopped_ticks) != len(self.actors):
            self.stopped_ticks = [0] * len(self.actors)
        for i, (actor, pose) in enumerate(zip(self.actors, poses)):
            v = actor.get_velocity()
            reported, wanted = math.hypot(v.x, v.y), float(pose.get("speed") or 0.0)
            if self.decisions >= 4:                  # after the spawn settles
                self.checks["actor_speed_err_max_mps"] = max(
                    self.checks["actor_speed_err_max_mps"], abs(reported - wanted))
            stuck = wanted > MOVING_MPS and reported < STOPPED_MPS
            self.stopped_ticks[i] = self.stopped_ticks[i] + 1 if stuck else 0
            if self.stopped_ticks[i] >= STOPPED_TICKS:
                raise RuntimeError(
                    f"agent {i} moves at {wanted:.1f} m/s but CARLA reports its speed as "
                    f"{reported:.2f} m/s for {STOPPED_TICKS} ticks: a policy reading "
                    "radar would see it as parked (is its physics off?)")
        frac = max((magenta_fraction(c) for c in cameras.values()), default=0.0)
        self.checks["magenta_max"] = max(self.checks["magenta_max"], frac)
        self.pink_ticks = self.pink_ticks + 1 if frac > PINK_FRACTION else 0
        if self.pink_ticks >= PINK_TICKS:
            raise RuntimeError(
                f"camera frames are {frac:.0%} magenta for {PINK_TICKS} decisions: CARLA "
                "is rendering missing materials (seen after a town reload). Restart the "
                "CARLA server in the episode's town and rerun")

    def _save_frame(self, cameras: Dict[str, Any], frame: int) -> None:
        if self.frames_dir is None or not cameras:
            return
        try:
            import cv2
        except ImportError:
            return
        views = [np.asarray(cameras[name])[..., :3] for name in sorted(cameras)
                 if getattr(cameras[name], "ndim", 0) == 3]
        if views:
            strip = np.concatenate(views, axis=1)
            cv2.imwrite(str(self.frames_dir / f"frame_{frame:08d}.jpg"),
                        cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    # -- close ----------------------------------------------------------
    def close(self) -> Dict[str, Any]:
        summary = {"ok": True, "decisions": self.decisions, "checks": dict(self.checks),
                   "dropped": dict(self.rig.dropped) if self.rig else {}}
        try:
            if (getattr(self, "signal_note", None) or {}).get("frozen"):
                try:
                    for light in self.world.get_actors().filter("traffic.traffic_light"):
                        light.freeze(False)
                        break                       # freeze(False) is scene-wide
                except Exception:
                    pass
            if self.rig is not None:
                self.rig.destroy()
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            if self.observer is not None:
                summary["state_observation"] = self.observer.describe()
            if self.bev is not None:
                summary["bev"] = self.bev.describe()
                self.bev.close()
            for actor in self.actors + ([self.ego] if self.ego else []):
                try:
                    actor.destroy()
                except Exception:
                    pass
            closer = getattr(self.policy, "close", None)
            if callable(closer):
                closer()
        finally:
            if self.world is not None and self.original_settings is not None:
                self.world.apply_settings(self.original_settings)
        return summary


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------

def serve(args, carla_mod=None, rig_mod=None, on_ready=None, signals_mod=None,
          route_mod=None) -> int:
    """Serve episodes. `carla_mod` / `rig_mod` / `on_ready(port)` exist so the
    whole loop can be exercised against fakes, with no server and no GPU."""
    if carla_mod is None:
        import carla as carla_mod  # fail here, not on the first request
    if rig_mod is None:
        rig_mod = load_rig_module(args.osc2runner)
        if signals_mod is None:
            signals_mod = load_signals_module(args.osc2runner)
    srv = socket.create_server((args.host, args.port))
    port = srv.getsockname()[1]
    print(f"sensor worker listening on {args.host}:{port} "
          f"(CARLA {args.carla_host}:{args.carla_port})", flush=True)
    if on_ready is not None:
        on_ready(port)
    while True:
        conn, peer = srv.accept()
        print(f"episode from {peer}", flush=True)
        stream = conn.makefile("rwb")
        session = Session(carla_mod, rig_mod, args.carla_host, args.carla_port,
                          allow_load_town=args.allow_load_town)
        session.signals = signals_mod
        session.route_mod = route_mod
        try:
            while True:
                try:
                    msg = receive(stream)
                except ConnectionError:
                    break
                if msg.get("protocol") != PROTOCOL_VERSION:
                    send(stream, {"error": f"protocol {msg.get('protocol')} != "
                                           f"{PROTOCOL_VERSION}"})
                    continue
                op = msg.get("op")
                try:
                    handler = {"init": session.init, "step": session.step,
                               "close": lambda m: session.close()}[op]
                    send(stream, handler(msg))
                except Exception:
                    error = traceback.format_exc()
                    print(f"[worker] {op} failed:\n{error}", file=sys.stderr, flush=True)
                    send(stream, {"error": error})
                if op == "close":
                    break
        finally:
            try:
                session.close()
            except Exception:
                pass
            stream.close()
            conn.close()
        if args.once:
            return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2100)
    ap.add_argument("--carla-host", default="127.0.0.1")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--osc2runner", default=DEFAULT_OSC2RUNNER)
    ap.add_argument("--allow-load-town", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="serve one episode, then exit")
    return serve(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
