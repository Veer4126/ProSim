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

The control loop follows the orchestration port's `PolicyEgoDriver`: apply the
held control, tick, capture the rig for that frame, ask the policy, hold the
answer for the next tick.

WHAT EACH PROSIM STEP DOES
-------------------------
ProSim steps at 0.1 s; the world ticks at the policy's own rate (20 Hz for
TFv6, two ticks per step). The other agents are placed on interpolated poses
between ProSim's -- physics off, like a replay -- so they appear where the
rollout put them; the ego keeps CARLA physics, so its motion is what the
policy's pedals and steering actually produce. The ego's pose after the last
tick goes back to ProSim.
"""

from __future__ import annotations

import argparse
import importlib.util
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


def route_in_ego_frame(route_world: np.ndarray, x: float, y: float,
                       yaw_rad: float) -> List[List[float]]:
    """The frozen world route as the policy reads it: ego frame (+x forward,
    +y right -- CARLA's left-handed convention, no flip needed), resampled by
    arc length measured FROM THE EGO, 20 points 1 m apart from 2.5 m ahead."""
    if route_world is None or len(route_world) < 2:
        return []
    d = np.asarray(route_world, dtype=float)[:, :2] - np.array([x, y])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    local = np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], axis=1)
    ahead = local[local[:, 0] > 0.0]
    if len(ahead) < 2:
        return []
    pts = np.concatenate([np.zeros((1, 2)), ahead])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    want = np.clip(ROUTE_FIRST_M + ROUTE_STEP_M * np.arange(ROUTE_POINTS), 0.0, arc[-1])
    return np.stack([np.interp(want, arc, pts[:, 0]),
                     np.interp(want, arc, pts[:, 1])], axis=1).tolist()


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
        self.original_settings = None
        self.control = None
        self.substeps = 1
        self.frames_dir: Optional[Path] = None
        self.decisions = 0
        self.last_meta: Dict[str, Any] = {}

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
            actor.set_simulate_physics(False)
            self.actors.append(actor)

        self.route_world = np.asarray(msg.get("route_world") or [], dtype=float)
        self.policy = load_policy(msg["policy_py"], msg["policy_request"])

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

        if msg.get("frames_dir"):
            self.frames_dir = Path(msg["frames_dir"])
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "map": self.world.get_map().name,
                "tick_hz": policy_hz, "substeps": self.substeps,
                "sensors": self.rig.names, "failed": self.rig.failed,
                "sensor_hz": self.rig.sensor_hz,
                "ego_light": getattr(self, "signal_note", None)}

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
            for actor, a, b in zip(self.actors, self.prev, target):
                pose = lerp_pose(a, b, f)
                actor.set_transform(self._transform(pose, 0.0))
                actor.set_target_velocity(self._forward_velocity(pose))
            if self.control is not None:
                self.ego.apply_control(self.control)
            frame = self.world.tick()
            cameras = self.rig.capture(frame) if self.rig.active else {}
            tf = self.ego.get_transform()
            v = self.ego.get_velocity()
            yaw = math.radians(tf.rotation.yaw)
            observation = {
                "t": self.decisions / max(1e-9, 1.0 / self.world.get_settings().fixed_delta_seconds),
                "ego": {"speed_mps": math.hypot(v.x, v.y)},
                "route": route_in_ego_frame(self.route_world, tf.location.x,
                                            tf.location.y, yaw),
            }
            if self.rig.active:
                observation["sensor"] = {"cameras": cameras}
            action = self.policy.act(observation)
            fields = to_control_fields(action)
            self.control = carla.VehicleControl(**fields)
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
                "meta": self.last_meta, "dropped": dict(self.rig.dropped)}

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
        summary = {"ok": True, "decisions": self.decisions,
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

def serve(args, carla_mod=None, rig_mod=None, on_ready=None, signals_mod=None) -> int:
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
                    send(stream, {"error": traceback.format_exc()})
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
