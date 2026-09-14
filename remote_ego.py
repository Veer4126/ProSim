"""ProSim's side of the sensor loop: an ego policy whose decisions come from CARLA.

Same surface as `ego_control.EgoPolicy` and `external_ego.ExternalEgoPolicy` --
`step(state, neighbors) -> VehicleState` -- so `prosim_ego.set_ego_policy(
policy=...)` takes it unchanged. The difference is where the ego's next pose
comes from: not from integrating an action here, but from CARLA's own vehicle
physics, after a vision policy has looked at a rendered world and applied its
throttle, brake and steer.

Per ProSim step (0.1 s) this sends the other agents' poses for the end of the
step; the worker places them, ticks the world at the policy's own rate (20 Hz
for TFv6 -- its lateral PID is tuned in decisions, not seconds; see the
harness's docs/policy_decision_rate.md), lets the policy decide on every tick,
and returns where the ego actually ended up. ProSim writes that back into the
rollout exactly as it writes back the IDM ego's pose, so its agents react to the
ego CARLA drove, not to a prediction of it.

The route is frozen once, in world coordinates, and sent at init: the policy
reads its target points and navigation command off it, and a route that
re-planned as the ego moved would stop meaning "where you were asked to go".
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ego_control import VehicleState
from sensor_protocol import Client, parse_address

ROUTE_LENGTH_M = 200.0


def _pose(v: VehicleState) -> Dict[str, float]:
    return {"x": float(v.x), "y": float(v.y), "yaw_rad": float(v.heading),
            "speed": float(v.speed), "length": float(v.length),
            "width": float(v.width)}


class RemoteSensorEgoPolicy:
    """An ego driven by a vision policy inside a CARLA world."""

    def __init__(self, address: str, route, lane_graph=None, *,
                 policy_py: str, policy_request_path: str, town: str,
                 dt: float = 0.1, policy_hz: float = 20.0,
                 frames_dir: Optional[str] = None, timeout_s: float = 300.0,
                 ego_light: Optional[str] = None):
        self.address = address
        self.route = route
        self.lane_graph = lane_graph
        self.policy_py = str(policy_py)
        self.policy_request_path = str(policy_request_path)
        self.town = str(town)
        self.dt = float(dt)
        self.policy_hz = float(policy_hz)
        self.frames_dir = frames_dir
        self.timeout_s = float(timeout_s)
        #: The ego's junction phase for the worker to set and freeze, or None.
        self.ego_light = ego_light
        self._client: Optional[Client] = None
        self._n_neighbors: Optional[int] = None
        self.init_reply: Dict[str, Any] = {}
        self.last_reply: Dict[str, Any] = {}
        self.steps = 0

    # ------------------------------------------------------------------ #
    def _connect(self, state: VehicleState, neighbors: Sequence[VehicleState]):
        host, port = parse_address(self.address)
        plan = np.asarray(self.route.path_ahead(state, ROUTE_LENGTH_M), dtype=float)
        with open(self.policy_request_path) as fh:
            request = json.load(fh)
        self._client = Client(host, port, timeout_s=self.timeout_s)
        self._n_neighbors = len(neighbors)
        self.init_reply = self._client.request({
            "op": "init",
            "town": self.town,
            "dt": self.dt,
            "policy_hz": self.policy_hz,
            "ego": _pose(state),
            "actors": [_pose(n) for n in neighbors],
            "route_world": plan[:, :2].tolist() if len(plan) else [],
            "policy_py": self.policy_py,
            "policy_request": request,
            "frames_dir": self.frames_dir,
            "ego_light": self.ego_light,
        })

    def step(self, state: VehicleState,
             neighbors: Sequence[VehicleState] = ()) -> VehicleState:
        if self._client is None:
            self._connect(state, neighbors)
        if len(neighbors) != self._n_neighbors:
            raise RuntimeError(
                f"the scene changed size mid-rollout ({self._n_neighbors} -> "
                f"{len(neighbors)} agents); the worker keys actors by position")
        reply = self._client.request({
            "op": "step", "actors": [_pose(n) for n in neighbors]})
        self.last_reply = reply
        self.steps += 1
        ego = reply["ego"]
        return VehicleState(x=float(ego["x"]), y=float(ego["y"]),
                            heading=float(ego["yaw_rad"]),
                            speed=float(ego["speed"]),
                            length=state.length, width=state.width)

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.request({"op": "close"})
        finally:
            self._client.close()
            self._client = None

    def metadata(self) -> Dict[str, Any]:
        return {"worker": self.address, "steps": self.steps,
                "init": self.init_reply,
                "last": {k: v for k, v in self.last_reply.items() if k != "ego"}}
