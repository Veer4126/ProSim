"""Drive the rule ego from a harness `ego_policy_v1` policy repository.

WHY THIS EXISTS
---------------
`configs/policy/idm.yaml` declares `requires: []`: the analytic IDM family is
realized NATIVELY by each execution method, which is what keeps integration
cost at M + P. `configs/policy/idm_mobil.yaml` declares the opposite --
"it is NOT realized natively by any execution method, so the repository has to
be present" -- because it carries a lateral law. So `idm_mobil` must be LOADED
from `third_party/idm` and driven through a bridge; substituting this repo's
own MOBIL would publish a different implementation's behaviour under that name.

This module is that bridge. It exposes `ExternalEgoPolicy`, which has the same
`step(state, neighbors) -> VehicleState` surface as `ego_control.EgoPolicy`,
so `prosim_ego.set_ego_policy(policy=...)` takes it unchanged.

THE TWO FRAMES, AND WHY THEY HAPPEN TO AGREE
--------------------------------------------
The harness `state` space is ego-relative: `+x` forward, `+y` to the ego's
RIGHT, `yaw_rad` relative to the ego heading (`idm/scene.py`, quoting
third_party/README.md). CARLA's world frame is LEFT-handed, so rotating a world
offset by -heading already yields (forward, right) -- measured on 11817 lane
points 2026-09-05, and the same convention `goal_control.world_to_body` uses.
So no sign flip is needed here, and positive `steer` (full right) turns toward
+y, which is a heading INCREASE in CARLA.

Object yaw is only ever used as `cos(yaw_rad)` (the along-projection of speed,
and the contraflow test), and cosine is even, so a sign error in relative yaw
could not change a decision. Route points are different: their lateral sign is
read directly, so those go through the same rotation as positions.

THE ROUTE MUST BE FROZEN
------------------------
`idm/policy.py` reads the ego's lateral drift as the route's offset in the
moving ego frame: "the route is a plan fixed at spawn, so its offset in the
moving ego frame IS our drift". `LaneGraphRoute` re-plans once the car is more
than `resnap_dist` (3.0 m) off its path -- and a lane change is 3.5 m, so a
live route would follow the ego into the new lane, report zero drift, and the
policy would never see its own manoeuvre. The route is therefore sampled ONCE,
in world coordinates, and only re-expressed in the ego frame thereafter.

ACTION
------
`{"acceleration_mps2": float, "steer": float}` -- an acceleration and a
normalised steer, NOT a throttle/brake pair (`policy.py --describe`). Steer is
inverted back through the bicycle model the policy used to produce it
(`idm/lateral.py`: WHEELBASE 2.8 m, DELTA_MAX 32 deg), so the motion realised
here is the motion the lateral law asked for.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ego_control import VehicleState, wrap_angle

#: From idm/lateral.py. Kept in sync deliberately: `steer` is a normalised
#: front-wheel angle, so integrating it with different constants would realise
#: a different manoeuvre from the one the policy computed.
WHEELBASE_M = 2.8
DELTA_MAX_RAD = math.radians(32.0)

#: (braking, accelerating) m/s^2 the bridge will realise when it has to turn a
#: target speed into an acceleration. Ours, not any policy's -- see
#: `_read_waypoints_action`.
ACCEL_LIMITS = (8.0, 4.0)

#: idm/scene.py::VEHICLE_TYPES. Anything outside it is dropped from the scene.
OBJECT_TYPE = "vehicle"

#: How much route to freeze, metres. The ego covers ~80 m in an 8 s rollout at
#: 10 m/s; LaneGraphRoute's own `max_lanes` caps the chain before this bites.
ROUTE_LENGTH_M = 200.0

#: PlanT samples its route one point per metre starting 2.5 m ahead
#: (`config.tf_first_checkpoint_distance`, `config.points_per_meter`) and its
#: route embedding is a fixed Linear(20*2, ...), so exactly 20 points are fed.
#: Matching that sampling is what makes the numbers comparable to the paper.
#: Applied only for waypoint policies: the IDM family reads the route as a
#: heading reference and was measured with the raw lane-graph sampling.
ROUTE_FIRST_M, ROUTE_STEP_M, ROUTE_POINTS = 2.5, 1.0, 20


def _resample(points: np.ndarray, first_m: float, step_m: float,
              count: int) -> np.ndarray:
    """Arc-length resampling of a route, `count` points from `first_m` ahead.

    Shorter routes are padded by repeating the last point, which is what the
    policy would do internally anyway.
    """
    # Arc length is measured from the EGO, not from the first sample: the plan
    # starts wherever the lane graph happened to be sampled, so measuring from
    # it would put the first route point `first_m` beyond that instead of
    # `first_m` ahead of the car.
    pts = np.concatenate([np.zeros((1, 2)), np.asarray(points, dtype=float)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = first_m + step_m * np.arange(count)
    want = np.clip(want, 0.0, s[-1])
    return np.stack([np.interp(want, s, pts[:, 0]),
                     np.interp(want, s, pts[:, 1])], axis=1)


def load_policy_module(policy_py: Path):
    """Import a method repository's ``scenario_orchestration/policy.py``.

    Loaded by path, with the repository root on ``sys.path`` (its own
    ``policy.py`` does the same), so nothing needs installing.
    """
    policy_py = Path(policy_py)
    if not policy_py.is_file():
        raise SystemExit(
            f"external ego policy not found: {policy_py}\n"
            "The submodule is probably not checked out. From the harness root:\n"
            "  git submodule update --init third_party/idm\n"
            "(on a node where https is blocked, add "
            "-c url.\"git@github.com:\".insteadOf=\"https://github.com/\")")
    import sys
    repo_root = str(policy_py.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location("_external_ego_policy", policy_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_external_ego_policy"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "build_policy"):
        raise SystemExit(f"{policy_py} exposes no build_policy(request) factory")
    return module


class ExternalEgoPolicy:
    """An `ego_policy_v1` policy driving the ego, with EgoPolicy's surface.

    `route` is an `ego_control.RouteProvider` (a LaneGraphRoute in practice).
    It is used ONCE, to sample the plan; `prosim_ego` also reads `.route` to
    recover the ego's intended path for the harness's reference path, so it
    stays attached rather than being consumed and dropped.
    """

    def __init__(self, policy, route, lane_graph=None, dt: float = 0.1,
                 route_length: float = ROUTE_LENGTH_M, speed_limit_kph: float = 50.0,
                 action_space: str = None, bev=None):
        self.policy = policy
        #: A `bev_raster.BevRasteriser`, or None. Every released PlanT 2.0
        #: checkpoint sets input_bev=True, so act() raises without one.
        self.bev = bev
        #: The harness's own declaration wins, because that is the contract both
        #: sides validate against; a policy object need not expose it at all
        #: (idm's classes do, PlanT2Policy keeps its request instead).
        self.action_space = str(action_space
                                or getattr(policy, "action_space", None)
                                or "control")
        self.speed_limit_kph = float(speed_limit_kph)
        self.route = route
        self.lane_graph = lane_graph
        self.dt = float(dt)
        self.route_length = float(route_length)
        self._plan: Optional[np.ndarray] = None      # world frame, frozen
        self._t = 0.0
        self.last_action: Dict[str, Any] = {}

    # -- observation ----------------------------------------------------

    @staticmethod
    def _to_ego(points: np.ndarray, state: VehicleState) -> np.ndarray:
        """World points -> the ego frame (+x forward, +y right)."""
        d = np.asarray(points, dtype=float)[:, :2] - np.array([state.x, state.y])
        c, s = math.cos(state.heading), math.sin(state.heading)
        return np.stack([d[:, 0] * c + d[:, 1] * s,
                         -d[:, 0] * s + d[:, 1] * c], axis=1)

    def _observation(self, state: VehicleState,
                     neighbors: Sequence[VehicleState]) -> Dict[str, Any]:
        if self._plan is None:
            plan = self.route.path_ahead(state, self.route_length)
            self._plan = np.asarray(plan, dtype=float)[:, :2] if len(plan) else None

        route = []
        if self._plan is not None and len(self._plan):
            local = self._to_ego(self._plan, state)
            # Only the part still ahead: the policy extrapolates the ego's
            # lateral offset back to x = 0 from the first two samples, and
            # points already behind would put that fit on the wrong side.
            ahead = local[local[:, 0] > 0.0]
            if self.action_space == "waypoints" and len(ahead) >= 2:
                ahead = _resample(ahead, ROUTE_FIRST_M, ROUTE_STEP_M, ROUTE_POINTS)
            route = [[float(x), float(y)] for x, y in ahead[:200]]

        objects = []
        for i, n in enumerate(neighbors):
            along, lat = self._to_ego(np.array([[n.x, n.y]]), state)[0]
            objects.append({
                "id": str(i),
                "type": OBJECT_TYPE,
                "position": [float(along), float(lat)],
                "yaw_rad": float(wrap_angle(n.heading - state.heading)),
                "speed_mps": float(n.speed),
                # idm/scene.py doubles these: they are CARLA HALF-extents.
                "extent": [float(n.length) / 2.0, float(n.width) / 2.0, 0.75],
            })

        obs = {
            "t": self._t,
            "ego": {"speed_mps": float(state.speed),
                    "extent": [float(state.length) / 2.0,
                               float(state.width) / 2.0, 0.75]},
            "objects": objects,
            "route": route,
        }
        if self.action_space == "waypoints":
            # PlanT snaps this to the nearest limit it was trained with
            # (50/80/100/120). No CARLA world here to read one from, so the
            # town's own limit is declared rather than invented per step.
            obs["speed_limit_kph"] = self.speed_limit_kph
        if self.bev is not None:
            # As ObsManager returns it: the receiving policy applies its own
            # rot90 and centre-crop (policy.py::_bev, PlanT_agent.py:162-164),
            # so handing it a pre-rotated raster would double the rotation.
            obs["bev"] = {"semantic_classes": self.bev.classes(
                state.x, state.y, state.heading)}
        return obs

    # -- action ---------------------------------------------------------

    def _read_waypoints_action(self, action: Mapping[str, Any],
                               state: VehicleState) -> tuple:
        """A `waypoints` action -> (acceleration, steer).

        Taken from the policy's OWN controllers wherever possible: `control.steer`
        is PlanT's lateral controller, which is how its published closed-loop
        numbers are produced, and `target_speed_mps` is the speed its own
        `_get_control` derives. Only the last step -- turning a target speed
        into an acceleration -- is ours, because this rollout integrates
        kinematically and has no CARLA vehicle model to invert a throttle
        against. Recorded as `longitudinal: target_speed tracker` so a reader
        knows which part is not PlanT's.
        """
        control = action.get("control") or {}
        steer = control.get("steer")
        if steer is None:
            raise SystemExit(
                "waypoint policy returned no control.steer, and this bridge "
                "will not re-derive a steering law: that would publish a "
                "different lateral controller under the policy's name. "
                "(policy.py omits `control` when its controllers cannot be "
                "constructed in this environment.)")
        target = action.get("target_speed_mps")
        if target is None:
            raise SystemExit("waypoint policy returned no target_speed_mps")
        accel = (float(target) - state.speed) / self.dt
        accel = max(-ACCEL_LIMITS[0], min(ACCEL_LIMITS[1], accel))
        return accel, max(-1.0, min(1.0, float(steer)))

    @staticmethod
    def _read_action(action: Mapping[str, Any]) -> tuple:
        if not isinstance(action, Mapping):
            raise SystemExit(f"policy returned {type(action).__name__}, "
                             "expected a control mapping")
        accel = action.get("acceleration_mps2", action.get("acceleration"))
        steer = action.get("steer", action.get("steer_norm"))
        if accel is None or steer is None:
            raise SystemExit(
                "policy action carries no acceleration_mps2/steer pair "
                f"(got keys {sorted(action)}); this bridge implements the "
                "'control' action space")
        return float(accel), max(-1.0, min(1.0, float(steer)))

    def _integrate(self, state: VehicleState, accel: float,
                   steer: float) -> VehicleState:
        """One dt of the bicycle model `idm/lateral.py` inverted to get `steer`.

        Midpoint on both speed and heading, matching `EgoPolicy.step`: a plain
        forward Euler on the new heading systematically cuts corners.
        """
        dt = self.dt
        new_speed = max(0.0, state.speed + accel * dt)
        v_mid = 0.5 * (state.speed + new_speed)
        yaw_rate = v_mid * math.tan(steer * DELTA_MAX_RAD) / WHEELBASE_M
        new_heading = wrap_angle(state.heading + yaw_rate * dt)
        mid = state.heading + 0.5 * wrap_angle(new_heading - state.heading)
        travel = v_mid * dt
        return VehicleState(x=state.x + travel * math.cos(mid),
                            y=state.y + travel * math.sin(mid),
                            heading=new_heading, speed=new_speed,
                            length=state.length, width=state.width)

    # -- the EgoPolicy surface ------------------------------------------

    def step(self, state: VehicleState,
             neighbors: Sequence[VehicleState] = ()) -> VehicleState:
        obs = self._observation(state, neighbors)
        action = self.policy.act(obs)
        self.last_action = dict(action) if isinstance(action, Mapping) else {}
        if self.action_space == "waypoints":
            accel, steer = self._read_waypoints_action(action, state)
        else:
            accel, steer = self._read_action(action)
        self._t += self.dt
        return self._integrate(state, accel, steer)

    def metadata(self) -> Dict[str, Any]:
        try:
            return dict(self.policy.metadata() or {})
        except Exception:
            return {}


def make_external_ego(policy_py, policy_request: Mapping[str, Any], route,
                      lane_graph=None, dt: float = 0.1,
                      bev_town: str = None) -> ExternalEgoPolicy:
    """Build the bridge from a policy repository and a PolicyRequest.

    The request is passed to the repository's own `build_policy` VERBATIM, as
    the process contract intends -- so every declared parameter reaches the
    policy, including the ones this repo's native IDM has no slot for
    (`lane_width_m`, `max_lane_offset`).
    """
    module = load_policy_module(policy_py)
    policy = module.build_policy(dict(policy_request))
    action_space = policy_request.get("action_space")

    # A BEV only when the policy's repository ships the town rasters AND the
    # policy is one that reads them; computing one per step for the IDM family
    # would be pure waste.
    bev = None
    if bev_town and action_space == "waypoints":
        from bev_raster import BevRasteriser, maps_dir_for
        maps = maps_dir_for(Path(policy_py).parent.parent)
        if maps.is_dir():
            bev = BevRasteriser(bev_town, maps)
            print(f"  BEV: {bev_town} raster from {maps.name} "
                  f"({bev.width}x{bev.width} at {bev.pixels_per_meter} px/m)")

    return ExternalEgoPolicy(policy, route, lane_graph=lane_graph, dt=dt,
                             action_space=action_space, bev=bev)
