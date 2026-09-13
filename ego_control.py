"""Pluggable rule-based ego controllers: IDM, pure pursuit, MOBIL.

DESIGN: this module is PURE numpy. It imports neither torch nor ProSim, so every
model here can be unit-tested on synthetic scenarios without a GPU, a checkpoint,
or a CARLA server. `prosim_ego.py` is the thin adapter that wires it into
ProSim's rollout.

Composition, so pieces can be swapped independently:

    EgoPolicy
      |- LongitudinalModel   how fast          -> IDM (or ConstantSpeed)
      |- LateralModel        where to point    -> PurePursuit (or ConstantHeading)
      |- LaneSelector        which lane        -> MOBIL (or KeepLane)
      `- RouteProvider       the path ahead    -> LaneGraphRoute (or StraightRoute)

Every component is an ABC with a trivial implementation alongside the real one,
so a controller can be built up one piece at a time and each piece isolated when
something misbehaves.

FRAMES: everything here is in a single 2D world frame, x/y in metres, heading in
radians measured CCW from +x with `heading = atan2(dy, dx)`. That matches CARLA's
native frame as used everywhere else in this project (see CLAUDE.md). The adapter
handles ProSim's per-agent local frames.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-6


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class VehicleState:
    x: float
    y: float
    heading: float
    speed: float
    length: float = 4.5
    width: float = 2.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class Lead:
    """What a longitudinal model needs to know about the vehicle in front."""
    gap: float          # bumper-to-bumper distance along the path, metres
    speed: float


# ---------------------------------------------------------------------------
# Longitudinal
# ---------------------------------------------------------------------------

class LongitudinalModel(ABC):
    @abstractmethod
    def accel(self, ego_speed: float, lead: Optional[Lead]) -> float:
        """Longitudinal acceleration in m/s^2. lead=None means free road."""


class ConstantSpeed(LongitudinalModel):
    """Trivial baseline: hold the current speed. Useful to isolate the lateral model."""

    def accel(self, ego_speed: float, lead: Optional[Lead]) -> float:
        return 0.0


@dataclass
class IDM(LongitudinalModel):
    """Intelligent Driver Model (Treiber, Hennecke & Helbing 2000).

        a = a_max * [ 1 - (v/v0)^delta - (s*/s)^2 ]
        s* = s0 + max(0, v*T + v*dv / (2*sqrt(a_max*b)))
        dv = v_ego - v_lead          <-- APPROACH rate

    NOTE the sign of dv. The prior implementation in animate_rollout.py used
    `rel_v = lead_speed - ego_speed`, i.e. the negation, which inverts the
    interaction term: closing on a slower lead REDUCED the desired gap instead of
    increasing it. Fixed here, and covered by a unit test.

    The `max(0, ...)` on the dynamic part is also required by the model -- without
    it, opening a gap on a faster lead can drive s* below s0 and even negative.
    """
    v0: float = 10.0        # desired free-road speed, m/s
    T: float = 1.5          # safe time headway, s
    s0: float = 2.0         # minimum jam gap, m
    a_max: float = 2.0      # max acceleration, m/s^2
    b: float = 1.5          # comfortable deceleration, m/s^2
    delta: float = 4.0      # free-road exponent
    b_emergency: float = 6.0  # hard clip on braking, m/s^2

    def accel(self, ego_speed: float, lead: Optional[Lead]) -> float:
        v = max(0.0, float(ego_speed))
        free = 1.0 - (v / max(self.v0, EPS)) ** self.delta

        if lead is None:
            a = self.a_max * free
        else:
            s = max(float(lead.gap), EPS)
            dv = v - float(lead.speed)                      # approach rate
            dynamic = v * self.T + (v * dv) / (2.0 * np.sqrt(self.a_max * self.b))
            s_star = self.s0 + max(0.0, dynamic)
            a = self.a_max * (free - (s_star / s) ** 2)

        return float(np.clip(a, -self.b_emergency, self.a_max))


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

class RouteProvider(ABC):
    @abstractmethod
    def path_ahead(self, state: VehicleState, distance: float) -> np.ndarray:
        """(N, 2) polyline starting at/near the vehicle, extending `distance` ahead."""


@dataclass
class StraightRoute(RouteProvider):
    """A straight path along the current heading. No map needed -- test fixture,
    and the fallback when a vehicle is off-map."""
    step: float = 1.0

    def path_ahead(self, state: VehicleState, distance: float) -> np.ndarray:
        n = max(2, int(distance / self.step) + 1)
        s = np.arange(n) * self.step
        return np.stack([state.x + s * np.cos(state.heading),
                         state.y + s * np.sin(state.heading)], axis=1)


class LaneGraphRoute(RouteProvider):
    """Follows a lane graph, choosing successors by heading agreement.

    `lane_graph` must expose:
        closest_lane(x, y, heading) -> lane_id
        centerline(lane_id)         -> (N, 2) array, in travel direction
        successors(lane_id)         -> iterable of lane_id
    `VecMapLaneGraph` in prosim_ego.py adapts trajdata's VectorMap to this.
    Keeping it an interface means the controller is testable against a hand-built
    graph with no trajdata dependency.
    """

    def __init__(self, lane_graph, max_lanes: int = 6, goal=None,
                 resnap_dist: float = 3.0):
        self.g = lane_graph
        self.max_lanes = max_lanes
        self.current_lane_id = None
        # Optional destination in the SAME frame as the lane graph. With it,
        # junctions are resolved toward the goal; without it, by staying as
        # straight as the map allows.
        self.goal = None if goal is None else np.asarray(goal, dtype=float)[:2]
        # The planned chain of lane ids. Held across steps so a junction is
        # decided once, on entry, instead of re-decided every 0.1 s.
        self._chain: List[object] = []
        # Re-plan only once the vehicle is this far off the planned path.
        self.resnap_dist = float(resnap_dist)

    def _chain_pts(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.g.centerline(l), dtype=float)
                               for l in self._chain], axis=0)

    def path_ahead(self, state: VehicleState, distance: float) -> np.ndarray:
        """Follow a REMEMBERED chain of lanes, re-planning only when off it.

        The previous version called closest_lane() every single step and rebuilt
        the route from whatever came back. Inside a junction several lanes
        overlap, so the snap would land on a turning connector and silently
        discard the straight-on plan that had just been chosen -- the route
        decision was made and then thrown away one step later. Measured on the
        ego at (-81.14, 24.47): the plan was 18049 -> 19049 -> 255049 (straight),
        and the car turned anyway.

        Keeping the chain means the junction is committed to once, on entry.
        """
        on_plan = False
        if self._chain:
            pts = self._chain_pts()
            if float(np.linalg.norm(pts - state.xy, axis=1).min()) <= self.resnap_dist:
                on_plan = True

        if not on_plan:
            lane_id = self.g.closest_lane(state.x, state.y, state.heading)
            if lane_id is None:
                self._chain = []
                return StraightRoute().path_ahead(state, distance)
            self._chain = [lane_id]

        # drop lanes we are already past, so `current_lane_id` stays meaningful
        while len(self._chain) > 1:
            first = np.asarray(self.g.centerline(self._chain[0]), dtype=float)
            rest = np.concatenate([np.asarray(self.g.centerline(l), dtype=float)
                                   for l in self._chain[1:]], axis=0)
            if (float(np.linalg.norm(rest - state.xy, axis=1).min())
                    < float(np.linalg.norm(first - state.xy, axis=1).min())):
                self._chain.pop(0)
            else:
                break

        # extend forward until there is `distance` of path ahead of the vehicle
        visited = set(self._chain)
        while len(self._chain) < self.max_lanes:
            pts = self._chain_pts()
            i = int(np.argmin(np.linalg.norm(pts - state.xy, axis=1)))
            ahead = _polyline_length(pts[i:])
            if ahead >= distance:
                break
            prev_seg = np.asarray(self.g.centerline(self._chain[-1]), dtype=float)
            nxt = self._best_successor(self._chain[-1], prev_seg, visited)
            if nxt is None:
                break
            self._chain.append(nxt)
            visited.add(nxt)

        self.current_lane_id = self._chain[0]
        return _trim_to_ahead(self._chain_pts(), state, distance)

    def _best_successor(self, lane_id, prev_seg, visited):
        """Choose the next lane: toward `self.goal` if set, else the straightest.

        THE BUG THIS REPLACES (measured 2026-09-07). The old version scored each
        candidate by the heading of its FIRST TWO POINTS -- 0.5 m of lane. Every
        branch of a junction leaves the entrance in the same direction and only
        diverges later, so they all scored the same and the winner was whichever
        the iterator yielded first. On lane 19049 at (-66.8, 24.5):

            296049  start-hdg +0.1  end-hdg -90.1     <- picked (first in order)
            255049  start-hdg +0.1  end-hdg  +0.0     <- the actual straight-on
            315049  start-hdg +0.1  end-hdg -90.0

        All three scored 0.1 deg. Asking for straight got a 90-degree turn, for
        the same reason turn_goal_from_lane_graph got one on 2026-09-05: a
        junction branch has to be judged by where it GOES, not where it starts.

        So candidates are scored on their END heading, which separates them
        cleanly, and on distance to the goal when there is one.
        """
        if len(prev_seg) < 2:
            return None
        heading = np.arctan2(prev_seg[-1, 1] - prev_seg[-2, 1],
                             prev_seg[-1, 0] - prev_seg[-2, 0])

        best, best_score = None, np.inf
        for cand in self.g.successors(lane_id):
            if cand in visited:
                continue
            seg = np.asarray(self.g.centerline(cand), dtype=float)
            if len(seg) < 2:
                continue
            # where the branch ACTUALLY goes, not how it starts
            end_h = np.arctan2(seg[-1, 1] - seg[-2, 1], seg[-1, 0] - seg[-2, 0])
            turn = abs(wrap_angle(end_h - heading))

            if self.goal is not None:
                # how close to the goal this branch gets us. Its own end point
                # is the honest measure: a branch that turns away from the goal
                # ends further from it.
                score = float(np.linalg.norm(seg - self.goal, axis=1).min())
                # tiny tie-break so a dead-straight branch wins an exact tie
                score += 1e-3 * turn
            else:
                score = turn

            if score < best_score:
                best, best_score = cand, score
        return best


def _polyline_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _trim_to_ahead(path: np.ndarray, state: VehicleState, distance: float) -> np.ndarray:
    """Drop path points behind the vehicle, then cut at `distance` of arc length."""
    if len(path) < 2:
        return path
    i = int(np.argmin(np.linalg.norm(path - state.xy, axis=1)))
    # step forward past any points that are behind us
    fwd = np.array([np.cos(state.heading), np.sin(state.heading)])
    while i < len(path) - 1 and np.dot(path[i] - state.xy, fwd) < 0:
        i += 1
    path = path[i:]
    if len(path) < 2:
        return path
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.searchsorted(np.cumsum(seg), distance) + 2
    return path[:max(2, keep)]


# ---------------------------------------------------------------------------
# Lateral
# ---------------------------------------------------------------------------

def _path_curvature(path: np.ndarray, origin: np.ndarray, window: float) -> float:
    """Mean |curvature| of the path within `window` metres of `origin`.

    Menger curvature over consecutive point triples: 1/R of the circle through
    them. Used to shorten the pure-pursuit lookahead in turns.
    """
    if len(path) < 3:
        return 0.0
    d = np.linalg.norm(path - origin, axis=1)
    sel = path[d <= max(window, 1.0)]
    if len(sel) < 3:
        sel = path[:3]
    a, b, c = sel[:-2], sel[1:-1], sel[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2.0
    denom = ab * bc * ca
    ok = denom > EPS
    if not ok.any():
        return 0.0
    return float(np.mean(4.0 * area[ok] / denom[ok]))


class LateralModel(ABC):
    @abstractmethod
    def heading(self, state: VehicleState, path: np.ndarray, dt: float) -> float:
        """New heading after dt seconds."""


class ConstantHeading(LateralModel):
    """Straight-line: what animate_rollout.py's IDM ego did. Cannot turn."""

    def heading(self, state: VehicleState, path: np.ndarray, dt: float) -> float:
        return state.heading


@dataclass
class PurePursuit(LateralModel):
    """Geometric path tracker.

        Ld    = clip(k*v + Ld0, Ld_min, Ld_max)      lookahead distance
        alpha = bearing to the lookahead point, in the vehicle frame
        kappa = 2*sin(alpha) / Ld                    path curvature
        yaw_rate = v * kappa                         (bicycle, clipped)

    Curvature is clipped to the vehicle's turning limit so the controller cannot
    demand a physically impossible turn at speed.
    """
    k: float = 0.6              # lookahead gain, seconds
    Ld0: float = 4.0            # base lookahead, m
    Ld_min: float = 3.0
    Ld_max: float = 25.0
    max_curvature: float = 0.2  # 1/m  (~5 m turning radius)
    max_yaw_rate: float = 1.0   # rad/s
    curve_gain: float = 0.35    # Ld <= curve_gain / path_curvature

    def heading(self, state: VehicleState, path: np.ndarray, dt: float) -> float:
        if path is None or len(path) < 2:
            return state.heading

        Ld = float(np.clip(self.k * max(state.speed, 0.0) + self.Ld0,
                           self.Ld_min, self.Ld_max))
        # Shrink the lookahead on curvy paths. A lookahead comparable to the
        # turn radius makes pure pursuit cut the corner; scaling it by the
        # local curvature keeps tracking tight through junctions while leaving
        # straight-line behaviour unchanged (curvature ~ 0 -> factor 1).
        kappa_path = _path_curvature(path, state.xy, Ld)
        if kappa_path > EPS:
            Ld = float(np.clip(min(Ld, self.curve_gain / kappa_path),
                               self.Ld_min, self.Ld_max))
        target = self._lookahead_point(state, path, Ld)
        if target is None:
            return state.heading

        d = target - state.xy
        # bearing in the vehicle frame
        alpha = wrap_angle(np.arctan2(d[1], d[0]) - state.heading)
        dist = max(float(np.linalg.norm(d)), EPS)

        kappa = 2.0 * np.sin(alpha) / dist
        kappa = float(np.clip(kappa, -self.max_curvature, self.max_curvature))

        yaw_rate = float(np.clip(state.speed * kappa,
                                 -self.max_yaw_rate, self.max_yaw_rate))
        return wrap_angle(state.heading + yaw_rate * dt)

    @staticmethod
    def _lookahead_point(state: VehicleState, path: np.ndarray, Ld: float):
        """First point at least Ld away, searching FORWARD from the nearest point.

        Searching from index 0 instead is a classic pure-pursuit bug: once the
        vehicle is past a curve, an early point on the path is still >= Ld away
        in a straight line, so the controller locks onto a target BEHIND itself
        and keeps turning. Anchoring the search at the nearest index makes the
        target monotonically advance along the path.
        """
        d = np.linalg.norm(path - state.xy, axis=1)
        near = int(np.argmin(d))
        idx = np.where(d[near:] >= Ld)[0]
        if len(idx):
            return path[near + idx[0]]
        return path[-1]


# ---------------------------------------------------------------------------
# Lane selection
# ---------------------------------------------------------------------------

class LaneSelector(ABC):
    @abstractmethod
    def select(self, state: VehicleState, neighbors: Sequence[VehicleState],
               lane_graph, current_lane_id) -> Optional[str]:
        """Lane id to occupy. None = stay put."""


class KeepLane(LaneSelector):
    def select(self, state, neighbors, lane_graph, current_lane_id):
        return current_lane_id


@dataclass
class MOBIL(LaneSelector):
    """Minimising Overall Braking Induced by Lane changes (Kesting et al. 2007).

    Change lane iff

      SAFETY    a_new_follower_after >= -b_safe
      INCENTIVE (a_ego_after - a_ego_before)
                + p * [ (a_new_fol_after - a_new_fol_before)
                      + (a_old_fol_after - a_old_fol_before) ] > a_threshold

    p is politeness: 0 = selfish, 1 = fully altruistic. Every acceleration is
    evaluated with the SAME longitudinal model the ego drives with, which is why
    IDM is injected rather than hard-coded.
    """
    longitudinal: LongitudinalModel = field(default_factory=IDM)
    politeness: float = 0.3
    a_threshold: float = 0.2     # m/s^2 gain needed to bother changing
    b_safe: float = 4.0          # m/s^2 worst braking we may impose on others
    corridor_halfwidth: float = 2.0

    def select(self, state, neighbors, lane_graph, current_lane_id):
        if current_lane_id is None or lane_graph is None:
            return current_lane_id

        candidates = list(lane_graph.adjacent(current_lane_id))
        if not candidates:
            return current_lane_id

        stay = self._accels(state, neighbors, lane_graph, current_lane_id)
        best_id, best_gain = current_lane_id, self.a_threshold

        for cand in candidates:
            move = self._accels(state, neighbors, lane_graph, cand)

            # safety: the follower we cut in front of must not brake harder than b_safe
            if move["new_follower_after"] < -self.b_safe:
                continue

            gain = (move["ego"] - stay["ego"]) + self.politeness * (
                (move["new_follower_after"] - move["new_follower_before"])
                + (stay["old_follower_after"] - stay["old_follower_before"])
            )
            if gain > best_gain:
                best_id, best_gain = cand, gain

        return best_id

    def _accels(self, state, neighbors, lane_graph, lane_id):
        """Accelerations relevant to occupying `lane_id`."""
        centre = np.asarray(lane_graph.centerline(lane_id), dtype=float)
        ahead, behind = _split_by_corridor(state, neighbors, centre,
                                           self.corridor_halfwidth)

        lead = _to_lead(state, ahead)
        ego_a = self.longitudinal.accel(state.speed, lead)

        # the follower in that lane, before and after we occupy it
        if behind is None:
            return {"ego": ego_a,
                    "new_follower_before": 0.0, "new_follower_after": 0.0,
                    "old_follower_before": 0.0, "old_follower_after": 0.0}

        fol, fol_gap = behind
        fol_lead_before = _to_lead(fol, ahead)
        before = self.longitudinal.accel(fol.speed, fol_lead_before)
        after = self.longitudinal.accel(
            fol.speed, Lead(gap=max(fol_gap - state.length, EPS), speed=state.speed))

        return {"ego": ego_a,
                "new_follower_before": before, "new_follower_after": after,
                "old_follower_before": before, "old_follower_after": after}


# ---------------------------------------------------------------------------
# Neighbour geometry
# ---------------------------------------------------------------------------

def _project_onto(path: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """(arc length along path, signed lateral offset) of point p.

    Signed lateral is +left of the direction of travel. Using arc length rather
    than a straight-line forward distance is what makes gap-finding correct
    around curves -- a straight-ahead test loses the lead vehicle in a turn.
    """
    if len(path) < 2:
        return 0.0, np.inf
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    ok = seg_len > EPS
    if not ok.any():
        return 0.0, np.inf
    seg, seg_len = seg[ok], seg_len[ok]
    starts = path[:-1][ok]
    u = seg / seg_len[:, None]

    rel = p[None, :] - starts
    t = np.clip((rel * u).sum(1), 0.0, seg_len)
    proj = starts + u * t[:, None]
    d = np.linalg.norm(p[None, :] - proj, axis=1)
    i = int(np.argmin(d))

    s = float(np.concatenate([[0.0], np.cumsum(seg_len)])[i] + t[i])
    # 2D cross product u x (p - proj): positive when p is LEFT of travel.
    cross = u[i, 0] * (p[1] - proj[i, 1]) - u[i, 1] * (p[0] - proj[i, 0])
    return s, float(cross)


def _extend_back(path: np.ndarray, distance: float = 60.0) -> np.ndarray:
    """Prepend a straight extension behind the path start.

    Without this, any vehicle behind the ego projects onto the path's first
    point and gets arc length 0 -- identical to the ego's own -- so it is
    classified as neither ahead nor behind and vanishes. That silently disabled
    MOBIL's safety criterion, which is entirely about the follower behind.
    """
    if len(path) < 2:
        return path
    d = path[0] - path[1]
    n = np.linalg.norm(d)
    if n < EPS:
        return path
    d = d / n
    steps = np.arange(int(distance), 0, -1)[:, None]
    return np.concatenate([path[0] + d * steps, path], axis=0)


def _split_by_corridor(state: VehicleState, neighbors: Sequence[VehicleState],
                       path: np.ndarray, halfwidth: float):
    """Nearest neighbour ahead of, and behind, `state` within the path corridor.

    Returns (ahead, behind) where each is (VehicleState, gap) or None.
    """
    if path is None or len(path) < 2:
        return None, None
    path = _extend_back(path)
    s_ego, _ = _project_onto(path, state.xy)

    ahead = behind = None
    ahead_gap = behind_gap = np.inf
    for n in neighbors:
        s_n, lat = _project_onto(path, n.xy)
        if abs(lat) > halfwidth:
            continue
        ds = s_n - s_ego
        # bumper-to-bumper
        gap = abs(ds) - 0.5 * (state.length + n.length)
        gap = max(gap, EPS)
        if ds > 0 and gap < ahead_gap:
            ahead, ahead_gap = n, gap
        elif ds < 0 and gap < behind_gap:
            behind, behind_gap = n, gap

    return ((ahead, ahead_gap) if ahead is not None else None,
            (behind, behind_gap) if behind is not None else None)


def _to_lead(follower: VehicleState, ahead) -> Optional[Lead]:
    if ahead is None:
        return None
    veh, gap = ahead
    return Lead(gap=gap, speed=veh.speed)


# ---------------------------------------------------------------------------
# Composed policy
# ---------------------------------------------------------------------------

@dataclass
class EgoPolicy:
    """Composes the four pieces into one closed-loop step."""
    longitudinal: LongitudinalModel = field(default_factory=IDM)
    lateral: LateralModel = field(default_factory=PurePursuit)
    lane_selector: LaneSelector = field(default_factory=KeepLane)
    route: RouteProvider = field(default_factory=StraightRoute)
    dt: float = 0.1
    lookahead: float = 60.0
    corridor_halfwidth: float = 2.0
    lane_graph: object = None

    def step(self, state: VehicleState,
             neighbors: Sequence[VehicleState] = ()) -> VehicleState:
        """One dt of closed-loop control. Returns the new state."""
        path = self.route.path_ahead(state, self.lookahead)

        # optional lane change; re-plan the path if the lane changed
        cur_lane = getattr(self.route, "current_lane_id", None)
        if self.lane_graph is not None and not isinstance(self.lane_selector, KeepLane):
            chosen = self.lane_selector.select(state, neighbors, self.lane_graph, cur_lane)
            if chosen is not None and chosen != cur_lane:
                self.route.current_lane_id = chosen
                path = _trim_to_ahead(
                    np.asarray(self.lane_graph.centerline(chosen), dtype=float),
                    state, self.lookahead)

        ahead, _ = _split_by_corridor(state, neighbors, path, self.corridor_halfwidth)
        accel = self.longitudinal.accel(state.speed, _to_lead(state, ahead))

        new_speed = max(0.0, state.speed + accel * self.dt)
        new_heading = self.lateral.heading(state, path, self.dt)

        # integrate with the average of old and new heading -- a plain forward
        # Euler on the new heading systematically cuts corners.
        mid = state.heading + 0.5 * wrap_angle(new_heading - state.heading)
        travel = 0.5 * (state.speed + new_speed) * self.dt

        return VehicleState(
            x=state.x + travel * np.cos(mid),
            y=state.y + travel * np.sin(mid),
            heading=new_heading,
            speed=new_speed,
            length=state.length,
            width=state.width,
        )

    def rollout(self, state: VehicleState, steps: int,
                neighbors_fn=None) -> List[VehicleState]:
        """Convenience for testing: run `steps` closed-loop steps."""
        out = [state]
        for i in range(steps):
            nb = neighbors_fn(i) if neighbors_fn is not None else ()
            out.append(self.step(out[-1], nb))
        return out


def make_policy(kind: str = "idm", dt: float = 0.1, lane_graph=None,
                **kwargs) -> EgoPolicy:
    """Convenience factory.

        'idm'          IDM + straight-line heading (matches the old behaviour)
        'idm_pursuit'  IDM + pure pursuit along the lane graph (turns)
        'idm_mobil'    the above + MOBIL lane changes

    route_goal=(x, y) makes junctions resolve toward that point, in the SAME
    frame as `lane_graph`. Without it the route stays as straight as it can.
    """
    idm = IDM(**{k: v for k, v in kwargs.items() if k in IDM.__dataclass_fields__})
    goal = kwargs.get("route_goal")
    route = (LaneGraphRoute(lane_graph, goal=goal,
                            resnap_dist=float(kwargs.get("resnap_dist", 3.0)))
             if lane_graph is not None else StraightRoute())

    if kind == "idm":
        return EgoPolicy(idm, ConstantHeading(), KeepLane(), route, dt=dt,
                         lane_graph=lane_graph)
    if kind == "idm_pursuit":
        return EgoPolicy(idm, PurePursuit(), KeepLane(), route, dt=dt,
                         lane_graph=lane_graph)
    if kind == "idm_mobil":
        return EgoPolicy(idm, PurePursuit(), MOBIL(longitudinal=idm), route, dt=dt,
                         lane_graph=lane_graph)
    raise ValueError(f"unknown policy kind: {kind!r}")
