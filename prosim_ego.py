"""Wire ego_control.py's rule-based controllers into ProSim's rollout.

Two pieces:

  VecMapLaneGraph  adapts trajdata's VectorMap to the small lane-graph interface
                   ego_control expects (centerline / successors / adjacent /
                   closest_lane), so the controllers stay trajdata-agnostic.

  ProSimRuleEgo    a ProSim model subclass that overrides step_agent_traj and
                   REPLACES one agent's predicted chunk with a rule-based
                   rollout. Every other agent keeps ProSim's prediction.

Replaces the inline IDM in animate_rollout.py's ProSimCustomEgo. Differences:
  - the IDM approach-rate sign bug is fixed (see ego_control.IDM)
  - the free-road term is the real IDM one, not a constant a_max
  - it can TURN (pure pursuit on the lane graph) instead of straight-line only
  - it can change lane (MOBIL)
  - speed is seeded from a_traj['vel'], not assumed 0 at the start
  - the velocity channel is written back, so the ego's own observation update
    stays consistent with its overridden position

FRAMES. ProSim stores each agent's trajectory in that agent's OWN frame:
    a_traj[task]['traj'][b, n, t] = [local_x, local_y, sin(local_h), cos(local_h)]
world = R(init_heading) @ local + init_pos. The controller works entirely in
world coordinates; this module converts in and out.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from ego_control import EgoPolicy, VehicleState, make_policy


# ---------------------------------------------------------------------------
# trajdata VectorMap -> ego_control lane-graph interface
# ---------------------------------------------------------------------------

class VecMapLaneGraph:
    """Adapts a trajdata VectorMap. Centrelines are cached: the same lanes are
    queried every step of every rollout, and .center.xy rebuilds an array."""

    def __init__(self, vec_map, search_radius: float = 30.0,
                 heading_weight: float = 8.0, max_heading_err: float = np.pi / 2):
        self.vec_map = vec_map
        self.search_radius = search_radius
        self.heading_weight = heading_weight
        self.max_heading_err = max_heading_err
        self._centerline_cache: Dict[str, np.ndarray] = {}

    # --- interface ---------------------------------------------------------

    def centerline(self, lane_id) -> np.ndarray:
        c = self._centerline_cache.get(lane_id)
        if c is None:
            c = np.asarray(self.vec_map.get_road_lane(lane_id).center.xy, dtype=float)
            self._centerline_cache[lane_id] = c
        return c

    def successors(self, lane_id):
        return list(self.vec_map.get_road_lane(lane_id).next_lanes)

    def adjacent(self, lane_id):
        lane = self.vec_map.get_road_lane(lane_id)
        return list(lane.adj_lanes_left) + list(lane.adj_lanes_right)

    def closest_lane(self, x: float, y: float, heading: float) -> Optional[str]:
        """Nearest lane that also AGREES WITH THE HEADING.

        Purely spatial nearest-lane is wrong here: the oncoming lane is often
        the closest one, and snapping to it would make the ego drive into
        traffic. Lanes more than max_heading_err off are rejected outright, and
        the rest are scored on distance + heading_weight * |angle error|.
        """
        p = np.array([x, y], dtype=float)
        try:
            near = self.vec_map.get_lanes_within(
                np.array([x, y, 0.0]), self.search_radius)
        except Exception:
            near = self.vec_map.lanes

        best, best_score = None, np.inf
        for lane in near:
            c = self.centerline(lane.id)
            if len(c) < 2:
                continue
            d = np.linalg.norm(c - p, axis=1)
            i = int(np.argmin(d))
            j = min(i + 1, len(c) - 1)
            k = max(i - 1, 0)
            if j == k:
                continue
            lane_h = np.arctan2(c[j, 1] - c[k, 1], c[j, 0] - c[k, 0])
            err = abs((lane_h - heading + np.pi) % (2 * np.pi) - np.pi)
            if err > self.max_heading_err:
                continue
            score = d[i] + self.heading_weight * err
            if score < best_score:
                best, best_score = lane.id, score
        return best


# ---------------------------------------------------------------------------
# frame conversion
# ---------------------------------------------------------------------------

def local_to_world(local_xy, local_h, init_pos, init_heading):
    c, s = np.cos(init_heading), np.sin(init_heading)
    return (float(local_xy[0] * c - local_xy[1] * s + init_pos[0]),
            float(local_xy[0] * s + local_xy[1] * c + init_pos[1]),
            float(local_h + init_heading))


def world_to_local(x, y, h, init_pos, init_heading):
    dx, dy = x - init_pos[0], y - init_pos[1]
    c, s = np.cos(-init_heading), np.sin(-init_heading)
    return (float(dx * c - dy * s), float(dx * s + dy * c),
            float(h - init_heading))


# ---------------------------------------------------------------------------
# CENTRED <-> WORLD.  The trap this module fell into on 2026-09-07.
#
# a_traj's init_pos / init_heading come from batch.extras['init_obs'], which is
# expressed relative to batch.centered_agent_state -- NOT in world coordinates.
# So local_to_world(..., init_pos, init_heading) above lands in the CENTRED
# frame, one transform short of world. The VectorMap, meanwhile, IS in world
# coordinates. Querying it with centred coordinates asks about a completely
# different piece of road: measured on example-idx 8, the two frames were
# 85.20 m apart, and the ego's own start position was 0.04 m from a lane in
# world and 13.18 m from one in centred.
#
# rollout_carla.py's export applies exactly this transform on the way out,
# which is why the exported CSV still looked self-consistent while the driving
# was nonsense.
# ---------------------------------------------------------------------------

def centred_to_world_pose(x, y, h, centre):
    cx, cy, ch = centre
    c, s = np.cos(ch), np.sin(ch)
    return (float(x * c - y * s + cx), float(x * s + y * c + cy), float(h + ch))


def world_to_centred_pose(x, y, h, centre):
    cx, cy, ch = centre
    dx, dy = x - cx, y - cy
    c, s = np.cos(-ch), np.sin(-ch)
    return (float(dx * c - dy * s), float(dx * s + dy * c), float(h - ch))


# ---------------------------------------------------------------------------
# ProSim model subclass
# ---------------------------------------------------------------------------

def make_rule_ego_class(prosim_base):
    """Build ProSimRuleEgo on top of a ProSim model class.

    Taken as an argument rather than imported so this module stays importable
    without ProSim (the lane-graph adapter and frame helpers are then testable
    on their own).
    """

    class ProSimRuleEgo(prosim_base):
        def set_ego_policy(self, ego_agent_id, policy: EgoPolicy = None,
                           vec_map=None, kind: str = "idm_pursuit",
                           centre=None, **kwargs):
            """Take `ego_agent_id` out of ProSim's control and drive it by rule.

            ego_agent_id : agent name as it appears in policy_agent_ids
                           ('ego', or a stringified CARLA id)
            policy       : a prebuilt EgoPolicy; else one is made from `kind`
            vec_map      : batch.vector_maps[0]; required for turning/MOBIL
            kind         : 'idm' | 'idm_pursuit' | 'idm_mobil'
            centre       : batch.centered_agent_state as (x, y, [z,] heading) in
                           WORLD. Required whenever vec_map is given -- a_traj is
                           in the centred frame and the map is in world.
            """
            self.ego_agent_id = str(ego_agent_id)
            self.ego_lane_graph = VecMapLaneGraph(vec_map) if vec_map is not None else None
            # The lane graph is in WORLD coordinates but a_traj is centred, so
            # the centre pose is REQUIRED to drive on the real map. Refuse to
            # run without it rather than silently drive 85 m off.
            self.ego_centre = None
            if centre is not None:
                c = np.asarray(centre, dtype=float).reshape(-1)
                self.ego_centre = (float(c[0]), float(c[1]), float(c[-1]))
            dt = float(self.config.DATASET.MOTION.DT)
            self.ego_policy = policy if policy is not None else make_policy(
                kind, dt=dt, lane_graph=self.ego_lane_graph, **kwargs)
            self.ego_debug: List[dict] = []
            # Ordered, de-duplicated lane ids the route has ever planned. The
            # route's own _chain drops lanes as they are passed, so by the end
            # of a rollout it holds only what is still ahead -- that is the
            # remaining plan, not the route driven. The evaluation harness
            # needs the WHOLE intended route as a reference path (station along
            # the route is undefined without it), so it is accumulated here.
            self.ego_route_lanes: List[object] = []

        # -- helpers --------------------------------------------------------

        def _agent_world_state(self, a_traj, task, bidx, nidx, step, extent=None):
            traj = a_traj[task]["traj"][bidx, nidx, step]
            init_pos = a_traj[task]["init_pos"][bidx, nidx].detach().cpu().numpy()
            init_h = float(a_traj[task]["init_heading"][bidx, nidx].reshape(-1)[0])

            lx, ly = float(traj[0]), float(traj[1])
            lh = float(torch.atan2(traj[2], traj[3]))
            # local -> CENTRED (this is as far as init_pos/init_heading go) ...
            x, y, h = local_to_world((lx, ly), lh, init_pos, init_h)
            # ... then CENTRED -> WORLD, so the pose matches the VectorMap.
            if getattr(self, "ego_centre", None) is not None:
                x, y, h = centred_to_world_pose(x, y, h, self.ego_centre)

            speed = self._agent_speed(a_traj, task, bidx, nidx, step)
            length, width = (extent if extent is not None else (4.5, 2.0))
            return VehicleState(x=x, y=y, heading=h, speed=speed,
                                length=length, width=width)

        def _neighbors(self, a_traj, task, bidx, nidx, ids, extents, step):
            """Every agent but the ego, as world VehicleStates at trajectory index `step`."""
            out = []
            for other in range(a_traj[task]["traj"].shape[1]):
                if other == nidx:
                    continue
                oname = ids[other] if other < len(ids) else None
                out.append(self._agent_world_state(a_traj, task, bidx, other, step,
                                                   extents.get(oname, (4.5, 2.0))))
            return out

        def _agent_speed(self, a_traj, task, bidx, nidx, step):
            """Speed in m/s. Prefer the stored velocity channel; fall back to
            differencing positions. The old implementation returned 0.0 for the
            first two steps, which made the ego launch from a standstill even
            when it was already moving."""
            dt = float(self.config.DATASET.MOTION.DT)
            vel = a_traj[task].get("vel") if hasattr(a_traj[task], "get") else None
            if vel is None and "vel" in a_traj[task]:
                vel = a_traj[task]["vel"]
            if vel is not None and step < vel.shape[2]:
                v = vel[bidx, nidx, step]
                sp = float(torch.linalg.norm(v))
                if np.isfinite(sp):
                    return sp
            if step < 1:
                return 0.0
            p0 = a_traj[task]["traj"][bidx, nidx, step - 1, :2]
            p1 = a_traj[task]["traj"][bidx, nidx, step, :2]
            return float(torch.linalg.norm(p1 - p0)) / dt

        def _extents(self, batch):
            try:
                names = list(batch.agent_names[0])
                ext = batch.agent_hist_extent[0, :, -1].detach().cpu().numpy()
                return {n: (float(ext[i][0]), float(ext[i][1]))
                        for i, n in enumerate(names)}
            except Exception:
                return {}

        # -- the override ---------------------------------------------------

        def step_agent_traj(self, a_traj, model_output, policy_agent_ids, t, mode):
            a_traj = super().step_agent_traj(a_traj, model_output, policy_agent_ids,
                                             t, mode)
            if not getattr(self, "ego_agent_id", None):
                return a_traj

            if (getattr(self, "ego_lane_graph", None) is not None
                    and getattr(self, "ego_centre", None) is None):
                raise SystemExit(
                    "\nRULE EGO HAS A MAP BUT NO CENTRE POSE.\n"
                    "  a_traj is in the ego-centred frame; the VectorMap is in "
                    "world coordinates.\n"
                    "  Without batch.centered_agent_state the policy queries the "
                    "map ~tens of metres\n"
                    "  from where the car actually is and drives off the road "
                    "while looking\n"
                    "  perfectly lane-abiding in the wrong frame.\n"
                    "  Pass centre=batch.centered_agent_state to set_ego_policy().\n")

            task = self.tasks[0]
            dt = float(self.config.DATASET.MOTION.DT)
            batch = getattr(self, "_ego_batch", None)
            extents = self._extents(batch) if batch is not None else {}

            for bidx, agent_ids in enumerate(policy_agent_ids[task]):
                ids = [str(a) for a in agent_ids]
                if self.ego_agent_id not in ids:
                    continue
                nidx = ids.index(self.ego_agent_id)

                tidx = int(a_traj[task]["last_step"])
                start = tidx - self.rollout_steps
                init_pos = a_traj[task]["init_pos"][bidx, nidx].detach().cpu().numpy()
                init_h = float(a_traj[task]["init_heading"][bidx, nidx].reshape(-1)[0])

                ego_ext = extents.get(self.ego_agent_id, (4.5, 2.0))
                state = self._agent_world_state(a_traj, task, bidx, nidx,
                                                start - 1, ego_ext)

                # A policy that decides from what it sees at the start of a step
                # (the rule ego, the no-CARLA bridge) is handed the others there.
                # The CARLA worker instead PLACES them in a world that runs
                # through the step, so it needs where they are at its END:
                # handed the start, every car it drew sat 0.1 s behind the ego
                # (measured 0.05 s, after each tick's own physics carried it
                # half of that; tools/audit_carla_cells.py A3).
                at_end = bool(getattr(self.ego_policy, "neighbors_at_step_end", False))
                for step in range(start, tidx):
                    if at_end:
                        state = self.ego_policy.step(
                            state, self._neighbors(a_traj, task, bidx, nidx, ids, extents, step),
                            neighbors_before=self._neighbors(a_traj, task, bidx, nidx, ids,
                                                             extents, step - 1))
                    else:
                        state = self.ego_policy.step(
                            state, self._neighbors(a_traj, task, bidx, nidx, ids, extents, step - 1))

                    # WORLD -> centred -> local, the exact inverse of the read
                    sx, sy, sh = state.x, state.y, state.heading
                    if getattr(self, "ego_centre", None) is not None:
                        sx, sy, sh = world_to_centred_pose(sx, sy, sh, self.ego_centre)
                    lx, ly, lh = world_to_local(sx, sy, sh, init_pos, init_h)
                    a_traj[task]["traj"][bidx, nidx, step] = torch.tensor(
                        [lx, ly, np.sin(lh), np.cos(lh)],
                        dtype=a_traj[task]["traj"].dtype,
                        device=a_traj[task]["traj"].device)

                    # Keep the velocity channel consistent with the overridden
                    # position -- step_env feeds it into the observation update,
                    # so a stale value would make the ego's own observation
                    # disagree with where it actually is.
                    if "vel" in a_traj[task] and a_traj[task]["vel"] is not None:
                        vw = np.array([state.speed * np.cos(state.heading),
                                       state.speed * np.sin(state.heading)])
                        c, s = np.cos(-init_h), np.sin(-init_h)
                        vloc = np.array([vw[0] * c - vw[1] * s,
                                         vw[0] * s + vw[1] * c])
                        if step < a_traj[task]["vel"].shape[2]:
                            a_traj[task]["vel"][bidx, nidx, step] = torch.tensor(
                                vloc, dtype=a_traj[task]["vel"].dtype,
                                device=a_traj[task]["vel"].device)

                    self.ego_debug.append(
                        {"t": step * dt, "x": state.x, "y": state.y,
                         "heading": state.heading, "speed": state.speed,
                         "lane": getattr(self.ego_policy.route, "current_lane_id", None)})

                    for lane_id in getattr(self.ego_policy.route, "_chain", []):
                        if lane_id not in self.ego_route_lanes:
                            self.ego_route_lanes.append(lane_id)

            return a_traj

        def ego_route_polyline(self):
            """The full intended route as (N, 2) world points, or None.

            Concatenated lane centrelines in plan order. This is the ROUTE, not
            the driven trajectory: handing the driven path to the harness would
            make path adherence trivially perfect and measure nothing.
            """
            if not getattr(self, "ego_route_lanes", None) or self.ego_lane_graph is None:
                return None
            import numpy as _np
            parts = []
            for lane_id in self.ego_route_lanes:
                try:
                    parts.append(_np.asarray(self.ego_lane_graph.centerline(lane_id),
                                             dtype=float))
                except Exception:
                    continue
            return _np.concatenate(parts, axis=0) if parts else None

        def forward(self, batch, mode):
            # stash the batch so step_agent_traj can read extents/vector_maps
            self._ego_batch = batch
            if (getattr(self, "ego_lane_graph", None) is None
                    and getattr(batch, "vector_maps", None)):
                self.ego_lane_graph = VecMapLaneGraph(batch.vector_maps[0])
                if hasattr(self, "ego_policy"):
                    self.ego_policy.lane_graph = self.ego_lane_graph
                    if hasattr(self.ego_policy.route, "g"):
                        self.ego_policy.route.g = self.ego_lane_graph
            return super().forward(batch, mode)

    return ProSimRuleEgo
