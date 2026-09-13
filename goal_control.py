"""Goal conditioning for ProSim -- a structured alternative to text prompts.

WHY THIS EXISTS. `animate_rollout.py` hardcodes
`PROMPT.CONDITION.TYPES = ['llm_text_OneText']` with the comment "skip
goal/v_action_tag/drag_point entirely", and rollout_carla.py inherited it. But
measured on a real 30-agent CARLA batch:

    goal            : input (1,30,3)     mask 30/30   finite
    drag_point      : input (1,30,16,2)  mask 30/30   ALL NaN (unusable)
    llm_text_OneText: mask  0/30                      no cached text

`goal` is populated and finite. It is also the most direct way to say "turn
left": put the goal on the left branch. No LLM involved.

FRAME -- the part that is easy to get wrong. From
`prosim/dataset/format_utils.py:606-611`:

    goal = batch.agent_fut[b, o, fut_len-1]                  # world-ish, centred frame
    if cfg.GOAL.LOCAL:                                       # True in waymo_demo.yaml
        goal = transform_to_frame_offset_rot(goal, local_state)

so `condition['goal']['input'][b, c, 0:2]` is the goal expressed in the AGENT'S
OWN BODY FRAME: +x forward along the agent's heading, +y lateral. Index 2 is the
timestep the goal applies at (the agent's future length).

WHICH SIDE IS +y? THE DRIVER'S RIGHT. CARLA's world frame is LEFT-handed (x
forward, y right, z up), so the usual "+y body = left" reading is MIRRORED here.
Decided from CARLA's own edge labels rather than from convention: for all 11817
sampled lane points in town10hd_lanes.json, CARLA's `left_edge` maps to body y
= -1.750 (100.0% negative) and its `right_edge` to +1.750 (100.0% positive).

The rotation maths below is NOT affected -- it round-trips and reproduces
ProSim's own goal values to 3.26e-06 m. Only the WORDS "left"/"right" were
wrong, and they were wrong everywhere: the --list-agents table, the "goal is
N m to the LEFT" line, the sidecar's "side" field, and the branch selector.

A goal written in the wrong frame steers the agent somewhere silently wrong,
which is exactly the failure mode this project keeps hitting -- so
`test_goal_control.py` round-trips against ProSim's OWN populated goal values
rather than trusting this docstring.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# In CARLA's left-handed world frame a POSITIVE lateral offset, and a POSITIVE
# heading change, both mean the driver's RIGHT. Verified against CARLA's own
# lane edge labels: 11817/11817 points agree.
LEFT_SIGN = -1.0

# A branch is only "straight" if it barely turns. Without this the selector
# happily labelled a 90-degree branch "straight" purely because it was the less
# extreme of two right turns -- which is what sent the ego left when 'straight'
# was requested.
STRAIGHT_MAX_DEG = 30.0


def side_name(lateral: float) -> str:
    """'LEFT' / 'RIGHT' for a body-frame lateral offset, correct for CARLA."""
    return "LEFT" if lateral * LEFT_SIGN > 0 else "RIGHT"


def turn_name(turn_rad: float) -> str:
    """Classify a branch by how much it actually turns."""
    deg = np.degrees(float(turn_rad))
    if abs(deg) <= STRAIGHT_MAX_DEG:
        return "straight"
    return "left" if deg * LEFT_SIGN > 0 else "right"


def world_to_body(goal_world: Sequence[float],
                  agent_xy: Sequence[float],
                  agent_heading: float) -> np.ndarray:
    """World point -> the agent's body frame.

    Returns (forward, lateral). CARLA's frame is LEFT-handed, so +lateral is the
    driver's RIGHT -- see the module docstring for the measurement that settled
    this. Use `side_name()` rather than writing the label by hand.
    """
    d = np.asarray(goal_world, dtype=float)[:2] - np.asarray(agent_xy, dtype=float)[:2]
    c, s = np.cos(-agent_heading), np.sin(-agent_heading)
    return np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])


def body_to_world(goal_body: Sequence[float],
                  agent_xy: Sequence[float],
                  agent_heading: float) -> np.ndarray:
    b = np.asarray(goal_body, dtype=float)[:2]
    c, s = np.cos(agent_heading), np.sin(agent_heading)
    return np.array([b[0] * c - b[1] * s, b[0] * s + b[1] * c]) + \
        np.asarray(agent_xy, dtype=float)[:2]


def agent_world_pose(batch, agent_index: int) -> Tuple[np.ndarray, float]:
    """(xy, heading) of one agent in CARLA world coordinates, at the rollout start.

    `agent_index` indexes the PROMPT agent list -- the same convention
    text_control's --text-agents uses, i.e. batch.extras['prompt']
    ['motion_pred']['agent_ids'].

    THERE ARE TWO DIFFERENT AGENT ORDERINGS IN A BATCH and they do not agree:

        batch.agent_names / init_obs['agent_ids'] : ['ego','123','125','124',...]
        prompt / io_pairs_batch['agent_names']    : ['ego','123','124','125',...]
        batch.tgt_agent_idxs maps io_pairs index -> agent_names index

    The goal condition is built from io_pairs_batch, so poses MUST be read from
    io_pairs_batch too. Reading init_obs with an io-pairs index silently returns
    a different agent's pose -- measured median error 25.6 m, max 83 m.

    io_pairs positions/headings are in the CENTRED frame; centered_agent_state is
    that frame's pose in world (same centred->world transform verified to
    0.0000 m in rollout_carla.py).
    """
    centre = batch.centered_agent_state.as_format("x,y,z,h").cpu().numpy()[0]
    cx, cy, _cz, ch = centre
    iop = batch.extras["io_pairs_batch"]
    p = iop["position"][0, 0, agent_index].detach().cpu().numpy()
    h = float(np.asarray(iop["heading"][0, 0, agent_index].detach().cpu()).reshape(-1)[0])
    c, s = np.cos(ch), np.sin(ch)
    return np.array([p[0] * c - p[1] * s + cx, p[0] * s + p[1] * c + cy]), h + ch


def agent_centred_pose(batch, agent_index: int) -> Tuple[np.ndarray, float]:
    """Same as agent_world_pose but left in the CENTRED frame, which is the frame
    io_pairs_batch['goal'] is expressed relative to."""
    iop = batch.extras["io_pairs_batch"]
    p = iop["position"][0, 0, agent_index].detach().cpu().numpy()
    h = float(np.asarray(iop["heading"][0, 0, agent_index].detach().cpu()).reshape(-1)[0])
    return np.asarray(p, dtype=float), h


def turn_goal_from_lane_graph(lane_graph, agent_xy, agent_heading,
                              direction: str = "left",
                              lookahead: float = 45.0,
                              max_depth: int = 6,
                              min_forward: float = 5.0):
    """Pick a world goal on the chosen branch at the next REAL junction.

    Returns (goal_world_xy, branch_turn_rad, n_options) or None.

    Walking only ONE lane forward is not enough in Town10HD: lanes are short
    (median 28.4 m) and a junction is its own lane, so the first successor is
    usually the only successor and 'left'/'right'/'straight' all return the same
    point. Measured on the real map before this fix: all three directions gave an
    identical goal 28.6 m to the RIGHT for a left request.

    So: walk forward through single-successor lanes until a lane offers >1
    successor, and branch THERE. n_options reports how many branches existed --
    1 means the agent had no choice and the direction was ignored.
    """
    lane_id = lane_graph.closest_lane(float(agent_xy[0]), float(agent_xy[1]),
                                      float(agent_heading))
    if lane_id is None:
        return None

    chain = [np.asarray(lane_graph.centerline(lane_id), dtype=float)]
    cur = lane_id
    visited = {lane_id}

    for _ in range(max_depth):
        succ = [c for c in lane_graph.successors(cur) if c not in visited]
        if not succ:
            cur = None
            break
        if len(succ) > 1:
            break                       # a real junction -- branch here
        cur = succ[0]
        visited.add(cur)
        chain.append(np.asarray(lane_graph.centerline(cur), dtype=float))
    else:
        succ = []

    if cur is None or not succ:
        return None

    prev = chain[-1]
    if len(prev) < 2:
        return None
    end_h = float(np.arctan2(prev[-1, 1] - prev[-2, 1], prev[-1, 0] - prev[-2, 0]))

    # Score each branch by HOW MUCH IT ACTUALLY TURNS, not by where its goal
    # point lands laterally.
    #
    # Scoring on the goal point's lateral offset (the previous version) is
    # wrong whenever the approach chain is long: the goal sits ~lookahead
    # metres along a path that may already have curved 90 degrees before the
    # junction, so the lateral offset is dominated by the approach, not by the
    # branch choice. Measured on the ego at example-idx 8: BOTH branches turned
    # about -90 deg (a right turn), with goal laterals of -81.8 and -83.8 m.
    # Lateral scoring picked -81.8 as "straight" and the same point again as
    # "left" -- so asking for 'straight' handed back a 90-degree turn, and
    # 'left' and 'straight' returned identical coordinates.
    #
    # The branch's own turn angle is the physically meaningful quantity and is
    # independent of the approach. n_options still reports how many branches
    # existed; turn_name(turn) says what the chosen one ACTUALLY does, so the
    # caller can shout when it does not match what was asked for.
    junction_start = sum(len(c) for c in chain)
    best, best_score, best_turn, best_lat = None, None, None, None
    for cand in succ:
        seg = np.asarray(lane_graph.centerline(cand), dtype=float)
        if len(seg) < 2:
            continue
        seg_end_h = float(np.arctan2(seg[-1, 1] - seg[-2, 1],
                                     seg[-1, 0] - seg[-2, 0]))
        turn = wrap_angle(seg_end_h - end_h)

        path = np.concatenate(chain + [seg], axis=0)
        d = np.linalg.norm(path - np.asarray(agent_xy, dtype=float)[:2], axis=1)
        start = int(np.argmin(d))
        seglen = np.linalg.norm(np.diff(path[start:], axis=0), axis=1)
        idx = start + int(np.searchsorted(np.cumsum(seglen), lookahead)) + 1
        # never place the goal before the junction, or direction is meaningless
        idx = min(max(idx, junction_start + len(seg) // 2), len(path) - 1)
        goal = path[idx]

        body = world_to_body(goal, agent_xy, agent_heading)
        fwd, lat = float(body[0]), float(body[1])
        # A goal BEHIND the agent is meaningless -- it cannot drive backwards to
        # reach it, and the model just ignores it. Observed in practice: a "left"
        # goal landed 72.9 m BEHIND the agent because the branch doubled back.
        if fwd < min_forward:
            continue
        # LEFT_SIGN carries CARLA's left-handed frame: a positive heading
        # change is the driver's RIGHT here, not their left.
        turn_signed = turn * LEFT_SIGN          # now +ve = genuinely LEFT

        # A junction does not owe you every direction. If no branch goes the way
        # that was asked for, say so (return None) instead of handing back the
        # closest one -- returning a 90-degree turn for 'straight' is how the
        # ego was sent left when 'straight' was requested, and the caller had no
        # way to tell. rollout_carla.py turns the None into an explicit
        # "no such branch, run --list-agents" error.
        if turn_name(turn) != direction:
            continue

        score = {"left": turn_signed,
                 "right": -turn_signed,
                 "straight": -abs(turn_signed)}[direction]
        if best_score is None or score > best_score:
            best, best_score, best_turn, best_lat = goal, score, turn, lat

    if best is None:
        # either every branch was behind the agent, or none of them goes the
        # requested way -- both mean "you cannot have this direction here"
        return None
    return best, float(best_turn), len(succ)


def set_goal_condition(batch, agent_index: int, goal_world,
                       exclusive: bool = True, min_forward: float = 5.0) -> dict:
    """Write a world-frame goal for one agent into the batch's goal condition.

    Mirrors what text_control does for the text channel: enable this agent's
    condition and (by default) disable every other one, so the rollout isolates
    the effect of this single instruction.

    Returns a dict describing what was written, for logging/assertions.
    """
    cond = batch.extras["condition"]
    if "goal" not in cond.keys():
        raise SystemExit(
            "no 'goal' condition on the batch -- PROMPT.CONDITION.TYPES must "
            "include 'goal' (rollout_carla.py used to force it to text only)")

    g = cond["goal"]
    prompt_idx = g["prompt_idx"][0, :, 0].detach().cpu().numpy()
    matches = np.nonzero(prompt_idx == agent_index)[0]
    if len(matches) == 0:
        raise SystemExit(
            f"agent index {agent_index} has no goal-condition slot "
            f"(prompt_idx values present: {sorted(set(prompt_idx.tolist()))})")
    cidx = int(matches[0])

    xy, heading = agent_world_pose(batch, agent_index)
    body = world_to_body(goal_world, xy, heading)

    # A goal BEHIND the agent is unreachable -- it cannot reverse to get there,
    # and the model effectively ignores it. Observed 2026-09-04: a goal 37.3 m
    # behind agent '89' left it unconditioned while a different car happened to
    # drive toward that point, which reads exactly like the condition went to
    # the wrong agent. Refuse it instead.
    if body[0] < min_forward:
        raise SystemExit(
            f"\nGOAL IS BEHIND AGENT {agent_index}.\n"
            f"  agent world  : ({xy[0]:.2f}, {xy[1]:.2f})  heading {heading:.3f} rad\n"
            f"  goal  world  : ({float(goal_world[0]):.2f}, {float(goal_world[1]):.2f})\n"
            f"  in its frame : {body[0]:+.1f} m forward, "
            f"{abs(body[1]):.1f} m to its {side_name(body[1])}\n"
            f"                 ^^^ forward must be >= {min_forward} m\n"
            "\nThe agent cannot reverse to reach it, so the condition does nothing\n"
            "and any car that moves toward that point is doing so on its own.\n"
            "Re-run with --list-agents: each row's [x,y] belongs to THAT row's\n"
            "agent only -- a goal taken from another agent's row is usually behind\n"
            "this one.\n")

    if exclusive:
        g["mask"][0, :] = False
        g["prompt_mask"][0, :] = False
    g["input"][0, cidx, 0] = float(body[0])
    g["input"][0, cidx, 1] = float(body[1])
    g["mask"][0, cidx] = True
    g["prompt_mask"][0, agent_index] = True

    return {"cond_idx": cidx, "agent_index": agent_index,
            "agent_world_xy": xy, "agent_heading": heading,
            "goal_world": np.asarray(goal_world, dtype=float)[:2],
            "goal_body": body,
            "timestep": float(g["input"][0, cidx, 2])}
