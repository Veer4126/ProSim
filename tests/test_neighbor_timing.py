"""Which instant of the other agents prosim_ego hands an ego policy.

    /scratch/veerk41/venvs/tfv6/bin/python tests/test_neighbor_timing.py

No model, no dataset: ProSimRuleEgo is built on a stand-in base model whose
step_agent_traj returns the trajectories it is given, and agent "b" drives
+1 m per step along x, so its x IS the trajectory index it was read at.

  the CARLA ego (RemoteSensorEgoPolicy declares neighbors_at_step_end) gets the
  others at the END of each step -- index `step` -- and their start, `step - 1`,
  as neighbors_before;
  CONTROL: an ego that decides from what it sees (no declaration) still gets
  `step - 1`, unchanged.
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import types

import torch

from ego_control import VehicleState
from prosim_ego import make_rule_ego_class
from remote_ego import RemoteSensorEgoPolicy
from external_ego import ExternalEgoPolicy

PASS, FAIL = [], []
START, TIDX, T = 10, 20, 25


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


class Base:
    def step_agent_traj(self, a_traj, model_output, policy_agent_ids, t, mode):
        return a_traj


class Recorder:
    route = types.SimpleNamespace(_chain=[], current_lane_id=None)

    def __init__(self, at_end):
        if at_end:
            self.neighbors_at_step_end = True
        self.calls = []

    def step(self, state, neighbors, neighbors_before=None):
        self.calls.append((neighbors[0].x, None if neighbors_before is None else neighbors_before[0].x))
        return state


def run(policy):
    cls = make_rule_ego_class(Base)
    m = cls.__new__(cls)
    m.config = types.SimpleNamespace(DATASET=types.SimpleNamespace(MOTION=types.SimpleNamespace(DT=0.1)))
    m.tasks, m.rollout_steps = ["task"], TIDX - START
    m.ego_agent_id, m.ego_policy = "ego", policy
    m.ego_lane_graph, m.ego_centre, m._ego_batch = None, None, None
    m.ego_debug, m.ego_route_lanes = [], []
    traj = torch.zeros(1, 2, T, 4)
    traj[..., 3] = 1.0                                   # heading 0: sin 0, cos 1
    traj[0, 1, :, 0] = torch.arange(T, dtype=torch.float32)   # agent b: x = index
    a_traj = {"task": {"traj": traj, "init_pos": torch.zeros(1, 2, 2),
                       "init_heading": torch.zeros(1, 2, 1), "last_step": TIDX}}
    m.step_agent_traj(a_traj, None, {"task": [["ego", "b"]]}, 0, "val")
    return policy.calls


def main():
    print("\n=== which instant of the other agents each ego is handed ===")
    remote = run(Recorder(at_end=True))
    check("CARLA ego: at step k the others are at index k (the end of the step)",
          [round(c[0]) for c in remote] == list(range(START, TIDX)), str([round(c[0]) for c in remote]))
    check("and their start of step, index k - 1, comes as neighbors_before",
          [round(c[1]) for c in remote] == list(range(START - 1, TIDX - 1)), str([c[1] for c in remote][:3]))
    plain = run(Recorder(at_end=False))
    check("CONTROL: an ego that decides from what it sees still gets index k - 1",
          [round(c[0]) for c in plain] == list(range(START - 1, TIDX - 1)) and all(c[1] is None for c in plain),
          str([round(c[0]) for c in plain][:3]))
    check("RemoteSensorEgoPolicy declares end-of-step neighbours; the no-CARLA bridge does not",
          getattr(RemoteSensorEgoPolicy, "neighbors_at_step_end", False) is True
          and not getattr(ExternalEgoPolicy, "neighbors_at_step_end", False))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
