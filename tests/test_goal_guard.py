"""Guard tests for set_goal_condition -- no model, no dataset, no GPU.

Stubs the four batch fields the function touches, so this runs in seconds
anywhere. Complements test_goal_control.py, which needs the real CARLA batch.

Reproduces the geometry of the 2026-09-04 run that looked like the goal had
gone to the wrong agent: agent at (-87.72, 16.46) heading -3.136, goal at
(-50, -60) -- 37.3 m BEHIND it.

    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 tests/test_goal_guard.py"
"""

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import sys

import numpy as np
import torch


from goal_control import (body_to_world, set_goal_condition, side_name,
                          world_to_body)

N = 4
AGENT_XY, AGENT_H, AGENT_IDX = (-87.72325897, 16.45824242), -3.1360315, 2

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


class _Centre:
    def as_format(self, fmt):
        return torch.tensor([[0.0, 0.0, 0.0, 0.0]])


class Batch:
    """Only the fields set_goal_condition actually reads."""

    def __init__(self, xy, h):
        self.centered_agent_state = _Centre()
        pos = torch.zeros(1, 1, N, 2)
        pos[0, 0, AGENT_IDX] = torch.tensor(xy)
        head = torch.zeros(1, 1, N, 1)
        head[0, 0, AGENT_IDX, 0] = h
        self.extras = {
            "io_pairs_batch": {"position": pos, "heading": head},
            "condition": {"goal": {
                # shuffled, as real batches are -- slot order != agent order
                "prompt_idx": torch.tensor([[[0], [3], [2], [1]]]),
                "input": torch.zeros(1, N, 3),
                "mask": torch.zeros(1, N, dtype=torch.bool),
                "prompt_mask": torch.zeros(1, N, dtype=torch.bool)}}}


print("=" * 72)
print("1. the geometry of the run that looked like a routing bug")
print("=" * 72)
b = world_to_body([-50.0, -60.0], AGENT_XY, AGENT_H)
print(f"  goal (-50,-60) in the agent's frame: fwd {b[0]:+.2f} m  "
      f"lat {b[1]:+.2f} m ({side_name(b[1])})")
check("reproduces the sidecar's goal_body exactly",
      abs(b[0] - (-37.2975)) < 1e-3 and abs(b[1] - 76.6668) < 1e-3,
      f"{np.round(b, 4)} vs sidecar [-37.2975, 76.6668]")

print()
print("=" * 72)
print("2. the guard")
print("=" * 72)
try:
    set_goal_condition(Batch(AGENT_XY, AGENT_H), AGENT_IDX, [-50.0, -60.0])
    check("a goal 37 m BEHIND the agent is refused", False, "it was ACCEPTED")
except SystemExit as e:
    msg = str(e)
    check("a goal 37 m BEHIND the agent is refused", True)
    check("the message names the agent and the shortfall",
          "-37.3 m forward" in msg and f"AGENT {AGENT_IDX}" in msg,
          repr(msg.splitlines()[1]))

# CONTROL: the same distance, mirrored to the FRONT, must be accepted --
# otherwise the guard could pass by rejecting everything.
ahead = body_to_world([37.3, 76.67], AGENT_XY, AGENT_H)
bt = Batch(AGENT_XY, AGENT_H)
try:
    info = set_goal_condition(bt, AGENT_IDX, ahead)
    check("CONTROL: the mirrored goal (same distance, AHEAD) is accepted", True,
          f"body={np.round(info['goal_body'], 2)}")
    g = bt.extras["condition"]["goal"]
    # prompt_idx = [0,3,2,1]; agent 2 lives in the slot whose prompt_idx is 2
    check("it lands in the slot whose prompt_idx == 2 (slot 2, not slot 3)",
          info["cond_idx"] == 2 and bool(g["mask"][0, 2])
          and int(g["mask"][0].sum()) == 1,
          f"cond_idx={info['cond_idx']} mask={g['mask'][0].tolist()}")
    check("prompt_mask marks agent 2 only",
          bool(g["prompt_mask"][0, 2]) and int(g["prompt_mask"][0].sum()) == 1)
except SystemExit as e:
    check("CONTROL: the mirrored goal (same distance, AHEAD) is accepted",
          False, str(e)[:120])

print()
print("=" * 72)
print("3. the boundary")
print("=" * 72)
for fwd, want_ok in ((5.001, True), (4.999, False), (0.0, False), (-0.001, False)):
    tgt = body_to_world([fwd, 0.0], AGENT_XY, AGENT_H)
    try:
        set_goal_condition(Batch(AGENT_XY, AGENT_H), AGENT_IDX, tgt)
        got = True
    except SystemExit:
        got = False
    check(f"fwd={fwd:+.3f} m -> {'accept' if want_ok else 'refuse'}", got == want_ok)

tgt = body_to_world([20.0, 0.0], AGENT_XY, AGENT_H)
try:
    set_goal_condition(Batch(AGENT_XY, AGENT_H), AGENT_IDX, tgt, min_forward=50.0)
    got = True
except SystemExit:
    got = False
check("min_forward=50 refuses a goal only 20 m ahead", not got)

print()
print("=" * 72)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("   FAILED:", f)
print("=" * 72)
sys.exit(1 if FAIL else 0)
