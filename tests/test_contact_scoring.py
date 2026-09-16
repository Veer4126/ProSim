"""Check 5 (offline): what the harness scores as a collision, on traces with a known answer.

    /scratch/veerk41/venvs/tfv6/bin/python tests/test_contact_scoring.py

The merged harness decides collisions from trajectories alone
(metrics/geometry/contact.py vehicle_contacts, counted past
contact_min_depth_m = 0.08 m). An ego (4.9 x 2.1 m) sits still; a 4.7 x 1.8 m car
drives straight at its nose and stops with a chosen final bumper gap:

  gap -0.50 m (0.50 m deep)  -> one contact, depth 0.50
  gap -0.19 m (the depth CARLA physics allowed in live_check_contact) -> one contact
  gap -0.05 m (a graze under 0.08 m) -> no contact at the 0.08 threshold, one at 0
  gap +0.30 m (near miss)    -> none at either threshold
CONTROL: the same 0.5 m-deep stop placed 3 m to the side -> none.
Then SimLingo's red_light cell, read from disk: its deepest overlap with every car,
so the "physical crash not scored" finding is a number, not a claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

H = Path("/scratch/veerk41/scenario_orchestration")
sys.path.insert(0, str(H))
from metrics.geometry.contact import contacts, vehicle_contacts   # noqa: E402
from metrics.ingest.canonical import read_run_dir                 # noqa: E402
from metrics.rollout import ActorTrace                             # noqa: E402

PASS, FAIL = [], []
EGO_L, EGO_W, CAR_L, CAR_W = 4.9, 2.1, 4.7, 1.8


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def scene(final_gap, lateral=0.0):
    t = np.arange(0, 4.0001, 0.1)
    stop_x = EGO_L / 2 + CAR_L / 2 + final_gap
    x = np.maximum(stop_x + 20.0 - 10.0 * t, stop_x)          # 10 m/s, then parked
    vx = np.where(x > stop_x, -10.0, 0.0)
    n = len(t)
    ego = ActorTrace("ego", np.zeros((n, 3)), np.zeros((n, 2)), np.ones(n, bool), (EGO_L, EGO_W), is_ego=True)
    car = ActorTrace("car", np.stack([x, np.full(n, lateral), np.full(n, np.pi)], 1),
                     np.stack([vx, np.zeros(n)], 1), np.ones(n, bool), (CAR_L, CAR_W),
                     attributes={"type_id": "vehicle.tesla.model3"})
    return ego, car, t


def depth(gap, lateral=0.0, min_depth=0.08):
    ego, car, t = scene(gap, lateral)
    found = contacts(ego, [car], t, substeps=4, min_depth_m=min_depth)
    return len(found), (found[0].max_depth_m if found else 0.0)


def main() -> int:
    n, d = depth(-0.50)
    check("0.50 m deep -> one contact of depth 0.50", n == 1 and abs(d - 0.50) < 0.02, f"{n}, {d:.3f} m")
    n, d = depth(-0.19)
    check("0.19 m deep (what CARLA physics allowed live) -> one contact", n == 1, f"{n}, {d:.3f} m")
    n8, _ = depth(-0.05)
    n0, d0 = depth(-0.05, min_depth=0.0)
    check("0.05 m graze -> not counted at 0.08 m, seen at 0", n8 == 0 and n0 == 1, f"{n8} / {n0} ({d0:.3f} m)")
    n8, _ = depth(0.30)
    n0, _ = depth(0.30, min_depth=0.0)
    check("0.30 m near miss -> nothing at either threshold", n8 == 0 and n0 == 0, f"{n8} / {n0}")
    n0, _ = depth(-0.50, lateral=3.0, min_depth=0.0)
    check("CONTROL: the same stop 3 m to the side -> nothing", n0 == 0, str(n0))

    cell = H / "results/raw_prosim_eval/red_light__prosim_carla__simlingo__s000"
    rollout = read_run_dir(str(cell))
    if rollout.evaluable:
        found = vehicle_contacts(rollout.trace, min_depth_m=0.0, merge_gap_s=0.5)
        deepest = max((c.max_depth_m for c in found), default=0.0)
        print(f"  SimLingo red_light ({cell.name}): {len(found)} box overlaps, deepest {deepest:.3f} m, "
              f"first at {min((c.t_start for c in found), default=float('nan')):.2f} s "
              f"-> {'counted' if deepest > 0.08 else 'NOT counted'} at 0.08 m")
    else:
        print(f"  SimLingo red_light cell not evaluable: {rollout.reason}")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
