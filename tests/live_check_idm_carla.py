"""Known-answer check: the harness's IDM, driven by the sensor worker in a live CARLA.

    CARLA_ROOT=<dist> /scratch/veerk41/venvs/simlingo/bin/python tests/live_check_idm_carla.py --host <node>

Uses a running server (loads Town04 if needed; run it when a town switch does not
matter, e.g. before the Town04 reruns). Needs no GPU of its own.

IDM's answers are known from its parameters (configs/policy/idm.yaml):
  1. on a free road it settles at `desired_speed_mps` (8 m/s);
  2. behind a STOPPED car it holds `laws.BLOCKED_STANDOFF_M` (8 m) back, not
     `min_gap_m`: idm/laws.py ego_s0 swaps in that standoff for a leader at
     <= 1 m/s within 12 m, so idm_mobil has room to go round it, and plain idm
     shares the law (idm/policy.py). The check first assumed min_gap and failed
     at 6.55 m -- the assumption was wrong, not the drive.
The drive goes through sensor_worker.Session -- osc2runner's state observation,
_to_command, SpawnGear and AccelerationTracker -- exactly as a harness cell does.
CONTROL: the same drive with the tracker replaced by the open-loop pedal map
(throttle a/3, brake -a/5) must fall short of 8 m/s: osc2runner measured that
map at 6.3 of 8 m/s (third_party/osc2runner commit 1d3edae), so a check that
passed both ways could not tell the actuation apart.
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

import sensor_worker as W

HARNESS = Path(_os.environ.get("HARNESS", "/scratch/veerk41/scenario_orchestration"))
PASS, FAIL = [], []

#: Town04 road 40, westbound lane 40_0_-1: 237 m dead straight at y = 16.3.
LANE_Y, START_X, END_X, LEAD_X = 16.3, -130.0, -357.0, -300.0


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def drive(carla, args, request, open_loop=False, steps=320):
    """Speeds and bumper gaps, one per ProSim step (0.1 s)."""
    import carla as carla_mod
    session = W.Session(carla_mod, W.load_rig_module(W.DEFAULT_OSC2RUNNER), args.host, args.port,
                        allow_load_town=True)
    route = np.stack([np.arange(START_X, END_X, -0.5), np.full(int((START_X - END_X) / 0.5), LANE_Y)], axis=1)
    lead = {"x": LEAD_X, "y": LANE_Y, "yaw_rad": math.pi, "speed": 0.0, "length": 4.7, "width": 1.8}
    session.init({"op": "init", "town": "Town04", "dt": 0.1, "policy_hz": 20.0,
                  "ego": {"x": START_X, "y": LANE_Y, "yaw_rad": math.pi, "speed": 0.0, "length": 4.9, "width": 2.1},
                  "actors": [lead], "route_world": route.tolist(),
                  "policy_py": str(HARNESS / "third_party/idm/scenario_orchestration/policy.py"),
                  "policy_request": request, "frames_dir": None})
    if open_loop:
        session.tracker.step = lambda accel, v, dt: ((accel / 3.0, 0.0) if accel >= 0 else (0.0, -accel / 5.0))
    speeds, gaps = [], []
    try:
        for _ in range(steps):
            out = session.step({"op": "step", "actors": [lead]})
            ego_x = out["ego"]["x"]
            speeds.append(out["ego"]["speed"])
            gaps.append(abs(ego_x - LEAD_X) - 2.45 - 2.35)       # MKZ and Model 3 half-lengths
    finally:
        session.close()
    return np.array(speeds), np.array(gaps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    cfg = yaml.safe_load((HARNESS / "configs/policy/idm.yaml").read_text())
    request = {"name": "idm", "implementation": cfg["implementation"], "interface": "ego_policy_v1",
               "observation_space": cfg["observation_space"], "action_space": cfg["action_space"],
               "seed": 0, "parameters": dict(cfg["parameters"])}
    v0 = float(cfg["parameters"]["desired_speed_mps"])
    sys.path.insert(0, str(HARNESS / "third_party/idm"))
    from idm import laws
    standoff = float(laws.BLOCKED_STANDOFF_M)
    print(f"idm: desired {v0} m/s, stopped-blocker standoff {standoff} m (min_gap "
          f"{cfg['parameters']['min_gap_m']} m is for a moving leader); lead stopped {START_X - LEAD_X:.0f} m ahead")

    speeds, gaps = drive(None, args, request)
    free = speeds[gaps > 60.0]
    cruise = float(np.max(free)) if len(free) else float("nan")
    settled = float(np.mean(free[-50:])) if len(free) >= 50 else float("nan")
    print("  free-road speed every 0.5 s:", np.round(free[::5], 2).tolist())
    print(f"  peak {cruise:.2f} m/s, settled mean over the last 5 s of free road {settled:.2f} m/s")
    check(f"free road: cruises at the desired {v0} m/s (within 5%)",
          abs(cruise - v0) <= 0.05 * v0, f"peak while > 60 m from the lead: {cruise:.2f} m/s")
    check(f"behind a stopped car: stops about the {standoff} m standoff short (5.5-9 m)",
          speeds[-1] < 0.2 and standoff - 2.5 <= gaps[-1] <= standoff + 1.0,
          f"final speed {speeds[-1]:.2f} m/s, bumper gap {gaps[-1]:.2f} m")
    check("never touches the stopped car", float(np.min(gaps)) > 0.0, f"closest {np.min(gaps):.2f} m")

    speeds_ol, gaps_ol = drive(None, args, request, open_loop=True)
    free_ol = speeds_ol[gaps_ol > 60.0]
    cruise_ol = float(np.max(free_ol)) if len(free_ol) else float("nan")
    check("CONTROL: the open-loop pedal map falls short of the desired speed",
          cruise_ol < 0.95 * v0, f"{cruise_ol:.2f} m/s (osc2runner measured 6.3 of 8 before its fix)")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
