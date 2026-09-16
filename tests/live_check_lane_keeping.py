"""B3: does a vision policy hold its lane through the worker's control loop?

    CARLA_ROOT=<dist> HF_HOME=/scratch/veerk41/hf_cache <policy venv>/bin/python \
        tests/live_check_lane_keeping.py --host <node> --policy tfv6|simlingo [--cwd DIR]

tfv6 and SimLingo steer through PDM-Lite's LateralPIDController, whose gains are
tuned per DECISION at 20 Hz (docs/policy_decision_rate.md). Both drive an empty
straight Town04 road (road 40, lane centre y = 16.3) for 15 s through
sensor_worker.Session, exactly as a harness cell does, at the worker's 20 Hz.

PASS: after 3 s it stays within 0.75 m of the lane centre and its steering is not
pinned (|steer| >= 0.9 on under 5% of decisions).
CONTROL: the same drive with the policy asked only every 10th decision (2 Hz,
its last answer held in between) -- the rate the harness doc measured as an
unstable limit cycle -- must be visibly worse (twice the lateral excursion, or
pinned steering), or this check could not tell a broken loop from a good one.
A policy that never gets moving (< 2 m/s) is reported, not passed. `--start-speed`
spawns the ego already moving, as scenario runs do: SimLingo holds full brake
from a standstill on an empty road, and its upstream agent only creeps after
800 stuck frames (40 s; team_code/config_simlingo.py).
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

import sensor_worker as W
from tests._live_policy import policy_py, policy_request

PASS, FAIL = [], []
LANE_Y, START_X = 16.3, -130.0


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def drive(args, every=1, steps=150):
    import carla
    run_dir = Path(tempfile.mkdtemp())
    s = W.Session(carla, W.load_rig_module(W.DEFAULT_OSC2RUNNER), args.host, args.port, allow_load_town=True)
    route = np.stack([np.arange(START_X, START_X - 230, -0.5), np.full(460, LANE_Y)], axis=1)
    try:
        s.init({"op": "init", "town": "Town04", "dt": 0.1, "policy_hz": 20.0,
                "ego": {"x": START_X, "y": LANE_Y, "yaw_rad": math.pi, "speed": float(args.start_speed),
                        "length": 4.9, "width": 2.1},
                "actors": [], "route_world": route.tolist(),
                "policy_py": str(policy_py(args.policy)), "policy_request": policy_request(args.policy),
                "frames_dir": str(run_dir / "ego_sensor_frames")})
        if every > 1:
            act, held = s.policy.act, {"n": 0, "last": None}

            def slow(observation):
                if held["n"] % every == 0 or held["last"] is None:
                    held["last"] = act(observation)
                held["n"] += 1
                return held["last"]
            s.policy.act = slow
        for _ in range(steps):
            s.step({"op": "step", "actors": []})
    finally:
        s.close()
    log = [json.loads(l) for l in (run_dir / "ego_policy_log.jsonl").read_text().splitlines()]
    late = [l for l in log if l["t"] >= 3.0]
    lateral = np.array([abs(l["ego"]["y"] - LANE_Y) for l in late])
    steer = np.array([l["control"]["steer"] for l in late])
    speed = np.array([l["ego"]["speed_mps"] for l in late])
    return {"max_lateral_m": float(lateral.max()), "pinned_frac": float(np.mean(np.abs(steer) >= 0.9)),
            "steer_flips_per_s": float(np.sum(np.diff(np.sign(steer[np.abs(steer) > 0.05])) != 0) / (len(late) * 0.05)),
            "mean_speed": float(speed.mean()), "decisions": len(log)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--policy", required=True, choices=["tfv6", "simlingo"])
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--start-speed", type=float, default=0.0,
                    help="spawn the ego moving at this speed (m/s), as scenario runs do")
    args = ap.parse_args()
    if args.cwd:
        os.chdir(args.cwd)

    good = drive(args)
    print(f"20 Hz: {good}")
    if good["mean_speed"] < 2.0:
        check("the policy got moving on an empty road (>= 2 m/s)", False, f"{good['mean_speed']:.2f} m/s")
    else:
        check("20 Hz: stays within 0.75 m of the lane centre after 3 s", good["max_lateral_m"] <= 0.75,
              f"max {good['max_lateral_m']:.2f} m at {good['mean_speed']:.1f} m/s")
        check("20 Hz: steering not pinned", good["pinned_frac"] < 0.05, f"{good['pinned_frac']:.1%} of decisions")
    slow = drive(args, every=10)
    print(f" 2 Hz: {slow}")
    check("CONTROL: at 2 Hz the same loop is visibly worse",
          slow["max_lateral_m"] >= 2 * max(good["max_lateral_m"], 0.1) or slow["pinned_frac"] >= 0.1,
          f"max lateral {slow['max_lateral_m']:.2f} m vs {good['max_lateral_m']:.2f} m; "
          f"pinned {slow['pinned_frac']:.1%} vs {good['pinned_frac']:.1%}")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
