"""B1: a vision policy's own sensor rig, attached and read through the sensor worker.

    CARLA_ROOT=<dist> HF_HOME=/scratch/veerk41/hf_cache <policy venv>/bin/python \
        tests/live_check_rigs.py --host <node> --policy tfv6|simlingo [--cwd DIR]

Loads the policy exactly as a harness cell does (configs/policy/<name>.yaml ->
sensor_worker.Session), drives 20 decisions on Town10HD with one car ahead, and
reads every capture the worker hands the policy. The same measurements as
osc2runner's rig validation (reports/sensor_rig_validation.md, section 2), whose
numbers are printed beside ours:

  every declared sensor attached, none failed, no dropped captures;
  each camera varied, not blank: std > 20 and > 50 distinct values;
  LiDAR > 5000 points; each radar > 10 detections.

CONTROL: the same thresholds applied to an all-black frame and an empty point
cloud must fail, or they could not catch a blank rig.
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
import os

import numpy as np

import sensor_worker as W
from tests._live_policy import policy_py, policy_request

PASS, FAIL = [], []
OSC2RUNNER_TABLE = {
    "simlingo": "rgb_front 1024x512: std 55.0, 183 unique values",
    "tfv6": "3 cameras std 44.0 / 50.6 / 64.4; LiDAR 15 684 points; radars 130-150 detections each",
}


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def summarize(arr):
    a = np.asarray(arr)
    if a.ndim == 3:
        return {"kind": "camera", "shape": list(a.shape), "mean": float(a.mean()), "std": float(a.std()),
                "unique": int(len(np.unique(a[::4, ::4, 0])))}
    return {"kind": "points", "n": int(len(a))}


def camera_ok(s):
    return s["std"] > 20.0 and s["unique"] > 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--policy", required=True, choices=["tfv6", "simlingo"])
    ap.add_argument("--cwd", default=None, help="working directory the policy needs (SimLingo: its checkpoint dir)")
    args = ap.parse_args()

    black = summarize(np.zeros((64, 64, 3), np.uint8))
    check("CONTROL: an all-black frame fails the camera thresholds", not camera_ok(black), str(black))
    check("CONTROL: an empty point cloud fails the LiDAR threshold", not summarize(np.zeros((0, 4))).get("n", 0) > 5000)

    import carla
    if args.cwd:
        os.chdir(args.cwd)
    s = W.Session(carla, W.load_rig_module(W.DEFAULT_OSC2RUNNER), args.host, args.port, allow_load_town=True)
    x0, y0 = -84.2, 24.45
    route = np.stack([np.arange(x0, x0 + 120, 0.5), np.full(240, y0)], axis=1)
    lead = {"x": x0 + 20.0, "y": y0, "yaw_rad": 0.0, "speed": 0.0, "length": 4.7, "width": 1.8}
    captures = []
    try:
        reply = s.init({"op": "init", "town": "Town10HD", "dt": 0.1, "policy_hz": 20.0,
                        "ego": {"x": x0, "y": y0, "yaw_rad": 0.0, "speed": 0.0, "length": 4.9, "width": 2.1},
                        "actors": [lead], "route_world": route.tolist(),
                        "policy_py": str(policy_py(args.policy)), "policy_request": policy_request(args.policy),
                        "frames_dir": None})
        declared = [d["name"] if isinstance(d, dict) else getattr(d, "name", str(d))
                    for d in (s.policy.sensors() if callable(getattr(s.policy, "sensors", None)) else [])]
        original = s.rig.capture

        def recording(frame=None):
            out = original(frame)
            captures.append({k: summarize(v) for k, v in out.items()})
            return out
        s.rig.capture = recording
        for _ in range(10):
            s.step({"op": "step", "actors": [lead]})
        dropped = dict(s.rig.dropped)
    finally:
        s.close()

    print(f"\n{args.policy}: declared {declared}; attached {reply.get('sensors')}; failed {reply.get('failed')}")
    print(f"osc2runner measured: {OSC2RUNNER_TABLE[args.policy]}")
    check("every declared sensor attached, none failed",
          sorted(reply.get("sensors") or []) == sorted(declared) and not reply.get("failed"),
          f"{len(reply.get('sensors') or [])} of {len(declared)}")
    check("no capture dropped", not any(dropped.values()), str(dropped))
    check("the policy was handed every sensor on every decision",
          len(captures) == 20 and all(set(c) == set(declared) for c in captures), f"{len(captures)} captures")
    for name in declared:
        stats = [c[name] for c in captures if name in c]
        if not stats:
            check(f"{name} delivered", False)
            continue
        if stats[0]["kind"] == "camera":
            std, uniq = np.median([x["std"] for x in stats]), np.median([x["unique"] for x in stats])
            check(f"{name} {stats[0]['shape']}: varied, not blank", std > 20 and uniq > 50,
                  f"median std {std:.1f}, {uniq:.0f} unique values, mean {np.median([x['mean'] for x in stats]):.1f}")
        else:
            n = np.median([x["n"] for x in stats])
            limit = 5000 if "lidar" in name.lower() else 10
            check(f"{name}: {'points' if limit > 10 else 'detections'} per capture", n > limit, f"median {n:.0f}")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
