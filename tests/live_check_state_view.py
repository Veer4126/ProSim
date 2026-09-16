"""Check 1: what PlanT 2.0 and IDM are told about the world, against CARLA's truth.

    CARLA_ROOT=<dist> /scratch/veerk41/venvs/simlingo/bin/python \
        tests/live_check_state_view.py --host <node> --policy plant2|idm

The policy is loaded and driven exactly as a harness cell does
(configs/policy/<name>.yaml -> sensor_worker.Session, osc2runner's
StateObservationBuilder and, for PlanT 2.0, its own BEV renderer). Every
observation it is handed is captured next to CARLA's own state at that tick.

On Town10HD (the red_light approach) three cars are placed: A 15 m ahead in
the ego's lane, B 25 m ahead one lane to the left, C 80 m ahead.

  objects: A and B, each at CARLA's position in the ego frame (+x forward,
  +y right), with CARLA's half-extents and speed; the ego is not among them;
  CONTROL: C, beyond osc2runner's 75 m range, is not either.
  route: 20 points starting ~2.5 m ahead; a speed limit.
  idm: the lead it picks from that observation is A.
  plant2 BEV: the raw raster's road classes (1 road, 3/4 lane markings) agree
  with CARLA's map -- a point is road when the map has a driving lane there --
  over a grid round the ego, for exactly one orientation of the raster.
  CONTROL: its left-right mirror agrees clearly less, so the check can see a
  flipped map.
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
import sys

import numpy as np

import sensor_worker as W
from tests._live_policy import HARNESS, policy_py, policy_request

PASS, FAIL = [], []
X0, Y0 = -84.2, 24.45
CARS = {"A": (15.0, 0.0), "B": (25.0, -3.5), "C": (80.0, 0.0)}   # (forward, right) from the ego


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def ego_frame(ego_tf, loc):
    yaw = math.radians(ego_tf.rotation.yaw)
    dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--policy", required=True, choices=["plant2", "idm"])
    args = ap.parse_args()
    import carla

    s = W.Session(carla, W.load_rig_module(W.DEFAULT_OSC2RUNNER), args.host, args.port, allow_load_town=True)
    route = np.stack([np.arange(X0, X0 + 140, 0.5), np.full(280, Y0)], axis=1)
    actors = [{"x": X0 + f, "y": Y0 + r, "yaw_rad": 0.0, "speed": 0.0, "length": 4.7, "width": 1.8}
              for f, r in CARS.values()]
    seen = []
    try:
        s.init({"op": "init", "town": "Town10HD", "dt": 0.1, "policy_hz": 20.0,
                "ego": {"x": X0, "y": Y0, "yaw_rad": 0.0, "speed": 0.0, "length": 4.9, "width": 2.1},
                "actors": actors, "route_world": route.tolist(),
                "policy_py": str(policy_py(args.policy)), "policy_request": policy_request(args.policy),
                "frames_dir": None})
        act = s.policy.act

        def recording(observation):
            ego_tf = s.ego.get_transform()
            truth = []
            for actor in s.actors:
                tf, v, e = actor.get_transform(), actor.get_velocity(), actor.bounding_box.extent
                truth.append({"pos": ego_frame(ego_tf, tf.location), "speed": math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
                              "extent": (e.x, e.y, e.z)})
            waypoints = {}
            if args.policy == "plant2" and "bev" in observation:
                cmap = s.world.get_map()
                yaw = math.radians(ego_tf.rotation.yaw)
                for f in range(-20, 31, 2):
                    for r in range(-20, 21, 2):
                        x = ego_tf.location.x + f * math.cos(yaw) - r * math.sin(yaw)
                        y = ego_tf.location.y + f * math.sin(yaw) + r * math.cos(yaw)
                        wp = cmap.get_waypoint(carla.Location(x, y, ego_tf.location.z), project_to_road=False,
                                               lane_type=carla.LaneType.Driving)
                        waypoints[(f, r)] = wp is not None
            seen.append({"obs": {k: observation[k] for k in observation if k != "sensor"},
                         "truth": truth, "road": waypoints})
            return act(observation)
        s.policy.act = recording
        for _ in range(3):
            s.step({"op": "step", "actors": actors})
        bev_note = {"config": dict(s.bev._obs_config)} if s.bev is not None else s.bev_note
    finally:
        s.close()

    last = seen[-1]
    obs, truth = last["obs"], last["truth"]
    cars = [o for o in obs.get("objects", []) if o.get("type") == "car"]
    print(f"{args.policy}: {len(seen)} observations; objects {[(round(o['position'][0], 2), round(o['position'][1], 2)) for o in cars]}")
    in_range = [t for t, name in zip(truth, CARS) if math.hypot(*t["pos"]) <= 75.0]
    check("exactly the cars within 75 m are objects (A and B; the ego is not one)",
          len(cars) == len(in_range) == 2, f"{len(cars)} objects, {len(in_range)} cars within range")
    worst = worst_ext = worst_v = 0.0
    for t in in_range:
        o = min(cars, key=lambda c: math.hypot(c["position"][0] - t["pos"][0], c["position"][1] - t["pos"][1]))
        worst = max(worst, math.hypot(o["position"][0] - t["pos"][0], o["position"][1] - t["pos"][1]))
        worst_ext = max(worst_ext, max(abs(a - b) for a, b in zip(o["extent"], t["extent"])))
        worst_v = max(worst_v, abs(o["speed_mps"] - t["speed"]))
    check("each object sits at CARLA's position in the ego frame", worst < 0.05, f"worst {worst:.3f} m")
    check("with CARLA's half-extents and speed", worst_ext < 1e-3 and worst_v < 0.05,
          f"extent error {worst_ext:.4f} m, speed error {worst_v:.3f} m/s")
    far = truth[list(CARS).index("C")]
    check("CONTROL: the car 80 m ahead is left out (osc2runner's 75 m range)",
          all(math.hypot(c["position"][0] - far["pos"][0], c["position"][1] - far["pos"][1]) > 5 for c in cars),
          f"C at {far['pos'][0]:.1f} m")
    r = obs.get("route") or []
    check("route: 20 points from ~2.5 m ahead on the ego's lane", len(r) == 20 and abs(r[0][0] - 2.5) < 0.5
          and abs(r[0][1]) < 0.5, str(r[:2]))
    check("a speed limit is given", (obs.get("speed_limit_kph") or 0) > 0, str(obs.get("speed_limit_kph")))

    if args.policy == "idm":
        sys.path.insert(0, str(HARNESS / "third_party/idm"))
        from idm import laws as L, scene as S
        lead, _ = L.lane_neighbours(0.0, S.neighbours(obs, L.EGO_LENGTH))
        a = truth[0]["pos"]
        check("idm picks car A (15 m ahead, same lane) as its lead", lead is not None and abs(lead.along - a[0]) < 0.5,
              f"lead along {None if lead is None else round(lead.along, 2)} m, A at {a[0]:.2f} m")

    if args.policy == "plant2":
        print(f"BEV: {bev_note}")
        raster = np.asarray((obs.get("bev") or {}).get("semantic_classes"))
        check("PlanT 2.0 was handed a BEV raster", raster.ndim == 2, str(raster.shape))
        if raster.ndim == 2:
            W_px = raster.shape[0]
            cfg = (bev_note or {}).get("config", {}) if isinstance(bev_note, dict) else {}
            span_m = float(cfg.get("width_in_pixels", 256)) / float(cfg.get("pixels_per_meter", 2.0))
            ppm, centre = W_px / span_m, W_px / 2.0
            variants = {"up=forward, right=right": lambda f, r: (centre - f * ppm, centre + r * ppm),
                        "up=forward, right=LEFT (mirror)": lambda f, r: (centre - f * ppm, centre - r * ppm),
                        "right=forward, down=right": lambda f, r: (centre + r * ppm, centre + f * ppm),
                        "right=forward, down=LEFT (mirror of that)": lambda f, r: (centre - r * ppm, centre + f * ppm),
                        "left=forward, up=right": lambda f, r: (centre - r * ppm, centre - f * ppm),
                        "down=forward, left=right": lambda f, r: (centre + f * ppm, centre - r * ppm)}
            score = {}
            for name, fn in variants.items():
                agree = n = 0
                for (f, rr), is_road in last["road"].items():
                    row, col = fn(f, rr)
                    if 0 <= int(row) < W_px and 0 <= int(col) < W_px:
                        n += 1
                        agree += (raster[int(row), int(col)] in (1, 3, 4)) == is_road
                score[name] = agree / n if n else 0.0
            for name, v in sorted(score.items(), key=lambda kv: -kv[1]):
                print(f"    {v:.1%}  {name}")
            best = max(score, key=score.get)
            ranked = sorted(score.values(), reverse=True)
            check("the raster agrees with CARLA's map in one orientation (>= 85%)",
                  ranked[0] >= 0.85 and ranked[0] - ranked[1] >= 0.05, f"best: {best} {ranked[0]:.1%}")
            mirror = {"up=forward, right=right": "up=forward, right=LEFT (mirror)",
                      "right=forward, down=right": "right=forward, down=LEFT (mirror of that)"}.get(best)
            check("CONTROL: the left-right mirror of the best orientation agrees clearly less",
                  mirror is not None and score[mirror] <= score[best] - 0.2,
                  f"{best} {score[best]:.1%} vs its mirror {score.get(mirror, float('nan')):.1%}")
            # What PlanT 2.0 is fed: the adapter's np.rot90 (policy.py _encode_bev), as PlanT_agent.py does.
            fwd = np.rot90(raster)[int(centre) - 20, int(centre)] , np.rot90(raster)[int(centre) + 20, int(centre)]
            print(f"    after np.rot90: class 10 m above centre {fwd[0]}, 10 m below {fwd[1]} (ego lane is road both ways)")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
