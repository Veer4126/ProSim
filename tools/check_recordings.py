"""Is each scenario's recording usable with its declared goals?

    python3 tools/check_recordings.py [family ...]

Pure stdlib + yaml; no model, no CARLA. Reads every family's prosim entry
from the harness's implementations.yaml and its recording
(history_<town>__<scene>.csv), and checks what a rollout would otherwise
discover 90 s in, or never:

  - one recorded actor per declared spawn, each within 5 m of its spawn
    (the recorder snaps to the lane centre), and the ego is the lowest id;
  - the START STATE -- the recording at scene_ts 10, i.e. 1 s of autopilot
    after spawn, which is what ProSim actually starts from -- still has every
    goal >= 5 m ahead (set_goal_condition's guard, which RAISES otherwise);
  - the mean speed each goal implies over the 8 s horizon (distance / 7.9 s),
    because the goal is learned as "be here at t + 8 s".
"""

# Run from anywhere: put the repo root on the import path and work from it,
# since this script reads prosim_demo/... and demo_dataset/... relatively.
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)
import csv, math, sys
from pathlib import Path
import yaml

HARNESS = Path("/home/veerk41/scratch/scenario_orchestration")
DATA = Path("/scratch/veerk41")
START_TS, HORIZON_S, MIN_FWD, SPAWN_TOL = 10, 7.9, 5.0, 5.0
TOWN = {"red_light": "town10hd", "left_turn": "town10hd", "right_turn": "town10hd",
        "cut_in": "town04", "lane_change": "town04", "overtake": "town04"}

def main(families):
    bad = 0
    for fam in families:
        p = yaml.safe_load((HARNESS / "scenarios" / fam / "implementations.yaml")
                           .read_text())["implementations"]["prosim"]["parameters"]
        scene = p.get("scene") or fam
        rec = DATA / f"history_{TOWN[fam]}__{scene}.csv"
        print(f"\n### {fam}  <-  {rec.name}")
        if not rec.is_file():
            print("   MISSING recording"); bad += 1; continue
        rows = {}
        for r in csv.DictReader(open(rec)):
            rows.setdefault(int(r["id"]), {})[int(r["frame"])] = r
        base = min(min(f) for f in rows.values())
        spawns = {int(s["agent"]): s["xy"] for s in p["spawns"]}
        ok = len(rows) == len(spawns)
        print(f"   actors {sorted(rows)}  declared spawns {len(spawns)}  "
              f"{'ok' if ok else 'COUNT MISMATCH'}")
        bad += not ok
        ids = {}
        for k, (sx, sy) in sorted(spawns.items()):
            d, aid = min((math.hypot(float(f[base]["x"]) - sx,
                                     float(f[base]["y"]) - sy), a)
                         for a, f in rows.items())
            ids[k] = aid
            tag = "ok" if d <= SPAWN_TOL else "TOO FAR"
            bad += d > SPAWN_TOL
            print(f"   agent {k} -> actor {aid}: recorded {d:4.2f} m from declared spawn  {tag}")
        ego_ok = ids.get(0) == min(rows)
        bad += not ego_ok
        print(f"   ego is lowest id: {'ok' if ego_ok else 'NO -- wrong car would be ego'}")
        pairs = [(g["agent"], g["xy"], "goal") for g in p.get("goals", [])]
        if p.get("ego_goal_xy"):
            pairs.append((0, p["ego_goal_xy"], "route hint"))
        for k, (gx, gy), kind in pairs:
            r = rows[ids[k]][base + START_TS]
            x, y, h = float(r["x"]), float(r["y"]), float(r["yaw"])
            v = math.hypot(float(r["vx"]), float(r["vy"]))
            dx, dy = gx - x, gy - y
            fwd = dx * math.cos(h) + dy * math.sin(h)
            dist = math.hypot(dx, dy)
            flag = "ok" if fwd >= MIN_FWD else "BEHIND -- guard will refuse"
            bad += fwd < MIN_FWD
            implied = (f"implies {dist / HORIZON_S:5.2f} m/s" if kind == "goal"
                       else "(IDM sets speed)")
            print(f"   agent {k} {kind:10s} start ({x:7.2f},{y:7.2f}) v={v:4.1f}  "
                  f"fwd {fwd:+7.1f} m  {implied}  {flag}")
    print(f"\n{'ALL USABLE' if not bad else f'{bad} PROBLEM(S)'}")
    return 1 if bad else 0

if __name__ == "__main__":
    fams = sys.argv[1:] or list(TOWN)
    sys.exit(main(fams))
