"""Audit finished prosim_carla cells from what they recorded -- no CARLA, no GPU.

    /scratch/veerk41/venvs/tfv6/bin/python tools/audit_carla_cells.py [RAW_DIR]

Every check reads a cell's own files: ego_policy_log.jsonl (one line per
policy decision, written by sensor_worker.py), states.jsonl / scene.json (the
trace ProSim wrote), and for A6 the orchestration method's reference cells.

  A1  controls pass through: the throttle/brake/steer applied to the car equal
      what the policy returned, once the spawn phase is over (idm/idm_mobil
      return an acceleration, so only their steer can be compared)
  A2  decision rate: 2 decisions per 0.1 s ProSim step, 160 in 8 s, evenly spaced
  A3  cars are where ProSim put them: the nearest car the worker saw, in the
      ego frame, equals ProSim's pose of that car at the same instant
  A4  the ego pose goes back unchanged: the worker's ego pose at the end of each
      ProSim step equals the ego's row in the trace
  A5  tfv6's navigation command agrees with the way its route turns
      (CONTROL: the mirrored mapping must disagree)
  A6  tfv6 / SimLingo against the orchestration method's cells of the same
      family: time to start, speeds, heading smoothness, net turn, commands
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "/scratch/veerk41/scenario_orchestration/results/raw_prosim_eval"
REF = "/scratch/veerk41/scenario_orchestration/results/raw_exp008_av_seeds"
SPAWN_S = 2.5          # SpawnGear: at most 2.0 s revving + 0.25 s gear hold (osc2runner actuation.py)
TICK = 0.05            # worker tick at 20 Hz
STEP = 0.1             # ProSim step
PEDAL = ("tfv6", "simlingo", "plant2")


def aligned_t(decision):
    """Trace time of the world after `decision`: measured on every cell, the ego
    the worker returns equals the trace at (d + 1) * TICK - STEP, to 0.01 m."""
    return (decision + 1) * TICK - STEP


def load_cell(path):
    scene = json.load(open(os.path.join(path, "scene.json")))
    ego = next(a["id"] for a in scene["actors"] if a.get("is_ego"))
    rows = [json.loads(l) for l in open(os.path.join(path, "states.jsonl"))]
    log_path = os.path.join(path, "ego_policy_log.jsonl")
    log = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
    name = os.path.basename(path)
    family, _algo, policy = name.split("__")[:3]
    return {"name": name, "family": family, "policy": policy, "ego": ego, "rows": rows, "log": log}


def pose_at(rows, actor, t):
    """(x, y) of `actor` at time t, linear between 0.1 s trace rows."""
    k = t / STEP
    i = int(math.floor(k + 1e-9))
    i = max(0, min(i, len(rows) - 1))
    j = min(i + 1, len(rows) - 1)
    f = min(max(k - i, 0.0), 1.0)
    a, b = rows[i]["a"].get(actor), rows[j]["a"].get(actor)
    if a is None or b is None:
        return None
    return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f


def a1(c):
    rows = [l for l in c["log"] if l["t"] >= SPAWN_S]
    if not rows:
        return None, "no decisions after the spawn phase"
    diffs = []
    for l in rows:
        act, ctl = l.get("action") or {}, l["control"]
        if c["policy"] in PEDAL:
            p = act.get("control") or {}
            want = (min(1, max(0, p.get("throttle", 0))), min(1, max(0, p.get("brake", 0))),
                    min(1, max(-1, p.get("steer", 0))))
            got = (ctl["throttle"], ctl["brake"], ctl["steer"])
        else:
            want, got = (min(1, max(-1, act.get("steer", 0))),), (ctl["steer"],)
        diffs.append(max(abs(w - g) for w, g in zip(want, got)))
    worst = max(diffs)
    what = "throttle/brake/steer" if c["policy"] in PEDAL else "steer (acceleration goes through the tracker)"
    return worst <= 2e-3, f"{what}: max |applied - policy| {worst:.4f} over {len(rows)} decisions after {SPAWN_S} s"


def a2(c):
    log = c["log"]
    t = np.array([l["t"] for l in log])
    even = len(t) > 1 and np.allclose(np.diff(t), TICK, atol=2e-3)
    seq = [l["decision"] for l in log] == list(range(len(log)))
    expected = 2 * len(c["rows"])
    ok = len(log) == expected and even and seq
    return ok, f"{len(log)} decisions for {len(c['rows'])} ProSim steps (want {expected}), evenly spaced {even}, in order {seq}"


def a3(c):
    errs = []
    others = [a for a in c["rows"][0]["a"] if a != c["ego"]]
    for l in c["log"]:
        na = l.get("nearest_agent")
        if not na or not others:
            continue
        tt = aligned_t(l["decision"])
        ex, ey, yaw = l["ego"]["x"], l["ego"]["y"], math.radians(l["ego"]["yaw_deg"])
        best = None
        for a in others:
            p = pose_at(c["rows"], a, tt)
            if p is None:
                continue
            dx, dy = p[0] - ex, p[1] - ey
            d = math.hypot(dx, dy)
            if best is None or d < best[0]:
                best = (d, dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))
        if best is None:
            continue
        errs.append(math.hypot(best[1] - na["forward_m"], best[2] - na["right_m"]))
    if not errs:
        return None, "no other car"
    e = np.array(errs)
    return float(np.percentile(e, 95)) <= 0.3, f"nearest car vs ProSim pose: median {np.median(e):.3f} m, p95 {np.percentile(e, 95):.3f} m, max {e.max():.3f} m ({len(e)} decisions)"


def a4(c):
    dpos, dyaw = [], []
    for l in c["log"]:
        if l["decision"] % 2 != 1:
            continue
        s = (l["decision"] - 1) // 2          # the step this decision ends; trace row s (aligned_t)
        if s >= len(c["rows"]):
            continue
        r = c["rows"][s]["a"][c["ego"]]
        dpos.append(math.hypot(r[0] - l["ego"]["x"], r[1] - l["ego"]["y"]))
        dyaw.append(abs((r[2] - l["ego"]["yaw_deg"] + 180) % 360 - 180))
    if not dpos:
        return None, "no step ends"
    ok = max(dpos) <= 0.05 and max(dyaw) <= 0.5
    return ok, f"worker ego vs trace ego at {len(dpos)} step ends: max {max(dpos):.3f} m, {max(dyaw):.2f} deg"


def a5(c):
    if c["policy"] != "tfv6":
        return None, "tfv6 only"
    agree = mirror = n = 0
    for l in c["log"]:
        last, cmd = (l.get("route") or {}).get("last"), l.get("command")
        if not last or cmd not in ("LEFT", "RIGHT", "LANEFOLLOW", "STRAIGHT"):
            continue
        if abs(last[1]) < 5.0:
            continue
        turn = "RIGHT" if last[1] > 0 else "LEFT"      # ego frame +y is the driver's right
        n += 1
        agree += cmd == turn
        mirror += cmd == ("LEFT" if turn == "RIGHT" else "RIGHT")
    seq = []
    for l in c["log"]:
        if l.get("command") and (not seq or seq[-1] != l["command"]):
            seq.append(l["command"])
    if n == 0:
        return None, f"route never turns more than 5 m; commands {seq}"
    junction = c["family"] in ("left_turn", "right_turn")
    # Never mirrored; at a junction the turn command must also appear (it may lag, hysteresis).
    # A lane change bends the route without being a turn, so LANEFOLLOW is right there.
    ok = mirror == 0 and (agree > 0 or not junction)
    return ok, f"route turning at {n} decisions: command matches {agree / n:.0%}, CONTROL mirrored matches {mirror / n:.0%}; sequence {seq}"


def features(rows, ego, dt=STEP):
    v = np.array([math.hypot(r["a"][ego][3], r["a"][ego][4]) for r in rows if ego in r["a"]])
    yaw = np.unwrap(np.radians([r["a"][ego][2] for r in rows if ego in r["a"]]))
    rate = np.diff(yaw) / dt
    moving = v[1:] > 2.0
    flips = int(np.sum(np.diff(np.sign(rate[moving])) != 0)) if moving.sum() > 2 else 0
    start = next((i * dt for i, s in enumerate(v) if s > 1.0), None)
    dur = moving.sum() * dt
    return {"start_s": start, "mean_v": float(v[v > 1.0].mean()) if (v > 1.0).any() else 0.0,
            "peak_v": float(v.max()), "turn_deg": float(np.degrees(yaw[-1] - yaw[0])),
            "yaw_flips_per_s": flips / dur if dur > 0 else 0.0}


def a6(c):
    if c["policy"] not in ("tfv6", "simlingo"):
        return None, "tfv6 / SimLingo only"
    refs = sorted(glob.glob(f"{REF}/{c['family']}__orchestration__{c['policy']}__*"))
    if not refs:
        return None, "no orchestration reference for this family"
    ours = features(c["rows"], c["ego"])
    rs = json.load(open(os.path.join(refs[0], "scene.json")))
    rego = next(a["id"] for a in rs["actors"] if a.get("is_ego"))
    rrows = [json.loads(l) for l in open(os.path.join(refs[0], "states.jsonl"))][:len(c["rows"])]
    ref = features(rrows, rego)
    fmt = lambda f: ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in f.items())
    same_turn = (abs(ours["turn_deg"]) < 30) == (abs(ref["turn_deg"]) < 30) and \
        (abs(ref["turn_deg"]) < 30 or np.sign(ours["turn_deg"]) == np.sign(ref["turn_deg"]))
    smooth = ours["yaw_flips_per_s"] <= max(2.0, 2 * ref["yaw_flips_per_s"])
    return same_turn and smooth, f"ours [{fmt(ours)}] | orchestration [{fmt(ref)}] (first {len(rrows)} ticks of {os.path.basename(refs[0])})"


def main():
    cells = [load_cell(p) for p in sorted(glob.glob(f"{RAW}/*__prosim_carla__*_s000"))]
    checks = [("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4), ("A5", a5), ("A6", a6)]
    tally = {k: [0, 0] for k, _ in checks}
    for c in cells:
        print(f"\n== {c['name']}")
        if not c["log"]:
            print("   no ego_policy_log.jsonl (run before the per-decision log existed)")
            continue
        for key, fn in checks:
            ok, detail = fn(c)
            if ok is None:
                continue
            tally[key][0 if ok else 1] += 1
            print(f"   {key} [{'ok  ' if ok else 'FAIL'}] {detail}")
    print("\nsummary (pass / fail):", {k: f"{p}/{f}" for k, (p, f) in tally.items()})
    return 1 if any(f for _, f in tally.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
