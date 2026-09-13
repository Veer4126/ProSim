"""Can a car spawned HERE reach THAT goal? Answer before spending a rollout.

Snaps a spawn point to its lane, says whether that lane is a turn connector,
and searches the real lane graph for a route to the goal's lane. Catches the
failure that costs a whole run: spawning on a turn-only lane, where no amount of
goal conditioning can help because the goal is not reachable at all.

    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 tools/check_route.py \
            --spawn -48.81 0 --goal -48.81 75"
"""

# Run from anywhere: put the repo root on the import path and work from it,
# since this script reads prosim_demo/... and demo_dataset/... relatively.
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import argparse
import math
import sys
from collections import deque

import numpy as np


from carla_dataset import register

register()

from torch.utils.data import DataLoader

from goal_control import turn_name
from prosim.config.default import get_config
from prosim.core.registry import registry
from prosim_ego import VecMapLaneGraph

HORIZON_S = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawn", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--example-idx", type=int, default=8)
    ap.add_argument("--max-lanes", type=int, default=12)
    args = ap.parse_args()

    cfg = get_config("prosim_demo/cfg/waymo_demo.yaml", cluster="local")
    cfg.defrost()
    cfg.PROMPT.CONDITION.TYPES = ["goal", "llm_text_OneText"]
    cfg.freeze()
    ds = registry.get_dataset(cfg.DATASET.TYPE)(cfg, "train")
    ds._data_index = [ds._data_index[args.example_idx]]
    ds._data_len = 1
    for batch in DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=ds.get_collate_fn(), num_workers=0):
        break

    lg = VecMapLaneGraph(batch.vector_maps[0])
    ids = [l.id for l in batch.vector_maps[0].lanes]

    def C(k):
        return np.asarray(lg.centerline(k), dtype=float)

    def h_start(c):
        return math.degrees(math.atan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0]))

    def h_end(c):
        return math.degrees(math.atan2(c[-1, 1] - c[-2, 1], c[-1, 0] - c[-2, 0]))

    def nearest(pt):
        return min(ids, key=lambda k: float(np.linalg.norm(C(k) - np.asarray(pt), axis=1).min()))

    sp, gl = np.asarray(args.spawn, float), np.asarray(args.goal, float)
    sk, gk = nearest(sp), nearest(gl)
    sc, gc = C(sk), C(gk)

    print("=" * 76)
    print("SPAWN")
    print("=" * 76)
    d = float(np.linalg.norm(sc - sp, axis=1).min())
    turn = (h_end(sc) - h_start(sc) + 180) % 360 - 180
    kind = turn_name(math.radians(turn))
    print(f"  ({sp[0]:.2f}, {sp[1]:.2f}) is {d:.2f} m from lane {sk}")
    print(f"  that lane runs ({sc[0,0]:7.2f},{sc[0,1]:7.2f}) hdg {h_start(sc):+6.1f}"
          f"  ->  ({sc[-1,0]:7.2f},{sc[-1,1]:7.2f}) hdg {h_end(sc):+6.1f}")
    print(f"  it turns {turn:+.1f} deg over {float(np.linalg.norm(np.diff(sc,axis=0),axis=1).sum()):.1f} m"
          f"  = a {kind.upper()} lane")
    if kind != "straight":
        print(f"  *** WARNING: this is a TURN CONNECTOR. A car spawned here is committed")
        print(f"      to that turn -- goal conditioning cannot re-route it. ***")

    # other lanes starting near the same point: the ambiguity that causes this
    near = [k for k in ids if k != sk and float(np.linalg.norm(C(k)[0] - sc[0])) < 3.0]
    if near:
        print(f"\n  OTHER lanes starting within 3 m of this one's start -- the snap is")
        print(f"  effectively a coin flip near there:")
        for k in near:
            c = C(k)
            t = (h_end(c) - h_start(c) + 180) % 360 - 180
            print(f"    {k:10s} -> ({c[-1,0]:7.2f},{c[-1,1]:7.2f}) turns {t:+6.1f} deg "
                  f"= {turn_name(math.radians(t))}")

    print()
    print("=" * 76)
    print("GOAL")
    print("=" * 76)
    print(f"  ({gl[0]:.2f}, {gl[1]:.2f}) is "
          f"{float(np.linalg.norm(gc - gl, axis=1).min()):.2f} m from lane {gk} "
          f"(hdg {h_end(gc):+.1f})")

    print()
    print("=" * 76)
    print("ROUTE")
    print("=" * 76)
    q, vis = deque([(sk, [sk])]), {sk}
    path = None
    while q:
        k, p = q.popleft()
        if k == gk:
            path = p
            break
        if len(p) > args.max_lanes:
            continue
        for s in lg.successors(k):
            if s not in vis:
                vis.add(s)
                q.append((s, p + [s]))

    if path:
        L = sum(float(np.linalg.norm(np.diff(C(x), axis=0), axis=1).sum()) for x in path)
        print(f"  REACHABLE: {' -> '.join(path)}")
        print(f"  lane-chain length {L:.1f} m -> needs {L / HORIZON_S:.2f} m/s to "
              f"cover it in the {HORIZON_S:.0f} s horizon")
        if L / HORIZON_S > 9.0:
            print(f"  NOTE: that is fast for a town. Bring the goal closer, or expect")
            print(f"  the car to be short of it when the clip ends.")
        sys.exit(0)
    else:
        best = min(vis, key=lambda k: float(np.linalg.norm(C(k) - gl, axis=1).min()))
        print(f"  NOT REACHABLE within {args.max_lanes} lanes "
              f"({len(vis)} lanes explored).")
        print(f"  Closest lane the car CAN reach is {best}, "
              f"{float(np.linalg.norm(C(best) - gl, axis=1).min()):.1f} m from the goal.")
        print(f"  Spawn somewhere whose lane leads to {gk}, or move the goal.")
        sys.exit(1)


if __name__ == "__main__":
    main()
