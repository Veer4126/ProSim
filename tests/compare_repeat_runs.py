"""Check 3: does the same cell, run twice, do the same thing?

    /scratch/veerk41/venvs/tfv6/bin/python tests/compare_repeat_runs.py \
        --family red_light --policy plant2 [--policy tfv6 ...]

Reads each run through the harness's own loader (metrics/ingest/canonical.py
read_run_dir), so the traces are the ones the metrics are scored from. Run 1 is
results/raw_prosim_eval (the cell the chain just reran); run 2 is
results/repeatability/r2, written by repeat_cells.sh, which never touches the
first.

Reported per policy: how far apart the two ego paths are at the same sim time,
the same for each agent, and whether the scored outcome (status, success) agrees.
A CARLA run is not bit-exact -- physics and the camera pipeline both vary -- so
this measures how much of a run's result is the policy and how much is noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

H = Path("/scratch/veerk41/scenario_orchestration")
sys.path.insert(0, str(H))
from metrics.ingest.canonical import read_run_dir            # noqa: E402

RAW = H / "results/raw_prosim_eval"
REPEAT = H / "results/repeatability"


def positions(rollout, actor_id, times):
    a = rollout.trace.actors.get(actor_id)
    if a is None:
        return None
    t = np.asarray(rollout.trace.times, float)
    return np.stack([np.interp(times, t, a.position[:, k]) for k in (0, 1)], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--run", default="r2")
    args = ap.parse_args()

    for policy in args.policy:
        cell = f"{args.family}__prosim_carla__{policy}__s000"
        a, b = read_run_dir(str(RAW / cell)), read_run_dir(str(REPEAT / args.run / cell))
        print(f"\n{cell}")
        if not (a.evaluable and b.evaluable):
            print(f"  not comparable: run1 {a.reason or 'ok'} / run2 {b.reason or 'ok'}")
            continue
        t_end = min(a.trace.times[-1], b.trace.times[-1])
        times = np.arange(0.0, t_end + 1e-9, 0.1)
        ea, eb = positions(a, a.trace.ego_id, times), positions(b, b.trace.ego_id, times)
        d = np.hypot(*(ea - eb).T)
        print(f"  status   {a.status} / {b.status}   duration {a.trace.times[-1]:.1f} / {b.trace.times[-1]:.1f} s")
        print(f"  ego apart: median {np.median(d):.3f} m, at 2 s {np.interp(2.0, times, d):.3f} m, "
              f"final {d[-1]:.3f} m, worst {d.max():.3f} m")
        for actor_id in sorted(set(x.actor_id for x in a.trace.others())):
            pa, pb = positions(a, actor_id, times), positions(b, actor_id, times)
            if pa is None or pb is None:
                print(f"  agent {actor_id}: only in one run")
                continue
            dd = np.hypot(*(pa - pb).T)
            print(f"  agent {actor_id}: median {np.median(dd):.3f} m, worst {dd.max():.3f} m")
        ma, mb = a.legacy_metrics or {}, b.legacy_metrics or {}
        keys = sorted(set(ma) | set(mb))
        same = [k for k in keys if ma.get(k) == mb.get(k)]
        differing = ", ".join(f"{k}={ma.get(k)!r}/{mb.get(k)!r}" for k in keys if k not in same)
        print(f"  result.json metrics identical on {len(same)}/{len(keys)}"
              + (f"; differing: {differing}" if differing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
