"""Drop the first N frames of a recording, so the scene STARTS already rolling.

    python3 tools/trim_recording.py carla_data/history_town04__lane_change.csv --frames 10

Why this exists. A recording made by `record_actor_history.py` starts from a
standstill, so the first seconds of every rollout are spent accelerating. The
rollout can be started later instead (`example_idx`), but that is a parameter
carried in the scenario yaml: two runs that differ only there stage different
scenes while looking identical in the results, and its value has to be chosen
per family.

Trimming moves that choice into the recording itself. `carla_dataset.py:191`
re-bases `scene_ts` off the lowest frame, so dropping N frames shifts every
sample start by N: the trimmed recording at `example_idx: 0` is the untrimmed
one at `example_idx: N`, tick for tick.

It also REWRITES THE DECLARED SPAWNS, which is the part that is easy to forget:
`run.py:assign_actor_ids` matches each declared spawn to the nearest recorded
first position within 5 m, so a trimmed recording whose yaml still names the
original spawns is refused (or, worse, matched to the wrong actor).

Prints the new first positions to paste into implementations.yaml, and refuses
a trim that would leave the scene shorter than ProSim needs (HISTORY_SEC +
FUTURE_SEC = 9.0 s).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

MIN_SECONDS = 9.0                     # HISTORY_SEC 1.0 + FUTURE_SEC 8.0


def read(path: Path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--frames", type=int, required=True,
                    help="how many leading frames to drop (0.1 s each)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: in place, after a .bak copy)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fields, rows = read(args.csv_path)
    frames = sorted({int(r["frame"]) for r in rows})
    dt = 0.1
    if len(frames) < 2:
        print(f"{args.csv_path}: only {len(frames)} frames; nothing to trim")
        return 1
    cut = frames[0] + args.frames
    kept = [r for r in rows if int(r["frame"]) >= cut]
    kept_frames = sorted({int(r["frame"]) for r in kept})
    seconds = len(kept_frames) * dt
    print(f"{args.csv_path.name}: {len(frames)} frames ({len(frames) * dt:.1f} s) "
          f"-> {len(kept_frames)} frames ({seconds:.1f} s) after dropping {args.frames}")
    if seconds < MIN_SECONDS:
        print(f"REFUSED: ProSim needs {MIN_SECONDS:.1f} s (1 s history + 8 s horizon); "
              f"trim fewer frames")
        return 1

    first = {}
    for r in kept:
        aid, fr = int(r["id"]), int(r["frame"])
        if aid not in first or fr < first[aid][0]:
            first[aid] = (fr, float(r["x"]), float(r["y"]),
                          (float(r["vx"]) ** 2 + float(r["vy"]) ** 2) ** 0.5)
    print("  new first positions -- PASTE THESE AS `spawns` IN implementations.yaml:")
    for i, (aid, (_, x, y, speed)) in enumerate(sorted(first.items())):
        print(f"        - {{agent: {i}, xy: [{x:.1f}, {y:.1f}]}}   # actor {aid}, "
              f"starts at {speed:.2f} m/s")
    if args.dry_run:
        print("  (dry run: nothing written)")
        return 0

    out = args.out or args.csv_path
    if out == args.csv_path:
        backup = args.csv_path.with_suffix(f".csv.bak-untrimmed")
        if not backup.exists():
            shutil.copy2(args.csv_path, backup)
            print(f"  kept the untrimmed original at {backup.name}")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    print(f"  wrote {out}")
    print("  now: set example_idx back to 0 (or remove it), update `spawns` above, and clear\n"
          "       demo_dataset/trajdata_cache/{<source>,data_indexes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
