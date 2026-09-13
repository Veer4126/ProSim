"""agent_history.json -> history_20hz.csv (which is actually 10 Hz -- see CLAUDE.md).

Reads both layouts:
  v2 (current): {"format_version": 2, "dt": 0.1, "agents": {id: {..., "states": [...]}}}
  v1 (legacy):  {id: [state, ...]}                 -- no extents, no z, type assumed VEHICLE

v1 files still convert, but emit placeholder extents; carla_dataset.py then falls back
to its DEFAULT_EXTENT table. Re-record with the current recorder to get real ones.
"""

import csv
import json

IN_PATH = "/workspace/agent_history.json"
OUT_PATH = "/workspace/history_20hz.csv"

# Only used for legacy v1 files that carry no bounding boxes.
FALLBACK_EXTENT = (4.5, 2.0, 1.5)

COLUMNS = [
    "frame", "time",
    "x", "y", "z",
    "vx", "vy",
    "ax", "ay",
    "yaw",
    "length", "width", "height",
    "id", "type",
]


def load(path):
    """Return (dt, {agent_id: (meta, states)}) for either layout."""
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and raw.get("format_version") == 2:
        dt = float(raw.get("dt", 0.1))
        out = {}
        for agent_id, rec in raw["agents"].items():
            meta = {
                "type": rec.get("type", "VEHICLE"),
                "length": rec["length"],
                "width": rec["width"],
                "height": rec["height"],
            }
            out[agent_id] = (meta, rec["states"])
        return dt, out

    # Legacy flat layout.
    print("WARNING: legacy v1 agent_history.json -- no bounding boxes recorded, "
          "writing placeholder extents and z=0.0")
    length, width, height = FALLBACK_EXTENT
    meta = {"type": "VEHICLE", "length": length, "width": width, "height": height}
    return 0.1, {aid: (dict(meta), states) for aid, states in raw.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=IN_PATH,
                    help=f"recorder output (default {IN_PATH})")
    ap.add_argument("--out", dest="out_path", default=OUT_PATH,
                    help=f"trajectory CSV to write (default {OUT_PATH}). For a "
                         "scenario, name it history_<town>__<scene>.csv so "
                         "carla_dataset serves it under carla_<town>__<scene>.")
    args = ap.parse_args()
    dt, agents = load(args.in_path)

    rows = []
    for agent_id, (meta, states) in agents.items():
        for t, s in enumerate(states):
            rows.append([
                t, round(t * dt, 3),
                s["x"], s["y"], s.get("z", 0.0),
                s["vx"], s["vy"],
                s.get("ax", 0.0), s.get("ay", 0.0),
                s["yaw_rad"],
                meta["length"], meta["width"], meta["height"],
                int(agent_id), meta["type"],
            ])

    with open(args.out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    n_steps = len(next(iter(agents.values()))[1]) if agents else 0
    print(f"wrote {args.out_path}: {len(agents)} agents x {n_steps} steps "
          f"= {len(rows)} rows, dt={dt}s ({n_steps * dt:.1f}s of scenario)")
    for agent_id, (meta, _) in sorted(agents.items(), key=lambda kv: int(kv[0])):
        print(f"  {agent_id}: {meta['type']:<10} "
              f"{meta['length']:.2f} x {meta['width']:.2f} x {meta['height']:.2f} m")


if __name__ == "__main__":
    main()
