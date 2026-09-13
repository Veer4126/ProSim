"""
Stage A of the CARLA -> trajdata bridge.

Converts an OpenDRIVE file into a plain-JSON lane graph, using CARLA's own
OpenDRIVE parser to evaluate the geometry (lines, arcs, spirals, poly3,
paramPoly3) instead of reimplementing it.

MUST run inside carla-ubuntu20.sif (python3.8 + carla from ~/.local).
Does NOT need a running CARLA server and does NOT need a GPU -- carla.Map is
built client-side from the .xodr text.

    module load apptainer/1.4.5
    apptainer exec -B /scratch/veerk41:/workspace \
        ~/projects/aip-gigor/veerk41/carla-ubuntu20.sif \
        python3 /workspace/export_lane_graph.py \
            --xodr /workspace/town10hd.xodr \
            --out  /workspace/town10hd_lanes.json

The output is consumed by carla_dataset.py, which runs inside prosim_v4.sif
where `carla` is NOT importable (PYTHONNOUSERSITE=1). Keeping the carla
dependency on this side of the boundary is the whole point of the split.

COORDINATE FRAME: waypoint.transform.location is in CARLA's native world
frame -- exactly the frame record_actor_history.py records agent x/y in. No
sign flip is applied here, and none must be applied downstream. (See the
CoordMap Y-flip section in CLAUDE.md for what happens when one creeps in.)
"""

import argparse
import json
import math

import carla

# Sampling step along each lane centerline, in metres.
#
# MUST stay ~0.5 to match the Waymo lane sampling the ProSim checkpoint was
# trained on (measured: Waymo median spacing 0.495 m). ProSim's map encoder is a
# PointNet over lane chunks capped at MAX_LANE_POINTS: 20, so the step directly
# sets how much road one map token covers:
#     step 2.0 -> 20-pt token spans 40 m   (4x out of distribution)
#     step 0.5 -> 20-pt token spans 10 m   (matches Waymo's 9.9 m)
# A coarse step also returns 4x fewer points for the policy's 50 m MAP_RADIUS
# query. Raising this is the single easiest way to silently degrade lane-keeping.
DEFAULT_STEP = 0.5


def lane_key(wp):
    """Stable identifier for a (road, section, lane) triple.

    MUST be parseable by int(): trajdata's VectorMap.get_traffic_light_status
    does `int(lane_id)` unconditionally (vec_map.py:377), so a readable string
    key like "939_0_-1" raises ValueError deep inside ProSim's map collation.
    The three components are packed into one integer instead; readable_key()
    keeps the human-facing form for debugging.

    Layout: road_id * 1000 + section_id * 100 + (lane_id + 50)
    Valid while section_id < 10 and -50 < lane_id < 50; asserted below.
    """
    assert 0 <= wp.section_id < 10, f"section_id {wp.section_id} breaks lane-id packing"
    assert -50 < wp.lane_id < 50, f"lane_id {wp.lane_id} breaks lane-id packing"
    return str(wp.road_id * 1000 + wp.section_id * 100 + (wp.lane_id + 50))


def readable_key(wp):
    """Human-readable form, kept alongside the packed id for debugging."""
    return f"{wp.road_id}_{wp.section_id}_{wp.lane_id}"


def same_direction(wp_a, wp_b):
    """OpenDRIVE lane ids share a sign iff the lanes run the same way."""
    return (wp_a.lane_id > 0) == (wp_b.lane_id > 0)


def centerline_from(start_wp, step):
    """Walk a lane section from its start waypoint to its end."""
    pts = [start_wp]
    try:
        pts.extend(start_wp.next_until_lane_end(step))
    except RuntimeError:
        # Very short sections can raise instead of returning an empty list.
        pass

    # Deduplicate: next_until_lane_end can repeat the seed point, and
    # coincident points make heading computation degenerate downstream.
    out = []
    for wp in pts:
        if out:
            prev = out[-1].transform.location
            cur = wp.transform.location
            if math.hypot(cur.x - prev.x, cur.y - prev.y) < 1e-3:
                continue
        out.append(wp)
    return out


def offset_point(wp, lateral):
    """Offset a waypoint perpendicular to its heading by `lateral` metres.

    +lateral is to the waypoint's right, matching CARLA's left-handed frame.
    """
    tf = wp.transform
    yaw = math.radians(tf.rotation.yaw)
    return [
        tf.location.x + lateral * math.cos(yaw + math.pi / 2.0),
        tf.location.y + lateral * math.sin(yaw + math.pi / 2.0),
        tf.location.z,
    ]


def build_lane_graph(xodr_path, step, driving_only=True):
    with open(xodr_path, "r") as f:
        xodr = f.read()

    carla_map = carla.Map("carla_export", xodr)
    topology = carla_map.get_topology()
    print(f"topology: {len(topology)} lane-section pairs")

    lanes = {}
    for start_wp, _end_wp in topology:
        key = lane_key(start_wp)
        if key in lanes:
            continue
        if driving_only and start_wp.lane_type != carla.LaneType.Driving:
            continue

        wps = centerline_from(start_wp, step)
        if len(wps) < 2:
            # trajdata's RoadLane needs >=2 points to derive headings.
            continue

        center, left_edge, right_edge = [], [], []
        for wp in wps:
            loc = wp.transform.location
            center.append([loc.x, loc.y, loc.z])
            half = wp.lane_width / 2.0
            left_edge.append(offset_point(wp, -half))
            right_edge.append(offset_point(wp, +half))

        first, last = wps[0], wps[-1]

        next_lanes, prev_lanes = set(), set()
        for nxt in last.next(step):
            if not driving_only or nxt.lane_type == carla.LaneType.Driving:
                next_lanes.add(lane_key(nxt))
        for prv in first.previous(step):
            if not driving_only or prv.lane_type == carla.LaneType.Driving:
                prev_lanes.add(lane_key(prv))

        # Adjacent lanes: only same-direction neighbours count as adjacent in
        # trajdata. An oncoming lane across a centre line is not reachable.
        adj_left, adj_right = set(), set()
        mid = wps[len(wps) // 2]
        lwp = mid.get_left_lane()
        if (
            lwp is not None
            and same_direction(mid, lwp)
            and (not driving_only or lwp.lane_type == carla.LaneType.Driving)
        ):
            adj_left.add(lane_key(lwp))
        rwp = mid.get_right_lane()
        if (
            rwp is not None
            and same_direction(mid, rwp)
            and (not driving_only or rwp.lane_type == carla.LaneType.Driving)
        ):
            adj_right.add(lane_key(rwp))

        lanes[key] = {
            "id": key,
            "readable_id": readable_key(start_wp),
            "road_id": start_wp.road_id,
            "section_id": start_wp.section_id,
            "lane_id": start_wp.lane_id,
            "is_junction": bool(start_wp.is_junction),
            "center": center,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "next_lanes": sorted(next_lanes),
            "prev_lanes": sorted(prev_lanes),
            "adj_lanes_left": sorted(adj_left),
            "adj_lanes_right": sorted(adj_right),
        }

    readable = {lane["readable_id"] for lane in lanes.values()}
    assert len(readable) == len(lanes), "packed lane ids collided -- check the layout"

    # Drop dangling references to lanes we filtered out, so the consumer never
    # has to defend against ids that don't exist.
    valid = set(lanes)
    dangling = 0
    for lane in lanes.values():
        for field in ("next_lanes", "prev_lanes", "adj_lanes_left", "adj_lanes_right"):
            kept = [k for k in lane[field] if k in valid]
            dangling += len(lane[field]) - len(kept)
            lane[field] = kept
    if dangling:
        print(f"dropped {dangling} references to filtered-out lanes")

    return lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xodr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=DEFAULT_STEP)
    ap.add_argument(
        "--all-lane-types",
        action="store_true",
        help="keep sidewalks/shoulders/parking too (default: Driving only)",
    )
    args = ap.parse_args()

    lanes = build_lane_graph(args.xodr, args.step, driving_only=not args.all_lane_types)

    xs = [p[0] for lane in lanes.values() for p in lane["center"]]
    ys = [p[1] for lane in lanes.values() for p in lane["center"]]
    payload = {
        "format_version": 2,  # v2: lane ids are int-parseable (see lane_key)
        "source_xodr": args.xodr,
        "step_m": args.step,
        "frame": "carla_native",
        "bounds": {
            "min_x": min(xs), "max_x": max(xs),
            "min_y": min(ys), "max_y": max(ys),
        },
        "lanes": lanes,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f)

    n_pts = sum(len(lane["center"]) for lane in lanes.values())
    print(f"wrote {args.out}")
    print(f"  lanes:      {len(lanes)}")
    print(f"  centre pts: {n_pts}")
    print(f"  bounds:     x[{min(xs):.1f}, {max(xs):.1f}]  y[{min(ys):.1f}, {max(ys):.1f}]")


if __name__ == "__main__":
    main()
