"""Spawn vehicles in CARLA, drive them on autopilot, record their trajectories.

Writes agent_history.json (format_version 2), which extract_csv.py turns into
history_20hz.csv.

NEEDS a running CARLA server on the same node. Run inside carla-ubuntu20.sif.

WHAT THE SPAWN POINTS ARE: world.get_map().get_spawn_points() returns locations
hand-authored into the Town10HD level by CARLA's map designers. They are NOT
random and NOT real-world data -- and they are not derivable from the .xodr
either (carla.Map built offline from the OpenDRIVE returns zero of them), so
--list-spawn-points needs the server.

The motion comes from CARLA's Traffic Manager (rule-based autopilot), so every
trajectory here is synthetic. Nothing in this pipeline is recorded real traffic.

An empty CARLA world contains NO vehicle actors -- Town10HD's parked cars are
static level meshes, not actors. So the agents recorded here are exactly the
ones this script spawns; there are no others to harvest.

Examples
--------
    # see what's available, then exit
    python3 record_actor_history.py --list-spawn-points

    # default: first 6 usable spawn points, deterministic (the old behaviour)
    python3 record_actor_history.py

    # exactly these spawn points, in this order
    python3 record_actor_history.py --spawn-points 12 13 44 45 61

    # 20 agents clustered near an intersection -- much closer to the agent
    # density ProSim was trained on than 6 agents spread across the whole town
    python3 record_actor_history.py --num-agents 20 --spawn-near -64 24 --radius 80

    # cars only, no bikes/motorbikes, randomised but reproducible
    python3 record_actor_history.py --num-agents 12 --four-wheels-only --seed 0

    # a hand-built scenario: exact coordinates, snapped to the nearest driving
    # lane. ORDER MATTERS -- the first one spawned gets the lowest actor id and
    # therefore becomes 'ego' downstream (carla_dataset maps min(id) -> 'ego').
    python3 record_actor_history.py --spawn-xy -85 24.5  -45 100 --steps 150
"""

import argparse
import json
import math
import random

import carla
import numpy as np

DEFAULT_OUT = "/workspace/agent_history.json"

# ProSim needs HISTORY_SEC 1.0 + FUTURE_SEC 8.0 = 9.0s per sample. At dt=0.1
# that is 91 steps; anything shorter yields zero samples (and trajdata dies in
# AgentDataIndex rather than reporting an empty dataset). See CLAUDE.md.
PROSIM_MIN_STEPS = 91


def classify(vehicle) -> str:
    """Map a CARLA vehicle blueprint onto a trajdata AgentType name.

    `vehicle.*` includes bicycles and motorbikes, which are NOT AgentType.VEHICLE
    downstream. number_of_wheels distinguishes them; the id distinguishes a
    pedal bike from a motorbike. Names must match TYPE_MAP in carla_dataset.py.
    """
    type_id = vehicle.type_id
    try:
        wheels = int(vehicle.attributes.get("number_of_wheels", 4))
    except (TypeError, ValueError):
        wheels = 4

    if wheels == 2:
        bicycles = ("crossbike", "omafiets", "century")
        return "BICYCLE" if any(b in type_id for b in bicycles) else "MOTORCYCLE"
    return "VEHICLE"


def bp_wheels(bp) -> int:
    try:
        return int(bp.get_attribute("number_of_wheels").as_int())
    except (RuntimeError, ValueError, AttributeError):
        return 4


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--out", default=DEFAULT_OUT)

    ap.add_argument("--steps", type=int, default=150,
                    help="timesteps to record (default 150 = 15.0s)")
    ap.add_argument("--dt", type=float, default=0.1,
                    help="fixed_delta_seconds; must match generate_video.py")

    # --- which spawn points ---
    ap.add_argument("--list-spawn-points", action="store_true",
                    help="print every spawn point with its index, then exit")
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--spawn-points", type=int, nargs="+", default=None,
                    metavar="IDX",
                    help="explicit spawn-point indices; overrides --num-agents")
    ap.add_argument("--spawn-near", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="only consider spawn points within --radius of this point")
    ap.add_argument("--spawn-xy", type=float, nargs="+", default=None,
                    metavar="X Y",
                    help="spawn at these exact world coordinates, given as X Y "
                         "pairs, each snapped to the nearest DRIVING lane so the "
                         "car faces the way that lane runs. Bypasses the "
                         "authored spawn points entirely, which is the only way "
                         "to build a scenario at a chosen intersection. ORDER "
                         "MATTERS: the first spawns first, gets the lowest actor "
                         "id, and so becomes 'ego' downstream.")
    ap.add_argument("--spawn-z", type=float, default=0.5,
                    help="metres above the lane surface to spawn at, for "
                         "--spawn-xy (default 0.5; too low and CARLA rejects the "
                         "spawn as colliding with the road)")
    ap.add_argument("--radius", type=float, default=100.0,
                    help="radius for --spawn-near, in metres")
    ap.add_argument("--seed", type=int, default=None,
                    help="shuffle candidates with this seed. Without it the "
                         "selection is deterministic (lowest indices first), "
                         "which is why repeated runs reproduce the same scene")

    # --- which vehicles ---
    ap.add_argument("--blueprints", default="vehicle.*",
                    help="blueprint filter, e.g. 'vehicle.tesla.*'")
    ap.add_argument("--four-wheels-only", action="store_true",
                    help="drop bicycles and motorbikes (they are not "
                         "AgentType.VEHICLE downstream)")
    return ap.parse_args()


def spawn_points_from_xy(world, coords, lift):
    """Turn a flat [x1,y1,x2,y2,...] list into lane-aligned spawn transforms.

    `map.get_waypoint(..., project_to_road=True)` snaps to the nearest DRIVING
    lane and returns that lane's full transform, so the car is placed on the
    road, in the lane, facing the way the lane runs -- which hand-writing a yaw
    would not get right on a curve. Returns [(label, transform), ...] in the
    order given.
    """
    if len(coords) % 2 != 0:
        raise SystemExit(
            f"--spawn-xy needs an even number of values (X Y pairs); got "
            f"{len(coords)}: {coords}")

    carla_map = world.get_map()
    print(f"  (the loaded map is {carla_map.name})")
    out = []
    for k in range(0, len(coords), 2):
        x, y = float(coords[k]), float(coords[k + 1])
        wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=0.0),
                                    project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            raise SystemExit(
                f"--spawn-xy ({x}, {y}) is not near any driving lane. Check it "
                "against town10hd_lanes.json (map_ref.py draws the map to "
                "scale) before re-running.")
        loc = wp.transform.location
        off = math.hypot(loc.x - x, loc.y - y)
        tf = carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + lift),
            wp.transform.rotation)
        print(f"  requested ({x:8.2f}, {y:8.2f}) -> lane "
              f"road {wp.road_id} lane {wp.lane_id} at "
              f"({loc.x:8.2f}, {loc.y:8.2f}) yaw {wp.transform.rotation.yaw:+7.1f} "
              f"  snapped {off:.2f} m")
        if off > 25.0:
            raise SystemExit(
                f"\n({x}, {y}) IS NOT ON THIS MAP.\n"
                f"  loaded map : {carla_map.name}\n"
                f"  nearest driving lane is {off:.1f} m away, at "
                f"({loc.x:.2f}, {loc.y:.2f}) -- the request was clamped to the\n"
                f"  edge of the map, which is why cars end up in the water or in\n"
                f"  mid-air.\n"
                "\nThis is almost always coordinates from ANOTHER TOWN. Load the "
                "town those\ncoordinates belong to first:\n"
                "    load_town_and_extract_xodr.py --town Town04\n"
                "and check the current one any time with --info.\n")
        if off > 5.0:
            print(f"    WARNING: snapped {off:.2f} m -- that point was not really "
                  "on a road; the car will start somewhere you did not ask for")
        out.append((f"xy({x:g},{y:g})", tf))
    return out


def select_spawn_points(spawn_points, args):
    """Return the ordered list of (index, transform) candidates to try."""
    if args.spawn_points is not None:
        bad = [i for i in args.spawn_points if not 0 <= i < len(spawn_points)]
        if bad:
            raise SystemExit(
                f"spawn point indices out of range {bad}; this map has "
                f"{len(spawn_points)} (0..{len(spawn_points) - 1})")
        return [(i, spawn_points[i]) for i in args.spawn_points]

    candidates = list(enumerate(spawn_points))

    if args.spawn_near is not None:
        cx, cy = args.spawn_near
        near = [(i, sp) for i, sp in candidates
                if math.hypot(sp.location.x - cx, sp.location.y - cy) <= args.radius]
        print(f"{len(near)}/{len(candidates)} spawn points within "
              f"{args.radius:.0f} m of ({cx:.1f}, {cy:.1f})")
        if not near:
            raise SystemExit("no spawn points in range -- widen --radius or move "
                             "--spawn-near (--list-spawn-points shows them all)")
        candidates = near

    if args.seed is not None:
        random.Random(args.seed).shuffle(candidates)

    return candidates


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()

    if args.list_spawn_points:
        print(f"{len(spawn_points)} spawn points in {world.get_map().name}")
        print(f"{'idx':>5}  {'x':>9} {'y':>9} {'z':>7}  {'yaw':>8}")
        for i, sp in enumerate(spawn_points):
            print(f"{i:5d}  {sp.location.x:9.2f} {sp.location.y:9.2f} "
                  f"{sp.location.z:7.2f}  {sp.rotation.yaw:8.2f}")
        return

    # defensive cleanup -- destroy any leftover vehicles from a previous run
    existing = world.get_actors().filter("vehicle.*")
    if existing:
        print(f"Cleaning up {len(existing)} leftover vehicle(s)...")
        for v in existing:
            v.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)

    vehicle_bps = list(world.get_blueprint_library().filter(args.blueprints))
    if args.four_wheels_only:
        before = len(vehicle_bps)
        vehicle_bps = [bp for bp in vehicle_bps if bp_wheels(bp) == 4]
        print(f"blueprints: {len(vehicle_bps)}/{before} after dropping 2-wheelers")
    if not vehicle_bps:
        raise SystemExit(f"no blueprints match {args.blueprints!r}")

    if args.spawn_xy is not None:
        print(f"spawning at {len(args.spawn_xy) // 2} explicit coordinate(s):")
        candidates = spawn_points_from_xy(world, args.spawn_xy, args.spawn_z)
        target = len(candidates)
    else:
        target = (len(args.spawn_points) if args.spawn_points is not None
                  else args.num_agents)
        candidates = select_spawn_points(spawn_points, args)

    actors, used = [], []
    for idx, sp in candidates:
        if len(actors) >= target:
            break
        bp = vehicle_bps[len(actors) % len(vehicle_bps)]
        vehicle = world.try_spawn_actor(bp, sp)
        if vehicle is None:
            print(f"  spawn point {idx} occupied/invalid, skipping"
                  + (" -- for --spawn-xy try nudging the coordinate along the "
                     "lane, or raise --spawn-z" if args.spawn_xy else ""))
            continue
        vehicle.set_autopilot(True, tm.get_port())
        actors.append(vehicle)
        used.append(idx)

    if not actors:
        raise SystemExit("could not spawn any vehicles")
    if len(actors) < target:
        print(f"WARNING: only spawned {len(actors)}/{target} agents")
    print(f"spawned {len(actors)} agents at spawn points {used}")
    if actors:
        # carla_dataset maps min(id) -> 'ego', and CARLA hands out ids in spawn
        # order, so this is decided here and nowhere else. Say so out loud.
        ego = min(actors, key=lambda v: v.id)
        et = ego.get_transform()
        print(f"  EGO will be actor {ego.id} (lowest id, spawned first): "
              f"{ego.type_id} at ({et.location.x:.2f}, {et.location.y:.2f})")

    world.tick()

    # Static per-agent info, captured once. bounding_box.extent is HALF-extents
    # in CARLA, so each dimension is doubled to get the full length/width/height
    # that trajdata expects.
    agents = {}
    for v, idx in zip(actors, used):
        ext = v.bounding_box.extent
        agents[str(v.id)] = {
            "blueprint": v.type_id,
            "type": classify(v),
            "spawn_point": idx,   # lets a scene be reproduced exactly
            "length": 2.0 * ext.x,
            "width": 2.0 * ext.y,
            "height": 2.0 * ext.z,
            "states": [],
        }
        print(f"  agent {v.id}: sp={idx:<4} {v.type_id} "
              f"[{agents[str(v.id)]['type']}] "
              f"{2*ext.x:.2f} x {2*ext.y:.2f} x {2*ext.z:.2f} m")

    try:
        for _ in range(args.steps):
            world.tick()
            for v in actors:
                t = v.get_transform()
                vel = v.get_velocity()
                acc = v.get_acceleration()
                agents[str(v.id)]["states"].append({
                    "x": t.location.x, "y": t.location.y, "z": t.location.z,
                    "yaw_rad": np.deg2rad(t.rotation.yaw),
                    "vx": vel.x, "vy": vel.y,
                    "ax": acc.x, "ay": acc.y,
                })

        history = {
            "format_version": 2,
            "dt": args.dt,
            "spawn_points": used,
            "agents": agents,
        }
        with open(args.out, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Recorded {args.steps} steps ({args.steps * args.dt:.1f}s) "
              f"for {len(actors)} agents -> {args.out}")
        if args.steps < PROSIM_MIN_STEPS:
            print(f"WARNING: {args.steps} steps is under ProSim's "
                  f"{PROSIM_MIN_STEPS}-step minimum (history 1.0s + future 8.0s "
                  "at dt=0.1); this recording will yield 0 samples")

    finally:
        # runs even if the loop above crashes -- prevents leftover collisions
        for v in actors:
            v.set_autopilot(False)
            v.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
