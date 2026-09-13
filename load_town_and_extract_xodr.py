"""Load a CARLA town on the running server and export its OpenDRIVE map.

NEEDS a running CARLA server on the same node. Run inside carla-ubuntu20.sif.

apptainer exec -B /scratch/veerk41:/workspace \
    ~/projects/aip-gigor/veerk41/carla-ubuntu20.sif \
    python3 /workspace/load_town_and_extract_xodr.py --town Town04

DESTRUCTIVE: client.load_world() replaces the map and destroys every actor on
it. Use --info to see what is currently loaded without touching anything.

The .xodr is written to /workspace/<town lowercased>.xodr, e.g. town04.xodr,
which is what export_lane_graph.py then turns into <town>_lanes.json.
"""

import argparse

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default=None,
                    help="map name, e.g. Town04 (highway loop), Town10HD. "
                         "REQUIRED -- there is deliberately no default, because "
                         "a default silently reloads the world and strands any "
                         "coordinates you had picked for another town.")
    ap.add_argument("--out", default=None,
                    help="output .xodr (default /workspace/<town lowercased>.xodr)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="map loads can take a while; 30 s is not always enough")
    ap.add_argument("--info", action="store_true",
                    help="report the current world and exit, changing nothing")
    ap.add_argument("--list-maps", action="store_true",
                    help="list the maps this server can load, then exit")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    if args.list_maps:
        for m in client.get_available_maps():
            print(m)
        return

    world = client.get_world()
    cur = world.get_map().name
    actors = world.get_actors()
    print(f"currently loaded : {cur}")
    print(f"  actors present : {len(actors.filter('vehicle.*'))} vehicles, "
          f"{len(actors.filter('walker.*'))} walkers, "
          f"{len(actors.filter('sensor.*'))} sensors")
    print(f"  spawn points   : {len(world.get_map().get_spawn_points())}")
    if args.info:
        return

    if args.town is None:
        raise SystemExit(
            "\n--town is required (e.g. --town Town04).\n"
            "There is deliberately no default: a default silently reloads the "
            "world and\nstrands any coordinates you had picked for another "
            "town.\n"
            "  --info       report the loaded world, change nothing\n"
            "  --list-maps  what this server can load\n")

    if args.town.lower() in cur.lower():
        print(f"\n{args.town} is already loaded; re-exporting its map without "
              "reloading (this preserves any actors).")
    else:
        print(f"\nloading {args.town} -- this DESTROYS the actors listed above ...")
        world = client.load_world(args.town)
        print(f"loaded: {world.get_map().name}")

    out = args.out or f"/workspace/{args.town.lower()}.xodr"
    xodr = world.get_map().to_opendrive()
    with open(out, "w") as f:
        f.write(xodr)
    sp = world.get_map().get_spawn_points()
    print(f"\nwrote {out}  ({len(xodr)} bytes)")
    print(f"spawn points on this map: {len(sp)}")
    if sp:
        xs = [p.location.x for p in sp]
        ys = [p.location.y for p in sp]
        print(f"  extent: x {min(xs):8.1f} .. {max(xs):8.1f}   "
              f"y {min(ys):8.1f} .. {max(ys):8.1f}")
    print(f"\nNEXT: export_lane_graph.py --xodr {out} "
          f"--out /workspace/{args.town.lower()}_lanes.json")


if __name__ == "__main__":
    main()
