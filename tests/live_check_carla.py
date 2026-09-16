"""Live checks against a RUNNING CARLA server -- the two things no fake can show.

    /scratch/veerk41/venvs/tfv6/bin/python tests/live_check_carla.py --host <node> [--port 2000]

Uses an existing server (does not start one), leaves its settings as it found
them and destroys what it spawns. Needs no GPU of its own.

1. A car the worker moves (`sensor_worker.place_actor`, physics on) reports its
   real speed, and a radar aimed at it measures it moving. CONTROL: the worker's
   previous way (teleport, physics off) must read 0 -- if it does not, this
   check could not have caught the bug.
2. The server renders without missing materials: a SimLingo-shaped camera
   (1024x512, fov 110) at a spawn point stays under `PINK_FRACTION` magenta.
   Run it after any town reload; a broken server fails here before an episode
   is wasted on it.
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
import queue

import numpy as np

import sensor_worker as W

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    library = world.get_blueprint_library()
    spawn = world.get_map().get_spawn_points()[10]
    yaw = math.radians(spawn.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    made = []
    print(f"server map {world.get_map().name}")
    try:
        # ------------------------------------------------------------ 1
        print("\n=== 1. a moved car's speed reaches CARLA and its radar ===")
        watcher = world.spawn_actor(library.find(W.EGO_BLUEPRINT), carla.Transform(
            carla.Location(spawn.location.x - 25 * fx, spawn.location.y - 25 * fy,
                           spawn.location.z + 0.3), spawn.rotation))
        made.append(watcher)
        watcher.set_simulate_physics(False)
        rb = library.find("sensor.other.radar")
        for key, value in (("horizontal_fov", "30"), ("vertical_fov", "10"),
                           ("range", "100"), ("points_per_second", "4000")):
            rb.set_attribute(key, value)
        radar = world.spawn_actor(rb, carla.Transform(carla.Location(x=2.5, z=1.0)), attach_to=watcher)
        made.append(radar)
        hits = {}
        radar.listen(lambda d: hits.__setitem__("v", [x.velocity for x in d if x.depth < 40]))
        speed = 8.0
        result = {}
        for label, physics in (("worker (place_actor, physics on)", True),
                               ("CONTROL: old way (teleport, physics off)", False)):
            car = world.spawn_actor(library.find(W.ACTOR_BLUEPRINT), carla.Transform(
                carla.Location(spawn.location.x, spawn.location.y, spawn.location.z + 0.3), spawn.rotation))
            made.append(car)
            car.set_simulate_physics(physics)
            world.tick()
            reported, radial = [], []
            for k in range(1, 21):
                pose = {"x": spawn.location.x + fx * speed * 0.05 * k,
                        "y": spawn.location.y + fy * speed * 0.05 * k,
                        "yaw_rad": yaw, "speed": speed}
                tf = carla.Transform(carla.Location(pose["x"], pose["y"], spawn.location.z + W.ACTOR_Z_OFFSET),
                                     spawn.rotation)
                if physics:
                    W.place_actor(carla, car, tf, pose)
                else:
                    car.set_transform(tf)
                    car.set_target_velocity(carla.Vector3D(speed * fx, speed * fy, 0.0))
                hits.clear()
                world.tick()
                v = car.get_velocity()
                if k > 5:
                    reported.append(math.hypot(v.x, v.y))
                    if hits.get("v"):
                        radial.append(float(np.mean(np.abs(hits["v"]))))
            result[physics] = (float(np.mean(reported)), float(np.mean(radial)) if radial else 0.0)
            print(f"  {label}: moved at {speed} m/s, CARLA reports {result[physics][0]:.2f} m/s, "
                  f"radar measures {result[physics][1]:.2f} m/s")
            car.destroy()
            made.remove(car)
            world.tick()
        check("the worker's way: CARLA reports the car's real speed (within 0.5 m/s)",
              abs(result[True][0] - speed) < 0.5, f"{result[True][0]:.2f} m/s")
        check("the worker's way: radar measures it moving (> 1 m/s)",
              result[True][1] > 1.0, f"{result[True][1]:.2f} m/s")
        check("CONTROL: the old physics-off way reads 0 m/s, the bug this guards against",
              result[False][0] < 0.5 and result[False][1] < 0.5,
              f"{result[False][0]:.2f} / {result[False][1]:.2f} m/s")

        # ------------------------------------------------------------ 2
        print("\n=== 2. the server renders without missing materials ===")
        cb = library.find("sensor.camera.rgb")
        for key, value in (("image_size_x", "1024"), ("image_size_y", "512"), ("fov", "110")):
            cb.set_attribute(key, value)
        frames = queue.Queue()
        camera = world.spawn_actor(cb, carla.Transform(carla.Location(x=-1.5, z=2.0)), attach_to=watcher)
        made.append(camera)
        camera.listen(frames.put)
        fractions = []
        for _ in range(8):
            world.tick()
            try:
                image = frames.get(timeout=10)
            except queue.Empty:
                continue
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            fractions.append(W.magenta_fraction(bgra[..., [2, 1, 0]]))
        check("camera frames arrived", len(fractions) >= 5, f"{len(fractions)} of 8")
        check(f"magenta under the worker's limit ({W.PINK_FRACTION:.0%})",
              bool(fractions) and max(fractions) < W.PINK_FRACTION,
              f"max {max(fractions) if fractions else float('nan'):.3f}")
    finally:
        for actor in made:
            try:
                actor.destroy()
            except Exception:
                pass
        world.apply_settings(original)
        print(f"\nserver settings restored (synchronous={world.get_settings().synchronous_mode})")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
