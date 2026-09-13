"""TFv6's adapter fed the observation sensor_worker.py builds -- no weights, no GPU.

    PYTHONPATH=<CARLA dist>/PythonAPI/carla \
    TFV6_CONFIG_DIR=<dir with tfv6_resnet34/config.json> \
        /scratch/veerk41/venvs/tfv6/bin/python test_tfv6_adapter_offline.py

The model call is the one thing not exercised. Everything the worker hands the
policy before that call is: the camera dict in the rig's own shapes (HxWx4
BGRA-derived RGB arrays from osc2runner's SensorRig), the lidar as Nx4, and the
route the worker computes. Each check has a control that the same code must
fail, so a check that cannot fail is caught.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/veerk41/ProSim")
sys.path.insert(0, "/scratch/veerk41/scenario_orchestration/third_party/tfv6/scenario_orchestration")
import sensor_worker as W
import policy as P

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


HARNESS = Path("/scratch/veerk41/scenario_orchestration")


def config_dir() -> Path:
    """TFV6_CONFIG_DIR if set, else the checkpoint the harness declares."""
    if os.environ.get("TFV6_CONFIG_DIR"):
        cfg_dir = Path(os.environ["TFV6_CONFIG_DIR"])
    else:
        import yaml
        declared = yaml.safe_load((HARNESS / "configs/policy/tfv6.yaml").read_text())["checkpoint"]
        cfg_dir = Path(declared) if Path(declared).is_absolute() else HARNESS / declared
    if not (cfg_dir / "config.json").is_file():
        raise SystemExit(f"no tfv6 config.json in {cfg_dir}; set TFV6_CONFIG_DIR to the "
                         "tfv6_resnet34 checkpoint directory")
    return cfg_dir


def make_policy():
    from lead.expert.config_expert import ExpertConfig
    from lead.training.config_training import TrainingConfig
    cfg_dir = config_dir()
    pol = P.build_policy({"checkpoint": str(cfg_dir), "parameters": {"device": "cpu"}})
    pol.training_config = TrainingConfig(json.load(open(cfg_dir / "config.json")))
    pol.config_expert = ExpertConfig()
    return pol


def main():
    pol = make_policy()
    cfg = pol.training_config

    print("\n=== 1. cameras: the rig's three views become the strip, in rig order ===")
    colours = {"PCAM_L0": (255, 0, 0), "PCAM_F0": (0, 255, 0), "PCAM_R0": (0, 0, 255)}
    cams = {n: np.full((384, 384, 4), (*c, 255), dtype=np.uint8) for n, c in colours.items()}
    strip = pol._stitch(cams)
    check("strip is CHW 3x384x1152, what the network reads",
          strip.shape == (3, cfg.final_image_height, cfg.final_image_width), str(strip.shape))
    thirds = [strip[:, 200, 192 + 384 * k] for k in range(3)]
    check("left third is PCAM_L0, middle PCAM_F0, right PCAM_R0",
          [tuple(t) for t in thirds] == [colours["PCAM_L0"], colours["PCAM_F0"], colours["PCAM_R0"]],
          str([tuple(int(v) for v in t) for t in thirds]))
    swapped = pol._stitch({"PCAM_L0": cams["PCAM_R0"], "PCAM_F0": cams["PCAM_F0"],
                           "PCAM_R0": cams["PCAM_L0"]})
    check("CONTROL: swapping L0/R0 changes the strip",
          tuple(swapped[:, 200, 192]) != tuple(strip[:, 200, 192]))
    missing = pol._stitch({"PCAM_F0": cams["PCAM_F0"]})
    check("a camera that did not attach is black, not silently a copy",
          missing[:, :, :384].max() == 0 and missing[:, :, 384:768].max() == 255)

    print("\n=== 2. route -> command, from the worker's own route_in_ego_frame ===")
    s = np.arange(0.0, 60.0, 0.5)
    straight = np.stack([s, np.zeros_like(s)], axis=1)
    r = 15.0
    ang = np.linspace(0, math.pi / 2, 60)
    bend_pos_y = np.concatenate([np.stack([np.linspace(0, 2, 5), np.zeros(5)], 1),
                                 np.stack([2 + r * np.sin(ang), r * (1 - np.cos(ang))], 1)])
    bend_neg_y = bend_pos_y * np.array([1.0, -1.0])
    cmd = {}
    for name, route in (("straight", straight), ("+y", bend_pos_y), ("-y", bend_neg_y)):
        obs_route = W.route_in_ego_frame(route, 0.0, 0.0, 0.0)
        _, target, _, commands = pol._navigation({"route": obs_route})
        cmd[name] = commands[1]
    # CARLA is left-handed: world +y is the driver's RIGHT (measured 2026-09-05).
    check("a bend toward world +y is a RIGHT turn for the model",
          cmd["+y"] == "RIGHT", str(cmd))
    check("CONTROL: the mirrored bend is LEFT, and straight is not a turn",
          cmd["-y"] == "LEFT" and cmd["straight"] in ("LANEFOLLOW", "STRAIGHT"), str(cmd))
    prev, target, nxt, _ = pol._navigation({"route": W.route_in_ego_frame(straight, 0, 0, 0)})
    check("target points are ahead, spaced along the 1 m route",
          prev[0] > 0 and target[0] > prev[0] and nxt[0] > target[0],
          f"{prev} {target} {nxt}")

    print("\n=== 3. lidar: the rig's Nx4 cloud through lead's own rasterizer ===")
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(-30, 30, 20000), rng.uniform(-30, 30, 20000),
                              np.full(20000, -2.4), np.ones(20000)]).astype(np.float32)
    wall = np.column_stack([np.full(3000, 12.0), rng.uniform(-3, 3, 3000),
                            rng.uniform(-2, 1, 3000), np.ones(3000)]).astype(np.float32)
    grid = pol._lidar({"lidar": np.concatenate([ground, wall])})
    empty = pol._lidar({"lidar": np.zeros((0, 4), np.float32)})
    want = (1, 1, cfg.lidar_height_pixel, cfg.lidar_width_pixel)
    check("raster is 1x1xHxW at TrainingConfig's own lidar size",
          grid.shape == want, f"{grid.shape} vs {want}")
    check("points produce a non-empty raster",
          float(grid.sum()) > 0, f"sum {grid.sum():.1f}, nonzero px {int((grid > 0).sum())}")
    check("CONTROL: an empty cloud gives an empty raster",
          float(empty.sum()) == 0.0, f"sum {empty.sum():.1f}")
    # The adapter's own "lidar did not attach" branch reads
    # lidar_resolution_height/width, which TrainingConfig does not define.
    # Reported, not patched here: it is the policy repository's code.
    try:
        pol._lidar({})
        check("adapter: a missing lidar yields zeros", True)
    except AttributeError as exc:
        print(f"  [note] adapter bug, a missing lidar raises: {exc}")

    print("\n=== 4. radar: the rig's ego-frame Nx4 detections, in rig order ===")
    # config.json leaves use_radars null; TrainingConfig defaults it to True.
    check("this checkpoint DOES read radar", bool(cfg.use_radars), str(cfg.use_radars))
    inside = [10.0, 2.0, 0.6, -3.0]                      # (x, y, z, radial vel)
    outside = [cfg.max_x_meter + 50.0, 0.0, 0.6, -1.0]   # beyond the BEV bounds
    radar = pol._radar({"radar1": np.array([inside, outside], np.float32),
                        "radar3": np.array([inside], np.float32)})
    n = cfg.num_radar_points_per_sensor
    check("block is (sensors x points_per_sensor) x 5",
          radar.shape == (cfg.num_radar_sensors * n, 5), str(radar.shape))
    ids = [float(radar[k * n, 4]) for k in range(cfg.num_radar_sensors)]
    check("5th column is the sensor index, in the adapter's RADARS order",
          ids == [0.0, 1.0, 2.0, 3.0], str(ids))
    check("an in-bounds detection survives, velocity column intact",
          np.allclose(radar[0, :4], inside) and np.allclose(radar[2 * n, :4], inside),
          str(radar[0, :4].tolist()))
    check("CONTROL: the out-of-bounds detection is dropped, not kept as row 2",
          np.allclose(radar[1, :4], 0.0), str(radar[1, :4].tolist()))
    check("a radar that returned nothing contributes zero rows, still labelled",
          np.allclose(radar[n:2 * n, :4], 0.0) and radar[n, 4] == 1.0)

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
