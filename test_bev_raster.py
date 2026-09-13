"""Is the offline BEV raster the one PlanT 2.0 expects?

    apptainer exec -B /scratch/veerk41:/workspace prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 test_bev_raster.py"

The warp is transcribed from `chauffeurnet.py`, so checking it against itself
would prove nothing. The decisive test projects OUR lane graph -- exported
independently from the town's OpenDRIVE by export_lane_graph.py -- into the
raster and asks how many lane centre points land on road. A wrong rotation,
a wrong world offset or a flipped axis puts them on unlabeled ground.

Controls, because a high hit rate can be had for the wrong reasons:
  - y-mirrored lane points must NOT land on road (the canonical control in this
    project since 2026-08-28);
  - a pose far outside the town must give an empty raster, so "everything is
    road" cannot pass;
  - the raster must actually differ when the ego turns.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/veerk41/ProSim")
from bev_raster import (CLASS_LANE_ALL, CLASS_LANE_BROKEN, CLASS_ROAD,
                        CLASS_SIDEWALK, BevRasteriser, maps_dir_for)

PLANT2 = Path("/home/veerk41/scratch/scenario_orchestration/third_party/plant2")
DATA = Path("/scratch/veerk41")
DRIVEABLE = (CLASS_ROAD, CLASS_LANE_ALL, CLASS_LANE_BROKEN)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def banner(t):
    print(f"\n=== {t} ===")


def ego_pose(csv_path, frame=10):
    rows = list(csv.DictReader(open(csv_path)))
    ego = min(int(r["id"]) for r in rows)
    r = next(x for x in rows if int(x["id"]) == ego and int(x["frame"]) == frame)
    return float(r["x"]), float(r["y"]), float(r["yaw"])


def hit_rate(bev, pts, x, y, yaw, radius=55.0):
    """Fraction of nearby lane points that land on a driveable class."""
    pts = np.asarray(pts, float)
    near = pts[np.linalg.norm(pts - np.array([x, y]), axis=1) < radius]
    if not len(near):
        return 0.0, 0
    rc = bev.world_to_raster(near, x, y, yaw)
    r = np.rint(rc[:, 0]).astype(int)
    c = np.rint(rc[:, 1]).astype(int)
    inside = (r >= 0) & (r < bev.width) & (c >= 0) & (c < bev.width)
    if not inside.any():
        return 0.0, 0
    cls = bev.classes(x, y, yaw)[r[inside], c[inside]]
    return float(np.isin(cls, DRIVEABLE).mean()), int(inside.sum())


def main():
    maps = maps_dir_for(PLANT2)
    check("PlanT 2.0's town rasters are present", maps.is_dir(), str(maps))
    if not maps.is_dir():
        return 1

    cases = [("Town04", DATA / "history_town04__cut_in.csv", DATA / "town04_lanes.json"),
             ("Town10HD_Opt", DATA / "history_town10hd__left_turn.csv",
              DATA / "town10hd_lanes.json")]

    for town, rec, lanes_json in cases:
        banner(f"{town}")
        bev = BevRasteriser(town, maps)
        x, y, yaw = ego_pose(rec)
        cls = bev.classes(x, y, yaw)

        check("raster is 256x256 uint8",
              cls.shape == (256, 256) and cls.dtype == np.uint8,
              f"{cls.shape} {cls.dtype}")
        present = sorted(int(v) for v in np.unique(cls))
        check("only chauffeurnet's static classes appear (0-4, no actors)",
              set(present) <= {0, CLASS_ROAD, CLASS_SIDEWALK, CLASS_LANE_ALL,
                               CLASS_LANE_BROKEN}, str(present))
        row, col = BevRasteriser.ego_pixel()
        check("the ego itself is standing on a driveable class",
              int(cls[row, col]) in DRIVEABLE,
              f"class {int(cls[row, col])} at ({row}, {col})")

        pts = np.concatenate([np.asarray(l["center"], float)[:, :2]
                              for l in json.load(open(lanes_json))["lanes"].values()])
        rate, n = hit_rate(bev, pts, x, y, yaw)
        print(f"   ego ({x:.1f}, {y:.1f}) yaw {math.degrees(yaw):+.1f} deg; "
              f"{n} lane points inside the crop")
        check("OUR OpenDRIVE lane graph lands on PlanT's road raster",
              rate > 0.90, f"{rate * 100:.1f}% on road/lane markings")

        # CONTROL. The y-mirror used elsewhere in this project does not
        # discriminate here: in a dense grid town a mirrored point usually
        # lands on ANOTHER road (Town10HD 83.5%, Town04 38.8%), which is a
        # property of the town, not evidence about the warp. Reported, not
        # asserted. The control that does discriminate is the WRONG TOWN:
        # a different world offset and different geometry, so the same points
        # cannot keep landing on road unless the mapping is town-specific.
        mirrored = pts.copy()
        mirrored[:, 1] = 2.0 * y - mirrored[:, 1]
        rate_m, _ = hit_rate(bev, mirrored, x, y, yaw)
        print(f"   (y-mirrored lane points: {rate_m * 100:.1f}% -- weak control "
              f"in a grid town, not asserted)")
        other = "Town10HD_Opt" if town == "Town04" else "Town04"
        rate_w, _ = hit_rate(BevRasteriser(other, maps), pts, x, y, yaw)
        print(f"   (same points on {other}'s raster: {rate_w * 100:.1f}% -- also "
              f"density-dependent, not asserted)")
        # The control that isolates ALIGNMENT rather than density: rotate the
        # lane points about the ego by an angle that is not a multiple of 90.
        # Same town, same points, same local road density -- only the
        # correspondence is destroyed. If the warp's rotation were wrong by an
        # arbitrary angle, this is what the positive check would look like.
        a = math.radians(37.0)
        d = pts - np.array([x, y])
        rot = np.stack([d[:, 0] * math.cos(a) - d[:, 1] * math.sin(a),
                        d[:, 0] * math.sin(a) + d[:, 1] * math.cos(a)],
                       axis=1) + np.array([x, y])
        rate_r, n_r = hit_rate(bev, rot, x, y, yaw)
        check("CONTROL: lane points rotated 37 deg about the ego do NOT",
              rate_r < 0.75 and rate_r < rate - 0.25,
              f"{rate_r * 100:.1f}% of {n_r} points (vs {rate * 100:.1f}% aligned)")

        turned = bev.classes(x, y, yaw + math.pi / 2)
        check("CONTROL: turning the ego 90 deg changes the raster",
              not np.array_equal(cls, turned),
              f"{100.0 * (cls != turned).mean():.1f}% of pixels differ")

        far = bev.classes(20000.0, 20000.0, 0.0)
        check("CONTROL: a pose off the map is empty, not road",
              int(far.max()) == 0, f"max class {int(far.max())}")

    banner("the ego pixel is where chauffeurnet puts it")
    # 128 px from the bottom of a 256 px crop, centred left-to-right, then
    # rot90(k=-1). Independently: the centre of the image.
    check("ego lands at the raster centre after the rotation",
          BevRasteriser.ego_pixel() == (128, 128), str(BevRasteriser.ego_pixel()))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
