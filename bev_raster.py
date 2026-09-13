"""PlanT 2.0's ego-centred BEV semantic raster, without a CARLA world.

WHY THIS CAN BE DONE OFFLINE
----------------------------
Every released PlanT 2.0 checkpoint carries ``model.training.input_bev = True``,
so its policy needs a BEV raster or ``act()`` raises. That raster looks like it
should need a live simulator -- and it does not, for one reason that is easy to
miss: in ``carla_garage/birds_eye_view/chauffeurnet.py`` the class raster is

    c_all = road_mask * 1
    c_all[sidewalk_mask] = 2
    c_all[lane_mask_all] = 3
    c_all[lane_mask_broken] = 4
    # c_all[vehicle_masks[-1]] = 9      <- commented out
    # c_all[walker_masks[-1]] = 10      <- commented out

The vehicle, walker, traffic-light and stop-line layers are COMMENTED OUT. What
PlanT actually sees in the BEV is STATIC MAP GEOMETRY ONLY; other cars reach it
through the object token list instead. And the static layers ship with the
repository, as pre-rendered town rasters in
``carla_garage/birds_eye_view/maps_2ppm_cv/<Town>.h5``.

So the whole raster is: load the town's masks, warp them to the ego pose, label
the four classes, rotate. No CARLA, no GPU, no rendering.

WHY THE WARP IS RE-IMPLEMENTED RATHER THAN IMPORTED
---------------------------------------------------
``chauffeurnet.py`` imports ``carla`` at module scope, and the ProSim container
has no ``carla`` module (PYTHONNOUSERSITE=1). The transform below is therefore
transcribed from ``_get_warp_transform`` / ``_world_to_pixel``, term for term,
and ``test_bev_raster.py`` checks it against an INDEPENDENT source -- the lane
graph exported from the town's own OpenDRIVE -- rather than against itself.

    from bev_raster import BevRasteriser
    bev = BevRasteriser("Town04", maps_dir)
    classes = bev.classes(x=-300.0, y=30.0, yaw_rad=0.0)   # (256, 256) uint8
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

#: PlanT's own BEV configuration (``PlanT/PlanT_agent.py`` with
#: ``carla_garage/config.py``): a 256 px square at 2 px/m, the ego half the
#: height up from the bottom, read from the 2-px-per-metre town rasters.
WIDTH_PX = 256
PIXELS_EV_TO_BOTTOM = 256 / 2.0
PIXELS_PER_METER = 2.0
MAP_FOLDER = "maps_2ppm_cv"

#: chauffeurnet.py's class indices. 0 is unlabeled.
CLASS_ROAD, CLASS_SIDEWALK, CLASS_LANE_ALL, CLASS_LANE_BROKEN = 1, 2, 3, 4

#: Which town raster goes with each of our recordings. Town10HD_Opt is the map
#: the CARLA server actually had loaded when they were recorded
#: ("the loaded map is Carla/Maps/Town10HD_Opt"), and it ships alongside the
#: plain Town10HD raster, so the matching one is named rather than guessed.
TOWN_RASTER = {"town04": "Town04", "town10hd": "Town10HD_Opt"}


def maps_dir_for(plant2_repo) -> Path:
    """Where the pre-rendered town rasters live inside the PlanT 2.0 repo."""
    return Path(plant2_repo) / "carla_garage" / "birds_eye_view" / MAP_FOLDER


class BevRasteriser:
    """Ego-centred semantic class raster for one town."""

    def __init__(self, town: str, maps_dir, width: int = WIDTH_PX,
                 pixels_ev_to_bottom: float = PIXELS_EV_TO_BOTTOM,
                 pixels_per_meter: float = PIXELS_PER_METER):
        import h5py

        self.town = str(town)
        self.width = int(width)
        self.pixels_ev_to_bottom = float(pixels_ev_to_bottom)
        self.pixels_per_meter = float(pixels_per_meter)

        path = Path(maps_dir) / f"{self.town}.h5"
        if not path.is_file():
            raise SystemExit(
                f"no BEV raster for {self.town}: {path}\n"
                "These ship with PlanT 2.0; check out the submodule:\n"
                "  git submodule update --init third_party/plant2")
        with h5py.File(path, "r", libver="latest", swmr=True) as hf:
            self._road = np.array(hf["road"], dtype=np.uint8)
            self._sidewalk = np.array(hf["sidewalk"], dtype=np.uint8)
            self._lane_all = np.array(hf["lane_marking_all"], dtype=np.uint8)
            self._lane_broken = np.array(hf["lane_marking_white_broken"],
                                         dtype=np.uint8)
            self._world_offset = np.array(hf.attrs["world_offset_in_meters"],
                                          dtype=np.float32)
            on_disk = float(hf.attrs["pixels_per_meter"])
        # chauffeurnet asserts this; a mismatch would scale the whole scene.
        if not np.isclose(self.pixels_per_meter, on_disk):
            raise SystemExit(
                f"{path.name} is {on_disk} px/m but this rasteriser is "
                f"configured for {self.pixels_per_meter} px/m")

    # -- geometry, transcribed from chauffeurnet.py ---------------------

    def _world_to_pixel(self, x: float, y: float) -> np.ndarray:
        """World metres -> map-raster pixels (``_world_to_pixel``)."""
        return np.array([self.pixels_per_meter * (x - self._world_offset[0]),
                         self.pixels_per_meter * (y - self._world_offset[1])],
                        dtype=np.float32)

    def warp(self, x: float, y: float, yaw_rad: float) -> np.ndarray:
        """The affine that crops the town raster around the ego.

        ``_get_warp_transform``: the ego sits `pixels_ev_to_bottom` up from the
        bottom edge and centred left-to-right, with +forward toward the top.
        """
        import cv2 as cv

        ev_px = self._world_to_pixel(x, y)
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
        right = np.array([math.cos(yaw_rad + 0.5 * math.pi),
                          math.sin(yaw_rad + 0.5 * math.pi)])
        w, ev2b = self.width, self.pixels_ev_to_bottom
        bottom_left = ev_px - ev2b * forward - (0.5 * w) * right
        top_left = ev_px + (w - ev2b) * forward - (0.5 * w) * right
        top_right = ev_px + (w - ev2b) * forward + (0.5 * w) * right
        src = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst = np.array([[0, w - 1], [0, 0], [w - 1, 0]], dtype=np.float32)
        return cv.getAffineTransform(src, dst)

    def classes(self, x: float, y: float, yaw_rad: float) -> np.ndarray:
        """The (width, width) uint8 class raster for an ego pose.

        Layer order and the final ``rot90(k=-1)`` ("Align with LiDAR
        voxelgrid") follow chauffeurnet.py exactly, because the checkpoint was
        trained on rasters in that orientation.
        """
        import cv2 as cv

        m = self.warp(x, y, yaw_rad)
        size = (self.width, self.width)
        road = cv.warpAffine(self._road, m, size).astype(bool)
        sidewalk = cv.warpAffine(self._sidewalk, m, size).astype(bool)
        lane_all = cv.warpAffine(self._lane_all, m, size).astype(bool)
        lane_broken = cv.warpAffine(self._lane_broken, m, size).astype(bool)

        c_all = road.astype(np.uint8) * CLASS_ROAD
        c_all[sidewalk] = CLASS_SIDEWALK
        c_all[lane_all] = CLASS_LANE_ALL
        c_all[lane_broken] = CLASS_LANE_BROKEN
        return np.rot90(c_all, k=-1).copy()

    # -- for checking and for overlays ----------------------------------

    def world_to_raster(self, pts: np.ndarray, x: float, y: float,
                        yaw_rad: float) -> np.ndarray:
        """World points -> (row, col) in the raster `classes` returns.

        The same affine, then the same ``rot90(k=-1)``: that rotation maps a
        pixel at (row, col) to (col, width-1-row).
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=float))[:, :2]
        px = np.stack([self.pixels_per_meter * (pts[:, 0] - self._world_offset[0]),
                       self.pixels_per_meter * (pts[:, 1] - self._world_offset[1])],
                      axis=1)
        m = self.warp(x, y, yaw_rad)
        uv = px @ m[:, :2].T + m[:, 2]              # (col, row) before rotation
        return np.stack([uv[:, 0], self.width - 1 - uv[:, 1]], axis=1)

    @staticmethod
    def ego_pixel(width: int = WIDTH_PX,
                  pixels_ev_to_bottom: float = PIXELS_EV_TO_BOTTOM) -> Tuple[int, int]:
        """Where the ego itself lands in the returned raster, as (row, col)."""
        row_before, col_before = width - 1 - pixels_ev_to_bottom, 0.5 * width
        return int(round(col_before)), int(round(width - 1 - row_before))
