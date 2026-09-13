"""Smoke test for CarlaDataset. Runs inside prosim_v4.sif. CPU only."""

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import os
import sys
import shutil
from pathlib import Path

# Resolve paths relative to THIS file, so the test works whether or not
# /scratch/veerk41 is bind-mounted at /workspace. Override with:
#   CARLA_DATA_DIR=/some/dir python3 tests/test_carla_dataset.py
HERE = Path(__file__).resolve().parent

import numpy as np

from carla_dataset import CarlaDataset, register
from trajdata import UnifiedDataset

DATA_DIR = os.environ.get("CARLA_DATA_DIR", str(HERE.parents[1]))  # recordings live one level above the repo
CACHE_DIR = Path(DATA_DIR) / "carla_trajdata_cache"

print(f"data dir:  {DATA_DIR}")
print(f"cache dir: {CACHE_DIR}")
for required in ("history_20hz.csv", "town10hd_lanes.json"):
    path = Path(DATA_DIR) / required
    print(f"  {'OK     ' if path.exists() else 'MISSING'} {path}")
print()

print("=" * 70)
print("1. metadata + dataset object")
print("=" * 70)
ds = CarlaDataset("carla_town10hd", DATA_DIR, parallelizable=False, has_maps=True)
print("  name:", ds.name, " dt:", ds.metadata.dt)
print("  scene_tags:", [str(t) for t in ds.scene_tags])
ds.load_dataset_obj(verbose=True)
print("  scene_length:", ds.dataset_obj.scene_length)

print()
print("=" * 70)
print("2. VectorMap construction")
print("=" * 70)
vm = CarlaDataset.build_vector_map(ds.dataset_obj.lane_graph, "carla_town10hd:test")
lanes = list(vm.iter_elems())
print("  map elements:", len(lanes))
print("  extent [min_xyz, max_xyz]:", np.round(vm.extent, 1))
lane0 = lanes[0]
print("  sample lane id:", lane0.id)
print("    centre pts:", lane0.center.points.shape, "(4th col = derived heading)")
print("    has left/right edge:", lane0.left_edge is not None, lane0.right_edge is not None)
print("    next:", sorted(lane0.next_lanes)[:3], " prev:", sorted(lane0.prev_lanes)[:3])
n_links = sum(len(l.next_lanes) + len(l.prev_lanes) for l in lanes)
n_adj = sum(len(l.adj_lanes_left) + len(l.adj_lanes_right) for l in lanes)
print("  total next/prev links:", n_links, " adjacency links:", n_adj)
assert n_links > 0, "lane graph has no connectivity -- topology export failed"

print()
print("=" * 70)
print("3. full UnifiedDataset build (this exercises get_agent_info + cache_maps)")
print("=" * 70)
if CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)

register()
dataset = UnifiedDataset(
    desired_data=["carla_town10hd"],
    data_dirs={"carla_town10hd": DATA_DIR},
    cache_location=str(CACHE_DIR),
    rebuild_cache=True,
    rebuild_maps=True,
    centric="agent",
    history_sec=(0.5, 0.5),
    future_sec=(1.0, 1.0),
    incl_vector_map=True,
    # NOTE: trajdata sets batch_element.vec_map = None on return unless
    # "collate" is True (dataset.py:1084). Without it the map is still cached
    # and reachable via MapAPI / extras fns -- it just isn't attached here.
    vector_map_params={"incl_road_lanes": True, "collate": True},
    num_workers=0,
    verbose=True,
)
print("  dataset length (agent-timestep samples):", len(dataset))
assert len(dataset) > 0, "no samples produced"

print()
print("=" * 70)
print("4. inspect one batch element")
print("=" * 70)
el = dataset[0]
print("  agent name:      ", el.agent_name)
print("  agent type:      ", el.agent_type)
print("  scene_ts:        ", el.scene_ts)
print("  dt:              ", el.dt)
print("  curr position:   ", np.round(np.asarray(el.curr_agent_state_np)[:2], 2))
print("  history shape:   ", np.asarray(el.agent_history_np).shape,
      f"({el.agent_history_len} steps)")
print("  future shape:    ", np.asarray(el.agent_future_np).shape,
      f"({el.agent_future_len} steps)")
print("  neighbours:      ", el.num_neighbors)
print("  extent (l,w,h):  ", np.round(np.asarray(el.agent_history_extent_np)[-1], 2))
if el.vec_map is not None:
    print("  vec_map id:      ", el.vec_map.map_id)
    print("  vec_map lanes:   ", len(el.vec_map.lanes))
    q = np.asarray(el.curr_agent_state_np)[:2]
    lane = el.vec_map.get_closest_lane(np.array([q[0], q[1], 0.0]))
    d = np.linalg.norm(lane.center.xy - q, axis=1).min()
    print(f"  closest lane:     {lane.id}  (agent is {d:.2f} m from its centreline)")

print()
print("=" * 70)
print("5. round-trip check: cached positions vs source CSV")
print("=" * 70)
import pandas as pd

src = pd.read_csv(Path(DATA_DIR) / "history_20hz.csv")
src["scene_ts"] = src["frame"] - src["frame"].min()
ego_id = int(src["id"].min())

scene = dataset.get_scene(0)
cache = dataset.cache_class(dataset.cache_path, scene)
cached = cache.scene_data_df

max_err = 0.0
for agent_id, group in src.groupby("id"):
    key = "ego" if agent_id == ego_id else str(agent_id)
    got = cached.loc[key]
    want = group.sort_values("scene_ts")
    err_x = np.abs(got["x"].to_numpy() - want["x"].to_numpy()).max()
    err_y = np.abs(got["y"].to_numpy() - want["y"].to_numpy()).max()
    err_h = np.abs(got["heading"].to_numpy() - want["yaw"].to_numpy()).max()
    max_err = max(max_err, err_x, err_y, err_h)
    print(f"  {key:>5}: max |dx|={err_x:.2e}  |dy|={err_y:.2e}  |dheading|={err_h:.2e}")

print()
print(f"  worst round-trip error: {max_err:.2e}")
assert max_err < 1e-6, "cached agent data does not match the source CSV"

print()
print("=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)
