"""
Stage B of the CARLA -> trajdata bridge.

A trajdata RawDataset subclass that presents a CARLA-recorded scenario
(town10hd_lanes.json + history CSV) as if it were a cached WOMD scene, so
ProSim can consume it unchanged.

MUST run inside prosim_v4.sif (/scratch/veerk41/containers/prosim_v4.sif),
which is where ProSim's trajdata lives. `carla` is NOT importable there
(PYTHONNOUSERSITE=1), which is exactly why the OpenDRIVE parsing happens in
export_lane_graph.py on the other side of the container boundary and arrives
here as plain JSON. Do not add a carla import to this file.

Modelled on trajdata/dataset_specific/waymo/waymo_dataset.py.

COORDINATE FRAME: both inputs are in CARLA's native world frame and are
consumed as-is. No mirroring, no yaw negation. See CLAUDE.md.

Registration: ProSim's trajdata is inside a read-only .sif, so
get_raw_dataset() cannot be edited. Call register() before constructing a
UnifiedDataset:

    from carla_dataset import register
    register()
    dataset = UnifiedDataset(desired_data=["carla_town10hd"], ...)
"""

import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import pandas as pd

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import EnvMetadata, Scene, SceneMetadata, SceneTag
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import Polyline, RoadLane
from trajdata.utils import arr_utils

# Matches fixed_delta_seconds in record_actor_history.py / generate_video.py.
# NOTE: history_20hz.csv is misnamed -- it is 10 Hz.
CARLA_DT: float = 0.1

# FALLBACK ONLY, for CSVs produced before the recorder captured bounding
# boxes. The current record_actor_history.py writes real length/width/height
# columns and those are used in preference to these.
DEFAULT_EXTENT: Dict[str, Tuple[float, float, float]] = {
    "VEHICLE": (4.5, 2.0, 1.5),
    "PEDESTRIAN": (0.8, 0.8, 1.8),
    "BICYCLE": (1.8, 0.7, 1.5),
    "MOTORCYCLE": (2.2, 0.9, 1.5),
}

TYPE_MAP: Dict[str, AgentType] = {
    "VEHICLE": AgentType.VEHICLE,
    "PEDESTRIAN": AgentType.PEDESTRIAN,
    "WALKER": AgentType.PEDESTRIAN,
    "BICYCLE": AgentType.BICYCLE,
    "CYCLIST": AgentType.BICYCLE,
    "MOTORCYCLE": AgentType.MOTORCYCLE,
}


class CarlaSceneRecord(NamedTuple):
    name: str
    length: str
    data_idx: int


def const_lambda(const_val: Any) -> Any:
    return const_val


def translate_agent_type(raw_type: str) -> AgentType:
    return TYPE_MAP.get(str(raw_type).strip().upper(), AgentType.UNKNOWN)


class CarlaSceneSource:
    """Holds the on-disk inputs for one CARLA-recorded scenario.

    The lane graph is chosen from the ENV NAME, not hardcoded:

        carla_town04    -> town04_lanes.json
        carla_town10hd  -> town10hd_lanes.json

    This used to be a class constant pinned to town10hd_lanes.json, so pointing
    SOURCE.TRAIN at 'carla_town04' silently kept loading the Town10HD map --
    the dataset name was a label and nothing more. Observed 2026-09-08.

    The trajectory CSV is per-town when a per-town file exists
    (history_<town>.csv), else the shared history_20hz.csv, so two towns can
    coexist in one data_dir instead of overwriting each other.

    SCENE-TAGGED names (2026-09-11) hold several scenarios per town:

        carla_town10hd__left_turn  -> history_town10hd__left_turn.csv
                                      + town10hd_lanes.json

    A tagged name REQUIRES its exact CSV and never falls back to the town file
    or history_20hz.csv -- a fallback would serve another scenario's recording
    under this scenario's name, which is the silent-wrong-scene failure this
    project has hit repeatedly. Each tagged name also gets its own trajdata
    cache directory (the cache is keyed by env name), so swapping recordings
    cannot leave one scenario reading another's stale cache.
    """

    SCENE_SEP = "__"

    TRAJ_CSV = "history_20hz.csv"          # fallback, shared
    LANES_JSON = None                      # derived from the env name

    # A CARLA autopilot recording sits ON the lanes; measured 0.10 m median for
    # a matched pair. A town/data mismatch reads ~2.4 m median, 20+ m worst.
    ON_MAP_MEDIAN_M = 3.0

    @staticmethod
    def split_env(env_name: str):
        """'carla_town10hd__left_turn' -> ('town10hd', 'left_turn');
        'carla_town04' -> ('town04', None)."""
        n = str(env_name).strip().lower()
        n = n[len("carla_"):] if n.startswith("carla_") else n
        town, _, scene = n.partition(CarlaSceneSource.SCENE_SEP)
        return town, (scene or None)

    @staticmethod
    def town_of(env_name: str) -> str:
        """'carla_town04' -> 'town04'; 'carla_town04__cut_in' -> 'town04'."""
        return CarlaSceneSource.split_env(env_name)[0]

    @classmethod
    def resolve_paths(cls, data_dir, env_name: str = None):
        """(trajectory CSV, lane graph) for an env name.

        The ONE place this rule lives. rollout_carla.py validates its export
        against the CSV this returns, so the dataset and the validator cannot
        read two different recordings -- which is exactly what a hardcoded
        history_20hz.csv in rollout_carla did for every Town04 run.
        """
        data_dir = Path(data_dir)
        town, scene = cls.split_env(env_name) if env_name else (None, None)
        lanes_path = data_dir / (cls.LANES_JSON or
                                 (f"{town}_lanes.json" if town
                                  else "town10hd_lanes.json"))
        if scene:
            csv_path = data_dir / f"history_{town}{cls.SCENE_SEP}{scene}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"no recording for scene {scene!r} in {town}: expected "
                    f"{csv_path}. A scene-tagged source never falls back to "
                    f"history_{town}.csv or {cls.TRAJ_CSV} -- that would serve "
                    f"another scenario's agents under this name. Record it:\n"
                    f"  record_actor_history.py --spawn-xy X Y [X Y ...] "
                    f"--out agent_history_{town}{cls.SCENE_SEP}{scene}.json\n"
                    f"  extract_csv.py --in agent_history_{town}{cls.SCENE_SEP}"
                    f"{scene}.json --out {csv_path.name}")
            return csv_path, lanes_path
        csv_name = cls.TRAJ_CSV
        if town and (data_dir / f"history_{town}.csv").exists():
            csv_name = f"history_{town}.csv"
        return data_dir / csv_name, lanes_path

    def __init__(self, data_dir: Path, env_name: str = None,
                 verbose: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.env_name = env_name

        csv_path, lanes_path = self.resolve_paths(self.data_dir, env_name)
        if verbose:
            print(f"CarlaSceneSource: env={env_name!r} -> {csv_path.name} + "
                  f"{lanes_path.name}", flush=True)

        if not csv_path.exists():
            raise FileNotFoundError(f"trajectory CSV not found: {csv_path}")
        if not lanes_path.exists():
            raise FileNotFoundError(
                f"lane graph not found: {lanes_path}\n"
                "Generate it first with export_lane_graph.py inside carla-ubuntu20.sif."
            )

        self.df = pd.read_csv(csv_path)
        self.df["id"] = self.df["id"].astype(int)
        self.df["frame"] = self.df["frame"].astype(int)

        # Re-base timesteps to 0 so scene_ts always starts at 0, whatever the
        # recording started at.
        self.df["scene_ts"] = self.df["frame"] - self.df["frame"].min()

        with open(lanes_path) as f:
            self.lane_graph = json.load(f)

        self.scene_length = int(self.df["scene_ts"].max()) + 1
        self.num_scenarios = 1

        self._assert_agents_on_map(lanes_path, csv_path)

        if verbose:
            print(
                f"CarlaSceneSource: {self.df['id'].nunique()} agents, "
                f"{self.scene_length} timesteps, "
                f"{len(self.lane_graph['lanes'])} lanes",
                flush=True,
            )


    def _assert_agents_on_map(self, lanes_path, csv_path) -> None:
        """Refuse a trajectory/map pair that does not belong together.

        The two are chosen independently -- the CSV by filename, the map by env
        name -- so nothing else notices when they disagree. A recording made in
        one town and scored against another town's lanes produces agents that
        are simply somewhere else, and every downstream number is then
        meaningless while looking perfectly well-formed.
        """
        pts = np.concatenate(
            [np.asarray(l["center"], dtype=float)[:, :2]
             for l in self.lane_graph["lanes"].values()], axis=0)
        xy = self.df[["x", "y"]].to_numpy()
        step = max(1, len(xy) // 400)              # a sample is plenty
        d = np.array([np.linalg.norm(pts - p, axis=1).min() for p in xy[::step]])
        med = float(np.median(d))
        if med <= self.ON_MAP_MEDIAN_M:
            return
        raise SystemExit(
            "\nTRAJECTORY AND MAP DO NOT MATCH.\n"
            f"  trajectories : {csv_path}\n"
            f"                 x [{xy[:,0].min():.1f}, {xy[:,0].max():.1f}]  "
            f"y [{xy[:,1].min():.1f}, {xy[:,1].max():.1f}]\n"
            f"  lane graph   : {lanes_path}\n"
            f"                 x [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]  "
            f"y [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]\n"
            f"  median distance from an agent to the nearest lane centreline: "
            f"{med:.2f} m (limit {self.ON_MAP_MEDIAN_M})\n"
            "\nA matched CARLA recording sits within ~0.1 m of a lane. This pair\n"
            "is from two different towns: re-record in the town you selected, or\n"
            "point SOURCE.TRAIN back at the town the recording was made in.\n")


class CarlaDataset(RawDataset):
    """Presents a CARLA recording to trajdata as a single-scene dataset."""

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # One scene, always in the "val" split -- ProSim's demo path loads
        # 'val', and there is nothing here to train on.
        dataset_parts: List[Tuple[str]] = [("val",)]
        scene_split_map = defaultdict(partial(const_lambda, const_val="val"))

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=CARLA_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)
        self.dataset_obj = CarlaSceneSource(self.metadata.data_dir,
                                            env_name=self.name, verbose=verbose)

    # ------------------------------------------------------------------
    # Scene enumeration
    # ------------------------------------------------------------------

    def _scene_name(self, idx: int) -> str:
        return f"scene_{idx}"

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[CarlaSceneRecord] = []
        scenes_list: List[SceneMetadata] = []

        for idx in range(self.dataset_obj.num_scenarios):
            scene_name = self._scene_name(idx)
            scene_split = self.metadata.scene_split_map[scene_name]
            scene_length = self.dataset_obj.scene_length

            all_scenes_list.append(
                CarlaSceneRecord(scene_name, str(scene_length), idx)
            )

            if scene_split in scene_tag and scene_desc_contains is None:
                scenes_list.append(
                    SceneMetadata(
                        env_name=self.metadata.name,
                        name=scene_name,
                        dt=self.metadata.dt,
                        raw_data_idx=idx,
                    )
                )

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[CarlaSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[Scene] = []
        for scene_record in all_scenes_list:
            scene_name, scene_length, data_idx = scene_record
            scene_split = self.metadata.scene_split_map[scene_name]

            if scene_split in scene_tag and scene_desc_contains is None:
                scenes_list.append(
                    Scene(
                        self.metadata,
                        scene_name,
                        f"{self.name}_{data_idx}",
                        scene_split,
                        int(scene_length),
                        data_idx,
                        None,  # unused once cached
                    )
                )

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, name, _, data_idx = scene_info
        scene_split = self.metadata.scene_split_map[name]

        return Scene(
            self.metadata,
            name,
            f"{self.name}_{data_idx}",
            scene_split,
            self.dataset_obj.scene_length,
            data_idx,
            None,
        )

    # ------------------------------------------------------------------
    # Agent tracks
    # ------------------------------------------------------------------

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        df = self.dataset_obj.df
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]

        has_z = "z" in df.columns
        has_extent = {"length", "width", "height"}.issubset(df.columns)
        has_accel = {"ax", "ay"}.issubset(df.columns)

        # The lowest-numbered agent becomes "ego". trajdata/ProSim expect one
        # agent under that name; which one is arbitrary for a CARLA recording
        # since every agent was on autopilot.
        ego_id = int(df["id"].min())

        frames: List[pd.DataFrame] = []
        agent_ids_per_row: List[np.ndarray] = []
        accel_blocks: List[np.ndarray] = []

        for agent_id, group in df.groupby("id", sort=True):
            group = group.sort_values("scene_ts")
            raw_type = str(group["type"].iloc[0])
            agent_type = translate_agent_type(raw_type)

            ts = group["scene_ts"].to_numpy()
            first_timestep = int(ts.min())
            last_timestep = int(ts.max())

            if has_extent:
                # Real bounding boxes, recorded from vehicle.bounding_box.extent.
                length = float(group["length"].iloc[0])
                width = float(group["width"].iloc[0])
                height = float(group["height"].iloc[0])
            else:
                length, width, height = DEFAULT_EXTENT.get(
                    raw_type.strip().upper(), DEFAULT_EXTENT["VEHICLE"]
                )

            block = pd.DataFrame(
                {
                    "x": group["x"].to_numpy(dtype=float),
                    "y": group["y"].to_numpy(dtype=float),
                    "z": (
                        group["z"].to_numpy(dtype=float)
                        if has_z
                        else np.zeros(len(group), dtype=float)
                    ),
                    "vx": group["vx"].to_numpy(dtype=float),
                    "vy": group["vy"].to_numpy(dtype=float),
                    # yaw is already radians (record_actor_history.py applies
                    # np.deg2rad to CARLA's degrees).
                    "heading": group["yaw"].to_numpy(dtype=float),
                    "length": (
                        group["length"].to_numpy(dtype=float)
                        if has_extent
                        else np.full(len(group), length)
                    ),
                    "width": (
                        group["width"].to_numpy(dtype=float)
                        if has_extent
                        else np.full(len(group), width)
                    ),
                    "height": (
                        group["height"].to_numpy(dtype=float)
                        if has_extent
                        else np.full(len(group), height)
                    ),
                },
                index=pd.MultiIndex.from_arrays(
                    [
                        np.full(len(group), str(agent_id)),
                        ts.astype(int),
                    ],
                    names=["agent_id", "scene_ts"],
                ),
            )
            frames.append(block)
            agent_ids_per_row.append(np.full(len(group), agent_id))
            if has_accel:
                accel_blocks.append(
                    group[["ax", "ay"]].to_numpy(dtype=float)
                )

            agent_name = "ego" if agent_id == ego_id else str(agent_id)
            agent_info = AgentMetadata(
                name=agent_name,
                agent_type=agent_type,
                first_timestep=first_timestep,
                last_timestep=last_timestep,
                extent=FixedExtent(length=length, width=width, height=height),
            )
            agent_list.append(agent_info)
            for timestep in range(first_timestep, last_timestep + 1):
                agent_presence[timestep].append(agent_info)

        all_agent_data_df = pd.concat(frames)
        flat_agent_ids = np.concatenate(agent_ids_per_row)

        if has_accel:
            # CARLA's own get_acceleration() is more faithful than differencing.
            all_agent_data_df[["ax", "ay"]] = np.concatenate(accel_blocks)
        else:
            # Fall back to finite-differencing velocity, without differencing
            # across the boundary between two different agents' rows.
            all_agent_data_df[["ax", "ay"]] = (
                arr_utils.agent_aware_diff(
                    all_agent_data_df[["vx", "vy"]].to_numpy(), flat_agent_ids
                )
                / CARLA_DT
            )

        all_agent_data_df.rename(index={str(ego_id): "ego"}, level="agent_id", inplace=True)
        all_agent_data_df.sort_index(inplace=True)

        final_cols = [
            "x", "y", "z",
            "vx", "vy",
            "ax", "ay",
            "heading",
            "length", "width", "height",
        ]
        cache_class.save_agent_data(
            all_agent_data_df.loc[:, final_cols], cache_path, scene
        )

        # No traffic-light states are recorded. trajdata expects the file to
        # exist, so write an empty, correctly-shaped frame.
        tls_df = pd.DataFrame(
            {"status": pd.Series(dtype="int64")},
            index=pd.MultiIndex.from_arrays(
                [np.array([], dtype=object), np.array([], dtype="int64")],
                names=["lane_id", "scene_ts"],
            ),
        )
        cache_class.save_traffic_light_data(tls_df, cache_path, scene)

        return agent_list, agent_presence

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------

    def cache_map(
        self,
        data_idx: int,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        map_name = f"{self.name}_{data_idx}"
        map_file = cache_path / f"{self.name}/maps" / f"{map_name}.pb"
        if map_file.exists():
            print(f"Map {map_file} already cached, skipping...", flush=True)
            return

        vector_map = self.build_vector_map(
            self.dataset_obj.lane_graph, f"{self.name}:{map_name}"
        )
        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        for data_idx in range(self.dataset_obj.num_scenarios):
            self.cache_map(data_idx, cache_path, map_cache_class, map_params)

    @staticmethod
    def build_vector_map(lane_graph: Dict[str, Any], map_name: str) -> VectorMap:
        """Turn export_lane_graph.py's JSON into a trajdata VectorMap."""
        vec_map = VectorMap(map_id=map_name)

        max_pt = np.full(3, -np.inf)
        min_pt = np.full(3, np.inf)

        for lane_id, lane in lane_graph["lanes"].items():
            center = np.asarray(lane["center"], dtype=float)
            if center.shape[0] < 2:
                # RoadLane.__post_init__ derives headings by differencing, so
                # a single point would produce a degenerate lane.
                continue

            left = np.asarray(lane["left_edge"], dtype=float)
            right = np.asarray(lane["right_edge"], dtype=float)

            road_lane = RoadLane(
                id=str(lane_id),
                center=Polyline(center),
                left_edge=Polyline(left) if left.shape[0] >= 2 else None,
                right_edge=Polyline(right) if right.shape[0] >= 2 else None,
                adj_lanes_left=set(lane.get("adj_lanes_left", [])),
                adj_lanes_right=set(lane.get("adj_lanes_right", [])),
                next_lanes=set(lane.get("next_lanes", [])),
                prev_lanes=set(lane.get("prev_lanes", [])),
            )
            vec_map.add_map_element(road_lane)

            for arr in (center, left, right):
                if arr.shape[0]:
                    max_pt = np.fmax(max_pt, arr[:, :3].max(axis=0))
                    min_pt = np.fmin(min_pt, arr[:, :3].min(axis=0))

        vec_map.extent = np.concatenate((min_pt, max_pt))
        return vec_map


def register(env_substring: str = "carla") -> None:
    """Teach trajdata about CarlaDataset.

    ProSim's trajdata lives inside a read-only .sif, so get_raw_dataset()
    cannot be edited on disk -- we wrap it at runtime instead. Idempotent.
    """
    from trajdata.utils import env_utils

    if getattr(env_utils.get_raw_dataset, "_carla_patched", False):
        return

    original = env_utils.get_raw_dataset

    def get_raw_dataset(dataset_name: str, data_dir: str):
        if env_substring in dataset_name:
            return CarlaDataset(
                dataset_name, data_dir, parallelizable=False, has_maps=True
            )
        return original(dataset_name, data_dir)

    get_raw_dataset._carla_patched = True
    env_utils.get_raw_dataset = get_raw_dataset
