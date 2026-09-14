"""Run a ProSim rollout on a CARLA-recorded scene and export a CARLA-replayable CSV.

Covers roadmap steps 1-3:
  1. load the CARLA scene through ProSim's dataset stack and the checkpoint
  2. optionally inject a free-text prompt (text_control)
  3. export the rollout to a CSV in CARLA's WORLD frame, with the full column
     set generate_video.py / wayla.py expect

Based on prosim_demo/animate_rollout.py.

Runs on CPU -- prosim_v4.sif's torch_cluster is a CPU-only build, so the model
cannot run on GPU in this container (see --device). Needs the Llama-3 weights.
Expect it to be slow; run it yourself, it is not launched unattended.

    module load apptainer/1.4.5
    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 rollout_carla.py --help"

Unconditional baseline:
module load apptainer/1.4.5
apptainer exec -B /scratch/veerk41:/workspace \
    /scratch/veerk41/containers/prosim_v4.sif \
    bash -c "cd /workspace/ProSim && python3 rollout_carla.py \
        --out outputs/rollout_uncond.csv"

Prompted:
module load apptainer/1.4.5
apptainer exec -B /scratch/veerk41:/workspace \
    /scratch/veerk41/containers/prosim_v4.sif \
    bash -c "cd /workspace/ProSim && python3 rollout_carla.py \
        --prompt 'The <A0> slows down and stops before the intersection.' \
        --text-agents 0 \
        --out outputs/rollout_prompted.csv"

--------------------------------------------------------------------------
COORDINATE FRAME -- THE IMPORTANT PART
--------------------------------------------------------------------------
`output['rollout_trajs'][id]['init_pos'] / ['init_heading']` are in the
EGO-CENTRED frame, NOT world. Verified against the source CSV: at scene_ts=10
the ego sits at init_pos (0, 0) with heading 0 while its true world pose is
(-64.06, 24.42, -0.033), and agent '25' sits at (-2.37, 3.47) which is exactly
the world delta (-2.256, 3.549) rotated by -h_ego.

animate_rollout.py's `export_wayla_csv` does `rotate(...) + init_pos` and stops
there, so it writes CENTRED coordinates into a file CARLA then reads as WORLD.
That is a real bug for the round trip -- it silently offsets and rotates the
whole scene. This script applies the centred -> world transform explicitly and
then ASSERTS the result against the source CSV at t=0 before writing anything.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# MUST precede dataset construction -- ProSim's trajdata is in a read-only .sif.
from carla_dataset import register

register()

from prosim.config.default import get_config
from prosim.core.registry import registry
from prosim.dataset.data_utils import rotate

SRC_CSV = "/scratch/veerk41/history_20hz.csv"
DT = 0.1

CSV_COLUMNS = [
    "frame", "time",
    "x", "y", "z",
    "vx", "vy",
    "ax", "ay",
    "yaw",
    "length", "width", "height",
    "id", "type",
]

AGENT_TYPE_NAMES = {0: "UNKNOWN", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "BICYCLE", 4: "MOTORCYCLE"}


def text_control(batch, text_input, text_idxs):
    """Inlined from animate_rollout.py:221 -- importing that module would execute
    its top-level script body (dataset build, model load, rendering)."""
    controlled = []
    for idx in text_idxs:
        name = batch.extras["prompt"]["motion_pred"]["agent_ids"][0][idx]
        if name not in controlled:
            controlled.append(name)

    cond = batch.extras["condition"]["llm_text_OneText"]
    cond["input"] = [text_input]
    cond["mask"][0] = True
    cond["prompt_mask"][0, :] = False
    for idx in text_idxs:
        cond["prompt_mask"][0, idx] = True

    return batch, controlled


def check_cache_is_fresh(agent_ids_list, cache_path, source_name="carla_town10hd"):
    """Fail loudly if ProSim served a cached scene that predates the CSV.

    ProSimDataset never sets rebuild_cache, so trajdata reuses whatever sits in
    DATASET.CACHE_PATH forever. Re-recording with a different agent count writes
    a new history_20hz.csv but does NOT invalidate that cache -- the rollout
    then silently runs on the OLD scene.

    The cache key is derived from UnifiedDataset's arguments, not from the CSV's
    contents, so nothing downstream notices on its own.
    """
    src = pd.read_csv(SRC_CSV)
    csv_ids = set(src["id"].astype(int).unique())
    ego_id = min(csv_ids)
    expected = {"ego"} | {str(i) for i in csv_ids - {ego_id}}
    got = set(agent_ids_list)
    if got == expected:
        return

    cache_path = Path(cache_path)
    raise SystemExit(
        "\nSTALE CACHE: the batch does not match the source CSV.\n"
        f"  {SRC_CSV}\n"
        f"    {len(expected)} agents: {sorted(expected)}\n"
        f"  batch served by ProSim\n"
        f"    {len(got)} agents: {sorted(got)}\n"
        "\nProSim reuses its trajdata cache unconditionally. Delete the CARLA\n"
        "entries and re-run (leave the waymo_* directories alone):\n"
        f"    rm -rf {cache_path / source_name}\n"
        f"    rm -rf {cache_path / 'data_indexes'}\n"
    )


def banner(msg):
    print()
    print("=" * 70)
    print(msg)
    print("=" * 70, flush=True)


def centred_to_world(xy_centred, heading_centred, centre_xyzh):
    """Map ego-centred poses back into CARLA's world frame.

    centre_xyzh is batch.centered_agent_state as (x, y, z, h) in world.
    """
    cx, cy, _cz, ch = centre_xyzh
    world_xy = rotate(xy_centred[..., 0], xy_centred[..., 1], ch) + np.array([cx, cy])
    return world_xy, heading_centred + ch


def export_rollout_csv(output, batch, out_path, src_csv=SRC_CSV, dt=DT):
    """Write the rollout to a CARLA-frame CSV matching extract_csv.py's schema.

    Unlike animate_rollout.py's export_wayla_csv this (a) converts to world
    coordinates, (b) keeps the ego, and (c) carries real extents and types.
    """
    src = pd.read_csv(src_csv)
    ego_id = int(src["id"].min())

    centre = batch.centered_agent_state.as_format("x,y,z,h").cpu().numpy()[0]
    scene_ts = int(batch.scene_ts[0])
    print(f"  centre (world x,y,z,h): {np.round(centre, 3)}   scene_ts={scene_ts}")

    # Per-agent extents and types straight off the batch.
    names = list(batch.agent_names[0])
    extents = batch.agent_hist_extent[0, :, -1].cpu().numpy()
    types = batch.agent_type[0].cpu().numpy()
    meta = {
        name: (extents[i], AGENT_TYPE_NAMES.get(int(types[i]), "VEHICLE"))
        for i, name in enumerate(names)
    }

    rows = []
    checks = []
    for rollout_id, data in output["rollout_trajs"].items():
        agent_token = rollout_id.split("-")[1]

        traj = data["traj"].cpu().detach().numpy()
        init_pos = data["init_pos"].cpu().detach().numpy()
        init_heading = data["init_heading"].cpu().detach().numpy()

        # local -> ego-centred (what animate_rollout.py stops at)
        pos_centred = rotate(traj[..., 0], traj[..., 1], init_heading) + init_pos
        head_centred = np.arctan2(traj[..., 2], traj[..., 3]) + init_heading
        head_centred = np.asarray(head_centred).reshape(-1)

        # ego-centred -> CARLA world
        pos_world, head_world = centred_to_world(pos_centred, head_centred, centre)

        agent_id = ego_id if agent_token == "ego" else int(agent_token)
        ext, atype = meta.get(agent_token, (np.array([4.5, 2.0, 1.5]), "VEHICLE"))

        # Self-check: rollout t=0 must coincide with the source CSV at scene_ts.
        src_row = src[(src["id"] == agent_id) & (src["frame"] == scene_ts)]
        if len(src_row):
            err = math.hypot(
                pos_world[0, 0] - float(src_row["x"].iloc[0]),
                pos_world[0, 1] - float(src_row["y"].iloc[0]),
            )
            checks.append((agent_token, err))

        # z is not modelled by ProSim; reuse the recorded ground height.
        z = float(src_row["z"].iloc[0]) if len(src_row) and "z" in src.columns else 0.0

        vel = np.zeros_like(pos_world)
        vel[1:] = (pos_world[1:] - pos_world[:-1]) / dt
        acc = np.zeros_like(pos_world)
        acc[1:] = (vel[1:] - vel[:-1]) / dt

        for t in range(len(pos_world)):
            rows.append([
                t, round(t * dt, 3),
                float(pos_world[t, 0]), float(pos_world[t, 1]), z,
                float(vel[t, 0]), float(vel[t, 1]),
                float(acc[t, 0]), float(acc[t, 1]),
                float(head_world[t]),
                float(ext[0]), float(ext[1]), float(ext[2]),
                agent_id, atype,
            ])

    banner("frame check: rollout t=0 vs source CSV at scene_ts")
    worst = 0.0
    for token, err in sorted(checks):
        print(f"  agent {token:>5}: |rollout[0] - csv[{scene_ts}]| = {err:.4f} m")
        worst = max(worst, err)
    print(f"  worst: {worst:.4f} m")
    assert checks, "no agents matched the source CSV -- id mapping is wrong"
    assert worst < 1.0, (
        f"rollout start is {worst:.2f} m from the recorded pose -- the centred->world "
        "transform is wrong; do NOT feed this CSV to CARLA"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r[0], r[13]))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)

    n_agents = len({r[13] for r in rows})
    n_steps = len(rows) // max(n_agents, 1)
    print(f"\n  wrote {out_path}: {n_agents} agents x {n_steps} steps = {len(rows)} rows")
    print(f"  set EXPECTED_FRAMES = {n_steps} in record_scenario_new.py")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg-data", default="prosim_demo/cfg/waymo_demo.yaml")
    ap.add_argument("--cfg-model", default="prosim_demo/cfg/with_text.yaml")
    ap.add_argument("--ckpt", default="prosim_demo/ckpt/prosim_demo_model.ckpt")
    ap.add_argument("--llama", default="./Meta-Llama-3-8B-Instruct-HF")
    ap.add_argument("--split", default="train",
                    help="animate_rollout.py uses 'train', so DATASET.SOURCE.TRAIN is read")
    ap.add_argument("--example-idx", type=int, default=0,
                    help="which sample (= which rollout start time) to use")
    ap.add_argument("--prompt", default="",
                    help="free text; refer to agents as <A0>, <A1>, ... (empty = unconditional)")
    ap.add_argument("--text-agents", type=int, nargs="*", default=[],
                    help="indices into agent_ids that the prompt refers to")
    ap.add_argument("--out", default="outputs/rollout_carla.csv")
    # DEFAULT IS CPU AND THAT IS DELIBERATE. prosim_v4.sif ships
    # torch_cluster 1.6.3+pt24cpu -- a CPU-only wheel (only *_cpu.so, no
    # _knn_cuda.so). ProSim's scene fusion calls knn_graph, so with CUDA
    # tensors it dies with "RuntimeError: Not compiled with CUDA support",
    # even though torch itself is a CUDA build (2.4.0+cu121). This is why
    # animate_rollout.py loads map_location='cpu' and never moves the model.
    # Only pass --device cuda in a container with a CUDA torch_cluster.
    ap.add_argument("--list-agents", action="store_true",
                    help="print every agent in this scene -- index, name, world "
                         "position, speed, and the goal points available from the "
                         "lane graph -- then EXIT before loading the model. "
                         "Use this to choose --goal-agent / --text-agents / --goal-xy.")
    ap.add_argument("--goal", action="append", nargs=3, type=float, default=None,
                    metavar=("IDX", "X", "Y"),
                    help="goal for ONE agent: its index plus a world point. "
                         "Repeatable, so several agents can be conditioned in "
                         "the same run: --goal 1 -300 35 --goal 2 -280 30. "
                         "Combine freely with --goal-agent/--goal-xy, which "
                         "remain the single-goal shorthand.")
    ap.add_argument("--goal-id", action="append", nargs=3, type=float, default=None,
                    metavar=("ID", "X", "Y"),
                    help="goal for ONE agent named by its recorded CARLA actor id "
                         "(the CSV 'id' column) rather than its batch index. "
                         "Repeatable. Prefer this for anything scripted: the batch "
                         "order is NOT spawn order (a 6-agent batch came back "
                         "['ego','25','27','26',...]), so an index can silently "
                         "hand one actor another's goal.")
    ap.add_argument("--goal-agent", type=int, default=None,
                    help="agent INDEX (same convention as --text-agents) to give a "
                         "GOAL condition to. Structured alternative to --prompt: no "
                         "LLM involved, and unlike text it IS populated for CARLA scenes.")
    ap.add_argument("--goal-turn", default=None,
                    choices=["left", "right", "straight"],
                    help="pick the goal off the lane graph: the left/right/straight "
                         "branch out of the agent's current lane")
    ap.add_argument("--goal-xy", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="explicit goal in CARLA WORLD coordinates (overrides --goal-turn)")
    ap.add_argument("--goal-lookahead", type=float, default=45.0,
                    help="how far along the chosen branch to place the goal, metres")
    ap.add_argument("--ego-agent", default=None,
                    help="agent name to drive by RULE instead of by ProSim "
                         "(e.g. 'ego'). Omit to let ProSim drive everything.")
    ap.add_argument("--ego-policy", default="idm_pursuit",
                    choices=["idm", "idm_pursuit", "idm_mobil"],
                    help="idm: straight-line IDM (cannot turn). "
                         "idm_pursuit: IDM + pure pursuit on the lane graph. "
                         "idm_mobil: the above + MOBIL lane changes.")
    ap.add_argument("--ego-external", default=None, metavar="POLICY_PY",
                    help="drive the rule ego from an external ego_policy_v1 "
                         "repository's scenario_orchestration/policy.py "
                         "(e.g. third_party/idm). Needs --ego-policy-request. "
                         "Overrides --ego-policy/--ego-v0/--ego-idm, which "
                         "configure THIS repo's native ego instead.")
    ap.add_argument("--ego-bev-town", default=None, metavar="TOWN",
                    help="town whose pre-rendered BEV raster to feed an external "
                         "policy that reads one (PlanT 2.0), e.g. Town04 or "
                         "Town10HD_Opt. The rasters ship with the policy's own "
                         "repository; no CARLA is involved.")
    ap.add_argument("--ego-remote", default=None, metavar="HOST:PORT",
                    help="drive the ego from a SENSOR policy running inside a "
                         "CARLA world, through sensor_worker.py listening at "
                         "HOST:PORT. --ego-external names the policy.py the "
                         "worker loads; the ego's pose each step is the one "
                         "CARLA's physics produced.")
    ap.add_argument("--ego-remote-town", default=None, metavar="TOWN",
                    help="map the worker's CARLA server must have loaded "
                         "(e.g. Town04, Town10HD_Opt)")
    ap.add_argument("--ego-light", default=None, choices=["green", "yellow", "red"],
                    help="with --ego-remote: set and freeze the traffic light "
                         "governing the ego's approach (and its junction group) "
                         "to this phase at episode start")
    ap.add_argument("--ego-frames-dir", default=None, metavar="DIR",
                    help="where the worker saves the camera strip the policy "
                         "saw, one JPEG per decision")
    ap.add_argument("--ego-policy-request", default=None, metavar="JSON",
                    help="the harness's policy.json, passed VERBATIM to the "
                         "external repository's build_policy()")
    ap.add_argument("--ego-goal-xy", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="give the RULE EGO a destination in CARLA world coords. "
                         "Junctions are then resolved toward it instead of by "
                         "'stay straightest', which cannot tell a junction's "
                         "branches apart (they all leave the entrance at the "
                         "same heading). Unrelated to --goal-xy, which conditions "
                         "the MODEL.")
    ap.add_argument("--ego-v0", type=float, default=10.0,
                    help="IDM desired free-road speed, m/s")
    ap.add_argument("--replan-freq", type=int, default=None, metavar="N",
                    help="how often the closed-loop rollout RE-OBSERVES the "
                         "scene, in steps of DT (0.1 s). The config ships 10 = "
                         "1.0 s; agents drive open-loop in between. Lower "
                         "values need the patched full_traj_xy sizing in "
                         "format_utils.py. MEASURED: 1 costs ~2-2.7x, not 10x "
                         "(Llama runs once, before the loop), but it also "
                         "roughly DOUBLES agent jerk (median 1.37 -> 2.39, max "
                         "5.93 -> 18.42 m/s^2) because the policy predicts "
                         "TARGET.STEPS=10 steps and only the first is executed, "
                         "which the checkpoint was not trained for. Use 1 for "
                         "coupling diagnostics only, NOT for CARLA playback. "
                         "Default: leave the config value alone.")
    ap.add_argument("--data-dir",
                    default=os.environ.get("PROSIM_DATA_DIR")
                    or str(Path(__file__).resolve().parent / "carla_data"),
                    help="directory holding the CARLA recordings and lane "
                         "graphs (history_<town>__<scene>.csv, <town>_lanes.json); "
                         "default: PROSIM_DATA_DIR, else this repo's carla_data/")
    ap.add_argument("--source", default=None, metavar="NAME",
                    help="trajdata source to roll out, e.g. carla_town04 or "
                         "carla_town10hd. Overrides DATASET.SOURCE.<SPLIT> from "
                         "--cfg-data. Town04 is the highway, Town10HD the "
                         "intersections; the yaml default is whatever was last "
                         "edited, which is exactly how the silent "
                         "town-switch bug happened. Name the town explicitly.")
    ap.add_argument("--ego-idm", action="append", default=None, metavar="K=V",
                    help="IDM parameter for the RULE ego, repeatable: T=1.5 "
                         "s0=2.0 a_max=1.5 b=2.0 delta=4.0. Unknown keys are "
                         "refused rather than silently dropped -- make_policy "
                         "filters kwargs by IDM.__dataclass_fields__, so a typo "
                         "would otherwise vanish without a trace.")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; required by this container's CPU-only "
                         "torch_cluster) or cuda")
    args = ap.parse_args()

    banner("1. dataset")
    config = get_config(args.cfg_data, cluster="local")
    config.defrost()

    # --source: name the town rather than inheriting whatever the yaml was last
    # left at. Applied to the split actually read (animate_rollout uses 'train').
    if args.source:
        split_key = args.split.upper()
        if split_key not in config.DATASET.SOURCE:
            raise SystemExit(f"--split {args.split!r} has no DATASET.SOURCE entry")
        config.DATASET.SOURCE[split_key] = [args.source]
        print(f"  --source: DATASET.SOURCE.{split_key} = ['{args.source}']")

    # Resolve the trajectory CSV the DATASET will read, by the dataset's own
    # rule (CarlaSceneSource.resolve_paths), and validate the export against
    # that same file. This was a hardcoded history_20hz.csv, so every run on a
    # source whose recording lives elsewhere (any carla_town04 run, any
    # scene-tagged source) checked its agents against the wrong recording.
    global SRC_CSV
    _src_name = config.DATASET.SOURCE[args.split.upper()][0]
    if "carla" in _src_name:
        from carla_dataset import CarlaSceneSource
        _key = _src_name.upper().replace("-", "_")   # exactly basic.py:115
        # Every CARLA source (plain or scene-tagged) reads from --data-dir, so
        # ProSim's path_cfg.py needs no CARLA entries.
        config.DATASET.DATA_PATHS[_key] = str(args.data_dir)
        try:
            _csv, _lanes = CarlaSceneSource.resolve_paths(
                config.DATASET.DATA_PATHS[_key], _src_name)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))
        SRC_CSV = str(_csv)
        print(f"  source {_src_name!r} -> {_csv.name} + {_lanes.name}")

    # --ego-idm K=V -> kwargs for make_policy's IDM. Validated HERE, before the
    # model loads, because make_policy silently drops keys it does not recognise.
    ego_idm_kwargs = {}
    if args.ego_idm:
        from ego_control import IDM as _IDM
        allowed = set(_IDM.__dataclass_fields__)
        for item in args.ego_idm:
            if "=" not in item:
                raise SystemExit(f"--ego-idm expects K=V, got {item!r}")
            k, _, v = item.partition("=")
            k = k.strip()
            if k not in allowed:
                raise SystemExit(
                    f"--ego-idm {k!r} is not an IDM parameter. "
                    f"Known: {sorted(allowed)}")
            try:
                ego_idm_kwargs[k] = float(v)
            except ValueError:
                raise SystemExit(f"--ego-idm {k}={v!r} is not a number")
        print(f"  --ego-idm: {ego_idm_kwargs}")
    # animate_rollout.py hardcodes text-only ("skip goal/v_action_tag/drag_point
    # entirely"). Keep that default, but include 'goal' when asked for -- goal is
    # the only structured channel actually populated and finite for a CARLA scene
    # (drag_point is all-NaN, llm_text has 0 cached samples).
    cond_types = ["llm_text_OneText"]
    # BOTH goal spellings must switch the channel on. --goal-agent/--goal-xy is
    # the one-goal shorthand; --goal IDX X Y is the repeatable form added with
    # multi-goal support -- and this test was not updated then, so
    # a --goal-only run built its goal_specs, found no 'goal' condition on the
    # batch, and died at set_goal_condition after a full model load. Every
    # multi-goal run and every harness cell takes the --goal path.
    if args.goal_agent is not None or args.goal or args.goal_id:
        cond_types = ["goal"] + cond_types
    config.PROMPT.CONDITION.TYPES = cond_types

    # --replan-freq: REPLAN_FREQ IS PINNED FOR THIS CHECKPOINT. Measured
    # by trying to set it to 1 and reading the traceback.
    #
    # Three quantities are locked together:
    #   H     = FUTURE_SEC/DT                       = 80   (the horizon)
    #   T     = len(all_t_indices) = ceil(H/SAMPLE_RATE)
    #   STEPS = DATASET.FORMAT.TARGET.STEPS         = 10
    #
    #   format_utils.py:629  full_traj_xy = zeros([B, N, STEPS*T, 2])
    #   format_utils.py:633  full_traj    = agent_fut[...][:, :STEPS*T]
    # so STEPS*T must equal H, i.e. T = H/STEPS = 8, i.e. SAMPLE_RATE = 10.
    #   traj_sam.py appends REPLAN_FREQ steps per index, so frames = T*REPLAN_FREQ
    # must also equal H, i.e. REPLAN_FREQ = H/T = 10 = STEPS.
    #
    # And STEPS is baked into the TRAINED WEIGHTS: act_decoder.py:29 sets
    # self.step = TARGET.STEPS and the motion head emits [b, K, step, dim].
    #
    # Setting SAMPLE_RATE=1 (what this flag used to do) raises
    #   RuntimeError: shape mismatch: value tensor of shape [3, 80, 2] cannot be
    #   broadcast to indexing result of shape [3, 800, 2]
    # and setting REPLAN_FREQ alone silently shortens the rollout to H/STEPS
    # frames
    _steps = int(config.DATASET.FORMAT.TARGET.STEPS)
    if args.replan_freq is not None:
        if args.replan_freq < 1:
            raise SystemExit("--replan-freq must be >= 1")
        config.DATASET.FORMAT.TARGET.SAMPLE_RATE = args.replan_freq
        if args.replan_freq != _steps:
            print(f"  NOTE: replan {args.replan_freq} != TARGET.STEPS {_steps}. "
                  "This relies on the PATCHED full_traj_xy sizing in\n"
                  "        format_utils.py (backup: format_utils.py.bak.20260910). "
                  "The policy still\n        PREDICTS "
                  f"{_steps} steps per decision and only the first "
                  f"{args.replan_freq} are executed -- receding horizon.")
    config.freeze()

    dataset = registry.get_dataset(config.DATASET.TYPE)(config, args.split)
    print(f"  samples available: {len(dataset)}")
    if not 0 <= args.example_idx < len(dataset):
        raise SystemExit(f"--example-idx must be in [0, {len(dataset)})")

    dataset._data_index = [dataset._data_index[args.example_idx]]
    dataset._data_len = 1
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=dataset.get_collate_fn(), num_workers=0)
    for batch in loader:
        break

    agent_ids_list = list(batch.extras["prompt"]["motion_pred"]["agent_ids"][0])
    print(f"  example_idx {args.example_idx} -> scene_ts {int(batch.scene_ts[0])}")
    print("  agents:", {i: n for i, n in enumerate(agent_ids_list)})

    check_cache_is_fresh(agent_ids_list, config.DATASET.CACHE_PATH,
                         config.DATASET.SOURCE[args.split.upper()][0])

    # animate_rollout.py runs everything on CPU, so it never needs this. Moving
    # the model without the batch gives:
    #   RuntimeError: Expected all tensors to be on the same device,
    #                 but found at least two devices, cuda:0 and cpu!
    # SceneBatch.to() walks extras and calls each value's __to__ (VecLanes,
    # BatchPrompt, InputMaskData, BatchCondition, ...), so this is sufficient.
    # Done BEFORE text_control so the prompt writes land on device tensors.
    batch.to(args.device)
    print(f"  batch moved to {args.device}")

    if args.device.startswith("cuda"):
        import torch_cluster
        if "cpu" in torch_cluster.__version__:
            raise SystemExit(
                f"torch_cluster is a CPU-only build ({torch_cluster.__version__}); "
                "ProSim's knn_graph will fail with 'Not compiled with CUDA support'. "
                "Re-run with --device cpu.")

    if args.list_agents:
        banner(f"agents in example-idx {args.example_idx} (scene_ts "
               f"{int(batch.scene_ts[0])})")
        from goal_control import (agent_world_pose, side_name,
                                  turn_goal_from_lane_graph, turn_name,
                                  world_to_body)
        from prosim_ego import VecMapLaneGraph

        lg = (VecMapLaneGraph(batch.vector_maps[0])
              if getattr(batch, "vector_maps", None) else None)
        iop = batch.extras["io_pairs_batch"]
        inp = batch.extras["init_obs"]["input"][0].detach().cpu().numpy()
        obs_names = list(batch.extras["init_obs"]["agent_ids"][0])

        print("  Each row is ONE agent. 'world x/y' is where it is now, in CARLA")
        print("  map coordinates -- the same numbers --goal-xy takes.")
        print()
        print("  The goal options are junction branches leading out of that agent's")
        print("  current lane. Each is printed as:")
        print("      <what the branch actually does>  [map x, map y]  (F+xx R+yy)")
        print("  where [map x, map y] is the goal point in CARLA world coordinates")
        print("  (copy this straight into --goal-xy), and F/R describe the same")
        print("  point relative to THAT AGENT, i.e. seen from its driving seat:")
        print("      F = metres straight AHEAD of it, along its current heading")
        print("      R = metres to its RIGHT (negative R = to its left)")
        print("  F and R are only there as a sanity check -- F must be positive or")
        print("  the agent would have to reverse. They are NOT what you pass in.")
        print()
        print("  The label is the branch's OWN turn angle, so 'straight' really")
        print("  means straight. A junction with no straight exit simply has no")
        print("  'straight' option listed.")
        print()
        print(f"  {'idx':>3}  {'name':>6}  {'world x':>9} {'world y':>9}  "
              f"{'hdg':>6}  {'spd':>5}   goal options")
        print("  " + "-" * 104)
        for i, nm in enumerate(agent_ids_list):
            xy, h = agent_world_pose(batch, i)
            # speed comes from init_obs, which uses a DIFFERENT ordering
            spd = float("nan")
            if nm in obs_names:
                k = obs_names.index(nm)
                spd = float(np.hypot(inp[k, -1, 4], inp[k, -1, 5]))

            # De-duplicate by the GOAL POINT: asking for three directions at a
            # junction that offers two branches must not print three options.
            opts, seen = [], {}
            if lg is not None:
                for d in ("left", "straight", "right"):
                    out = turn_goal_from_lane_graph(lg, xy, h, d)
                    if out is None:
                        continue
                    gx, gy = float(out[0][0]), float(out[0][1])
                    key = (round(gx, 2), round(gy, 2))
                    if key in seen:
                        continue
                    seen[key] = True
                    b = world_to_body(out[0], xy, h)
                    # label by what the branch ACTUALLY does, not by what was asked
                    opts.append(f"{turn_name(out[1]):>8} [{gx:7.1f},{gy:7.1f}] "
                                f"(F{b[0]:+.0f} R{b[1]:+.0f})")
            opt_s = "   ".join(opts) if opts else "none (no junction ahead)"
            print(f"  {i:>3}  {nm:>6}  {xy[0]:9.2f} {xy[1]:9.2f}  {h:6.2f}  "
                  f"{spd:5.1f}   {opt_s}")
        print()
        print("  !! Each row's [x,y] is a goal for THAT ROW'S AGENT ONLY.")
        print("     Using one agent's coordinates with another --goal-agent")
        print("     usually puts the goal BEHIND it, which does nothing.")

        print()
        print("  Pick an agent with a moving speed and >1 goal option, then either:")
        print("    --goal-agent <idx> --goal-turn left          (uses the lane graph)")
        print("    --goal-agent <idx> --goal-xy <X> <Y>         (the [x,y] shown above)")
        print("  --goal-turn only works if that direction is actually listed for")
        print("  the agent; a junction with no left exit has no 'left' option, and")
        print("  asking for one is an error rather than a silent nearest match.")
        print("  A goal behind the agent (F negative) is refused explicitly.")
        print("  --text-agents uses the SAME index; --ego-agent uses the NAME.")
        raise SystemExit(0)

    banner("2. model")
    mcfg = get_config(args.cfg_model, cluster="local")
    mcfg.defrost()
    mcfg.PROMPT.CONDITION.TYPES = cond_types
    mcfg.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.MODEL_PATH[
        "LLAMA3_8B_INSTRUCT"] = args.llama
    _replan_cfg = int(mcfg.ROLLOUT.POLICY.REPLAN_FREQ)
    if args.replan_freq is not None:
        if args.replan_freq < 1:
            raise SystemExit("--replan-freq must be >= 1 (steps of DT)")
        mcfg.ROLLOUT.POLICY.REPLAN_FREQ = args.replan_freq
    mcfg.freeze()
    _dt = float(config.DATASET.MOTION.DT)
    _replan = int(mcfg.ROLLOUT.POLICY.REPLAN_FREQ)
    print(f"  REPLAN_FREQ: {_replan} steps = {_replan * _dt:.1f} s between "
          f"re-observations" + ("" if args.replan_freq is None
                                else f"   (config default was {_replan_cfg})"))
    if _replan > 1:
        # Report the blind distance for THIS scene, not a hardcoded speed.
        # The old message quoted 30 m/s -- the Town04 highway --dego-v0 -- which
        # is 10x wrong for a town scene and read as if it described the run.
        try:
            _inp = batch.extras["init_obs"]["input"][0].detach().cpu().numpy()
            _spd = np.hypot(_inp[:, -1, 4], _inp[:, -1, 5])
            _vmax = float(np.nanmax(_spd))
            print(f"    agents drive {_replan * _dt:.1f} s open-loop between "
                  f"decisions -- the fastest agent here is {_vmax:.1f} m/s, "
                  f"so {_vmax * _replan * _dt:.1f} m blind")
            if args.ego_agent and args.ego_v0 > _vmax:
                print(f"    (the RULE EGO may reach --ego-v0 {args.ego_v0:.1f} m/s, "
                      f"i.e. up to {args.ego_v0 * _replan * _dt:.0f} m blind)")
        except Exception:
            print(f"    agents drive {_replan * _dt:.1f} s open-loop between "
                  f"decisions")
    if args.replan_freq is not None and args.replan_freq < _replan_cfg:
        print(f"    NOTE: the checkpoint was trained/evaluated at "
              f"{_replan_cfg}; behaviour at {_replan} is not something this "
              "project has characterised. Compare against a {_replan_cfg} run."
              .replace("{_replan_cfg}", str(_replan_cfg)))

    model_cls = registry.get_model(mcfg.MODEL.TYPE)
    if args.ego_agent:
        from prosim_ego import make_rule_ego_class
        model_cls = make_rule_ego_class(model_cls)

    model = model_cls.load_from_checkpoint(
        args.ckpt, config=mcfg, strict=False, map_location="cpu")
    model = model.to(args.device).eval()
    print(f"  loaded {args.ckpt} on {args.device}")

    _remote_ego = None
    if args.ego_remote and not (args.ego_agent and args.ego_external):
        raise SystemExit("--ego-remote needs --ego-agent and --ego-external")
    if args.ego_agent:
        if args.ego_agent not in agent_ids_list:
            raise SystemExit(
                f"--ego-agent {args.ego_agent!r} not in this scene. "
                f"Available: {agent_ids_list}")
        vec_map = batch.vector_maps[0] if getattr(batch, "vector_maps", None) else None
        if vec_map is None and (args.ego_external or args.ego_policy != "idm"):
            raise SystemExit(
                f"--ego-policy {args.ego_policy} needs the lane graph, but the batch "
                "carries no vector_maps. Use --ego-policy idm, or enable the map.")
        # The centre pose is NOT optional when a map is used: a_traj is in the
        # ego-centred frame and the VectorMap is in world coordinates.
        centre = batch.centered_agent_state.as_format("x,y,z,h").cpu().numpy()[0]
        if args.ego_external:
            # An external ego_policy_v1 policy, loaded and driven through
            # external_ego.py. idm_mobil must come this way: the harness
            # declares it NOT natively realizable, so running our own MOBIL
            # under that name would publish someone else's behaviour.
            import json as _json
            from ego_control import LaneGraphRoute
            from external_ego import make_external_ego
            from prosim_ego import VecMapLaneGraph as _VMLG
            if not args.ego_policy_request:
                raise SystemExit("--ego-external needs --ego-policy-request")
            with open(args.ego_policy_request) as _fh:
                _preq = _json.load(_fh)
            _lg = _VMLG(vec_map)
            if args.ego_remote:
                from remote_ego import RemoteSensorEgoPolicy
                if not args.ego_remote_town:
                    raise SystemExit("--ego-remote needs --ego-remote-town")
                _ext = RemoteSensorEgoPolicy(
                    args.ego_remote, LaneGraphRoute(_lg, goal=args.ego_goal_xy),
                    lane_graph=_lg, policy_py=args.ego_external,
                    policy_request_path=args.ego_policy_request,
                    town=args.ego_remote_town, dt=DT,
                    frames_dir=args.ego_frames_dir, ego_light=args.ego_light)
                _remote_ego = _ext
            else:
                _ext = make_external_ego(args.ego_external, _preq,
                                         LaneGraphRoute(_lg, goal=args.ego_goal_xy),
                                         lane_graph=_lg, dt=DT,
                                         bev_town=args.ego_bev_town)
            model.set_ego_policy(args.ego_agent, policy=_ext, vec_map=vec_map,
                                 centre=(centre[0], centre[1], centre[3]))
            print(f"  RULE EGO: {args.ego_agent!r} driven by EXTERNAL policy "
                  f"{_preq.get('name')!r} ({_preq.get('implementation')}) "
                  f"from {args.ego_external}")
            print(f"    parameters passed verbatim: {_preq.get('parameters')}")
        else:
            model.set_ego_policy(args.ego_agent, vec_map=vec_map,
                                 kind=args.ego_policy, v0=args.ego_v0,
                                 centre=(centre[0], centre[1], centre[3]),
                                 route_goal=args.ego_goal_xy,
                                 **ego_idm_kwargs)
            print(f"  RULE EGO: {args.ego_agent!r} driven by {args.ego_policy} "
                  f"(v0={args.ego_v0} m/s); ProSim still drives the other "
                  f"{len(agent_ids_list)-1} agents")
        print(f"    centred frame origin (world): ({centre[0]:.2f}, {centre[1]:.2f}) "
              f"h={centre[3]:+.4f} rad")
        if args.ego_goal_xy:
            print(f"    route goal: {tuple(args.ego_goal_xy)} -- junctions are "
                  "resolved toward this point")
        else:
            print("    no --ego-goal-xy: at a junction the route keeps as straight "
                  "as the map allows")

    banner("3. prompt")
    if args.prompt:
        idxs = args.text_agents or [agent_ids_list.index("ego")]
        for i in idxs:
            if not 0 <= i < len(agent_ids_list):
                raise SystemExit(
                    f"--text-agents index {i} out of range [0, {len(agent_ids_list)})")
        batch, controlled = text_control(batch, args.prompt, idxs)
        print(f"  prompt: {args.prompt!r}")
        print(f"  applied to indices {idxs} -> agents {controlled}")
    else:
        controlled = []
        print("  UNCONDITIONAL (no text) -- baseline run")
        print("  condition mask left as collated:",
              batch.extras["condition"]["llm_text_OneText"]["mask"].tolist())


    # Collect every requested goal. --goal-agent/--goal-xy is the one-goal
    # shorthand; --goal IDX X Y is repeatable.
    goal_specs = []
    if args.goal_agent is not None:
        goal_specs.append((args.goal_agent, args.goal_xy, args.goal_turn))
    for entry in (args.goal or []):
        idx_f, gx, gy = entry
        if abs(idx_f - round(idx_f)) > 1e-9:
            raise SystemExit(
                f"--goal takes an INTEGER agent index first; got {idx_f}. "
                "Usage: --goal IDX X Y")
        goal_specs.append((int(round(idx_f)), [gx, gy], None))
    if args.goal_id:
        _ids = set(pd.read_csv(SRC_CSV)["id"].astype(int))
        for aid_f, gx, gy in args.goal_id:
            aid = int(round(aid_f))
            if aid not in _ids:
                raise SystemExit(
                    f"--goal-id {aid}: no such actor in {SRC_CSV} "
                    f"(recorded ids: {sorted(_ids)})")
            name = "ego" if aid == min(_ids) else str(aid)
            if name not in list(agent_ids_list):
                raise SystemExit(
                    f"--goal-id {aid} ({name!r}) is not in this batch: "
                    f"{list(agent_ids_list)}")
            idx = list(agent_ids_list).index(name)
            print(f"  --goal-id {aid} -> batch agent {name!r} at index {idx}")
            goal_specs.append((idx, [gx, gy], None))

    if goal_specs:
        from goal_control import (agent_world_pose, set_goal_condition,
                                  side_name, turn_goal_from_lane_graph,
                                  turn_name)
        from prosim_ego import VecMapLaneGraph
        import json as _json

        src_csv_df = pd.read_csv(SRC_CSV)
        _ego_id = int(src_csv_df["id"].min())
        seen, goal_records = {}, []

        for n, (gi, gxy, gturn) in enumerate(goal_specs):
            if not 0 <= gi < len(agent_ids_list):
                raise SystemExit(
                    f"goal agent index {gi} out of range "
                    f"[0, {len(agent_ids_list)}) -- run --list-agents")
            if gi in seen:
                raise SystemExit(
                    f"agent index {gi} ({agent_ids_list[gi]!r}) was given two "
                    "goals. Each agent has ONE goal slot; the second would "
                    "overwrite the first.")
            seen[gi] = True

            axy, ah = agent_world_pose(batch, gi)
            if gxy is not None:
                goal_world, turn, n_opts = np.asarray(gxy, dtype=float), None, None
            else:
                if not getattr(batch, "vector_maps", None):
                    raise SystemExit(
                        "--goal-turn needs the lane graph; no vector_maps on batch")
                lg = VecMapLaneGraph(batch.vector_maps[0])
                out = turn_goal_from_lane_graph(lg, axy, ah, gturn or "straight",
                                                lookahead=args.goal_lookahead)
                if out is None:
                    raise SystemExit(
                        f"no {gturn!r} branch from agent {agent_ids_list[gi]!r}'s "
                        "lane -- try --goal-xy or another --example-idx")
                goal_world, turn, n_opts = out

            # exclusive=True CLEARS every other agent's goal mask, so it may
            # only be used on the FIRST goal. Passing it again would silently
            # erase the goals set before it -- the whole point of this loop.
            info = set_goal_condition(batch, gi, goal_world, exclusive=(n == 0))

            print(f"  GOAL {n + 1}/{len(goal_specs)}: agent "
                  f"{agent_ids_list[gi]!r} (index {gi}) at world "
                  f"{np.round(info['agent_world_xy'], 2)} "
                  f"heading {info['agent_heading']:.3f}")
            print(f"        goal world {np.round(info['goal_world'], 2)}  ->  body "
                  f"{np.round(info['goal_body'], 2)} (+x fwd, +y lateral)")
            if turn is not None:
                actual = turn_name(turn)
                print(f"        chosen branch actually goes {actual.upper()} "
                      f"({np.degrees(turn) * -1:+.1f} deg, +ve = left), "
                      f"{n_opts} option(s) at the junction")
                if n_opts < 2:
                    print("        WARNING: only one branch -- the direction "
                          "had no effect")
                if actual != gturn:
                    print(f"        WARNING: you asked for {str(gturn).upper()} but "
                          f"no branch at this junction goes that way; the "
                          f"closest\n                 available one goes "
                          f"{actual.upper()}. Use --list-agents.")
            side = side_name(info["goal_body"][1])
            print(f"        goal is {abs(info['goal_body'][1]):.1f} m to the {side}, "
                  f"{info['goal_body'][0]:.1f} m ahead")

            _tok = agent_ids_list[gi]
            goal_records.append({
                "goal_world": [float(info["goal_world"][0]),
                               float(info["goal_world"][1])],
                "goal_body": [float(info["goal_body"][0]),
                              float(info["goal_body"][1])],
                "agent_index": gi,
                "agent_token": _tok,
                "agent_csv_id": _ego_id if _tok == "ego" else int(_tok),
                "agent_world_xy": [float(info["agent_world_xy"][0]),
                                   float(info["agent_world_xy"][1])],
                "agent_heading": float(info["agent_heading"]),
                "side": side,
                "turn_deg": (float(np.degrees(turn)) if turn is not None else None),
                "n_options": n_opts,
            })

        n_on = int(batch.extras["condition"]["goal"]["mask"][0].sum())
        print(f"  {len(goal_records)} goal(s) requested; "
              f"{n_on} condition slot(s) enabled on the batch")
        if n_on != len(goal_records):
            raise SystemExit(
                f"expected {len(goal_records)} enabled goal slots but the batch "
                f"has {n_on}. exclusive=True on a later goal would do this.")

        # Sidecar. The first goal is ALSO written at the top level so readers
        # that predate multi-goal keep working unchanged.
        goal_meta = dict(goal_records[0])
        goal_meta["goals"] = goal_records
        goal_meta["scene_ts"] = int(batch.scene_ts[0])
        goal_meta["example_idx"] = args.example_idx
        goal_path = Path(args.out).with_suffix(".goal.json")
        goal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(goal_path, "w") as f:
            _json.dump(goal_meta, f, indent=2)
        print(f"        wrote {goal_path}  (annotate_frames.py --goal-json)")

    banner("4. rollout")
    try:
        with torch.no_grad():
            output = model.forward(batch, "val")["motion_pred"]
    finally:
        # Hand the CARLA world back (actors destroyed, async restored) even
        # when the rollout dies -- a world left synchronous wedges the server.
        if _remote_ego is not None:
            _remote_ego.close()
    print(f"  rollout_trajs: {len(output['rollout_trajs'])} agents")
    dbg = getattr(model, "ego_debug", None)
    if dbg:
        sp = [d["speed"] for d in dbg]
        lanes = [d["lane"] for d in dbg if d["lane"] is not None]
        print(f"  rule ego: {len(dbg)} control steps, "
              f"speed {min(sp):.1f}-{max(sp):.1f} m/s, "
              f"{len(set(lanes))} distinct lanes traversed")

    banner("5. export")
    _horizon = int(round(float(config.DATASET.MOTION.FUTURE_SEC["TRAIN"]) / DT))
    _n_out = min(len(d["traj"]) for d in output["rollout_trajs"].values())
    print(f"  rollout produced {_n_out} steps; the configured horizon is "
          f"{_horizon} ({_horizon * DT:.1f} s)")
    if _n_out != _horizon:
        raise SystemExit(
            f"\nROLLOUT IS THE WRONG LENGTH: {_n_out} steps, expected "
            f"{_horizon}.\n"
            f"  frames = len(all_t_indices) x REPLAN_FREQ\n"
            f"  REPLAN_FREQ                     = "
            f"{int(mcfg.ROLLOUT.POLICY.REPLAN_FREQ)}\n"
            f"  DATASET.FORMAT.TARGET.SAMPLE_RATE = "
            f"{int(config.DATASET.FORMAT.TARGET.SAMPLE_RATE)}\n"
            "\nThese two must match. Set both with --replan-freq (which now "
            "does), or\nleave both at the config default. Exporting a short "
            "CSV here is how a\n0.8 s rollout ends up spliced onto stale "
            "frames in the video.\n")
    export_rollout_csv(output, batch, args.out, src_csv=SRC_CSV)

    # Run manifest, written for EVERY run. Two rollouts are only comparable if
    # they share --example-idx: it selects the START TIME, so a different value
    # moves every agent's t=0 and any A/B difference then measures the offset,
    # not the condition. Observed 2026-09-04: rollout_goal.csv (example-idx ->
    # source frame 19) was compared against rollout_unprompted.csv (frame 11),
    # and every agent already differed by 2-4.5 m at t=0.
    import json as _json
    meta_path = Path(args.out).with_suffix(".meta.json")
    # The rule ego's intended route, for the evaluation harness's
    # reference path. None when no rule ego was used.
    _ego_route_polyline = None
    if getattr(model, 'ego_route_polyline', None) is not None:
        _p = model.ego_route_polyline()
        if _p is not None:
            _ego_route_polyline = [[round(float(a), 3), round(float(b), 3)]
                                   for a, b in _p]
            print(f'  ego route: {len(_ego_route_polyline)} points'
                  f' over {len(model.ego_route_lanes)} lanes')

    with open(meta_path, "w") as f:
        _json.dump({
            "out": args.out,
            "example_idx": args.example_idx,
            "scene_ts": int(batch.scene_ts[0]),
            "src_csv": SRC_CSV,
            "agents": list(agent_ids_list),
            "prompt": args.prompt,
            "text_agents": args.text_agents,
            "goal_agent": args.goal_agent,
            "goal_xy": args.goal_xy,
            "goal_turn": args.goal_turn,
            "ego_agent": args.ego_agent,
            "ego_policy": args.ego_policy,
            "ego_goal_xy": args.ego_goal_xy,
            "ego_v0": args.ego_v0,
            "ego_idm": ego_idm_kwargs,
            "ego_external": args.ego_external,
            "ego_remote": args.ego_remote,
            "ego_remote_town": args.ego_remote_town,
            "ego_light": args.ego_light,
            "ego_remote_session": (_remote_ego.metadata()
                                   if _remote_ego is not None else None),
            "ego_route_polyline": _ego_route_polyline,
            "source": args.source,
            "replan_freq": int(mcfg.ROLLOUT.POLICY.REPLAN_FREQ),
            "replan_seconds": round(int(mcfg.ROLLOUT.POLICY.REPLAN_FREQ) * DT, 3),
        }, f, indent=2)
    print(f"  wrote {meta_path}")
    print(f"  COMPARE ONLY against runs with the same example_idx "
          f"({args.example_idx}) and the same source CSV.")

    banner("DONE")
    print("Replay it with generate_video.py (point it at this CSV), then")
    print("record_scenario_new.py with EXPECTED_FRAMES set as printed above.")


if __name__ == "__main__":
    main()
