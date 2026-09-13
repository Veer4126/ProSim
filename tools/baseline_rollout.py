"""Measure ProSim's off-road drift on ANY source, so CARLA can be compared to Waymo.

This is the control experiment. The lane-deviation numbers from a CARLA rollout are
meaningless in isolation -- they only mean something next to the same measurement on a
scene from the checkpoint's own training distribution.

Runs the model closed-loop, converts the rollout to world coordinates, then measures
distance from each predicted position to the nearest lane centreline of THAT source's
own map. Identical metric both sides.

CPU only (prosim_v4.sif ships a CPU-only torch_cluster). Model inference -- run manually.

module load apptainer/1.4.5
apptainer exec -B /scratch/veerk41:/workspace \
    /scratch/veerk41/containers/prosim_v4.sif \
    bash -c "cd /workspace/ProSim && python3 tools/baseline_rollout.py --source waymo_train"

    ... and then --source carla_town10hd, and compare.

INTERPRETATION
  If Waymo drifts about as much as CARLA  -> this is the model's normal 8s closed-loop
      behaviour. The bridge is fine; the horizon is the limit.
  If Waymo stays near the lane and CARLA does not -> something about the CARLA scene is
      out of distribution. Map resolution was one such factor; scene
      composition (stationary fraction) is the next suspect.
"""

# Run from anywhere: put the repo root on the import path and work from it,
# since this script reads prosim_demo/... and demo_dataset/... relatively.
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import argparse
import sys


import numpy as np
import torch
from torch.utils.data import DataLoader

from carla_dataset import register

register()

from prosim.config.default import get_config
from prosim.core.registry import registry
from prosim.dataset.data_utils import rotate


def banner(m):
    print()
    print("=" * 74)
    print(m)
    print("=" * 74, flush=True)


def lane_points_from_batch(batch):
    """All lane centreline points of the map THIS batch carries, as (N,2).

    Taken from batch.vector_maps rather than re-opening the cache via MapAPI, so
    the metric is measured against exactly the map the model was shown.
    (SceneBatch has no .cache attribute, and .map_names is None unless
    incl_raster_map is on -- neither is a usable handle here.)
    """
    vm = batch.vector_maps[0]
    pts = np.concatenate([lane.center.xy for lane in vm.lanes], axis=0)
    return pts, len(vm.lanes), vm.map_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="carla_town10hd",
                    help="carla_town10hd | waymo_train | waymo_val")
    ap.add_argument("--cfg-data", default="prosim_demo/cfg/waymo_demo.yaml")
    ap.add_argument("--cfg-model", default="prosim_demo/cfg/with_text.yaml")
    ap.add_argument("--ckpt", default="prosim_demo/ckpt/prosim_demo_model.ckpt")
    ap.add_argument("--llama", default="./Meta-Llama-3-8B-Instruct-HF")
    ap.add_argument("--n-scenes", type=int, default=3,
                    help="how many samples to roll out and average over")
    ap.add_argument("--on-lane-thresh", type=float, default=3.0,
                    help="only measure agents that START within this many metres of a "
                         "lane. ESSENTIAL for a fair comparison: Waymo ground truth "
                         "contains agents 30 m from any lane (parked in lots, on "
                         "sidewalks), which would otherwise be counted as model drift")
    ap.add_argument("--vehicles-only", action="store_true", default=True,
                    help="restrict to AgentType.VEHICLE (default on)")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="only measure agents whose speed at the last history step is "
                         "at least this (m/s). Removes the PARKED-CAR CONFOUND: 41%% of "
                         "Waymo's on-lane vehicles are stationary and have trivially "
                         "zero drift, which drags its median down. Use 2.0 to compare "
                         "genuinely moving vehicles on both sides. 0 = keep everything")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    banner(f"1. dataset: {args.source}")
    cfg = get_config(args.cfg_data, cluster="local")
    cfg.defrost()
    cfg.PROMPT.CONDITION.TYPES = ["llm_text_OneText"]
    cfg.DATASET.SOURCE.TRAIN = [args.source]
    cfg.freeze()

    ds = registry.get_dataset(cfg.DATASET.TYPE)(cfg, "train")
    n = min(args.n_scenes, len(ds))
    print(f"  samples available: {len(ds)}, using {n}")
    ds._data_index = ds._data_index[:n]
    ds._data_len = n
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=ds.get_collate_fn(), num_workers=0)
    batches = [b for b in dl]

    banner("2. model")
    mcfg = get_config(args.cfg_model, cluster="local")
    mcfg.defrost()
    mcfg.PROMPT.CONDITION.TYPES = ["llm_text_OneText"]
    mcfg.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.MODEL_PATH[
        "LLAMA3_8B_INSTRUCT"] = args.llama
    mcfg.freeze()
    model = registry.get_model(mcfg.MODEL.TYPE).load_from_checkpoint(
        args.ckpt, config=mcfg, strict=False, map_location="cpu")
    model = model.to(args.device).eval()
    print(f"  loaded {args.ckpt}")

    banner(f"3. rollouts on {args.source}")
    all_mean, all_max, per_t = [], [], {}
    drift_max, d0_all = [], []
    kept = skipped_offlane = skipped_type = skipped_slow = 0
    speeds_kept = []
    for i, batch in enumerate(batches):
        batch.to(args.device)
        with torch.no_grad():
            out = model.forward(batch, "val")["motion_pred"]

        centre = batch.centered_agent_state.as_format("x,y,z,h").cpu().numpy()[0]
        cx, cy, _cz, ch = centre
        P, nlanes, mapname = lane_points_from_batch(batch)

        names = list(batch.agent_names[0])
        types = batch.agent_type[0].cpu().numpy()
        type_of = {n: int(types[k]) for k, n in enumerate(names)}

        # Speed at the last history step, straight off the encoder input.
        # init_obs['input'] channels are x,y,s,c,xd,yd,... in the agent's own frame,
        # so speed = hypot(ch4, ch5) at t = -1.
        io = batch.extras["init_obs"]
        inp = io["input"][0].cpu().numpy()
        speed_of = {
            name: float(np.hypot(inp[i, -1, 4], inp[i, -1, 5]))
            for i, name in enumerate(io["agent_ids"][0])
        }

        for rid, data in out["rollout_trajs"].items():
            token = rid.split("-")[1]
            if args.vehicles_only and type_of.get(token, 1) != 1:  # 1 = VEHICLE
                skipped_type += 1
                continue

            spd = speed_of.get(token, float("inf"))
            if spd < args.min_speed:
                skipped_slow += 1
                continue

            traj = data["traj"].cpu().numpy()
            ip = data["init_pos"].cpu().numpy()
            ih = data["init_heading"].cpu().numpy()
            pos_c = rotate(traj[..., 0], traj[..., 1], ih) + ip
            pos_w = rotate(pos_c[..., 0], pos_c[..., 1], ch) + np.array([cx, cy])

            d = np.sqrt(((pos_w[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)

            # An agent that STARTS far from a lane was never on the road -- counting
            # it would measure the dataset, not the model.
            if d[0] > args.on_lane_thresh:
                skipped_offlane += 1
                continue

            kept += 1
            speeds_kept.append(spd)
            d0_all.append(d[0])
            all_mean.append(d.mean())
            all_max.append(d.max())
            drift_max.append((d - d[0]).max())
            for t in range(0, len(d), 10):
                per_t.setdefault(t, []).append(d[t])

        print(f"  scene {i}: {len(out['rollout_trajs'])} rollouts, "
              f"map {mapname} ({nlanes} lanes)")

    print(f"\n  kept {kept} agent-rollouts; skipped {skipped_offlane} that started "
          f">{args.on_lane_thresh} m off-lane, {skipped_type} non-vehicle, "
          f"{skipped_slow} slower than {args.min_speed} m/s")
    if kept == 0:
        raise SystemExit("nothing measurable -- raise --on-lane-thresh")

    banner(f"RESULT -- {args.source}")
    am, ax_, dr, d0 = (np.array(all_mean), np.array(all_max),
                       np.array(drift_max), np.array(d0_all))
    print(f"  agent-rollouts measured : {len(am)}")
    print(f"  start-of-rollout offset : median={np.median(d0):.2f} m  max={d0.max():.2f} m"
          f"   (the on-lane baseline)")
    sk = np.array(speeds_kept)
    print(f"  speed of kept agents    : median={np.median(sk):.2f} m/s  "
          f"p95={np.percentile(sk,95):.2f} m/s  stationary(<0.5)={100*np.mean(sk<0.5):.0f}%")
    print()
    print("  ABSOLUTE distance to nearest lane centreline:")
    print(f"    mean per agent : median={np.median(am):.2f} m  p95={np.percentile(am,95):.2f} m")
    print(f"    max  per agent : median={np.median(ax_):.2f} m  p95={np.percentile(ax_,95):.2f} m  worst={ax_.max():.2f} m")
    print(f"    ever >3 m off  : {np.mean(ax_>3)*100:.0f}%")
    print(f"    ever >6 m off  : {np.mean(ax_>6)*100:.0f}%")
    print()
    print("  DRIFT (max increase over each agent's own t=0 offset) -- the fair metric:")
    print(f"    median={np.median(dr):.2f} m  p95={np.percentile(dr,95):.2f} m  worst={dr.max():.2f} m")
    print()
    print("  deviation vs rollout time:")
    for t in sorted(per_t):
        v = np.array(per_t[t])
        print(f"    t={t*0.1:4.1f}s  median={np.median(v):5.2f} m  p95={np.percentile(v,95):6.2f} m")


if __name__ == "__main__":
    main()
