"""Confirm a CARLA-recorded scene batches correctly through ProSim's dataset stack.

Mirrors stage 1 of prosim_demo/animate_rollout.py (dataset -> DataLoader -> one
batch) and stops before the model, so it is CPU-only and needs no GPU or LLM.

Run from the ProSim directory, inside prosim_v4.sif:

    module load apptainer/1.4.5
    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 load_carla_batch.py"
"""

import sys

# carla_dataset.py lives one level up, alongside the recording.

import numpy as np
import torch
from torch.utils.data import DataLoader

# MUST come before the dataset is constructed: ProSim's trajdata is inside the
# read-only .sif, so get_raw_dataset() is patched at runtime rather than edited.
from carla_dataset import register

register()

from prosim.config.default import get_config
from prosim.core.registry import registry

CFG = "prosim_demo/cfg/waymo_demo.yaml"
SPLIT = "train"  # animate_rollout.py uses 'train', so DATASET.SOURCE.TRAIN is read


def banner(msg):
    print()
    print("=" * 70)
    print(msg)
    print("=" * 70)


banner("1. config")
config = get_config(CFG, cluster="local")
config.defrost()
config.PROMPT.CONDITION.TYPES = ["llm_text_OneText"]
config.freeze()

data_cfg = config.DATASET
print("  SOURCE.%s:      %s" % (SPLIT.upper(), data_cfg.SOURCE[SPLIT.upper()]))
for source in data_cfg.SOURCE[SPLIT.upper()]:
    key = source.upper().replace("-", "_")
    print(f"  DATA_PATHS.{key}: {data_cfg.DATA_PATHS[key]}")
print("  CACHE_PATH:       ", data_cfg.CACHE_PATH)
print("  dt / hist / fut:   %.1f / %.1f / %.1f s" % (
    data_cfg.MOTION.DT,
    data_cfg.MOTION.HISTORY_SEC,
    data_cfg.MOTION.FUTURE_SEC[SPLIT.upper()],
))
print("  needs %.1f s per sample" % (
    data_cfg.MOTION.HISTORY_SEC + data_cfg.MOTION.FUTURE_SEC[SPLIT.upper()]))
print("  CACHE_MAP:        ", data_cfg.CACHE_MAP)
print("  LOAD_VEC_MAP.%s: %s" % (SPLIT.upper(), data_cfg.LOAD_VEC_MAP[SPLIT.upper()]))
print("  USE_EGO_CENTER.%s: %s" % (SPLIT.upper(), data_cfg.USE_EGO_CENTER[SPLIT.upper()]))

banner("2. build ProSim dataset")
dataset = registry.get_dataset(data_cfg.TYPE)(config, SPLIT)
print("  dataset class:  ", type(dataset).__name__)
print("  samples:        ", len(dataset))
assert len(dataset) > 0, "empty dataset -- scene too short, or SOURCE/DATA_PATHS wrong"

banner("3. one batch through the collate fn")
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=dataset.get_collate_fn(),
    num_workers=0,
)
for batch in loader:
    break
print("  batch type:     ", type(batch).__name__)

banner("4. batch contents")
print("  scene_ids:      ", batch.scene_ids)
print("  dt:             ", batch.dt)

hist = batch.agent_hist
print("  num_agents:     ", batch.num_agents.tolist())
print("  agent_names:    ", batch.agent_names)
print("  agent_type:     ", batch.agent_type.tolist())
print("  agent_hist:     ", tuple(hist.shape), hist.dtype, "(B, A, T_hist, F)")
print("  agent_hist_len: ", batch.agent_hist_len.tolist())
print("  agent_fut:      ", tuple(batch.agent_fut.shape), "(B, A, T_fut, F)")
print("  agent_fut_len:  ", batch.agent_fut_len.tolist())
print("  agent_hist_extent (last step):")
print("     ", np.round(batch.agent_hist_extent[0, :, -1].cpu().numpy(), 2).tolist())

centered = batch.centered_agent_state.as_format("x,y,z,h").cpu().numpy()[0]
print("  centered agent (x,y,z,h):", np.round(centered, 2))

# The prompt structure the text conditioning attaches to. BatchPrompt is a
# dict-like wrapper, not a dict -- iterate its keys().
banner("5. prompt + condition slots (what text_control() writes into)")
prompt = batch.extras["prompt"]
for task in prompt.keys():
    pdata = prompt[task]
    names = pdata["agent_ids"][0]
    print(f"  prompt['{task}']: {len(names)} prompt agents -> {list(names)}")
    for k, v in pdata.items():
        if hasattr(v, "shape"):
            print(f"      {k}: {tuple(v.shape)}")

cond = batch.extras["condition"]
print("  condition types:", list(cond.keys()))
for ctype in cond.keys():
    cdata = cond[ctype]
    shapes = {k: tuple(v.shape) for k, v in cdata.items() if hasattr(v, "shape")}
    print(f"    {ctype}: {shapes}")

banner("6. vector map")
vmaps = getattr(batch, "vector_maps", None)
if vmaps:
    vec_map = vmaps[0]
    print("  vector_maps[0]: ", vec_map.map_id, f"({len(vec_map.lanes)} lanes)")
    from prosim.demo.vis import extract_lane_vecs

    vis_vecs = extract_lane_vecs(vec_map, centered, 150)
    print("  extract_lane_vecs keys:", list(vis_vecs.keys()))
    for k, v in vis_vecs.items():
        print(f"    {k}: {np.asarray(v).shape}")
    # Frame sanity: the ego must sit on the road it is driving on.
    lane = vec_map.get_closest_lane(np.array([centered[0], centered[1], 0.0]))
    d = np.linalg.norm(lane.center.xy - centered[:2], axis=1).min()
    print(f"  closest lane to centred agent: {lane.id} at {d:.2f} m")
    assert d < 3.0, "centred agent is far from every lane -- coordinate frame mismatch"
else:
    print("  NO vector map on the batch.")
    print("  -> DATASET.LOAD_VEC_MAP.%s is %s; animate_rollout.py needs it True."
          % (SPLIT.upper(), data_cfg.LOAD_VEC_MAP[SPLIT.upper()]))

banner("BATCHING OK")
print("A CARLA-recorded scene now flows through ProSim's dataset stack.")
print("Next: load the model and run the rollout (GPU + Llama-3 required).")
