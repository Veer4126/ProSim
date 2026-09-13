"""Compare the INPUT features ProSim receives from a CARLA scene vs a real Waymo scene.

CPU-only, no model, no GPU. Builds one batch from each source through ProSim's own
dataset stack and compares the tensors the network actually consumes.

The point: if a feature channel is out of distribution (wrong units, wrong scale,
degenerate), the Waymo-trained checkpoint will produce garbage on CARLA data no matter
how correct the coordinate frames are. Lane-distance checks cannot see this; only a
side-by-side of the encoder inputs can.

    module load apptainer/1.4.5
    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 tools/compare_inputs.py"
"""

# Run from anywhere: put the repo root on the import path and work from it,
# since this script reads prosim_demo/... and demo_dataset/... relatively.
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import sys


import numpy as np
import torch
from torch.utils.data import DataLoader

from carla_dataset import register

register()

from prosim.config.default import get_config
from prosim.core.registry import registry

# From waymo_demo.yaml: DATASET.FORMAT.HISTORY.ELEMENTS
HIST_CHANNELS = "x,y,s,c,xd,yd,xdd,ydd".split(",")


def build_batch(source, n_batches=8):
    cfg = get_config("prosim_demo/cfg/waymo_demo.yaml", cluster="local")
    cfg.defrost()
    cfg.PROMPT.CONDITION.TYPES = ["llm_text_OneText"]
    cfg.DATASET.SOURCE.TRAIN = [source]
    cfg.freeze()

    ds = registry.get_dataset(cfg.DATASET.TYPE)(cfg, "train")
    n = min(n_batches, len(ds))
    ds._data_index = ds._data_index[:n]
    ds._data_len = n
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=ds.get_collate_fn(), num_workers=0)
    return [b for b in dl], len(ds)


def summarise(batches, label):
    print()
    print("=" * 78)
    print(f"{label}")
    print("=" * 78)

    hists, futs, nagents = [], [], []
    for b in batches:
        h = b.agent_hist[0].cpu().numpy()          # (A, T, F)
        f = b.agent_fut[0].cpu().numpy()
        nagents.append(int(b.num_agents[0]))
        hists.append(h)
        futs.append(f)

    H = np.concatenate([h.reshape(-1, h.shape[-1]) for h in hists], axis=0)
    F = np.concatenate([f.reshape(-1, f.shape[-1]) for f in futs], axis=0)
    print(f"agents/scene: {nagents}")
    print(f"agent_hist stacked: {H.shape}   agent_fut stacked: {F.shape}")

    print()
    print(f"{'ch':>3} {'name':<6} {'min':>10} {'p05':>10} {'median':>10} "
          f"{'p95':>10} {'max':>10} {'nan%':>6}")
    for i in range(H.shape[-1]):
        col = H[:, i]
        name = HIST_CHANNELS[i] if i < len(HIST_CHANNELS) else f"f{i}"
        finite = col[np.isfinite(col)]
        nanpct = 100.0 * (1 - len(finite) / max(len(col), 1))
        if len(finite) == 0:
            print(f"{i:>3} {name:<6} {'ALL NAN':>10}")
            continue
        print(f"{i:>3} {name:<6} {finite.min():>10.3f} "
              f"{np.percentile(finite,5):>10.3f} {np.median(finite):>10.3f} "
              f"{np.percentile(finite,95):>10.3f} {finite.max():>10.3f} {nanpct:>6.1f}")

    # Speed, the most interpretable physical quantity.
    if H.shape[-1] >= 6:
        spd = np.hypot(H[:, 4], H[:, 5])
        spd = spd[np.isfinite(spd)]
        print()
        print(f"speed |(xd,yd)|: median={np.median(spd):.2f}  "
              f"p95={np.percentile(spd,95):.2f}  max={spd.max():.2f}  "
              f"frac>0.5={np.mean(spd>0.5):.2f}   (STATIONARY frac={np.mean(spd<0.1):.2f})")
    if H.shape[-1] >= 8:
        acc = np.hypot(H[:, 6], H[:, 7])
        acc = acc[np.isfinite(acc)]
        print(f"accel |(xdd,ydd)|: median={np.median(acc):.2f}  "
              f"p95={np.percentile(acc,95):.2f}  max={acc.max():.2f}")

    return H, F, nagents


def main():
    print("Building CARLA batch...")
    carla_b, carla_n = build_batch("carla_town10hd")
    print("Building Waymo batch...")
    waymo_b, waymo_n = build_batch("waymo_train")

    print(f"\nsamples available -- carla: {carla_n}   waymo: {waymo_n}")

    HC, FC, nc = summarise(carla_b, "CARLA (carla_town10hd)")
    HW, FW, nw = summarise(waymo_b, "WAYMO (waymo_train) -- the training distribution")

    print()
    print("=" * 78)
    print("SIDE BY SIDE: median / p95 per history channel")
    print("=" * 78)
    print(f"{'ch':>3} {'name':<6} {'CARLA med':>12} {'WAYMO med':>12} "
          f"{'CARLA p95':>12} {'WAYMO p95':>12}  flag")
    for i in range(min(HC.shape[-1], HW.shape[-1])):
        name = HIST_CHANNELS[i] if i < len(HIST_CHANNELS) else f"f{i}"
        c = HC[:, i][np.isfinite(HC[:, i])]
        w = HW[:, i][np.isfinite(HW[:, i])]
        if len(c) == 0 or len(w) == 0:
            continue
        cm, wm = np.median(c), np.median(w)
        cp, wp = np.percentile(c, 95), np.percentile(w, 95)
        # flag order-of-magnitude disagreement in spread
        flag = ""
        if wp != 0 and abs(cp) > 1e-6:
            ratio = abs(cp) / max(abs(wp), 1e-6)
            if ratio > 3 or ratio < 1 / 3:
                flag = f"<-- p95 differs {ratio:.1f}x"
        print(f"{i:>3} {name:<6} {cm:>12.3f} {wm:>12.3f} {cp:>12.3f} {wp:>12.3f}  {flag}")

    print()
    print("agents per scene -- CARLA:", nc)
    print("agents per scene -- WAYMO:", nw)


if __name__ == "__main__":
    main()
