"""Trace a text prompt layer-by-layer through ProSim's LLM conditioning.

Answers "is my prompt actually doing anything?" at five checkpoints, each of
which can fail independently:

  L1  TOKENISATION    does '<A0>' become ONE special token, or five characters?
  L2  GROUNDING       is that token's embedding actually REPLACED by agent 0's
                      prompt embedding? (REPLACE_AGENT_TOKEN)
  L3  ATTENTION       does the <A0> position attend to the instruction words?
                      (the llm call hardcodes output_attentions=False; this
                      script forces it True)
  L4  LLM OUTPUT      does the conditioned agent embedding CHANGE when the
                      prompt changes? This is the decisive test -- if 'turn
                      left' and 'turn right' produce identical embeddings,
                      information is not flowing and nothing downstream can help.
  L5  TRAJECTORY      does the rolled-out path change, and in the right direction?

Runs three conditions through ONE model load: unconditional, prompt A, prompt B.
Using two OPPOSITE prompts is the control -- a prompt that merely perturbs the
output would move both the same way; a prompt that is understood should move
them apart.

CPU only, but it loads Llama-3-8B in fp32 (~32 GB) and runs three rollouts.

    module load apptainer/1.4.5
    apptainer exec -B /scratch/veerk41:/workspace \
        /scratch/veerk41/containers/prosim_v4.sif \
        bash -c "cd /workspace/ProSim && python3 tools/prompt_flow.py"
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

CAPTURE = {}


def _deep_clone(x):
    """CAPTURE holds a mix of tensors and dicts-of-tensors (the 'replace' hook
    records a dict). A blanket .clone() assumes tensors and dies with
    AttributeError: 'dict' object has no attribute 'clone'."""
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _deep_clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_deep_clone(v) for v in x)
    return x


def banner(m):
    print()
    print("=" * 78)
    print(m)
    print("=" * 78, flush=True)


def install_hooks(model):
    """Capture the LLM's inputs, outputs and attention for each forward."""
    ct = model.condition_transformers["policy_decoder"]
    if not hasattr(ct, "text_attn"):
        raise SystemExit("no text_attn on policy_decoder -- is llm_text_OneText "
                         "in PROMPT.CONDITION.TYPES?")
    ta = ct.text_attn

    # 1. capture the assembled LLM input (post agent-token replacement)
    orig_replace = ta._replace_agent_token_with_prompt_emd

    def replace_spy(llm_input, llm_ids, prompt_emd, prompt_nidxs):
        before = llm_input.detach().clone()
        out_input, out_ids = orig_replace(llm_input, llm_ids, prompt_emd, prompt_nidxs)
        CAPTURE.setdefault("replace", []).append({
            "ids": out_ids.detach().cpu().clone(),
            "changed": (out_input - before).abs().sum(-1).detach().cpu().clone(),
            "prompt_emd": prompt_emd.detach().cpu().clone(),
            "input_after": out_input.detach().cpu().clone(),
        })
        return out_input, out_ids

    ta._replace_agent_token_with_prompt_emd = replace_spy

    # 2. force attention output on, and capture it
    orig_llm = ta.llm_model

    class LLMSpy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

        def forward(self, *a, **kw):
            kw["output_attentions"] = True      # the call site hardcodes False
            out = self.inner(*a, **kw)
            if getattr(out, "attentions", None) is not None:
                # mean over heads of the LAST layer: (T, L, L)
                CAPTURE.setdefault("attn", []).append(
                    out.attentions[-1].mean(dim=1).detach().cpu().float().clone())
            return out

    ta.llm_model = LLMSpy(orig_llm)

    # 3. capture the conditioned agent embedding the policy receives
    orig_query = ta._query_prompt_cond_from_llm

    orig_obtain = ta._obtain_llm_input_with_prompt

    def obtain_spy(*a, **kw):
        out = orig_obtain(*a, **kw)
        CAPTURE.setdefault("prompt_lidx", []).append(out[3].detach().cpu().clone())
        return out

    ta._obtain_llm_input_with_prompt = obtain_spy

    def query_spy(t_p_emd, t_bidxs, t_nidxs, t_mask, text_cond):
        out = orig_query(t_p_emd, t_bidxs, t_nidxs, t_mask, text_cond)
        CAPTURE.setdefault("llm_emd", []).append(out.detach().cpu().float().clone())
        CAPTURE.setdefault("p_emd_in", []).append(t_p_emd.detach().cpu().float().clone())
        CAPTURE.setdefault("nidxs", []).append(t_nidxs.detach().cpu().clone())
        return out

    ta._query_prompt_cond_from_llm = query_spy
    return ta


def text_control(batch, text, idxs):
    cond = batch.extras["condition"]["llm_text_OneText"]
    cond["input"] = [text]
    cond["mask"][0] = True
    cond["prompt_mask"][0, :] = False
    for i in idxs:
        cond["prompt_mask"][0, i] = True
    return batch


def clear_control(batch):
    cond = batch.extras["condition"]["llm_text_OneText"]
    cond["input"] = [""]
    cond["mask"][0] = False
    cond["prompt_mask"][0, :] = False
    return batch


def world_traj(output, batch, token):
    """Rollout of one agent in the ego-centred frame -> (T, 2) world xy."""
    from prosim.dataset.data_utils import rotate
    centre = batch.centered_agent_state.as_format("x,y,z,h").cpu().numpy()[0]
    cx, cy, _cz, ch = centre
    for rid, d in output["rollout_trajs"].items():
        if rid.split("-")[1] != token:
            continue
        traj = d["traj"].cpu().detach().numpy()
        ip = d["init_pos"].cpu().detach().numpy()
        ih = d["init_heading"].cpu().detach().numpy()
        pc = rotate(traj[..., 0], traj[..., 1], ih) + ip
        return rotate(pc[..., 0], pc[..., 1], ch) + np.array([cx, cy])
    return None


def signed_area(p):
    """+ve = net LEFT turn, -ve = net RIGHT turn. Cross product of the
    displacement from start against each step, summed."""
    d = p - p[0]
    return float(np.sum(d[:-1, 0] * np.diff(d[:, 1]) - d[:-1, 1] * np.diff(d[:, 0])))


def net_heading_change(p):
    """Total turn in radians from the first to the last motion direction."""
    v0 = p[min(5, len(p) - 1)] - p[0]
    v1 = p[-1] - p[max(0, len(p) - 6)]
    if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(v1) < 1e-6:
        return 0.0
    a = np.arctan2(v0[1], v0[0])
    b = np.arctan2(v1[1], v1[0])
    return float((b - a + np.pi) % (2 * np.pi) - np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-data", default="prosim_demo/cfg/waymo_demo.yaml")
    ap.add_argument("--cfg-model", default="prosim_demo/cfg/with_text.yaml")
    ap.add_argument("--ckpt", default="prosim_demo/ckpt/prosim_demo_model.ckpt")
    ap.add_argument("--llama", default="./Meta-Llama-3-8B-Instruct-HF")
    ap.add_argument("--example-idx", type=int, default=0)
    ap.add_argument("--agent-idx", type=int, default=0,
                    help="index into agent_ids; <A{idx}> is what the prompt references")
    ap.add_argument("--prompt-a", default="Turn left at the intersection, <A0>")
    ap.add_argument("--prompt-b", default="Turn right at the intersection, <A0>")
    args = ap.parse_args()

    # ---------------- L1: tokenisation ----------------
    banner("L1  TOKENISATION")
    from transformers import AutoTokenizer

    from prosim.dataset.text_utils import AGENT_TEMPLATE
    tok = AutoTokenizer.from_pretrained(args.llama, use_fast=False, truncation_side="left")
    tok.add_special_tokens({"additional_special_tokens":
                            [AGENT_TEMPLATE.format(i) for i in range(128)]})
    a2n = {tok.convert_tokens_to_ids(AGENT_TEMPLATE.format(i)): i for i in range(128)}
    for label, text in (("A", args.prompt_a), ("B", args.prompt_b)):
        ids = tok(text)["input_ids"]
        hits = [(i, a2n[t]) for i, t in enumerate(ids) if t in a2n]
        print(f"  prompt {label}: {text!r}")
        print(f"    pieces: {[tok.decode([i]) for i in ids]}")
        print(f"    agent tokens at (pos, nidx): {hits}"
              f"   {'OK' if hits else 'NO AGENT TOKEN -- prompt has no grounding!'}")

    # ---------------- dataset ----------------
    banner("dataset + model")
    cfg = get_config(args.cfg_data, cluster="local")
    cfg.defrost()
    cfg.PROMPT.CONDITION.TYPES = ["llm_text_OneText"]
    cfg.freeze()
    ds = registry.get_dataset(cfg.DATASET.TYPE)(cfg, "train")
    ds._data_index = [ds._data_index[args.example_idx]]
    ds._data_len = 1
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=ds.get_collate_fn(), num_workers=0)
    for batch in dl:
        break
    names = list(batch.extras["prompt"]["motion_pred"]["agent_ids"][0])
    print("  agents:", {i: n for i, n in enumerate(names)})
    token = names[args.agent_idx]
    print(f"  prompt references <A{args.agent_idx}> -> agent {token!r}")

    mcfg = get_config(args.cfg_model, cluster="local")
    mcfg.defrost()
    mcfg.PROMPT.CONDITION.TYPES = ["llm_text_OneText"]
    mcfg.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.MODEL_PATH[
        "LLAMA3_8B_INSTRUCT"] = args.llama
    mcfg.freeze()
    model = registry.get_model(mcfg.MODEL.TYPE).load_from_checkpoint(
        args.ckpt, config=mcfg, strict=False, map_location="cpu").eval()
    ta = install_hooks(model)
    print("  hooks installed on policy_decoder.text_attn")

    # ---------------- three runs ----------------
    runs = {}
    for label, text in (("uncond", None), ("A", args.prompt_a), ("B", args.prompt_b)):
        CAPTURE.clear()
        for b in dl:
            batch = b
            break
        batch.to("cpu")
        if text is None:
            clear_control(batch)
        else:
            text_control(batch, text, [args.agent_idx])

        banner(f"run: {label}" + (f"   {text!r}" if text else "   (no prompt)"))
        with torch.no_grad():
            out = model.forward(batch, "val")["motion_pred"]
        runs[label] = {
            "out": out,
            "cap": {k: [_deep_clone(t) for t in v] for k, v in CAPTURE.items()},
            "traj": world_traj(out, batch, token),
        }
        print(f"  captured: {[f'{k}x{len(v)}' for k, v in CAPTURE.items()]}")

    # ---------------- L2: grounding ----------------
    banner("L2  GROUNDING -- was <A0>'s embedding replaced by the agent embedding?")
    for label in ("A", "B"):
        reps = runs[label]["cap"].get("replace", [])
        if not reps:
            print(f"  {label}: REPLACE HOOK NEVER FIRED "
                  "-> REPLACE_AGENT_TOKEN is off, no grounding")
            continue
        r = reps[0]
        changed = r["changed"]          # (T, L) L1 change per token position
        n_changed = int((changed > 1e-6).sum())
        ids = r["ids"][0].tolist()
        pos = [i for i, t in enumerate(ids) if t in a2n]
        print(f"  {label}: {n_changed} token embedding(s) replaced; "
              f"agent-token positions {pos}")
        if pos:
            amt = {a2n[ids[p]]: float(changed[0, p]) for p in pos}
            # Only the REFERENCED agents should be replaced. Other <An> tokens in
            # the sequence belong to agents the prompt did not mention and are
            # correctly left alone -- demanding that all of them change was wrong.
            ref = [n for n, v in amt.items() if v > 1e-6]
            print(f"      replaced for agent nidx {ref}"
                  f"   {'OK' if ref else 'NOTHING REPLACED'}")
            print(f"      (untouched agent tokens present: "
                  f"{sorted(n for n, v in amt.items() if v <= 1e-6)[:8]} ... "
                  "-- expected, they were not referenced)")

    # ---------------- L3: attention ----------------
    banner("L3  ATTENTION -- what does the <A0> position look at?")
    for label in ("A", "B"):
        attn = runs[label]["cap"].get("attn", [])
        reps = runs[label]["cap"].get("replace", [])
        if not attn or not reps:
            print(f"  {label}: no attention captured")
            continue
        A = attn[0][0]                       # (L, L) last layer, head-mean
        ids = reps[0]["ids"][0].tolist()
        pos = [i for i, t in enumerate(ids) if t in a2n]
        if not pos:
            print(f"  {label}: no agent token to read attention from")
            continue
        # THE READOUT POSITION. With PROMPT_TAIL=True the agent embeddings are
        # APPENDED AFTER the text (`llm_input = cat([text, prompt])`, then
        # `prompt_lidx += L`), so the conditioning is read from the TAIL and can
        # see the whole instruction. The inline <An> token found by scanning ids
        # is a different position -- causally masked, but NOT where the
        # conditioning comes from. Reading attention there was my error.
        plidx = runs[label]["cap"].get("prompt_lidx", [])
        q = int(plidx[0][0, 0]) if plidx else pos[0]
        src = "prompt_lidx (readout)" if plidx else "inline <An> (NOT the readout)"
        print(f"  {label}: reading attention from {src}")
        row = A[q].numpy()
        order = np.argsort(-row)[:8]
        after = float(row[q + 1:].sum()) if q + 1 < len(row) else 0.0
        print(f"  {label}: attention FROM position {q} (<A{a2n[ids[q]]}>), top 8:")
        print(f"      attention mass on tokens AFTER this position: {after:.4f}")
        if after < 1e-6 and q + 1 < len(row):
            print("      *** CAUSAL MASKING: this token cannot see anything after it.")
            print("      *** Any instruction placed AFTER <An> is invisible to it.")
            print("      *** Put the agent token LAST: 'Turn left at the "
                  "intersection, <A0>'")
        for j in order:
            piece = tok.decode([ids[j]]) if ids[j] < len(tok) else f"<id {ids[j]}>"
            print(f"      {row[j]:.4f}  pos {int(j):3d}  {piece!r}")

    # ---------------- L4: LLM output ----------------
    banner("L4  LLM OUTPUT -- does the conditioned embedding depend on the prompt?")
    e = {k: runs[k]["cap"].get("llm_emd", [None])[0] for k in runs}
    p_in = runs["A"]["cap"].get("p_emd_in", [None])[0]
    if e.get("A") is None or e.get("B") is None:
        print("  llm_emd not captured for A and B -- cannot compare")
    else:
        def rel(u, v):
            d = (u - v).norm().item()
            return d, d / max(v.norm().item(), 1e-9)

        dab, rab = rel(e["A"], e["B"])
        print(f"  ||emd_A - emd_B||       = {dab:.4f}   ({100*rab:.2f}% of ||emd_B||)")
        if e.get("uncond") is not None:
            dau, rau = rel(e["A"], e["uncond"])
            print(f"  ||emd_A - emd_uncond||  = {dau:.4f}   ({100*rau:.2f}%)")
        else:
            print("  (uncond not captured: with mask all-False text_attn returns "
                  "early -- 'no valid text' -- so A vs B is the control)")
        if p_in is not None:
            din = (e["A"] - p_in).norm().item()
            print(f"  ||emd_A - prompt_emd_in|| = {din:.4f}"
                  "   (how much the LLM changed the agent embedding at all)")
        print()
        if dab < 1e-6:
            print("  VERDICT: opposite prompts give IDENTICAL embeddings.")
            print("           Information is NOT flowing through the LLM.")
        elif rab < 0.01:
            print("  VERDICT: embeddings differ by <1%. Information flows but is")
            print("           very weak relative to the embedding's magnitude.")
        else:
            print("  VERDICT: the LLM output IS prompt-dependent. If the trajectory")
            print("           does not follow, the bottleneck is downstream (the")
            print("           policy's use of the conditioning), not the LLM.")

    # ---------------- L5: trajectory ----------------
    banner("L5  TRAJECTORY -- does the path change, and in the right direction?")
    base = runs["uncond"]["traj"]
    for label in ("uncond", "A", "B"):
        p = runs[label]["traj"]
        if p is None:
            print(f"  {label}: agent {token!r} not in rollout_trajs")
            continue
        turn = net_heading_change(p)
        area = signed_area(p)
        d = (np.linalg.norm(p - base, axis=1).max() if base is not None else 0.0)
        print(f"  {label:7s} net turn = {np.degrees(turn):+7.1f} deg   "
              f"signed area = {area:+9.1f}   max dev from uncond = {d:6.2f} m")
    print()
    print("  A prompt asking for LEFT should give a POSITIVE net turn / signed area,")
    print("  RIGHT a negative one. If A and B are nearly identical, the policy is")
    print("  ignoring the conditioning even though L4 says it changed.")


if __name__ == "__main__":
    main()
