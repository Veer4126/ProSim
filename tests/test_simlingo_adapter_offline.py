"""SimLingo's adapter fed what sensor_worker.py builds -- no weights, no GPU, no CARLA server.

    CARLA_ROOT=<CARLA dist> /scratch/veerk41/venvs/simlingo/bin/python tests/test_simlingo_adapter_offline.py

Exercises everything around the model call with the real adapter
(third_party/simlingo/scenario_orchestration/policy.py): the harness's own
policy declaration resolved the way the worker resolves it, the rig it
declares, the bonnet crop, the two route target points taken from the worker's
own route, the prompt (two <TARGET_POINT> tokens, image-context budget), and
SimLingo's real PID controller. Every check has a control the same code must
fail. The model forward pass is the one thing not run.
"""

import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import yaml

import sensor_worker as W

HARNESS = Path(_os.environ.get("SCENARIO_ORCHESTRATION_ROOT")
               or "/scratch/veerk41/scenario_orchestration")
POLICY_PY = HARNESS / "third_party/simlingo/scenario_orchestration/policy.py"
BACKBONE = Path(_os.environ.get("SIMLINGO_BACKBONE_DIR")
                or "/scratch/veerk41/av_checkpoints/simlingo/pretrained/InternVL2-1B")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def banner(t):
    print(f"\n=== {t} ===")


def load_adapter():
    spec = importlib.util.spec_from_file_location("_simlingo_policy", POLICY_PY)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(POLICY_PY.parent.parent))
    spec.loader.exec_module(module)
    return module


def main():
    M = load_adapter()

    banner("1. the harness's declaration, resolved the way the worker resolves it")
    declared = yaml.safe_load((HARNESS / "configs/policy/simlingo.yaml").read_text())
    request = {"name": "simlingo", "checkpoint": declared["checkpoint"],
               "parameters": dict(declared.get("parameters") or {})}
    resolved = W.resolve_request_paths(request, POLICY_PY)
    ckpt, weights = Path(resolved["checkpoint"]), Path(resolved["parameters"]["weights"])
    check("checkpoint resolves to a run dir with .hydra/config.yaml",
          (ckpt / ".hydra/config.yaml").is_file(), str(ckpt))
    check("weights resolve to the downloaded pytorch_model.pt",
          weights.is_file() and weights.stat().st_size == 2569679322,
          f"{weights} ({weights.stat().st_size if weights.is_file() else 'missing'} bytes)")
    policy = M.build_policy(resolved)
    check("the adapter takes exactly those paths", policy.root == ckpt and policy.weights == weights)
    check("CONTROL: an unresolved request would point elsewhere",
          M.build_policy(request).root != ckpt, str(M.build_policy(request).root))

    banner("2. the rig it declares, through osc2runner's own rig code")
    rig = W.load_rig_module(W.DEFAULT_OSC2RUNNER)
    specs = rig.specs_from(policy.sensors())
    s = specs[0] if specs else None
    check("one camera, rgb_front 1024x512 fov 110 at (-1.5, 0, 2.0)",
          len(specs) == 1 and type(s).__name__ == "CameraSpec" and s.name == "rgb_front"
          and (s.width, s.height, float(s.fov)) == (1024, 512, 110.0)
          and (float(s.x), float(s.y), float(s.z)) == (-1.5, 0.0, 2.0),
          str([(type(x).__name__, x.name) for x in specs]))

    banner("3. the bonnet crop, upstream's arithmetic")
    frame = np.repeat(np.arange(512, dtype=np.uint16)[:, None], 1024, axis=1)
    frame = np.stack([frame % 256] * 3, axis=-1).astype(np.uint8)
    frame[:, :, 1] = (np.arange(512)[:, None] >= 359).astype(np.uint8) * 255  # mark the band
    cropped = M._crop_bonnet(frame)
    check("512 rows -> 359 kept (512 - (512*4.8)//16)", cropped.shape == (359, 1024, 3), str(cropped.shape))
    check("CONTROL: the kept rows are the top ones and none of the marked band survives",
          int(cropped[:, :, 1].max()) == 0 and int(cropped[-1, 0, 0]) == 358 % 256)

    banner("4. target points from the worker's own route")
    s_line = np.arange(0.0, 60.0, 0.5)
    straight = W.route_in_ego_frame(np.stack([s_line, np.zeros_like(s_line)], 1), 0.0, 0.0, 0.0)
    tp = M._target_points({"route": straight})
    check("straight route -> (9.5, 0) and (21.5, 0): points 7 and 19 of 2.5 m + 1 m steps",
          np.allclose(tp, [(9.5, 0.0), (21.5, 0.0)]), str(np.round(tp, 3).tolist()))
    ang = np.linspace(0, math.pi / 2, 80)
    bend = np.stack([15 * np.sin(ang), 15 * (1 - np.cos(ang))], 1)
    tp_r = M._target_points({"route": W.route_in_ego_frame(bend, 0.0, 0.0, 0.0)})
    tp_l = M._target_points({"route": W.route_in_ego_frame(bend * [1, -1], 0.0, 0.0, 0.0)})
    check("a bend toward world +y puts the far point on the RIGHT (+y)",
          tp_r[1][1] > 1.0 and tp_l[1][1] < -1.0,
          f"right {np.round(tp_r[1], 2).tolist()}, mirrored {np.round(tp_l[1], 2).tolist()}")
    check("CONTROL: no route -> the adapter's straight-ahead defaults",
          M._target_points({"route": []}) == [(8.0, 0.0), (16.0, 0.0)])

    banner("5. the prompt, with the backbone's own template and tokenizer")
    from transformers import AutoConfig, AutoTokenizer
    spec = importlib.util.spec_from_file_location("get_conv_template", BACKBONE / "conversation.py")
    conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
    cfg = AutoConfig.from_pretrained(str(BACKBONE), trust_remote_code=True)
    image_size = cfg.force_image_size or cfg.vision_config.image_size
    num_image_token = int((image_size // cfg.vision_config.patch_size) ** 2 * cfg.downsample_ratio ** 2)
    policy._conv, policy._num_image_token = conv, num_image_token
    prompt = policy._prompt(5.0, 2)
    check("two <TARGET_POINT> tokens, image-context budget = tokens x tiles",
          prompt.count("<TARGET_POINT>") == 2 and prompt.count("<IMG_CONTEXT>") == num_image_token * 2,
          f"{prompt.count('<TARGET_POINT>')} TP, {prompt.count('<IMG_CONTEXT>')} IMG_CONTEXT ({num_image_token} x 2)")
    check("speed, route block and trained task phrasing in the <SAFETY> mode",
          "<SAFETY> Current speed: 5.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. "
          "What should the ego do next?" in prompt)
    policy.mode_token, policy.instruction = "", "Turn left at the junction."
    p2 = policy._prompt(5.0, 2)
    check("CONTROL: empty mode_token drops the prefix; instruction replaces only the task",
          "<SAFETY>" not in p2 and "Current speed: 5.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. "
          "Turn left at the junction." in p2 and "What should the ego do next?" not in p2)
    tok = AutoTokenizer.from_pretrained(str(BACKBONE), trust_remote_code=True)
    before = len(tok.tokenize("<TARGET_POINT>"))
    tok.add_special_tokens({"additional_special_tokens": M.EXTRA_TOKENS})
    after = tok.tokenize("<TARGET_POINT>")
    check("<TARGET_POINT> is ONE token once the adapter's special tokens are added",
          len(after) == 1 and before > 1, f"{before} pieces before, {after} after")

    banner("6. SimLingo's own PID controller")
    import torch
    ctrl = M._Controller()
    c = ctrl._agent.config
    one = int(c.carla_fps // (c.wp_dilation * c.data_save_freq))
    half = one // 2
    idx_gap = (one - 2) - (half - 2)

    def speed_wps(v):                       # desired = |wp[half-2]-wp[one-2]| * 2
        step = v / 2.0 / idx_gap
        return torch.tensor([[[k * step, 0.0] for k in range(11)]], dtype=torch.float32)

    xs = np.arange(1.0, 21.0)
    straight_route = torch.tensor([[[x, 0.0] for x in xs]], dtype=torch.float32)
    curve = lambda sign: torch.tensor([[[x, sign * 0.02 * x * x] for x in xs]], dtype=torch.float32)
    st, th, br = ctrl.step(straight_route, 3.0, speed_wps(6.0))
    check("straight route, 3 m/s wanting 6 m/s: steer ~0, throttle > 0, no brake",
          abs(st) < 0.05 and th > 0.0 and br == 0.0, f"steer {st} throttle {th:.3f} brake {br}")
    ctrl.reset(); sr, _, _ = ctrl.step(curve(+1), 3.0, speed_wps(6.0))
    ctrl.reset(); sl, _, _ = ctrl.step(curve(-1), 3.0, speed_wps(6.0))
    check("a route curving to +y (right) steers right, its mirror steers left",
          sr > 0.0 and sl < 0.0, f"right {sr}, left {sl}")
    ctrl.reset(); _, th0, br0 = ctrl.step(straight_route, 3.0, speed_wps(0.0))
    check("CONTROL: stationary speed waypoints brake", br0 == 1.0 and th0 == 0.0, f"throttle {th0} brake {br0}")
    check("CONTROL: no plan is a stop, not a guess", ctrl.step(None, 3.0, None) == (0.0, 0.0, 1.0))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
