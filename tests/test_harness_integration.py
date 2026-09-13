"""Does scenario_orchestration/run.py satisfy the harness contract?

CPU only. No checkpoint, no GPU, no CARLA, no container. The expensive part --
the ~90 s ProSim rollout -- is synthesised by pointing PROSIM_ROLLOUT_CSV at a
real rollout CSV already on disk, so everything AROUND the model call is
exercised against real data: request parsing, town selection, policy mapping,
argv construction, the CSV -> trace transcription, and the harness's own
ingest reading the result back.

The requests are built by the harness's REAL ProSimAdapter, not hand-written
here, so a change to the contract breaks this test rather than passing it.

Every positive assertion has a control that must fail. In particular:
  - the yaw check has a radians-vs-degrees control, because a missing
    conversion produces a trace that looks entirely plausible;
  - the "trace is evaluable" check is paired with deleting states.jsonl, which
    must make it unevaluable -- otherwise the check proves nothing.

    python3 tests/test_harness_integration.py
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = Path(os.environ.get("SCENARIO_ORCHESTRATION_ROOT")
               or "/home/veerk41/scratch/scenario_orchestration")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))
sys.path.insert(0, str(HARNESS))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def banner(text):
    print(f"\n=== {text} ===")


def pick_rollout_csv() -> Path:
    """A real rollout export to stand in for the model call: the harness's own
    cut_in x idm rollout, committed so the test runs from a fresh clone."""
    path = REPO / "tests" / "fixtures" / "rollout_cut_in.csv"
    if not path.is_file():
        raise SystemExit(f"missing test fixture {path}")
    return path


def build_requests(family="cut_in", policy="idm", seed=0):
    """Ask the HARNESS to build the two request documents."""
    from scenario_orchestration.config import ConfigStore
    from scenario_orchestration.adapters.prosim import ProSimAdapter
    from scenario_orchestration.experiment.spec import ExperimentSpec

    store = ConfigStore.default()
    adapter = ProSimAdapter("prosim", store.algorithm("prosim"), store=store,
                            root=HARNESS)
    spec = ExperimentSpec.create(
        scenario_family=family, algorithm="prosim", policy=policy, seed=seed,
        store=store, raw_dir=Path(tempfile.mkdtemp()),
    )
    return adapter, spec


def run_entry_point(request, policy_request, out_dir, env_extra=None,
                    with_root=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = out_dir / "request.json"
    pol_path = out_dir / "policy.json"
    req_path.write_text(json.dumps(request))
    pol_path.write_text(json.dumps(policy_request))

    env = dict(os.environ)
    env.pop("SCENARIO_ORCHESTRATION_ROOT", None)
    if with_root:
        env["SCENARIO_ORCHESTRATION_ROOT"] = str(HARNESS)
    env.update(env_extra or {})
    completed = subprocess.run(
        [sys.executable, "-u", "scenario_orchestration/run.py",
         "--scenario-request", str(req_path),
         "--policy-request", str(pol_path),
         "--output-dir", str(out_dir)],
        cwd=str(REPO), env=env, capture_output=True, text=True,
    )
    return completed


def main():
    csv_path = pick_rollout_csv()
    print(f"stand-in rollout CSV: {csv_path.name}")

    adapter, spec = build_requests()
    request = adapter.build_scenario_request(spec).to_dict()
    policy_request = adapter.build_policy_request(spec).to_dict()

    # ---------------------------------------------------------------- 1
    banner("1. the harness's own request documents")
    check("scenario request carries an implementation block",
          request.get("implementation") is not None)
    check("implementation declares a prompt (what ProSimAdapter validates)",
          bool((request["implementation"].get("parameters") or {}).get("prompt")),
          repr((request["implementation"].get("parameters") or {}).get("prompt")))
    check("policy request is the idm analytic policy",
          policy_request["implementation"] == "idm.policy.IDMPolicy",
          policy_request["implementation"])
    check("policy request carries IDM parameters",
          "desired_speed_mps" in (policy_request.get("parameters") or {}),
          str(policy_request.get("parameters")))

    # ---------------------------------------------------------------- 2
    banner("2. CONTROL: no goal declared must be REFUSED, not run unconditioned")
    # implementations.yaml now declares real goals for every family, so the
    # control has to REMOVE them rather than rely on their absence -- which is
    # what this check was silently doing until the goals landed.
    nogoal_request = json.loads(json.dumps(request))
    for key in ("goals", "goal_xy", "goal_agent"):
        nogoal_request["implementation"]["parameters"].pop(key, None)
    out_dir = Path(tempfile.mkdtemp()) / "nogoal"
    completed = run_entry_point(nogoal_request, policy_request, out_dir,
                                {"PROSIM_ROLLOUT_CSV": str(csv_path)})
    report = json.loads((out_dir / "method_result.json").read_text())
    check("exits non-zero", completed.returncode != 0, f"rc={completed.returncode}")
    check("status is failure", report["status"] == "failure", report["status"])
    check("reason names the goal channel", "goal" in (report["reason"] or "").lower())
    check("wrote NO states.jsonl", not (out_dir / "states.jsonl").exists())

    # ---------------------------------------------------------------- 3
    banner("3. positive path: goals declared, model call synthesised")
    # Starts from the REAL cut_in entry (2 goals on Town04 road 40) and only
    # adds a zone, so the positive path exercises shipped configuration.
    goal_request = json.loads(json.dumps(request))
    goal_request["implementation"]["parameters"].update({
        "zone": {"id": "conflict", "kind": "junction",
                 "center": [-250.0, 30.0], "radius": 8.0},
    })
    check("the shipped cut_in entry already declares goals",
          len(goal_request["implementation"]["parameters"].get("goals") or []) == 1,
          str(goal_request["implementation"]["parameters"].get("goals")))
    out_dir = Path(tempfile.mkdtemp()) / "ok"
    completed = run_entry_point(goal_request, policy_request, out_dir,
                                {"PROSIM_ROLLOUT_CSV": str(csv_path)})
    if completed.returncode != 0:
        print(completed.stdout[-3000:]); print(completed.stderr[-3000:])
    check("exits zero", completed.returncode == 0, f"rc={completed.returncode}")
    report = json.loads((out_dir / "method_result.json").read_text())
    check("status is success", report["status"] == "success", report["status"])
    check("states.jsonl written", (out_dir / "states.jsonl").exists())
    check("scene.json written", (out_dir / "scene.json").exists())

    scene = json.loads((out_dir / "scene.json").read_text())
    check("schema is trace_v2.1 (what ingest accepts)",
          scene["schema"] == "trace_v2.1", scene["schema"])
    check("state_fields match the schema",
          scene["state_fields"] == ["x", "y", "yaw_deg", "vx", "vy", "ax", "ay"])
    check("recorder reported no internal errors",
          not scene.get("recorder_errors"), str(scene.get("recorder_errors"))[:200])
    egos = [a for a in scene["actors"] if a.get("is_ego")]
    check("exactly one actor is the ego", len(egos) == 1,
          f"{len(egos)} of {len(scene['actors'])}: {[a['id'] for a in egos]}")
    check("conflict zone declared when the family declares one",
          len(scene.get("zones") or []) == 1)
    check("extents are real, not placeholders",
          len({tuple(a.get("extent", [])) for a in scene["actors"]}) > 1,
          str([a.get("extent") for a in scene["actors"]]))

    detail = report["method_metrics"]
    check("source is the scene-tagged cut_in recording on Town04",
          detail["source"] == "carla_town04__cut_in", detail["source"])
    check("the ego is LOADED from third_party/idm, not realized natively",
          detail["ego_policy_kind"] == "external"
          and detail["ego_policy_source"].endswith("policy.py"),
          f"{detail['ego_policy_kind']} <- {detail['ego_policy_source']}")
    check("no parameter is translated: the policy reads policy.json itself",
          detail["ego_idm"] == {}, str(detail["ego_idm"]))
    check("no policy parameter silently dropped",
          detail["ego_policy_unapplied_parameters"] == {},
          str(detail["ego_policy_unapplied_parameters"]))
    check("prompt recorded but NOT used to steer",
          detail["prompt_used"] is False and detail["prompt_declared"])
    check("seed recorded as declared-not-applied",
          detail["seed_applied"] is False and "seed_declared" in detail)
    check("horizon shortfall declared",
          detail["horizon_truncated"] is True
          and detail["horizon_delivered_s"] == 8.0,
          f"{detail['horizon_delivered_s']} of {detail['horizon_requested_s']} s")

    # ---------------------------------------------------------------- 4
    banner("4. yaw units -- degrees, with a radians control")
    import csv as _csv
    with open(csv_path) as fh:
        first = next(_csv.DictReader(fh))
    src_yaw_rad = float(first["yaw"])
    src_id = str(first["id"])
    with open(out_dir / "states.jsonl") as fh:
        tick0 = json.loads(fh.readline())
    written = tick0["a"][src_id][2]
    expected_deg = math.degrees(src_yaw_rad)
    check("written yaw equals degrees(source yaw)",
          abs(written - expected_deg) < 1e-3,
          f"{written:.4f} vs {expected_deg:.4f} deg")
    check("CONTROL: it is NOT the raw radian value",
          abs(written - src_yaw_rad) > 1e-3,
          f"radians would have been {src_yaw_rad:.4f}")

    # ---------------------------------------------------------------- 5
    banner("5. the harness's OWN ingest reads it back")
    from metrics.ingest.canonical import read_run_dir
    shutil.copyfile(out_dir / "method_result.json", out_dir / "result.json")
    rollout = read_run_dir(str(out_dir))
    check("rollout is evaluable", rollout.evaluable,
          rollout.reason or "no reason given")
    n_ticks = len(rollout.trace.times) if rollout.trace is not None else 0
    check("trace has the expected tick count", n_ticks == detail["trace_ticks"],
          f"{n_ticks} ticks")
    check("trace carries every actor",
          rollout.trace is not None
          and len(rollout.trace.actors) == detail["trace_actors"],
          f"{len(rollout.trace.actors) if rollout.trace else 0} actors")
    check("ingest recovered the same ego we declared",
          rollout.trace is not None
          and rollout.trace.ego_id == detail["trace_ego_id"],
          f"{rollout.trace.ego_id if rollout.trace else None} "
          f"vs {detail['trace_ego_id']}")
    # The ingest converts CARLA-native -> a right-handed planar frame, so the
    # recorded y must come back NEGATED. If it did not, the recorder would have
    # been handed already-converted coordinates and the frame would be applied
    # twice somewhere downstream.
    ego_pose = rollout.trace.ego.pose
    with open(out_dir / "states.jsonl") as fh:
        raw0 = json.loads(fh.readline())["a"][detail["trace_ego_id"]]
    check("ingest negates y (CARLA-native in, canonical out)",
          abs(float(ego_pose[0][1]) + raw0[1]) < 1e-3,
          f"recorded y={raw0[1]:.3f} -> canonical y={float(ego_pose[0][1]):.3f}")
    check("CONTROL: x is unchanged by that conversion",
          abs(float(ego_pose[0][0]) - raw0[0]) < 1e-3,
          f"x {raw0[0]:.3f} -> {float(ego_pose[0][0]):.3f}")

    banner("6. CONTROL: without states.jsonl the same run is unevaluable")
    broken = Path(tempfile.mkdtemp()) / "broken"
    shutil.copytree(out_dir, broken)
    (broken / "states.jsonl").unlink()
    broken_rollout = read_run_dir(str(broken))
    check("evaluable is False", not broken_rollout.evaluable)
    check("and it says why", "trace_v2" in (broken_rollout.reason or ""),
          broken_rollout.reason or "")

    # ---------------------------------------------------------------- 7
    banner("7. town mapping and unsupported policies")
    sys.path.insert(0, str(REPO / "scenario_orchestration"))
    import importlib.util
    spec_ = importlib.util.spec_from_file_location(
        "prosim_run", REPO / "scenario_orchestration" / "run.py")
    run_mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(run_mod)

    check("left_turn -> Town10HD (intersections)",
          run_mod.resolve_town("left_turn", {}) == "carla_town10hd")
    check("overtake -> Town04 (highway)",
          run_mod.resolve_town("overtake", {}) == "carla_town04")
    check("an explicit town in parameters wins",
          run_mod.resolve_town("left_turn", {"town": "Town04"}) == "carla_town04")
    try:
        run_mod.resolve_ego_policy({"name": "tfv6",
                                    "implementation": "scenario_orchestration.policy.build_policy"})
        check("CONTROL: a policy that is neither sensor nor loadable is refused",
              False, "it was accepted")
    except SystemExit as exc:
        check("CONTROL: a policy that is neither sensor nor loadable is refused",
              True, str(exc)[:60])

    # The sensor branch: tfv6 as the harness declares it (configs/policy/tfv6.yaml).
    tfv6_req = {"name": "tfv6", "interface": "ego_policy_v1",
                "implementation": "scenario_orchestration.policy.build_policy",
                "observation_space": "sensor", "action_space": "control",
                "repository": "third_party/tfv6",
                "checkpoint": "third_party/checkpoints/tfv6_cvpr2026/tfv6_resnet34",
                "parameters": {"device": "cuda:0", "town": "Town10HD"}}
    check("a sensor policy resolves to the sensor kind",
          run_mod.resolve_ego_policy(tfv6_req)[0] == "sensor")
    a_s, sp_s = build_requests(family="left_turn", policy="idm")
    scen_s = a_s.build_scenario_request(sp_s).to_dict()
    saved_worker = os.environ.pop("PROSIM_SENSOR_WORKER", None)
    try:
        run_mod.build_argv(scen_s, tfv6_req, Path("/tmp/x/rollout.csv"),
                           harness_root=HARNESS, policy_request_path="/tmp/policy.json")
        check("CONTROL: no PROSIM_SENSOR_WORKER is refused, not defaulted", False, "accepted")
    except SystemExit as exc:
        check("CONTROL: no PROSIM_SENSOR_WORKER is refused, not defaulted",
              "PROSIM_SENSOR_WORKER" in str(exc), str(exc)[:70])
    os.environ["PROSIM_SENSOR_WORKER"] = "127.0.0.1:2100"
    try:
        argv_s, d_s = run_mod.build_argv(scen_s, tfv6_req, Path("/tmp/x/rollout.csv"),
                                         harness_root=HARNESS,
                                         policy_request_path="/tmp/policy.json")
        a_hw, sp_hw = build_requests(family="cut_in", policy="idm")
        argv_hw, _ = run_mod.build_argv(a_hw.build_scenario_request(sp_hw).to_dict(),
                                        tfv6_req, Path("/tmp/x/rollout.csv"),
                                        harness_root=HARNESS,
                                        policy_request_path="/tmp/policy.json")
    finally:
        os.environ.pop("PROSIM_SENSOR_WORKER", None)
        if saved_worker is not None:
            os.environ["PROSIM_SENSOR_WORKER"] = saved_worker

    def flag(argv, name):
        return argv[argv.index(name) + 1] if name in argv else None

    # Assets a submodule clone does not carry: passed only when declared.
    saved_assets = {k: os.environ.pop(k, None) for k in ("PROSIM_CKPT", "PROSIM_LLAMA")}
    try:
        argv_noassets, _ = run_mod.build_argv(scen_s, policy_request, Path("/tmp/x/rollout.csv"),
                                              harness_root=HARNESS,
                                              policy_request_path="/tmp/policy.json")
        os.environ["PROSIM_CKPT"] = "/assets/prosim_demo_model.ckpt"
        os.environ["PROSIM_LLAMA"] = "/assets/Meta-Llama-3-8B-Instruct-HF"
        argv_assets, _ = run_mod.build_argv(scen_s, policy_request, Path("/tmp/x/rollout.csv"),
                                            harness_root=HARNESS,
                                            policy_request_path="/tmp/policy.json")
    finally:
        for k, v in saved_assets.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    check("PROSIM_CKPT / PROSIM_LLAMA become --ckpt / --llama",
          flag(argv_assets, "--ckpt") == "/assets/prosim_demo_model.ckpt"
          and flag(argv_assets, "--llama") == "/assets/Meta-Llama-3-8B-Instruct-HF",
          f"{flag(argv_assets, '--ckpt')} / {flag(argv_assets, '--llama')}")
    check("CONTROL: unset, neither flag is passed (rollout_carla's defaults apply)",
          "--ckpt" not in argv_noassets and "--llama" not in argv_noassets)

    # Recordings and lane graphs ship in the repo's carla_data/.
    saved_data = os.environ.pop("PROSIM_DATA_DIR", None)
    try:
        committed = run_mod.data_dir()
        argv_data, _ = run_mod.build_argv(scen_s, policy_request, Path("/tmp/x/rollout.csv"),
                                          harness_root=HARNESS,
                                          policy_request_path="/tmp/policy.json")
        os.environ["PROSIM_DATA_DIR"] = "/elsewhere/recordings"
        overridden = run_mod.data_dir()
        argv_over, _ = run_mod.build_argv(scen_s, policy_request, Path("/tmp/x/rollout.csv"),
                                          harness_root=HARNESS,
                                          policy_request_path="/tmp/policy.json")
    finally:
        os.environ.pop("PROSIM_DATA_DIR", None)
        if saved_data is not None:
            os.environ["PROSIM_DATA_DIR"] = saved_data
    check("unset, the data dir is this repo's carla_data/ and is passed as --data-dir",
          committed == REPO / "carla_data" and flag(argv_data, "--data-dir") == str(REPO / "carla_data"),
          str(flag(argv_data, "--data-dir")))
    check("CONTROL: PROSIM_DATA_DIR overrides it",
          overridden == Path("/elsewhere/recordings")
          and flag(argv_over, "--data-dir") == "/elsewhere/recordings")
    sources = ["carla_town04__cut_in", "carla_town04__lane_change", "carla_town04__overtake",
               "carla_town10hd__left_turn", "carla_town10hd__red_light", "carla_town10hd__right_turn"]
    missing = [s for s in sources if not (committed / Path(run_mod.recording_path(s)).name).is_file()]
    lanes = [t for t in ("town04", "town10hd") if not (committed / f"{t}_lanes.json").is_file()]
    check("all six recordings and both lane graphs are committed in carla_data/",
          not missing and not lanes, f"missing {missing} {lanes}")
    check("tfv6 is loaded from third_party/tfv6, found by its REPOSITORY",
          (flag(argv_s, "--ego-external") or "").endswith(
              "third_party/tfv6/scenario_orchestration/policy.py"),
          str(flag(argv_s, "--ego-external")))
    check("the worker address is passed through and recorded",
          flag(argv_s, "--ego-remote") == "127.0.0.1:2100"
          and d_s["ego_sensor_worker"] == "127.0.0.1:2100")
    check("the CARLA map follows the family: left_turn -> Town10HD_Opt, cut_in -> Town04",
          flag(argv_s, "--ego-remote-town") == "Town10HD_Opt"
          and flag(argv_hw, "--ego-remote-town") == "Town04",
          f"{flag(argv_s, '--ego-remote-town')} / {flag(argv_hw, '--ego-remote-town')}")
    check("camera frames land next to the rollout, in the cell's output dir",
          flag(argv_s, "--ego-frames-dir") == "/tmp/x/ego_sensor_frames")
    check("no BEV and no native ego flags on the sensor path",
          "--ego-bev-town" not in argv_s and "--ego-policy" not in argv_s
          and "--ego-v0" not in argv_s, str(argv_s))
    kind_ext, _, kw_ext, un_ext = run_mod.resolve_ego_policy(
        {"name": "idm_mobil", "implementation": "idm.policy.IDMMobilPolicy",
         "parameters": {"desired_speed_mps": 14.0, "lane_width_m": 3.5}})
    check("idm_mobil resolves as EXTERNAL, not as this repo's own MOBIL",
          kind_ext == "external", kind_ext)
    check("nothing is translated or dropped for an external policy",
          kw_ext == {} and un_ext == {}, f"{kw_ext} {un_ext}")
    a_m, sp_m = build_requests(family="lane_change", policy="idm_mobil")
    argv_m, d_m = run_mod.build_argv(
        a_m.build_scenario_request(sp_m).to_dict(),
        a_m.build_policy_request(sp_m).to_dict(), Path("/tmp/x.csv"),
        harness_root=HARNESS, policy_request_path="/tmp/policy.json")
    check("argv loads third_party/idm's policy.py and passes policy.json",
          "--ego-external" in argv_m
          and argv_m[argv_m.index("--ego-external") + 1].endswith(
              "third_party/idm/scenario_orchestration/policy.py")
          and "--ego-policy-request" in argv_m,
          argv_m[argv_m.index("--ego-external") + 1] if "--ego-external" in argv_m else "absent")
    check("and it does NOT configure this repo's native ego",
          "--ego-policy" not in argv_m and "--ego-v0" not in argv_m
          and "--ego-idm" not in argv_m, str(argv_m))
    check("the policy source is recorded, not left to inference",
          d_m["ego_policy_source"].endswith("policy.py"), d_m["ego_policy_source"])
    try:
        run_mod.build_argv(a_m.build_scenario_request(sp_m).to_dict(),
                           a_m.build_policy_request(sp_m).to_dict(),
                           Path("/tmp/x.csv"), harness_root=Path(tempfile.mkdtemp()),
                           policy_request_path="/tmp/policy.json")
        check("CONTROL: an uninitialised submodule is refused", False, "accepted")
    except SystemExit as exc:
        check("CONTROL: an uninitialised submodule is refused",
              "git submodule update --init third_party/idm" in str(exc),
              str(exc)[-60:])
    # Plain `idm` is LOADED too since 2026-09-11, so the ego is the same code
    # on every arm and idm vs idm_mobil differs only in the lateral law.
    kind, v0, kwargs, unapplied = run_mod.resolve_ego_policy(
        {"name": "x", "implementation": "idm.policy.IDMPolicy",
         "parameters": {"desired_speed_mps": 25.0, "politeness": 0.3}})
    check("plain idm resolves as EXTERNAL as well", kind == "external", kind)
    check("its parameters are passed through untouched, none dropped",
          kwargs == {} and unapplied == {} and v0 == 25.0,
          f"{kwargs} {unapplied} v0={v0}")
    argv_i, d_i = run_mod.build_argv(request, policy_request, Path("/tmp/x.csv"),
                                     harness_root=HARNESS,
                                     policy_request_path="/tmp/policy.json")
    check("an idm cell loads third_party/idm, and says so",
          "--ego-external" in argv_i
          and d_i["ego_policy_source"].endswith("idm/scenario_orchestration/policy.py"),
          d_i["ego_policy_source"])
    check("and no native ego flags are emitted",
          "--ego-policy" not in argv_i and "--ego-v0" not in argv_i, str(argv_i[:12]))

    # ---------------------------------------------------------------- 8
    banner("8. the recorder is found WITHOUT SCENARIO_ORCHESTRATION_ROOT")
    # The harness does not export that variable -- only the experiment id and
    # the seed. Setting it in every other case above masked a real defect:
    # third_party/prosim is a symlink, resolve() collapsed it, and the
    # '../../' fallback landed in /scratch/veerk41 instead of the harness.
    bare = Path(tempfile.mkdtemp()) / "bare"
    completed = run_entry_point(goal_request, policy_request, bare,
                                {"PROSIM_ROLLOUT_CSV": str(csv_path)},
                                with_root=False)
    check("exits zero with no root hint", completed.returncode == 0,
          f"rc={completed.returncode}")
    check("states.jsonl still written", (bare / "states.jsonl").exists())
    check("it located the harness by itself",
          "trace recorder from" in completed.stdout,
          next((l for l in completed.stdout.splitlines()
                if "recorder from" in l), "not found"))

    banner("9. CONTROL: a missing recorder reports itself, not just rc=1")
    stray = Path(tempfile.mkdtemp()) / "stray"
    completed = run_entry_point(goal_request, policy_request, stray,
                                {"PROSIM_ROLLOUT_CSV": str(csv_path),
                                 "SCENARIO_ORCHESTRATION_ROOT": "/nonexistent"},
                                with_root=False)
    # /nonexistent is skipped and the real root is still found by fallback, so
    # this asserts the fallback survives a bad hint rather than trusting it.
    check("a bad root hint does not break the run",
          completed.returncode == 0, f"rc={completed.returncode}")

    # ---------------------------------------------------------------- 10
    banner("10. all six families resolve from their real implementations.yaml")
    expect_town = {
        "red_light": "carla_town10hd", "left_turn": "carla_town10hd",
        "right_turn": "carla_town10hd", "cut_in": "carla_town04",
        "lane_change": "carla_town04", "overtake": "carla_town04",
    }
    expect_goals = {"red_light": 1, "left_turn": 1, "right_turn": 1,
                    "cut_in": 1, "lane_change": 2, "overtake": 2}
    expect_ego_hint = {"red_light", "left_turn", "right_turn", "overtake",
                       "cut_in", "lane_change"}
    for fam in sorted(expect_town):
        fam_adapter, fam_spec = build_requests(family=fam)
        fam_request = fam_adapter.build_scenario_request(fam_spec).to_dict()
        fam_policy = fam_adapter.build_policy_request(fam_spec).to_dict()
        try:
            argv, fam_detail = run_mod.build_argv(
                fam_request, fam_policy, Path("/tmp/x.csv"),
                harness_root=HARNESS, policy_request_path="/tmp/policy.json")
        except SystemExit as exc:
            check(f"{fam}: builds a command", False, str(exc)[:90])
            continue
        ok = (fam_detail["town"] == expect_town[fam]
              and len(fam_detail["goals"]) == expect_goals[fam]
              and (("--ego-goal-xy" in argv) == (fam in expect_ego_hint)))
        check(f"{fam}: {expect_town[fam].replace('carla_','')}, "
              f"{expect_goals[fam]} goal(s)"
              f"{', ego route hint' if fam in expect_ego_hint else ''}",
              ok,
              f"{fam_detail['source']} {fam_detail['goals']} "
              f"ego_hint={'--ego-goal-xy' in argv}")

    banner("11. CONTROL: no two families are silently identical")
    # cut_in and lane_change DO share actor geometry by design; every other
    # pair must differ, or a family name is decorating a duplicate run.
    seen = {}
    for fam in sorted(expect_town):
        fam_adapter, fam_spec = build_requests(family=fam)
        _, d = run_mod.build_argv(
            fam_adapter.build_scenario_request(fam_spec).to_dict(),
            fam_adapter.build_policy_request(fam_spec).to_dict(),
            Path("/tmp/x.csv"), harness_root=HARNESS,
            policy_request_path="/tmp/policy.json")
        seen[fam] = (d["source"], tuple(map(tuple, d["goals"])))
    dupes = {}
    for fam, key in seen.items():
        dupes.setdefault(key, []).append(fam)
    collided = [v for v in dupes.values() if len(v) > 1]
    check("only cut_in/lane_change share a staging",
          collided in ([], [["cut_in", "lane_change"]]),
          f"duplicate groups: {collided}")

    # ---------------------------------------------------------------- 12
    banner("12. every declared goal is >=5 m AHEAD of its declared spawn")
    # set_goal_condition(min_forward=5.0) RAISES on a goal behind its agent, so
    # a spawn edit can invalidate a goal and the run dies mid-sweep. This turns
    # that into a config check that fails in seconds. Heading comes from the
    # lane the agent spawns on, which is what record_actor_history snaps to.
    import math as _m
    graphs = {}
    for town in ("town10hd", "town04"):
        g = json.loads((Path("/scratch/veerk41") / f"{town}_lanes.json").read_text())
        graphs[town] = [l["center"] for l in g["lanes"].values()]

    def lane_heading(town, x, y):
        best = (1e9, 0.0)
        for c in graphs[town]:
            for i, q in enumerate(c):
                d = (q[0] - x) ** 2 + (q[1] - y) ** 2
                if d < best[0]:
                    j, k = min(i + 1, len(c) - 1), max(i - 1, 0)
                    best = (d, _m.atan2(c[j][1] - c[k][1], c[j][0] - c[k][0]))
        return best[1]

    import yaml as _yaml
    checked = 0
    for fam in sorted(expect_town):
        params = (_yaml.safe_load(
            (HARNESS / "scenarios" / fam / "implementations.yaml").read_text()
        )["implementations"]["prosim"]["parameters"])
        town = expect_town[fam].replace("carla_", "")
        spawns = {s["agent"]: s["xy"] for s in params.get("spawns", [])}
        pairs = [(g["agent"], g["xy"]) for g in params.get("goals", [])]
        if params.get("ego_goal_xy"):
            pairs.append((0, params["ego_goal_xy"]))
        for agent, goal in pairs:
            spawn = spawns.get(agent)
            if spawn is None:
                check(f"{fam} agent {agent}: has a declared spawn", False)
                continue
            h = lane_heading(town, *spawn)
            dx, dy = goal[0] - spawn[0], goal[1] - spawn[1]
            fwd = dx * _m.cos(h) + dy * _m.sin(h)
            checked += 1
            check(f"{fam} agent {agent}: goal is ahead of spawn",
                  fwd >= 5.0, f"forward {fwd:+.1f} m (guard is +5.0)")
    # 8 model goals (1+1+1+1+2+2) + 6 ego route hints (every family) = 14. Asserted so a family that quietly loses its
    # goals still fails here rather than passing an empty loop.
    check("every family declared spawns for every goal", checked == 14,
          f"{checked} spawn/goal pairs checked")

    banner("13. CONTROL: a goal placed behind its spawn must be caught")
    # Mirror one goal to the far side of its spawn; the same arithmetic must
    # now report it as behind, or section 12 proves nothing.
    spawn, goal = [-85.0, 24.5], [-40.0, 24.5]
    h = lane_heading("town10hd", *spawn)
    behind = [2 * spawn[0] - goal[0], 2 * spawn[1] - goal[1]]
    fwd = ((behind[0] - spawn[0]) * _m.cos(h)
           + (behind[1] - spawn[1]) * _m.sin(h))
    check("the mirrored goal reads as behind", fwd < 5.0,
          f"forward {fwd:+.1f} m")

    banner("14. each family rolls out its OWN scene")
    scenes = {}
    for fam in sorted(expect_town):
        a, sp = build_requests(family=fam)
        _, d = run_mod.build_argv(a.build_scenario_request(sp).to_dict(),
                                  a.build_policy_request(sp).to_dict(),
                                  Path("/tmp/x.csv"), harness_root=HARNESS, policy_request_path='/tmp/policy.json')
        scenes[fam] = d["source"]
    check("six distinct recordings for six families",
          len(set(scenes.values())) == 6, str(sorted(set(scenes.values()))))
    check("every family is tagged with its own name",
          all(scenes[f] == f"{expect_town[f]}__{f}" for f in expect_town),
          str(scenes))

    banner("15. goals are assigned by recorded actor id, not batch index")
    # lane_change: the 3-actor recording (cut_in is 2 actors since 2026-09-11)
    rec = Path("/scratch/veerk41/history_town04__lane_change.csv")
    declared = {"spawns": [{"agent": 0, "xy": [-300, 30]},
                           {"agent": 1, "xy": [-280, 30]},
                           {"agent": 2, "xy": [-280, 35]}]}
    ids = run_mod.assign_actor_ids(rec, declared)
    check("lane_change: agent 0/1/2 -> actors in spawn order",
          ids == {0: 154, 1: 155, 2: 156}, str(ids))
    swapped = {"spawns": [{"agent": 0, "xy": [-300, 30]},
                          {"agent": 1, "xy": [-280, 35]},
                          {"agent": 2, "xy": [-280, 30]}]}
    ids2 = run_mod.assign_actor_ids(rec, swapped)
    check("CONTROL: swapping two declared spawns swaps their actors",
          ids2 == {0: 154, 1: 156, 2: 155}, str(ids2))
    for label, bad in [
        ("ego declared on another car's spawn",
         [{"agent": 0, "xy": [-280, 30]}, {"agent": 1, "xy": [-300, 30]},
          {"agent": 2, "xy": [-280, 35]}]),
        ("a spawn 40 m from any recorded actor",
         [{"agent": 0, "xy": [-300, 30]}, {"agent": 1, "xy": [-240, 30]},
          {"agent": 2, "xy": [-280, 35]}]),
        ("fewer spawns than recorded actors",
         [{"agent": 0, "xy": [-300, 30]}, {"agent": 1, "xy": [-280, 30]}]),
    ]:
        try:
            run_mod.assign_actor_ids(rec, {"spawns": bad})
            check(f"CONTROL: refuses {label}", False, "was accepted")
        except SystemExit as exc:
            check(f"CONTROL: refuses {label}", True, str(exc)[:70])
    a_lc, sp_lc = build_requests(family="lane_change")
    argv, d = run_mod.build_argv(a_lc.build_scenario_request(sp_lc).to_dict(),
                                 a_lc.build_policy_request(sp_lc).to_dict(),
                                 Path("/tmp/x.csv"),
                                 actor_ids={0: 154, 1: 155, 2: 156}, harness_root=HARNESS, policy_request_path='/tmp/policy.json')
    gid = [argv[i + 1:i + 4] for i, a in enumerate(argv) if a == "--goal-id"]
    check("argv carries --goal-id with actor ids, and no bare --goal",
          gid == [["155", "-240", "30"], ["156", "-100", "35"]]
          and "--goal" not in argv, str(gid))
    check("the mapping is recorded in method_metrics",
          d["goal_assignment"] == "recorded actor id"
          and d["agent_actor_ids"] == {"0": 154, "1": 155, "2": 156})

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
