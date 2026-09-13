"""ProSim's standardized entry point for the scenario_orchestration harness.

Invoked across a process boundary (DESIGN.md section 5) as::

    <launcher...> python3 scenario_orchestration/run.py \
        --scenario-request request.json \
        --policy-request  policy.json \
        --output-dir      <results/raw/<experiment_id>>

with cwd set to this repository. Nothing is imported from the harness's
``src/`` package -- only ``metrics.recording``, which is stdlib-only and exists
precisely to be imported by method repositories inside their own interpreters.

WHAT THIS WRITES, and why the second one is the one that matters
---------------------------------------------------------------
``method_result.json``  the harness's own report (status + metrics).
``states.jsonl`` +
``scene.json``          the canonical rollout, schema ``trace_v2.1``.

``metrics/ingest/canonical.py::read_run_dir`` reads ONLY the second pair. A run
that writes ``method_result.json`` and no ``states.jsonl`` comes back as
``Rollout(evaluable=False)`` and every canonical metric is unavailable -- while
the cell still reports ``status: success``. That is the failure this file is
most careful about, so the trace is written before the report and the report
records how many ticks reached disk.

Canonical metrics are NOT self-reported. ``scenario_realized``,
``time_to_event`` and ``collision`` are derived from the trace by
``metrics.scenario``; anything this file put in those slots would be ProSim
marking its own homework. What goes into ``metrics`` here is provenance.

CONDITIONING: GOALS, NOT PROMPTS
-------------------------------
``ProSimAdapter.validate`` requires a ``prompt`` in the family's
implementation entry, so one is always present -- but text conditioning does
not steer this checkpoint. Measured 2026-09-04/09-07: two opposite prompts
('turn left' / 'turn right') produced trajectories differing by 0.0 degrees of
net heading, while the structured ``goal`` channel put an agent 0.24 m from a
requested point. So the prompt is recorded as provenance and the run is driven
by goals.

A run with no goal declared is REFUSED rather than run unconditioned. An
unconditioned ProSim rollout does not stage the scenario, and would be scored
as an honest ``scenario_realized: false`` -- a null result manufactured by a
missing config key. Declare goals in
``scenarios/<family>/implementations.yaml`` under ``prosim.parameters``.

SEED: recorded, not applied. ProSim's rollout exposes no seed; the sampling
path is not seeded from this entry point. The seed axis on this arm therefore
measures nothing and repeats are near-identical -- same situation
``configs/algorithm/osc2runner_carla.yaml`` documents in ``notes.seed``.

HORIZON: this checkpoint's rollout is FUTURE_SEC = 8.0 s, and every scenario
family asks for 15-25 s. The shortfall is real, is reported in
``method_metrics.horizon_*``, and is printed. Set
``PROSIM_STRICT_HORIZON=1`` to make it a failure instead of a truncation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The same directory reached WITHOUT resolving symlinks. third_party/prosim is
#: a symlink to the ProSim checkout, and resolve() collapses it -- so the
#: resolved parent is the checkout's neighbourhood, not the harness's. Both
#: spellings are searched for metrics/recording/; keeping only the resolved one
#: silently defeats the "../../" fallback whenever the submodule is a symlink.
REPO_ROOT_UNRESOLVED = Path(__file__).absolute().parent.parent

#: Which town each scenario family is staged on. cut_in and lane_change use
#: Town04's road 40 (4 lanes per direction, 237 m dead straight -- the only
#: stretch long enough for the 8 s horizon at highway speed); overtake uses a
#: two-lane stretch elsewhere on Town04. The three junction families are on
#: Town10HD. Overridable per implementation with a `town` parameter.
#:
#: red_light is staged here as GEOMETRY only: the bridge writes an empty
#: traffic-light frame (every lane NO_DATA) and the analytic ego reads no
#: signal, so there is no right-of-way to violate. metrics/definitions.py maps
#: red_light to the `intersection` kernel with role `crossing`, which is what
#: actually gets measured.
FAMILY_TOWN = {
    "red_light": "carla_town10hd",
    "cut_in": "carla_town04",
    "lane_change": "carla_town04",
    "overtake": "carla_town04",
    "left_turn": "carla_town10hd",
    "right_turn": "carla_town10hd",
}

#: Harness policy ``implementation`` -> this repository's OWN ego-policy kind,
#: for policies realized natively. Empty since 2026-09-11: the whole IDM family
#: is loaded from third_party/idm instead (see EXTERNAL_POLICY), so that the ego
#: is the same code on every arm and `idm` vs `idm_mobil` differs only in the
#: lateral law. The native path is kept because the harness's declared default
#: is native realization (configs/policy/idm.yaml: `requires: []`), and a future
#: policy may take it.
POLICY_KIND = {}

#: Policies the harness declares must be LOADED, never realized natively, and
#: the third_party/ directory each is loaded from.
#: configs/policy/idm_mobil.yaml: "it is NOT realized natively by any execution
#: method, so the repository has to be present" -- because it carries a lateral
#: law. Running this repo's own MOBIL under that name would publish a different
#: implementation's behaviour, which that yaml warns is "comparing two
#: implementations as well as two lateral laws". external_ego.py drives the
#: repository's own policy object instead.
EXTERNAL_POLICY = {
    "idm.policy.IDMPolicy": "idm",
    "idm.policy.IDMMobilPolicy": "idm",
    # A learned ego. Declared observation space is `state`, but every released
    # PlanT 2.0 checkpoint carries input_bev=True, so it also needs a BEV
    # semantic raster -- see external_ego.py. Checkpoints are not in the repo
    # (huggingface.co/SimonGer/PlanT2); without one, act() raises on load.
    "plant2.agent.PlanTAgent": "plant2",
}

#: Harness policy parameter -> IDM dataclass field (``ego_control.IDM``).
#: ``desired_speed_mps`` is handled separately because rollout_carla exposes it
#: as its own flag (--ego-v0).
IDM_PARAM_MAP = {
    "time_headway_s": "T",
    "min_gap_m": "s0",
    "max_accel_mps2": "a_max",
    "comfort_decel_mps2": "b",
}

#: ProSim's fixed rollout horizon, seconds. DATASET.FORMAT.FUTURE_SEC.
PROSIM_HORIZON_S = 8.0

#: Rollout tick, seconds. Matches the recording and every family's tick_rate_hz.
DT = 0.1

#: Which pre-rendered BEV raster goes with each town, for a policy that reads
#: one. Town10HD_Opt is the map the recordings were actually made on.
BEV_TOWN = {"town04": "Town04", "town10hd": "Town10HD_Opt"}


# --------------------------------------------------------------------------
# the harness's stdlib-only recorder
# --------------------------------------------------------------------------

def load_recorder(output_dir: Path):
    """Import ``metrics.recording.TraceRecorder`` from the harness checkout.

    Loaded as a standalone package rather than via ``import metrics`` so that
    ``metrics/__init__.py`` -- which pulls in numpy through ``.evaluate`` -- is
    never executed. ``metrics.recording`` is documented as stdlib-only for
    exactly this reason; going through the parent package would quietly
    reintroduce the dependency it was split out to avoid.
    """
    candidates = []
    env = os.environ.get("SCENARIO_ORCHESTRATION_ROOT")
    if env:
        candidates.append(Path(env))
    # results/raw/<experiment_id>/ -> walk up to the harness root.
    candidates.extend(Path(output_dir).resolve().parents)
    # third_party/prosim -> ../../, by both spellings of this file's location.
    candidates.extend(REPO_ROOT_UNRESOLVED.parents)
    candidates.extend(REPO_ROOT.parents)
    # Last resort: a sibling checkout beside the method repository, which is
    # where it sits when third_party/prosim is a symlink into /scratch.
    for parent in list(REPO_ROOT.parents)[:2]:
        try:
            candidates.extend(sorted(p for p in parent.iterdir() if p.is_dir()))
        except OSError:
            pass

    for root in candidates:
        pkg_init = root / "metrics" / "recording" / "__init__.py"
        if not pkg_init.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_prosim_trace_recording", pkg_init,
            submodule_search_locations=[str(pkg_init.parent)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["_prosim_trace_recording"] = module
        spec.loader.exec_module(module)
        return module, root

    raise SystemExit(
        "could not locate the harness's metrics/recording/ package. Without it "
        "no states.jsonl is written and every metric for this run is "
        "unavailable, so this is a hard failure rather than a degraded run. "
        "Set SCENARIO_ORCHESTRATION_ROOT to the harness checkout. Searched: "
        + ", ".join(str(c) for c in candidates[:8])
    )


# --------------------------------------------------------------------------
# request -> rollout_carla.py argv
# --------------------------------------------------------------------------

def resolve_town(family: str, parameters: dict) -> str:
    """Which trajdata source to roll out."""
    declared = parameters.get("town") or parameters.get("source")
    if declared:
        name = str(declared).lower()
        return name if name.startswith("carla_") else f"carla_{name}"
    if family not in FAMILY_TOWN:
        raise SystemExit(
            f"scenario family {family!r} has no town mapping and none is "
            f"declared in its implementation parameters. Known: "
            f"{sorted(FAMILY_TOWN)}"
        )
    return FAMILY_TOWN[family]


def resolve_source(family: str, parameters: dict) -> str:
    """The scene-tagged trajdata source, ``carla_<town>__<scene>``.

    Every scenario gets its OWN recording and its own trajdata cache: one CSV
    per town cannot hold three Town10HD scenes, and swapping files under one
    dataset name is how the stale-cache failure of 2026-09-02 happened. The
    scene defaults to the family name; a family may share another's recording
    by declaring ``scene`` (lane_change stages the same actors as cut_in).
    """
    explicit = parameters.get("source")
    if explicit and "__" in str(explicit):
        return str(explicit).lower()
    town = resolve_town(family, parameters).split("__")[0]
    scene = str(parameters.get("scene") or family).lower()
    return f"{town}__{scene}"


def recording_path(source: str) -> Path:
    """Where carla_dataset expects the recording for a scene-tagged source."""
    data_dir = Path(os.environ.get("PROSIM_DATA_DIR", "/scratch/veerk41"))
    town, _, scene = source[len("carla_"):].partition("__")
    return data_dir / f"history_{town}__{scene}.csv"


def record_recipe(source: str, parameters: dict) -> str:
    """The exact commands that produce the missing recording."""
    town, _, scene = source[len("carla_"):].partition("__")
    spawns = sorted(parameters.get("spawns") or [], key=lambda s: int(s["agent"]))
    xy = " ".join(f"{float(s['xy'][0]):g} {float(s['xy'][1]):g}" for s in spawns)
    return (f"  load_town_and_extract_xodr.py --town {town.replace('town', 'Town').replace('hd', 'HD')}\n"
            f"  record_actor_history.py --spawn-xy {xy or 'X Y ...'} --steps 150 "
            f"--four-wheels-only --out /workspace/agent_history_{town}__{scene}.json\n"
            f"  extract_csv.py --in /workspace/agent_history_{town}__{scene}.json "
            f"--out /workspace/history_{town}__{scene}.csv")


def assign_actor_ids(recording: Path, parameters: dict, tol_m: float = 5.0) -> dict:
    """Declared agent index -> recorded CARLA actor id, matched by position.

    Goals are declared per agent INDEX in implementations.yaml, but ProSim's
    batch order is not spawn order, so an index handed straight to the model
    can give one actor another's goal -- silently, when both goals are
    plausible (cut_in's two actors spawn ~20 m from the ego either way). The
    recording knows which actor started where, so match each declared spawn
    to the actor whose first recorded position is nearest.

    Refuses: an actor count that differs from the declared spawns, a spawn
    with no recorded actor within `tol_m` (the recorder snaps to the lane
    centre, measured <= 1.61 m), two spawns claiming one actor, and a
    declared ego that is not the lowest id -- rollout_carla drives 'ego',
    which is min(id), so that would put the ego policy on the wrong car.
    """
    import csv
    first = {}
    with open(recording, newline="") as fh:
        for row in csv.DictReader(fh):
            aid, fr = int(row["id"]), int(row["frame"])
            if aid not in first or fr < first[aid][0]:
                first[aid] = (fr, float(row["x"]), float(row["y"]))
    spawns = sorted(parameters.get("spawns") or [], key=lambda s: int(s["agent"]))
    if not spawns:
        raise SystemExit("no `spawns` declared; cannot tell which recorded actor "
                         "is which, and batch order is not spawn order")
    if len(first) != len(spawns):
        raise SystemExit(f"{recording.name} holds {len(first)} actors but "
                         f"{len(spawns)} spawns are declared")
    mapping, taken = {}, set()
    for sp in spawns:
        sx, sy = float(sp["xy"][0]), float(sp["xy"][1])
        dist, aid = min((math.hypot(x - sx, y - sy), a)
                        for a, (_, x, y) in first.items())
        if dist > tol_m:
            raise SystemExit(f"declared spawn {sp['agent']} at ({sx:g}, {sy:g}): "
                             f"nearest recorded actor is {dist:.2f} m away "
                             f"(limit {tol_m} m) -- wrong recording?")
        if aid in taken:
            raise SystemExit(f"declared spawns share actor {aid}; ambiguous")
        taken.add(aid)
        mapping[int(sp["agent"])] = aid
    if mapping.get(0) != min(first):
        raise SystemExit(f"declared ego (agent 0) is actor {mapping.get(0)} but "
                         f"the lowest id is {min(first)}; the ego policy would "
                         f"drive the wrong car")
    return mapping


def resolve_goals(parameters: dict) -> list:
    """Goals as a list of ``(agent_index, x, y)``.

    Accepts either::

        goals:
          - {agent: 1, xy: [-300.0, 35.0]}
          - [2, -280.0, 30.0]

    Refuses an empty result. See the module docstring for why an unconditioned
    rollout is not an acceptable fallback.
    """
    raw = parameters.get("goals")
    goals = []
    if raw:
        for entry in raw:
            if isinstance(entry, dict):
                xy = entry.get("xy") or [entry.get("x"), entry.get("y")]
                goals.append((int(entry["agent"]), float(xy[0]), float(xy[1])))
            else:
                goals.append((int(entry[0]), float(entry[1]), float(entry[2])))
    elif parameters.get("goal_xy") is not None:
        xy = parameters["goal_xy"]
        goals.append((int(parameters.get("goal_agent", 1)),
                      float(xy[0]), float(xy[1])))

    if not goals:
        raise SystemExit(
            "no goal declared for this scenario family. ProSim is steered "
            "through the structured 'goal' condition channel, not through the "
            "prompt -- text conditioning was measured not to move this "
            "checkpoint's trajectories at all. Add to "
            "scenarios/<family>/implementations.yaml under prosim.parameters:\n"
            "    goals:\n"
            "      - {agent: 1, xy: [-300.0, 35.0]}\n"
            "Running unconditioned would report scenario_realized: false and "
            "look like a result rather than a missing key, so it is refused."
        )

    seen = set()
    for agent, _, _ in goals:
        if agent in seen:
            raise SystemExit(
                f"two goals declared for agent index {agent}; ProSim holds one "
                f"goal slot per agent")
        seen.add(agent)
    return goals


def resolve_ego_policy(policy_request: dict) -> tuple:
    """(kind, v0, idm_kwargs, unapplied) from the harness's policy request."""
    implementation = str(policy_request.get("implementation") or "")
    if policy_request.get("observation_space") == "sensor":
        # A vision policy: loaded by sensor_worker.py inside a CARLA world, and
        # every parameter goes to it verbatim, as for any loaded policy.
        return ("sensor", 0.0, {}, {})
    if implementation in EXTERNAL_POLICY:
        # Loaded, not realized: every declared parameter goes to the policy
        # verbatim, so nothing is translated here and nothing is unapplied.
        parameters = dict(policy_request.get("parameters") or {})
        return ("external", float(parameters.get("desired_speed_mps", 0.0)),
                {}, {})
    if implementation not in POLICY_KIND:
        raise SystemExit(
            f"ego policy {policy_request.get('name')!r} has implementation "
            f"{implementation!r}, which this arm can neither load nor "
            f"realize natively. Loadable: {sorted(EXTERNAL_POLICY)}; "
            f"native: {sorted(POLICY_KIND)}."
        )
    kind = POLICY_KIND[implementation]

    parameters = dict(policy_request.get("parameters") or {})
    v0 = float(parameters.pop("desired_speed_mps", 10.0))

    idm_kwargs, unapplied = {}, {}
    for key, value in parameters.items():
        if key in IDM_PARAM_MAP:
            idm_kwargs[IDM_PARAM_MAP[key]] = float(value)
        else:
            # Reported rather than dropped: a policy axis that silently loses
            # half its parameters produces two 'different' policies that are
            # the same policy, and nothing in the numbers would say so.
            unapplied[key] = value
    return kind, v0, idm_kwargs, unapplied


def external_policy_py(policy_request: dict, harness_root) -> Path:
    """Where the declared external policy repository's entry point lives."""
    implementation = str(policy_request.get("implementation") or "")
    # Sensor policies share a generic implementation string
    # (scenario_orchestration.policy.build_policy), so the repository they
    # declare is what names them.
    repo = (EXTERNAL_POLICY.get(implementation)
            or Path(str(policy_request.get("repository") or "")).name)
    if not repo:
        raise SystemExit(f"ego policy {policy_request.get('name')!r} declares "
                         "no repository to load it from")
    if harness_root is None:
        raise SystemExit(
            f"ego policy {policy_request.get('name')!r} is loaded from "
            f"third_party/{repo}, so build_argv needs harness_root to find it")
    policy_py = (Path(harness_root) / "third_party" / repo
                 / "scenario_orchestration" / "policy.py")
    if not policy_py.is_file():
        raise SystemExit(
            f"ego policy {policy_request.get('name')!r} must be loaded from "
            f"third_party/{repo}, which is not checked out ({policy_py} is "
            f"missing). From the harness root:\n"
            f"  git submodule update --init third_party/{repo}")
    return policy_py


def build_argv(request: dict, policy_request: dict, out_csv: Path,
               actor_ids: dict = None, harness_root=None,
               policy_request_path=None) -> tuple:
    """The rollout_carla.py command line for this cell."""
    implementation = request.get("implementation") or {}
    parameters = dict(implementation.get("parameters") or {})
    family = str(request.get("scenario_family") or "")

    source = resolve_source(family, parameters)
    goals = resolve_goals(parameters)
    kind, v0, idm_kwargs, unapplied = resolve_ego_policy(policy_request)

    argv = [
        sys.executable, "-u", "rollout_carla.py",
        "--source", source,
        "--out", str(out_csv),
        "--ego-agent", str(parameters.get("ego_agent", "ego")),
        "--example-idx", str(int(parameters.get("example_idx", 0))),
        "--device", "cpu",
    ]
    # The checkpoint and Llama weights are gitignored, so a submodule clone
    # lacks them: the harness declares where they live, otherwise rollout_carla
    # uses its repo-relative defaults.
    for flag, env_name in (("--ckpt", "PROSIM_CKPT"), ("--llama", "PROSIM_LLAMA")):
        if os.environ.get(env_name):
            argv += [flag, os.environ[env_name]]
    policy_py = None
    sensor_worker = None
    if kind == "sensor":
        # The worker runs natively on the CARLA node, in the policy's own
        # environment; this process only talks to it. Never guessed: a default
        # port on a shared node is somebody else's world.
        sensor_worker = os.environ.get("PROSIM_SENSOR_WORKER")
        if not sensor_worker:
            raise SystemExit(
                f"ego policy {policy_request.get('name')!r} observes `sensor`, "
                "so it runs inside a CARLA world through sensor_worker.py. Set "
                "PROSIM_SENSOR_WORKER=HOST:PORT to the worker, started on the "
                "CARLA node in the policy's environment.")
        policy_py = external_policy_py(policy_request, harness_root)
        world_town = BEV_TOWN.get(source.split("__")[0].replace("carla_", ""))
        if not world_town:
            raise SystemExit(f"no CARLA map known for source {source!r}")
        argv += ["--ego-external", str(policy_py),
                 "--ego-policy-request", str(policy_request_path),
                 "--ego-remote", sensor_worker,
                 "--ego-remote-town", world_town,
                 "--ego-frames-dir", str(Path(out_csv).parent / "ego_sensor_frames")]
    elif kind == "external":
        policy_py = external_policy_py(policy_request, harness_root)
        argv += ["--ego-external", str(policy_py),
                 "--ego-policy-request", str(policy_request_path)]
        bev_town = BEV_TOWN.get(source.split("__")[0].replace("carla_", ""))
        if bev_town:
            argv += ["--ego-bev-town", bev_town]
    else:
        argv += ["--ego-policy", kind, "--ego-v0", f"{v0:g}"]
    for agent, x, y in goals:
        if actor_ids is not None:
            argv += ["--goal-id", str(actor_ids[agent]), f"{x:g}", f"{y:g}"]
        else:
            argv += ["--goal", str(agent), f"{x:g}", f"{y:g}"]
    for key, value in sorted(idm_kwargs.items()):
        argv += ["--ego-idm", f"{key}={value:g}"]
    if parameters.get("ego_goal_xy"):
        xy = parameters["ego_goal_xy"]
        argv += ["--ego-goal-xy", f"{float(xy[0]):g}", f"{float(xy[1]):g}"]
    if parameters.get("replan_freq"):
        argv += ["--replan-freq", str(int(parameters["replan_freq"]))]

    detail = {
        "source": source,
        "town": source.split("__")[0],
        "scene": source.split("__")[1] if "__" in source else None,
        "goals": [list(g) for g in goals],
        "ego_policy_kind": kind,
        "ego_sensor_worker": sensor_worker,
        "ego_policy_source": (str(policy_py) if policy_py is not None
                              else "native (ego_control.make_policy)"),
        "ego_v0_mps": v0,
        "ego_idm": idm_kwargs,
        "ego_policy_unapplied_parameters": unapplied,
        # Declared by the harness, deliberately not used to steer. See the
        # module docstring.
        "goal_assignment": ("recorded actor id" if actor_ids is not None
                            else "batch index"),
        "agent_actor_ids": ({str(k): v for k, v in actor_ids.items()}
                            if actor_ids is not None else None),
        "prompt_declared": parameters.get("prompt"),
        "prompt_used": False,
    }
    return argv, detail


# --------------------------------------------------------------------------
# rollout CSV -> canonical trace
# --------------------------------------------------------------------------

def read_rollout_csv(path: Path) -> tuple:
    """(rows_by_time, extents) from a rollout_carla export.

    Columns: frame,time,x,y,z,vx,vy,ax,ay,yaw,length,width,height,id,type.
    ``yaw`` is RADIANS (it comes from ``yaw_rad`` in agent_history.json); the
    trace schema wants DEGREES, and that conversion is the one thing in this
    function that silently produces a plausible-looking wrong answer.
    """
    import csv

    by_time = {}
    extents = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            actor = str(row["id"])
            t = round(float(row["time"]), 4)
            by_time.setdefault(t, {})[actor] = (
                float(row["x"]),
                float(row["y"]),
                math.degrees(float(row["yaw"])),
                float(row["vx"]),
                float(row["vy"]),
                float(row["ax"]) if row.get("ax") not in (None, "") else None,
                float(row["ay"]) if row.get("ay") not in (None, "") else None,
            )
            if actor not in extents:
                extents[actor] = {
                    "extent": (float(row.get("length") or 4.5),
                               float(row.get("width") or 2.0)),
                    "type_id": str(row.get("type") or "VEHICLE"),
                }
    return by_time, extents


def resolve_ego_id(meta: dict, actor_ids) -> str:
    """Which CSV id is the rule-driven ego.

    rollout_carla names the rule ego 'ego' in the batch, and 'ego' maps back to
    ``min(id)`` in the source CSV -- so the manifest's ``ego_agent`` is a batch
    name, not a CSV id, and the two must not be confused.
    """
    declared = str(meta.get("ego_agent") or "ego")
    if declared != "ego" and declared in actor_ids:
        return declared
    numeric = [a for a in actor_ids if a.lstrip("-").isdigit()]
    return min(numeric, key=int) if numeric else sorted(actor_ids)[0]


def write_trace(recording, output_dir: Path, csv_path: Path, meta: dict,
                request: dict, detail: dict) -> dict:
    """Transcribe the rollout into states.jsonl + scene.json."""
    by_time, extents = read_rollout_csv(csv_path)
    times = sorted(by_time)
    actor_ids = sorted({a for row in by_time.values() for a in row})
    ego_id = resolve_ego_id(meta, actor_ids)

    evaluation = request.get("evaluation") or {}
    rate_hz = float(evaluation.get("tick_rate_hz") or 10.0)
    parameters = dict((request.get("implementation") or {}).get("parameters") or {})

    recorder = recording.TraceRecorder(
        str(output_dir), rate_hz=rate_hz,
        context={
            "method": "prosim",
            "experiment_id": request.get("experiment_id"),
            "scenario_family": request.get("scenario_family"),
            "source": detail["source"],
            "rollout_csv": csv_path.name,
            "prosim_manifest": meta,
        },
    )
    for actor in actor_ids:
        info = extents.get(actor, {})
        recorder.declare(
            actor,
            extent=info.get("extent"),
            type_id=info.get("type_id"),
            role="ego" if actor == ego_id else "background",
            is_ego=(actor == ego_id),
        )

    # The ego's REFERENCE PATH -- its intended route, taken from the rule ego's
    # lane-graph plan and written into the manifest by rollout_carla.
    #
    # Not optional decoration: metrics.scenario refuses to evaluate an
    # intersection family without it ("ego has no declared reference path;
    # station along the route is undefined"), so a run without this succeeds,
    # ingests cleanly, and still scores nothing.
    #
    # And it must be the ROUTE, never the driven trajectory. Declaring where
    # the car actually went would make path adherence trivially perfect and
    # turn a deviation metric into a tautology.
    route = meta.get("ego_route_polyline")
    if route and len(route) >= 2:
        recorder.declare_path(ego_id, route, source="lane_graph_route")

    # A conflict zone, when the family's implementation entry declares one. Not
    # invented from the trajectories: a zone derived from where the cars
    # actually went would make every run realize its own scenario.
    zone = parameters.get("zone")
    if zone:
        recorder.declare_zone(
            str(zone.get("id", "conflict")), str(zone.get("kind", "junction")),
            center=zone.get("center"), radius=zone.get("radius"),
            polygon=zone.get("polygon"),
        )

    for t in times:
        recorder.tick(t, by_time[t])
    recorder.close()

    return {
        "trace_ego_path_points": len(route) if route else 0,
        "trace_ticks": recorder.n_ticks,
        "trace_actors": len(actor_ids),
        "trace_ego_id": ego_id,
        "trace_zone_declared": bool(zone),
        "trace_recorder_errors": recorder.errors,
        "scenario_duration": (round(times[-1] - times[0], 4) if len(times) > 1
                              else 0.0),
    }


# --------------------------------------------------------------------------

def write_report(output_dir: Path, status: str, metrics: dict,
                 reason=None, **extra) -> None:
    report = {"status": status, "metrics": metrics, "reason": reason}
    report.update(extra)
    (output_dir / "method_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-request", required=True)
    parser.add_argument("--policy-request", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.scenario_request).read_text())
    policy_request = json.loads(Path(args.policy_request).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = request.get("experiment_id", "?")
    evaluation = request.get("evaluation") or {}
    horizon_s = float(evaluation.get("horizon_s") or PROSIM_HORIZON_S)

    print(f"prosim: {experiment_id} "
          f"family={request.get('scenario_family')} "
          f"policy={policy_request.get('name')} seed={request.get('seed')}")

    try:
        recording, harness_root = load_recorder(output_dir)
    except SystemExit as exc:
        # Report it rather than dying silently: without this the harness sees
        # only a non-zero exit code and has to guess why.
        write_report(output_dir, "error", {}, reason=str(exc))
        print(f"prosim: {exc}", file=sys.stderr)
        return 1
    print(f"prosim: trace recorder from {harness_root}")

    try:
        out_csv = output_dir / "rollout.csv"
        argv, detail = build_argv(request, policy_request, out_csv,
                                  harness_root=harness_root,
                                  policy_request_path=args.policy_request)
    except SystemExit as exc:
        write_report(output_dir, "failure", {}, reason=str(exc))
        print(f"prosim: {exc}", file=sys.stderr)
        return 1

    rec = recording_path(detail["source"])
    if not os.environ.get("PROSIM_ROLLOUT_CSV") and not rec.is_file():
        params = dict((request.get("implementation") or {}).get("parameters") or {})
        reason = (f"no recording for {detail['source']}: expected {rec}. Each "
                  f"scenario needs its own CARLA recording at its declared spawn "
                  f"points; nothing falls back to another scene's. Record it "
                  f"(carla-ubuntu20.sif, on the CARLA node):\n"
                  + record_recipe(detail["source"], params))
        write_report(output_dir, "failure", {}, reason=reason,
                     method_metrics=detail)
        print(f"prosim: {reason}", file=sys.stderr)
        return 1

    # Goals by recorded actor id whenever the model will actually run. (With a
    # PROSIM_ROLLOUT_CSV stand-in the argv is never executed.)
    if not os.environ.get("PROSIM_ROLLOUT_CSV"):
        params = dict((request.get("implementation") or {}).get("parameters") or {})
        try:
            ids = assign_actor_ids(rec, params)
            argv, detail = build_argv(request, policy_request, out_csv,
                                      actor_ids=ids,
                                      harness_root=harness_root,
                                      policy_request_path=args.policy_request)
        except SystemExit as exc:
            write_report(output_dir, "failure", {}, reason=str(exc),
                         method_metrics=detail)
            print(f"prosim: {exc}", file=sys.stderr)
            return 1
        print(f"prosim: agent -> actor id {ids}")

    if detail["ego_policy_unapplied_parameters"]:
        print(f"prosim: WARNING policy parameters not applied: "
              f"{detail['ego_policy_unapplied_parameters']}", file=sys.stderr)

    # Horizon. Declared loudly either way -- a 8 s episode scored against a
    # 15 s protocol is not a fair 'scenario did not happen'.
    truncated = horizon_s > PROSIM_HORIZON_S + 1e-6
    if truncated:
        message = (f"family asks for {horizon_s:g} s; this ProSim checkpoint "
                   f"rolls out {PROSIM_HORIZON_S:g} s (DATASET.FORMAT."
                   f"FUTURE_SEC, and TARGET.STEPS is baked into the trained "
                   f"decoder weights)")
        print(f"prosim: HORIZON SHORTFALL -- {message}", file=sys.stderr)
        if os.environ.get("PROSIM_STRICT_HORIZON"):
            write_report(output_dir, "failure", {}, reason=message,
                         method_metrics=detail)
            return 1

    # The rollout. PROSIM_ROLLOUT_CSV substitutes a prepared CSV for the model
    # call, so the request parsing, the trace writing and the report can be
    # exercised without a checkpoint, a GPU or ~90 s of CPU inference.
    started = time.time()
    prepared = os.environ.get("PROSIM_ROLLOUT_CSV")
    if prepared:
        print(f"prosim: PROSIM_ROLLOUT_CSV set -- using {prepared}, model NOT run")
        import shutil
        shutil.copyfile(prepared, out_csv)
        meta_src = Path(str(prepared).replace(".csv", ".meta.json"))
        meta = json.loads(meta_src.read_text()) if meta_src.is_file() else {}
        returncode = 0
    else:
        print("prosim: " + " ".join(argv))
        completed = subprocess.run(argv, cwd=str(REPO_ROOT))
        returncode = completed.returncode
        meta_path = Path(str(out_csv).replace(".csv", ".meta.json"))
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    rollout_s = time.time() - started

    if returncode != 0 or not out_csv.is_file():
        reason = (f"rollout_carla.py exited {returncode} "
                  f"and wrote {'no' if not out_csv.is_file() else 'an'} CSV")
        write_report(output_dir, "failure", {}, reason=reason,
                     method_metrics=detail)
        print(f"prosim: {reason}", file=sys.stderr)
        return 1

    trace_info = write_trace(recording, output_dir, out_csv, meta,
                             request, detail)
    print(f"prosim: wrote {trace_info['trace_ticks']} ticks for "
          f"{trace_info['trace_actors']} actors "
          f"(ego={trace_info['trace_ego_id']})")

    if trace_info["trace_ticks"] < 2:
        reason = ("the rollout produced fewer than 2 usable ticks; the trace "
                  "would be unevaluable")
        write_report(output_dir, "failure", {}, reason=reason,
                     method_metrics={**detail, **trace_info})
        return 1

    detail.update(trace_info)
    detail.update({
        "rollout_seconds": round(rollout_s, 2),
        "horizon_requested_s": horizon_s,
        "horizon_delivered_s": PROSIM_HORIZON_S,
        "horizon_truncated": truncated,
        # Recorded, not applied -- ProSim's rollout exposes no seed.
        "seed_declared": request.get("seed"),
        "seed_applied": False,
        "prosim_manifest": meta,
    })

    write_report(
        output_dir, "success",
        # Provenance only. scenario_realized / collision / time_to_event are
        # derived from the trace by metrics.scenario, never self-reported.
        {"scenario_duration": trace_info["scenario_duration"]},
        method_metrics=detail,
        trace_path="trace.json" if (output_dir / "trace.json").exists() else None,
    )
    print("prosim: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
