"""The traffic-light record ProSim writes into its trace, read back by the harness.

    /scratch/veerk41/venvs/tfv6/bin/python tests/test_signal_record.py

The harness scores red_light against `context.signal_plan`
(metrics/scenario/scenes.py `light_plan`) and nothing else. So every record here
is written with the harness's REAL recorder (loaded the way run.py loads it)
through run.py's `record_signals`, and read back with the harness's REAL
`light_plan` -- the writer and the reader the scoring uses. No CARLA, no GPU.
"""

from __future__ import annotations

# Run from anywhere: the repo root goes on the import path and becomes the
# working directory (tests read prosim_demo/..., demo_dataset/... relatively).
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))  # realpath: works via symlinks
_sys.path.insert(0, _REPO)
_os.chdir(_REPO)

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HARNESS = Path(os.environ.get("HARNESS", "/scratch/veerk41/scenario_orchestration"))
os.environ["SCENARIO_ORCHESTRATION_ROOT"] = str(HARNESS)

spec = importlib.util.spec_from_file_location("_prosim_run", "scenario_orchestration/run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
sys.path.insert(0, str(HARNESS))
from metrics.scenario.scenes import light_plan  # noqa: E402

PASS, FAIL = [], []
TIMES = [round(0.1 * k, 4) for k in range(80)]


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def record(family, parameters, meta):
    """(record_signals detail, scene.json context, the harness's light_plan of it)."""
    out = Path(tempfile.mkdtemp())
    recording, _root = R.load_recorder(out)
    rec = recording.TraceRecorder(str(out), rate_hz=10.0, context={"method": "prosim"})
    rec.declare("ego", extent=(4.5, 2.0), type_id="VEHICLE", role="ego", is_ego=True)
    detail = R.record_signals(rec, family, parameters, meta, TIMES)
    for t in TIMES:
        rec.tick(t, {"ego": (0.0, 0.0, 0.0, 0.0, 0.0, None, None)})
    rec.close()
    context = json.loads((out / "scene.json").read_text()).get("context") or {}
    return detail, context, light_plan(context)


def carla_meta(log, frozen=True):
    return {"ego_remote_session": {
        "init": {"ego_light": {"requested": "green", "applied": "green", "frozen": frozen}},
        "signal_roles": {"1": "ego", "2": "opposing", "3": "crossing"},
        "light_log": log}}


def entry(step, ego="Green", crossing="Red"):
    return {"step": step, "ego": [ego], "opposing": ["Green"], "crossing": [crossing],
            "lights": {"1": ego, "2": "Green", "3": crossing}}


def main():
    print("\n=== 1. CARLA arm: the lights the worker read back off CARLA ===")
    detail, ctx, plan = record("red_light", {}, carla_meta([entry(k) for k in range(81)]))
    check("recorded as 'simulator', one entry per readout offered",
          detail == {"signal_source": "simulator", "signal_entries": 81}, str(detail))
    check("the harness reads it: ego green, crossing red, frozen, from the simulator",
          plan is not None and plan["ego"] == "green" and plan["crossing"] == "red"
          and plan["frozen"] is True and plan.get("recorded_from") == "simulator", str(plan))
    check("an unchanging light is ONE timeline entry, at the first tick",
          [e[0] for e in plan["timeline"]] == [0.0], str(plan["timeline"]))
    check("the light ids and their roles are kept beside the record",
          ctx["signal_plan"].get("roles") == {"1": "ego", "2": "opposing", "3": "crossing"},
          str(ctx["signal_plan"].get("roles")))

    log = [entry(k) for k in range(40)] + [entry(k, ego="Red", crossing="Green") for k in range(40, 81)]
    _, _, plan = record("red_light", {}, carla_meta(log, frozen=False))
    check("a light that changes at step 40 starts a new entry at t = 4.0 s",
          [(e[0], e[1], e[2]) for e in plan["timeline"]] == [(0.0, "green", "red"), (4.0, "red", "green")],
          str(plan["timeline"]))

    print("\n=== 2. no-CARLA arm: the scenario's declared phase ===")
    detail, ctx, plan = record("red_light", {}, {})
    check("red_light with no simulator is 'declared': ego green, crossing red, frozen",
          detail["signal_source"] == "declared" and plan is not None and plan["ego"] == "green"
          and plan["crossing"] == "red" and plan["frozen"] and plan.get("recorded_from") == "declared",
          str(plan))
    _, _, plan = record("left_turn", {"ego_light": "red"}, {})
    check("an implementation's own ego_light wins: red for the ego, green across",
          plan is not None and plan["ego"] == "red" and plan["crossing"] == "green", str(plan))
    detail, ctx, plan = record("cut_in", {}, {})
    check("a family with no light is 'none', and the harness finds no plan",
          detail["signal_source"] == "none" and plan is None
          and ctx["signal_plan"].get("source") == "none", str(ctx.get("signal_plan")))

    print("\n=== 3. CONTROLS: lights are never claimed that were not read ===")
    detail, ctx, plan = record("red_light", {}, carla_meta([entry(k, ego=None) | {"ego": []}
                                                            for k in range(81)]))
    check("CONTROL: a CARLA run that found no light facing the ego is 'none', not 'declared'",
          detail["signal_source"] == "none" and plan is None, str(detail))
    old = {"ego_remote_session": {"init": {"ego_light": {"requested": "green", "frozen": True}}}}
    detail, ctx, plan = record("red_light", {}, old)
    check("CONTROL: a CARLA run from before the light log is 'none' (it must be rerun)",
          detail["signal_source"] == "none" and plan is None, str(detail))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
