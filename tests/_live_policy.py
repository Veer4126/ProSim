"""Shared by the live policy checks: a policy request built the way a harness cell
builds it, from configs/policy/<name>.yaml, and where its entry point lives."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

HARNESS = Path(os.environ.get("HARNESS", "/scratch/veerk41/scenario_orchestration"))


def policy_request(name: str) -> dict:
    cfg = yaml.safe_load((HARNESS / "configs" / "policy" / f"{name}.yaml").read_text())
    request = {"name": name, "implementation": cfg["implementation"], "interface": cfg["interface"],
               "observation_space": cfg["observation_space"], "action_space": cfg["action_space"],
               "seed": 0, "parameters": dict(cfg.get("parameters") or {})}
    if cfg.get("checkpoint"):
        request["checkpoint"] = cfg["checkpoint"]
    return request


def policy_py(name: str) -> Path:
    cfg = yaml.safe_load((HARNESS / "configs" / "policy" / f"{name}.yaml").read_text())
    return HARNESS / cfg["repository"] / cfg["entry_point"]
