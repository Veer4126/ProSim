"""Scene-tagged sources resolve to their OWN recording, and never fall back.

    apptainer exec -B /scratch/veerk41:/workspace prosim_v4.sif \
        python3 /workspace/test_scene_names.py

No model, no CARLA. The resolver only touches the filesystem, so it is tested
against a throwaway data dir where every fallback file EXISTS -- a resolver
that silently falls back would pass a test in a directory where the fallbacks
are missing.
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from carla_dataset import CarlaSceneSource as S

P, F = [], []
def check(name, ok, detail=""):
    (P if ok else F).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

print("=== split_env / town_of ===")
for env, want in [("carla_town10hd__left_turn", ("town10hd", "left_turn")),
                  ("carla_town04__cut_in",      ("town04", "cut_in")),
                  ("carla_town04",              ("town04", None)),
                  ("CARLA_Town10HD__Red_Light", ("town10hd", "red_light"))]:
    check(f"{env} -> {want}", S.split_env(env) == want, str(S.split_env(env)))
check("town_of drops the scene", S.town_of("carla_town10hd__x") == "town10hd")

d = Path(tempfile.mkdtemp())
for f in ["history_20hz.csv", "history_town04.csv", "history_town10hd.csv",
          "history_town10hd__red_light.csv", "town10hd_lanes.json", "town04_lanes.json"]:
    (d / f).write_text("x")

print("=== tagged sources, with EVERY fallback file present ===")
csv, lanes = S.resolve_paths(d, "carla_town10hd__red_light")
check("tagged -> its own CSV", csv.name == "history_town10hd__red_light.csv", csv.name)
check("tagged -> the TOWN's lane graph", lanes.name == "town10hd_lanes.json", lanes.name)
try:
    csv, _ = S.resolve_paths(d, "carla_town10hd__left_turn")
    check("CONTROL: missing scene CSV refuses (no fallback)", False,
          f"fell back to {csv.name}")
except FileNotFoundError as e:
    check("CONTROL: missing scene CSV refuses (no fallback)", True)
    check("  and the error names the expected file",
          "history_town10hd__left_turn.csv" in str(e))
    check("  and gives the record + extract commands",
          "record_actor_history.py" in str(e) and "extract_csv.py" in str(e))

print("=== untagged sources behave exactly as before ===")
check("carla_town04 -> history_town04.csv",
      S.resolve_paths(d, "carla_town04")[0].name == "history_town04.csv")
check("carla_town10hd -> history_town10hd.csv when present",
      S.resolve_paths(d, "carla_town10hd")[0].name == "history_town10hd.csv")
(d / "history_town10hd.csv").unlink()
check("carla_town10hd -> history_20hz.csv otherwise",
      S.resolve_paths(d, "carla_town10hd")[0].name == "history_20hz.csv")

print("=== the real data dir ===")
csv, lanes = S.resolve_paths("/workspace", "carla_town10hd__red_light")
check("red_light resolves to its copied recording", csv.is_file(), str(csv))

print(f"\n{len(P)}/{len(P) + len(F)} checks passed")
sys.exit(1 if F else 0)
