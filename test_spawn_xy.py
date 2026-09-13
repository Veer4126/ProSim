"""Tests for record_actor_history.py --spawn-xy. NO CARLA server, NO GPU.

`carla` is stubbed with a fake map whose get_waypoint() snaps to a hand-built
road grid, so the selection/snapping/ordering logic is exercised for real while
the simulator is synthesised. Ordering is the part that actually matters:
carla_dataset maps min(actor id) -> 'ego' and CARLA hands out ids in spawn
order, so --spawn-xy's order silently decides which car is the ego.

    apptainer exec -B /scratch/veerk41:/workspace \
        ~/projects/aip-gigor/veerk41/carla-ubuntu20.sif \
        python3 /workspace/test_spawn_xy.py
"""

import math
import sys
import types

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


# --------------------------------------------------------------------------
# fake carla: two roads crossing, matching Town10HD's real geometry near the
# intersection the scenario uses -- a southbound carriageway at x=-45 and an
# eastbound one at y=24.5.
# --------------------------------------------------------------------------
class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()


class LaneType:
    Driving = "Driving"


class Waypoint:
    def __init__(self, x, y, z, yaw, road_id, lane_id):
        self.transform = Transform(Location(x, y, z), Rotation(0.0, yaw, 0.0))
        self.road_id, self.lane_id = road_id, lane_id


class FakeMap:
    """Southbound lane along x=-45.1 (yaw -90) and eastbound along y=24.5 (yaw 0)."""

    name = "Carla/Maps/Town10HD_Opt"      # record_actor_history prints it

    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        cands = [
            (Waypoint(-45.1, loc.y, 0.2, -90.0, 934, -1), abs(loc.x - (-45.1))),
            (Waypoint(loc.x, 24.5, 0.2, 0.0, 255, -1), abs(loc.y - 24.5)),
        ]
        wp, d = min(cands, key=lambda t: t[1])
        return wp if d < 40.0 else None      # nothing else is road


class FakeWorld:
    def get_map(self):
        return FakeMap()


carla = types.ModuleType("carla")
carla.Location, carla.Rotation, carla.Transform = Location, Rotation, Transform
carla.LaneType = LaneType
sys.modules["carla"] = carla

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import record_actor_history as rah

world = FakeWorld()

print("=" * 74)
print("1. snapping to the lane")
print("=" * 74)
got = rah.spawn_points_from_xy(world, [-85.0, 24.5, -45.0, 100.0], 0.5)
check("one transform per coordinate pair", len(got) == 2, f"{len(got)}")

(lab0, tf0), (lab1, tf1) = got
check("the ego point lands on the EASTBOUND lane (yaw ~0)",
      abs(tf0.rotation.yaw) < 1e-6, f"yaw={tf0.rotation.yaw:+.1f}")
check("the other point lands on the SOUTHBOUND lane (yaw ~-90)",
      abs(tf1.rotation.yaw + 90.0) < 1e-6, f"yaw={tf1.rotation.yaw:+.1f}")
check("lateral snap moved it onto the lane centre",
      abs(tf0.location.y - 24.5) < 1e-6 and abs(tf1.location.x - (-45.1)) < 1e-6,
      f"ego y={tf0.location.y:.2f}, other x={tf1.location.x:.2f}")
check("along-lane coordinate is preserved, not snapped away",
      abs(tf0.location.x - (-85.0)) < 1e-6 and abs(tf1.location.y - 100.0) < 1e-6,
      f"ego x={tf0.location.x:.2f}, other y={tf1.location.y:.2f}")
check("--spawn-z lifts the car off the road surface",
      abs(tf0.location.z - 0.7) < 1e-9, f"z={tf0.location.z:.2f} (lane 0.2 + 0.5)")

print()
print("=" * 74)
print("2. ORDER: --spawn-xy decides which car becomes 'ego'")
print("=" * 74)
labels = [lab for lab, _ in got]
print(f"  labels in order: {labels}")
check("order is preserved exactly as given",
      labels == ["xy(-85,24.5)", "xy(-45,100)"], f"{labels}")
# reversing the argument list must reverse the spawn order
rev = rah.spawn_points_from_xy(world, [-45.0, 100.0, -85.0, 24.5], 0.5)
check("CONTROL: reversing the arguments reverses the spawn order",
      [l for l, _ in rev] == list(reversed(labels)), f"{[l for l, _ in rev]}")

print()
print("=" * 74)
print("3. bad input fails loudly")
print("=" * 74)
try:
    rah.spawn_points_from_xy(world, [-85.0, 24.5, -45.0], 0.5)
    check("an odd number of values raises", False, "it was accepted")
except SystemExit as e:
    check("an odd number of values raises", "even number" in str(e))

try:
    rah.spawn_points_from_xy(world, [5000.0, 5000.0], 0.5)   # far off any road
    check("a point with no driving lane raises", False, "it was accepted")
except SystemExit as e:
    check("a point with no driving lane raises", "not near any driving lane" in str(e))

print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("   FAILED:", f)
print("=" * 74)
sys.exit(1 if FAIL else 0)
