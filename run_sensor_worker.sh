#!/usr/bin/env bash
# Start the sensor worker on the CARLA node, natively (not in Apptainer), in the
# policy's own Python 3.10 environment. It serves ProSim's run.py, which reaches
# it through PROSIM_SENSOR_WORKER=HOST:PORT.
#
#   ./run_sensor_worker.sh [--port 2100] [--carla-port 2000] [--allow-load-town] [--once]
#
# The CARLA dist's PythonAPI/carla supplies `agents.navigation`, which lead
# imports; it is the 0.9.16 copy that matches the server, not the 0.9.15 tree
# tfv6 expects under 3rd_party/ (not checked out).
set -euo pipefail
CARLA_DIST=${CARLA_DIST:-/home/veerk41/scratch/Carla/Dist/CARLA_Shipping_294096eb1-dirty/LinuxNoEditor}
VENV=${VENV:-/scratch/veerk41/venvs/tfv6}
export PYTHONPATH="$CARLA_DIST/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
export PIP_CONFIG_FILE=/dev/null
cd "$(dirname "$0")"
exec "$VENV/bin/python" -u sensor_worker.py "$@"
