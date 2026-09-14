#!/usr/bin/env bash
# Start the sensor worker on the CARLA node, natively (not in Apptainer), in the
# policy's own environment. ProSim's run.py reaches it through
# PROSIM_SENSOR_WORKER=HOST:PORT.
#
#   ./run_sensor_worker.sh [--port 2100] [--carla-port 2000] [--allow-load-town] [--once]
#
#   VENV        the policy's environment (default venvs/tfv6; SimLingo: venvs/simlingo)
#   CARLA_DIST  CARLA install; its PythonAPI/carla supplies `agents.navigation`,
#               and it is exported as CARLA_ROOT for policies that look it up
#   WORKER_CWD  working directory (SimLingo reads pretrained/InternVL2-1B from it)
#   HF_HOME     Hugging Face cache (default on scratch rather than $HOME)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CARLA_DIST=${CARLA_DIST:-/home/veerk41/scratch/Carla/Dist/CARLA_Shipping_294096eb1-dirty/LinuxNoEditor}
VENV=${VENV:-/scratch/veerk41/venvs/tfv6}
export CARLA_ROOT="${CARLA_ROOT:-$CARLA_DIST}"
export PYTHONPATH="$CARLA_DIST/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/veerk41/hf_cache}"
export PIP_CONFIG_FILE=/dev/null
cd "${WORKER_CWD:-$HERE}"
exec "$VENV/bin/python" -u "$HERE/sensor_worker.py" "$@"
