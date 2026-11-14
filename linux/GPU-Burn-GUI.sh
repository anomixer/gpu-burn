#!/bin/sh
# Launch the Linux GPU Burn GUI from any working directory.
# The Python script finds the native gpu_burn backend next to the repository
# root (or through GPU_BURN_GUI_BACKEND), so it is safe here.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "${SCRIPT_DIR}/GPU-Burn-GUI.py" "$@"
