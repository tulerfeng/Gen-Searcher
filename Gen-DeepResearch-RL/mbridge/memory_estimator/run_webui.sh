#!/bin/bash

# Set the path to your Megatron-LM directory
export MEGATRON_PATH="/PATH/TO/Megatron-LM"

# Get the absolute path of the directory containing this script.
# This makes the script runnable from any location.
ESTIMATOR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Add Megatron-LM and the estimator directory to PYTHONPATH
export PYTHONPATH="${ESTIMATOR_DIR}:${MEGATRON_PATH}:${PYTHONPATH}"

# Change to the estimator directory so uvicorn can find the 'webui' module.
cd "${ESTIMATOR_DIR}"
uvicorn webui.main:app --host 0.0.0.0 --port 8000 