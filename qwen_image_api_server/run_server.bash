#!/bin/bash

# Prefix for services.yaml entry "${SERVICE_NAME}_edit" (optional override: $0 <name>)
SERVICE_NAME="${1:-combined_qwen_services_1}"

# Start qwen-image-edit service (8 GPUs, port 8001)
echo "Starting qwen-image-edit service..."

# Register qwen-image-edit service
python3 "./register_service.py" "${SERVICE_NAME}_edit" "qwen-image-edit" 8001

# Run qwen-image-edit in the background: 8 GPUs, one model per GPU for concurrent requests
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./qwen-image-edit/api.py --num_gpus 8 --port 8001 &
QWEN_EDIT_PID=$!
echo "qwen-image-edit service started, PID: $QWEN_EDIT_PID, port: 8001"

echo "All services are up."
echo "qwen-image-edit: http://localhost:8001 (8 GPUs)"

# Wait while services run
wait
