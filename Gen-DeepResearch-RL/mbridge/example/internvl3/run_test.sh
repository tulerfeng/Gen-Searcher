#!/bin/bash
ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

DIR="$(cd "$( dirname "$0" )" && pwd)"
# mbridge
cd ${DIR}/../..

export PYTHONPATH==$DIR/../..:$DIR/../../../Megatron-LM:$PYTHONPATH
echo "PYTHONPATH ${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

readonly GPUS_PER_NODE=2
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

# only support tp for demo
readonly TP_SIZE=2
readonly PP_SIZE=1
readonly CP_SIZE=1

echo "INFO
__POD_IP__ $__POD_IP__
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

torchrun $DISTRIBUTED_ARGS \
    example/internvl3/load_model_and_forward.py \
    --tp $TP_SIZE \
    --ep $PP_SIZE \
    --model_path ../hf-hub/OpenGVLab/InternVL3-2B


torchrun $DISTRIBUTED_ARGS \
    example/internvl3/load_model_and_inference.py \
    --tp $TP_SIZE \
    --ep $PP_SIZE \
    --model_path ../hf-hub/OpenGVLab/InternVL3-2B
