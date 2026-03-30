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

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly TP_SIZE=2
readonly PP_SIZE=2
readonly CP_SIZE=2
readonly EP_SIZE=2

echo "INFO
__POD_IP__ $__POD_IP__
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

readonly SAMPLE_TYPE="mix"
# run the huggingface fwd
python example/qwen3vl/hf_fwd_moe.py \
    --model_path ../hf-hub/Qwen/Qwen3-VL-30B-A3B-Instruct \
    --sample_type $SAMPLE_TYPE

torchrun $DISTRIBUTED_ARGS \
    example/qwen3vl/load_model_and_forward.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --ep $EP_SIZE \
    --etp 1 \
    --cp $CP_SIZE \
    --model_path ../hf-hub/Qwen/Qwen3-VL-30B-A3B-Instruct \
    --sample_type $SAMPLE_TYPE \
    --check_export

torchrun $DISTRIBUTED_ARGS \
    example/qwen3vl/load_model_and_inference.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --ep $EP_SIZE \
    --etp 1 \
    --cp $CP_SIZE \
    --model_path ../hf-hub/Qwen/Qwen3-VL-30B-A3B-Instruct \
    --sample_type $SAMPLE_TYPE


# run the huggingface fwd
python example/qwen3vl/hf_fwd.py \
    --model_path ../hf-hub/Qwen/Qwen3-VL-4B-Instruct \
    --sample_type $SAMPLE_TYPE

torchrun $DISTRIBUTED_ARGS \
    example/qwen3vl/load_model_and_forward.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --cp $CP_SIZE \
    --model_path ../hf-hub/Qwen/Qwen3-VL-4B-Instruct \
    --sample_type $SAMPLE_TYPE \
    --check_export

torchrun $DISTRIBUTED_ARGS \
    example/qwen3vl/load_model_and_inference.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --cp $CP_SIZE \
    --model_path ../hf-hub/Qwen/Qwen3-VL-4B-Instruct \
    --sample_type $SAMPLE_TYPE
