#!/usr/bin/env bash
# Gen Image training script: image generation task
set -xeuo pipefail


# Ensure we run under the rllm directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${RLLM_DIR}"

# Load Gen Image environment variables
if [ -f .env.gen_image ]; then 
    echo "[GenImageTrain] Loading .env.gen_image"
    set -a
    source .env.gen_image
    set +a
else
    echo "[GenImageTrain] WARNING: .env.gen_image not found"
fi


# ========= rollout =========
rollout_mode="async"
rollout_name="sglang"
#rollout_name="vllm"
if [ "$rollout_mode" = "async" ]; then
  return_raw_chat="True"
else
  return_raw_chat="False"
fi

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1

dtype="bfloat16"
adv_estimator="grpo"
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
clip_ratio_high=0.28

train_prompt_bsz=8
n_resp_per_prompt=6
train_prompt_mini_bsz=8
n_parallel_tasks=64
n_parallel_tools=128

max_prompt_length=4096
max_response_length=30000
# SGLang per-call generation limit = GEN_MAX_NEW_TOKENS_PER_TURN + 128, aligned with agent truncation to reduce OOM
sglang_response_length=$((${GEN_MAX_NEW_TOKENS_PER_TURN:-4096} + 128))
use_dynamic_bsz=True
actor_ppo_max_token_len_per_gpu=37000
infer_ppo_max_token_len_per_gpu=37000
offload=True
gen_tp=2

temperature=0.7
top_p=0.95
top_k=-1
val_top_p=0.95
loss_agg_mode="seq-mean-token-sum"

NNODES=1
project_name='deepresearch-gen-image'
exp_name='gen_rl'

# ========= Dataset =========
# Dataset name (must be registered to DatasetRegistry via register_gen_rl_dataset.py)
# Default is Vision-DeepResearch-Gen; can be overridden via DATASET_NAME
export DATASET_NAME="GenRL"

# ========= paths (local model) =========

MODEL_PATH=Gen-Searcher-SFT-8B
CKPTS_DIR=checkpoints/${project_name}/${exp_name}

echo "[GenImageTrain] ===== Starting Training ====="
echo "[GenImageTrain] Project: ${project_name}"
echo "[GenImageTrain] Experiment: ${exp_name}"
echo "[GenImageTrain] Dataset: ${DATASET_NAME}"
echo "[GenImageTrain] Model: ${MODEL_PATH}"
echo "[GenImageTrain] Checkpoints: ${CKPTS_DIR}"
echo "[GenImageTrain] ============================="


CUDA_LAUNCH_BLOCKING=1 python3 -m vision_deepresearch_async_workflow.train_image_deepresearch_workflow_fsdp_gen \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=32 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=${return_raw_chat} \
    data.seed=42 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.response_length=${sglang_response_length} \
    actor_rollout_ref.rollout.dtype=${dtype} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=${n_parallel_tasks} \
    rllm.workflow.n_parallel_tools=${n_parallel_tools} \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_unknown=True \
    rllm.compact_filtering.mask_error=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_repeated_response=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=20 \
    trainer.test_freq=0 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}"

echo "[GenImageTrain] ===== Training Completed ====="
