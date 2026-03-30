# DeepSWE Software Engineering Agent Example

This example demonstrates training and running [DeepSWE](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art[%E2%80%A6]-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33), a software-engineering agent trained on top of Qwen3-32B to search, view, and navigate codebases. The model achieves an impressive **59.0%** on SWE-Bench-Verified, which is currently #1 in the open-weights category.

## Overview

The DeepSWE examples demonstrate:

- How to use rLLM's SWEAgent for software engineering tasks.
- How to train DeepSWE with compact filtering.
- How to evaluate DeepSWE over SWE-Bench-Verified.

## Quick Start

### Setup Coding Data

First, prepare your coding datasets:

```bash
cd examples/swe
python prepare_swe_data.py
```

### Model Hosting

Start a model server using vLLM:

```bash
# Start VLLM server with tensor parallelism across 8 GPUs
export MAX_CONTEXT_LEN=65536
export TENSOR_PARALLEL_SIZE=8
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve agentica-org/DeepSWE-Preview \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching
```

### Run/Evaluate DeepSWE Agent on SWE-Bench-Verified

```bash
python run_deepswe.py
```

To fully reproduce DeepSWE's evaluation, see the official R2E-Gym [repo](https://github.com/agentica-project/R2E-Gym/tree/master/reproduction) for more details.

### Train DeepSWE Agent

To train DeepSWE, we suggest deploying a Kubernetes (K8) cluster on AWS/GCP/Azure. Each node should have a large number of CPUs and diskspace. Each node in our K8 cluster contains 200 CPUs and over 6 TB+ of disk space to store 1000s of Docker images.

To run Kubernetes locally, we suggest installing [`kind`](https://kind.sigs.k8s.io/) and launching it with `kind create cluster`. However, please do note that this is not sufficient to launch a full training run.

Next, run the bash script below:

```bash
# Train with 16K context
bash train_deepswe_32b.sh
```

## Code Reference

### SWE Agent Runner

Main script for evaluating SWE-Bench performance:

```python title="examples/deepcoder/run_deepswe.py"
--8<-- "examples/swe/run_deepswe.py"
```

### Training Script

DeepSWE training configuration:

```python title="examples/deepcoder/train_deepswe_agent.py"
--8<-- "examples/swe/train_deepswe_agent.py"
```

For detailed setup instructions, see the [README](https://github.com/rllm-org/rllm/blob/main/examples/swe/README.md) in the deepswe example directory.
