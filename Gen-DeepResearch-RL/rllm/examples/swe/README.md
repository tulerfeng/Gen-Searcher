<h1 align="center"> DeepSWE-Preview - Training a State-of-the-Art Coding Agent by Scaling RL </h1>

<!-- paper . data and models . project page -->
<p align="center">
<a href="https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art[…]-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33?pvs=73">📃 Blog Post</a>
•
<a href="https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset" > 🤗 HF Dataset (R2E-Gym) </a>
•
<!-- project page -->
<a href="https://wandb.ai/mluo/deepswe" >🔥 WandB Logs</a>
•
<a href="https://huggingface.co/agentica-org/DeepSWE-Preview" > 🤗 DeepSWE-Preview</a>
•
<a href="https://drive.google.com/file/d/10LIwpJeaFuiX6Y-qEG2a4a335PEuQJeS/view?usp=sharing" > 📈 Evaluation Logs</a>
•
<a href="https://agentica-project.com/" > 🌐 Project Page</a>
•
<a href="https://github.com/rllm-org/rllm" > 🧑‍💻 Code</a>
</p>

<div align="center">

[![Github](https://img.shields.io/badge/RLLM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/rllm-org/rllm)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://www.agentica-project.com) 
[![Twitter](https://img.shields.io/badge/Agentica-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/Agentica_)
[![Hugging Face Collection](https://img.shields.io/badge/Agentica-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/agentica-org)

</div>

We introduce DeepSWE-Preview, a reasoning-enabled coding agent trained from scratch from Qwen3-32B with only reinforcement learning (RL). It achieves 59.2% on SWE-Bench-Verified with test-time scaling, reaching SOTA for open-weight coding agents (42.2% Pass@1, 71.0% Pass@16).

DeepSWE is trained using [**rLLM**](https://github.com/rllm-org/rllm), our framework for post-training language agents using high-quality SWE environments from [**R2E-Gym**](https://github.com/R2E-Gym/R2E-Gym). We’ve open-sourced everything—our dataset, code, training, and evaluation logs, for everyone to progress on scaling and improving agents with RL.

## Quick Start 🎯

### 1. 📦 Installation
```bash
# Installing Python 3.10 Environment.
conda create -n rllm python=3.10 -y
conda activate rllm

# Installing RLLM dependencies.
cd rllm
pip install -e ./verl
pip install -e ./verl[vllm]
pip install -e .
```

Also, install [**R2E-Gym**](https://github.com/R2E-Gym/R2E-Gym) for high-quality SWE-Bench environments used for RL training.
```bash
git clone https://github.com/agentica-project/R2E-Gym.git
cd R2E-Gym
pip install -e .
```


### 2. 🤗 Data and Agent Scaffold

We provide two ways to interface with SWE-based environments.

**rLLM**

rLLM's `SWEEnv` provides a nice wrapper and abstraction on top of `R2E-Gym`. Here is a short code snippet:
```python
from rllm.environments.swe.swe import SWEEnv
from datasets import load_dataset

# load gym dataset
ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
idx = 0

env = SWEEnv(entry=ds[idx], backend='kubernetes', scaffold='r2egym')
env.reset()
env.close()
```

For further integration with rLLM's agents, see `run_deepswe.py`.

**R2E-Gym**

R2E-Gym environment can be simply used as:
```python
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from pathlib import Path
from datasets import load_dataset

# load gym dataset
ds = load_dataset("R2E-Gym/R2E-Gym-Subset")
split = 'train'

# load gym environment
env_index = 100 # index of the environment [0, len(ds)]
env_args = EnvArgs(ds = ds[split][env_index])
env = RepoEnv(env_args)

# load agent
agent_args = AgentArgs.from_yaml(Path('./src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml'))
# define llm: ['claude-3-5-sonnet-20241022', 'gpt-4o', 'vllm/agentica-org/DeepSWE-Preview']
agent_args.llm_name = 'vllm/agentica-org/DeepSWE-Preview'
agent = Agent(name="EditingAgent", args=agent_args)

# run the agent
output = agent.run(env, max_steps=40)
```

## 🤖 3. Running DeepSWE-Preview Inference

First, start the VLLM server to serve the DeepCoder model:

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

> ⚠️ **Important**: Wait for the server to fully load before proceeding to the next step. You should see logs indicating the server is ready to accept requests.


In a new terminal session, run the DeepSWE agent evaluation:

```bash
# Activate the virtual environment (if in new terminal)
source .venv/bin/activate

export EXP_NAME="deepswe-run"
export TEMP=1.0

# Run the DeepSWE agent on SWE-Bench Verified
time python src/r2egym/agenthub/run/edit.py runagent_multiple \
    --traj_dir "./traj" \
    --max_workers 48 \
    --start_idx 0 \
    --k 500 \
    --dataset "R2E-Gym/SWE-Bench-Verified" \
    --split "test" \
    --llm_name "openai/agentica-org/DeepSWE-Preview" \
    --scaffold "r2egym" \
    --use_fn_calling False \
    --exp_name "$EXP_NAME" \
    --temperature "$TEMP" \
    --max_steps_absolute 100 \
    --backend "docker" \
    --condense_history False \
    --max_reward_calc_time 1200 \
    --max_tokens 65536
```

**Parameter Explanation:**
- `--max_workers 54`: Number of parallel workers for processing, reduce if you hit trajectory time limit errors
- `--k 500`: Number of instances to evaluate (max 500 for SWE-Bench Verified)
- `--temperature 1`: Sampling temperature for model responses
- `--max_steps 40`: Soft maximum steps per trajectory. Outputs warnings after steps exceed `max_steps`.
- `--max_steps_absolute 100`: Absolute maximum steps limit

> 📊 **Expected Runtime**: This evaluation may take several hours depending on your hardware configuration.

**Trajectory Visualization:** 
The generated trajectories are saved in `./traj` directory. You can visualize the trajectories using the trajectory visualization tool in [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym),
```bash
cd R2E-Gym
python app/app.py --traj_dir "./traj"
```

## 🔥 4. Training DeepSWE-Preview with rLLM and R2E-Gym

To train DeepSWE-Preview, we suggest deploying a Kubernetes (K8) cluster on AWS/GCP/Azure. Each node should have a large number of CPUs and diskspace. Each node in our K8 cluster contains 200 CPUs and over 6 TB+ of disk space to store 1000s of Docker images.

To run Kubernetes locally, we suggest installing [`kind`](https://kind.sigs.k8s.io/) and launching it with `kind create cluster`. However, please do note that this is not sufficient to launch a full training run.

We provide the exact scripts to replicate our training curves in the DeepSWE-Preview blog post. It requires at least 64 GPUs and launches 512 Docker containers in parallel.
```bash
cd rllm/examples/swe
bash deepswe_32b.sh
```

## 🔬 5. DeepSWE-Preview Reproduction Guide

Please refer the following for detailed reproduction guide for DeepSWE-Preview.
* [DeepSWE-Preview Reproduction Guide](https://github.com/agentica-project/R2E-Gym/blob/master/reproduction/DEEPSWE_REPRODUCTION.MD)
* [DeepSWE-Preview with Hybrid Test-time Scaling](https://github.com/agentica-project/R2E-Gym/blob/master/reproduction/DEEPSWE_TTS_REPRODUCTION.MD)

## Citation

```
@misc{deepswe2025,
  title={DeepSWE: Training a State-of-the-Art Coding Agent from Scratch by Scaling RL},
  author={Michael Luo and Naman Jain and Jaskirat Singh and Sijun Tan and Ameen Patel and Qingyang Wu and Alpay Ariyak and Colin Cai and Tarun Venkat and Shang Zhu and Ben Athiwaratkun and Manan Roongta and Ce Zhang and Li Erran Li and Raluca Ada Popa and Koushik Sen and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33}},
  note={Notion Blog},
  year={2025}
}
```
