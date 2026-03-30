# DeepCoder Training Examples

This directory contains examples for training and running DeepCoder, a code reasoning LLM fine-tuned from DeepSeek-R1-Distill-Qwen-14B using distributed reinforcement learning (RL). 

Our examples uses the following:
* DeepSeek-R1-Distill-Qwen-14B as the base model
* agentica-org/DeepCoder-Preview-Dataset (lcbv5 subset) for training and evaluation



## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model agentica-org/DeepCoder-14B-Preview \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --max-model-len 65536
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path agentica-org/DeepCoder-14B-Preview \ 
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU 
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the DeepCoder Preview Dataset:

```bash
cd examples/deepcoder
python prepare_deepcoder_data.py
```

This will:
- Download the agentica-org/DeepCoder-Preview-Dataset (lcbv5 subset)
- Register both train/test splits with the RLLM DatasetRegistry

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/deepcoder
python run_deepcoder.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 64)
- `model_name`: Model to use (default: "agentica-org/DeepCoder-14B-Preview")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 64000)
- `max_prompt_length`: Maximum prompt length (default: 2048)
- `temperature`: Sampling temperature (default: 0.6)
- `top_p`: Top-p sampling (default: 0.95)

The script will:
1. Load the DeepCoder Preview test dataset
2. Run parallel and async trajectory collection using the agent execution engine
3. Evaluate results and report accuracy metrics

## Training

### Basic Training

To train DeepCoder with iterative context lengthening (16K -> 32K -> 64K):

```bash
bash examples/deepcoder/train_deepcoder_16k.sh

# modify MODEL_PATH to the 16k checkpoint path before running the script.
bash examples/deepcoder/train_deepcoder_32k.sh
```
