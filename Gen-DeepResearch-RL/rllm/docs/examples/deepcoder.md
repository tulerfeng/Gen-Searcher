# DeepCoder Programming Agent Example

This example demonstrates training and running [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), a code reasoning LLM fine-tuned from DeepSeek-R1-Distill-Qwen-14B on coding competition problems with RL. The model achieves 60.6% Pass@1 accuracy on LiveCodeBench v5, representing an 8% improvement over the base model.

## Overview

The DeepCoder examples demonstrate:

- How to use rLLM's CompetitionCodingAgent for programming tasks
- How to train agents with iterative context lengthening (16K -> 32K)
- How to evaluate coding performance on LiveCodeBench

## Quick Start

### Setup Coding Data

First, prepare your coding datasets:

```bash
cd examples/deepcoder
python prepare_deepcoder_data.py
```

### Model Hosting

Start a model server (choose one option):

**Option 1: Using vLLM**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model agentica-org/DeepCoder-14B-Preview \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --max-model-len 32768
```

**Option 2: Using SGLang**
```bash
python -m sglang_router.launch_server \
    --model-path agentica-org/DeepCoder-14B-Preview \ 
    --dp-size 1 \
    --dtype bfloat16
```

### Run DeepCoder Agent

Execute the coding agent for evaluation:

```bash
python evaluate_deepcoder.py
```

### Train DeepCoder Agent

Train your own DeepCoder agent with iterative context lengthening:

```bash
# Train with 16K context
bash train_deepcoder_16k.sh

# Train with 32K context (modify MODEL_PATH to 16k checkpoint)
bash train_deepcoder_32k.sh
```

## Code Reference

### Code Agent Evaluator

Main script for evaluating coding performance:

```python title="examples/deepcoder/run_deepcoder.py"
--8<-- "examples/deepcoder/run_deepcoder.py"
```

### Training Script

DeepCoder training configuration:

```python title="examples/deepcoder/train_deepcoder.py"
--8<-- "examples/deepcoder/train_deepcoder.py"
```

For detailed setup instructions, see the [README](https://github.com/rllm-org/rllm/blob/main/examples/deepcoder/README.md) in the deepcoder example directory.
