# DeepScaler Math Agent Example

This example demonstrates training and running [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), a reasoning LLM finetuned from Deepseek-R1-Distill-1.5B on math competition problems using RL. The model achieves >40% Pass@1 on AIME2024, reaching o1-preview performance despite its small size. 

## Overview

The DeepScaler examples demonstrate:

- How to use rLLM's MathAgent for mathematical reasoning
- How to train agents with iterative context lengthening (8K -> 16K -> 24K)
- How to evaluate mathematical reasoning with Pass@K metrics

## Quick Start

### Setup Math Data

First, prepare your mathematical datasets:

```bash
cd examples/deepscaler
python prepare_math_data.py
```

### Model Hosting

Start a model server (choose one option):

**Option 1: Using vLLM**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 
```

**Option 2: Using SGLang**
```bash
python -m sglang_router.launch_server \
    --model-path agentica-org/DeepScaleR-1.5B-Preview \ 
    --dp-size 1 \
    --dtype bfloat16
```

### Run DeepScaler Agent

Execute the math reasoning agent:

```bash
python run_deepscaler.py
```

### Train DeepScaler Agent

Train your own DeepScaler agent with iterative context lengthening:

```bash
# Train with 8K context
bash train_deepscaler_8k.sh

# Train with 16K context (modify MODEL_PATH to 8k checkpoint)
bash train_deepscaler_16k.sh

# Train with 24K context (modify MODEL_PATH to 16k checkpoint)
bash train_deepscaler_24k.sh
```

## Code Reference

### Math Agent Runner

Main script for running mathematical reasoning:

```python title="examples/deepscaler/run_deepscaler.py"
--8<-- "examples/deepscaler/run_deepscaler.py"
```

### Training Script

DeepScaler training configuration:

```python title="examples/deepscaler/train_deepscaler.py"
--8<-- "examples/deepscaler/train_deepscaler.py"
```

For detailed setup instructions, see the [README](https://github.com/rllm-org/rllm/blob/main/examples/deepscaler/README.md) in the deepscaler example directory.
