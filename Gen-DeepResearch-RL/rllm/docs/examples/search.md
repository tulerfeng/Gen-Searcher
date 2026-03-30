# Search Agent Example

This example follows the set up from [Search-R1](https://github.com/PeterGriffinJin/Search-R1) to train a agent that can perform interleaved reasoning and search to answer multi-hop QA questions.

## Overview

The search examples demonstrate:
- How to use rLLM's ToolAgent and ToolEnvironment
- How to write custom tools in rLLM

## Quick Start

### Setup Search Data

First, prepare your search data:

```bash
cd examples/search
python prepare_search_data.py
```

### Run Search Agent

Execute the search agent:

```bash
python run_search_agent.py
```

### Train Search Agent

Train your own search agent:

```bash
bash train_search_agent.sh
```

## Code Reference

### Search Agent Runner

Main script for running search operations:

```python title="examples/search/run_search_agent.py"
--8<-- "examples/search/run_search_agent.py"
```

### Training Script

Search agent training configuration:

```python title="examples/search/train_search_agent.py"
--8<-- "examples/search/train_search_agent.py"
```

For detailed setup instructions, see the [README](https://github.com/rllm-org/rllm/blob/main/examples/search/README.md) in the search example directory. 