# FrozenLake Agent Example

This example shows how we could train a RL agent to play the FrozenLake game.

**FrozenLake** is a classic RL environment where:

- Agent navigates a frozen lake grid
- Goal is to reach the frisbee without falling into holes
- Slippery surface adds stochasticity to actions
- Discrete action space: UP, DOWN, LEFT, RIGHT

## Quick Start

### Prepare Environment Data

```bash
cd examples/frozenlake
python prepare_frozenlake_data.py
```

### Run FrozenLake Agent

```bash
python run_frozenlake_agent.py
```

### Train Agent

```bash
bash train_frozenlake_agent.sh
```

## Code Reference

### Agent Runner

Main script for running the FrozenLake agent:

```python title="examples/frozenlake/run_frozenlake_agent.py"
--8<-- "examples/frozenlake/run_frozenlake_agent.py"
```

### Training Script

Agent training implementation:

```python title="examples/frozenlake/train_frozenlake_agent.py"
--8<-- "examples/frozenlake/train_frozenlake_agent.py"
```

For more details, see the [FrozenLake README](https://github.com/rllm-org/rllm/blob/main/examples/frozenlake/README.md). 