# Solver-Judge Workflow Example

This example demonstrates a **solver-judge workflow** using rLLM's `AgentWorkflowEngine`. The workflow generates multiple candidate solutions to countdown problems and uses a judge to select the best one, showcasing multi-agent coordination.

## Overview

The solver-judge workflow demonstrates:

- How to implement custom workflows extending the `Workflow` base class
- Multi-agent coordination between solver and judge agents
- Parallel solution generation and quality assessment
- Integration with rLLM's workflow engine

## Quick Start

### Setup Countdown Data

First, prepare the countdown dataset:

```bash
cd examples/solver_judge
python prepare_countdown_data.py
```

### Run Solver-Judge Workflow

Execute the workflow on countdown problems:

```bash
python run_solver_judge_flow.py
```

### Train with Solver-Judge Workflow

Train an agent using the solver-judge workflow:

```bash
bash train_solver_judge_flow.sh
```

## Code Reference

### Solver-Judge Workflow Implementation

The core workflow that coordinates solver and judge agents:

```python title="examples/solver_judge/solver_judge_flow.py"
--8<-- "examples/solver_judge/solver_judge_flow.py"
```

### Workflow Runner

Main script for running the solver-judge workflow:

```python title="examples/solver_judge/run_solver_judge_flow.py"
--8<-- "examples/solver_judge/run_solver_judge_flow.py"
```

### Training Script

Training configuration using the solver-judge workflow:

```python title="examples/solver_judge/train_solver_judge_flow.py"
--8<-- "examples/solver_judge/train_solver_judge_flow.py"
```

## How It Works

1. **Solver Phase**: Generate multiple candidate solutions in parallel
2. **Judge Phase**: Evaluate solutions and select the best one
3. **Episode Completion**: Determine overall correctness based on both solver and judge results

The workflow uses the countdown dataset where agents must use given numbers and basic arithmetic operations to reach a target number.