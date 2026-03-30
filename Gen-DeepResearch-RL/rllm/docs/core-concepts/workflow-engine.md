# Agent Workflow Engine

The Agent Workflow Engine is the high-level orchestrator in rLLM that manages complex multi-step workflows and episode rollout. Unlike the `AgentExecutionEngine` which handles simple agent-environment interactions, the `AgentWorkflowEngine` coordinates sophisticated workflows that can involve multiple agents and complex orchestration logic.

## Architecture Overview

At its core, the `AgentWorkflowEngine` manages a dynamic set of concurrent workflow instances:

1. **Workflow Pool**: Maintains a pool of workflow instances for parallel processing
2. **Task Queue**: Manages task distribution across available workflows
3. **Retry Logic**: Handles failures with configurable retry mechanisms
4. **Episode Management**: Collects and processes complete episodes rather than individual trajectories

## Core Components

### Batch Execution

The engine supports both standard batch processing (i.e., on tasks) and Verl-compatible batch processing (i.e., on DataProtos):

```python
# Standard batch execution
episodes = await engine.execute_tasks(tasks, task_ids)

# Verl-compatible batch execution for RL training
verl_batch = await engine.execute_tasks_verl(verl_data_proto)
```

## Workflow Types

The `AgentWorkflowEngine` supports various workflow types, each designed for different use cases:

### 1. Backward-Compatible Workflows
`SingleTurnWorkflow`, `MultiTurnWorkflow`, and `CumulativeWorkflow` reimplement the core functionality of `AgentExecutionEngine`. These provide a migration path from the execution engine to the workflow engine while maintaining compatibility.

### 2. Simple Workflow
`SimpleWorkflow` implements a basic workflow to optimize an LLM against a reward function, perfect for straightforward RL training scenarios.

### 3. Custom Workflows
Users can create custom workflows by extending the `Workflow` base class:

```python
from rllm.workflows.workflow import Workflow

class CustomWorkflow(Workflow):
    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        # Custom workflow logic
        # Multi-agent coordination
        # Complex reasoning chains
        # Tool usage and external API calls
        pass
```

For a complete example, see the [Solver-Judge Workflow](../examples/solver_judge.md) documentation.

## Basic Usage

### Initialization

```python
from transformers import AutoTokenizer
from rllm.engine import AgentWorkflowEngine, OpenAIEngine

model = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model)

# Initialize rollout engine
rollout_engine = OpenAIEngine(
    model=model,
    tokenizer=tokenizer,
    base_url="http://localhost:30000/v1",
    api_key="your-api-key",
    max_prompt_length=2048,
    max_response_length=1024,
    sampling_params={"temperature": 0.6, "top_p": 0.5}
)

# Initialize the workflow engine
engine = AgentWorkflowEngine(
    workflow_cls=SolverJudgeWorkflow,
    workflow_args={
        "n_solutions": n_solutions,
        "reward_function": countdown_reward_fn,
    },
    rollout_engine=rollout_engine,
    n_parallel_tasks=128,  # Number of parallel workflow instances
    retry_limit=3,        # Retry failed tasks up to 3 times
    raise_on_error=True   # Raise exceptions on permanent failures
)
```

### Task Execution

```python
from rllm.datasets import DatasetRegistry

# Load tasks from dataset registry
dataset = DatasetRegistry.load_dataset("countdown", "test").data

# Execute tasks asynchronously
episodes = await engine.execute_tasks(tasks)

# Process results
for episode in episodes:
    print(f"Episode {episode.id}: {episode.termination_reason}")
    print(f"Success: {episode.is_correct}")
```