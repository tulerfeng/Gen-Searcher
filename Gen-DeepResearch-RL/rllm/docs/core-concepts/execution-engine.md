# Agent Execution Engine

The Agent Execution Engine is the central orchestrator in rLLM that manages interactions between agents and environments. It handles batched execution, trajectory collection, and provides both synchronous and asynchronous execution modes for scalable RL training.

## Batch Trajectory Generation with `execute_tasks`

The `AgentExecutionEngine` provides a method called `execute_tasks` for performing offline batch inference. Each task is a dictionary containing either a dataset entry (e.g., AIME problems) or necessary information (e.g., random seed) to construct an environment instance from the environment class. 

Here's how the workflow operates:

1. **Initialization**: The `AgentExecutionEngine` initializes `N=n_parallel_agents` agent-environment pairs. For each agent, it initializes using `agent_args`, and for each environment, it uses the `env_class.from_json({**env_args, **task})` method to merge both task information and environment arguments when creating the environment instance.

2. **Parallel Execution**: Each agent-environment pair performs trajectory generation asynchronously. rLLM supports the OpenAI Completions interface for LLM inference and structures requests sent to the inference engine endpoint.

3. **Task Queue Management**: After each agent-environment pair completes its task, the `AgentExecutionEngine` initializes a new agent-environment pair to process the next task from the queue. This process continues until all tasks are processed.


### Basic Usage

```python
import asyncio
from rllm.engine.agent_execution_engine import AgentExecutionEngine

engine = AgentExecutionEngine(
    agent_class=CustomAgent,
    env_class=CustonEnvironment,
    engine_name="openai",  # or "verl"
    tokenizer=tokenizer,
    n_parallel_agents=64,
    max_steps=10,
    max_response_length=4096,
    max_prompt_length=2048,
    sampling_params={"temperature": 0.7, "top_p": 0.9},
    rollout_engine_args={
        "base_url": "http://localhost:8000/v1",
        "api_key": "your_api_key"
    }
)

# Execute tasks asynchronously
results = await engine.execute_tasks(tasks)
```
