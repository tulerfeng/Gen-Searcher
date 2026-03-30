# Agent and Environment

This guide explains how the two main components in rLLM work together: **agents** and **environments**. You'll learn about their core interfaces, interaction patterns, and how to implement your own custom agents and environments.

## Overview

rLLM uses a modular approach where agents act as intelligent decision-makers and environments provide tasks and feedback. This separation enables:

- **Modular development**: Components can be built and tested independently
- **Flexible reuse**: The same agent can work across different environments
- **Consistent patterns**: Standardized interfaces ensure compatibility

## Agents

Agents are the core components in rLLM that generate intelligent actions based on environmental observations. They serve as the bridge between language models and interactive environments, enabling autonomous problem-solving and decision-making.

### BaseAgent Interface

All agents inherit from the `BaseAgent` class, which defines the essential methods for environment interaction:

```python
from rllm.agents.agent import BaseAgent, Step, Trajectory

class BaseAgent(ABC):
    @abstractmethod
    def update_from_env(self, observation, reward, done, info, **kwargs):
        """Updates agent state after receiving environment feedback."""
        pass
        
    @abstractmethod
    def update_from_model(self, response, **kwargs):
        """Updates agent state after receiving model response."""
        pass
        
    @abstractmethod
    def reset(self):
        """Resets agent's internal state for new episode."""
        pass
        
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns messages formatted for chat completion."""
        return []
    
    @property
    def trajectory(self) -> Trajectory:
        """Returns complete interaction history."""
        return Trajectory()

    def get_current_state(self) -> Step | None:
        """Return the most recent step."""
        if not self.trajectory.steps:
            return None
        return self._trajectory.steps[-1]
```

### Key Agent Responsibilities

Each agent manages:
- **State tracking**: Maintaining conversation history and internal state through trajectories
- **Model interaction**: Formatting messages for language model consumption
- **Response processing**: Handling and storing model outputs
- **Environment adaptation**: Updating state based on environmental feedback

## Environments

Environments complement agents by defining the tasks, evaluation criteria, and interaction rules. They provide the context within which agents operate and learn.

### BaseEnv Interface

All environments in rLLM inherit from the `BaseEnv` class, which follows the Gymnasium interface with rLLM-specific extensions:

```python
from rllm.environments.base.base_env import BaseEnv

class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        """Reset environment and return initial observation and info."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute action and return (observation, reward, done, info)."""
        pass

    def close(self):
        """Clean up environment resources."""
        pass

    @staticmethod
    @abstractmethod
    def from_dict(env_args: Dict) -> "BaseEnv":
        """Create environment instance from dictionary. This function is used both during inference and training to instantiate a new environment instance, so it has to be implemented properly."""
        pass
```

### Key Environment Responsibilities

Each environment handles:
- **Task definition**: Providing problems or scenarios for agents to solve
- **Observation generation**: Creating meaningful inputs for agent decision-making
- **Action evaluation**: Assessing agent responses and providing rewards
- **Episode management**: Determining when interactions should terminate

## The Interaction Cycle

Understanding how agents and environments work together is crucial for effective rLLM usage. The interaction follows a structured cycle that enables learning and adaptation.

### Step-by-Step Flow

The agent-environment interaction follows this pattern:

1. **Initialization**: Agent calls `agent.reset()` to prepare for a new episode, environment calls `env.reset()` to provide initial observation
2. **State Update**: Agent processes environment observation via `update_from_env()`
3. **Model Interaction**: Language model generates response using agent's `chat_completions`
4. **Response Processing**: Agent updates state via `update_from_model()`
5. **Environment Feedback**: Environment evaluates the response via `step()` and provides reward/next observation
6. **Repeat**: Process continues until episode completion or termination criteria are met

This cycle enables sophisticated behaviors like self-correction, multi-turn reasoning, and adaptive problem-solving.

## Complete Implementation Example

To illustrate these concepts in action, let's examine a complete implementation that demonstrates self-correction behavior. The `MathAgent` and `MathEnv` example shows how agents can learn from mistakes and improve their responses.

### MathAgent Implementation

The `MathAgent` demonstrates how to implement the core agent interface for mathematical problem-solving:

```python
from typing import Any, Dict, List

from rllm.agents.agent import BaseAgent, Step, Trajectory

class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """
    def __init__(self, accumulate_thinking=True):
        """
        Initialize the MathAgent.
        """
        self.instruction = "Let's think step by step and put the final answer within \\boxed{}."
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.accumulate_thinking = accumulate_thinking
        
    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict, **kwargs):
        """Process environment feedback and update internal state."""
        
        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self.trajectory.steps:
            # Initial problem presentation
            assert isinstance(observation, dict) and 'question' in observation
            question = observation['question']
            formatted_observation = f'{question} {self.instruction}'
        else:
            # Follow-up correction prompt
            formatted_observation = (
                "Your previous answer may contain a mistake. "
                "Please review it carefully and answer again. "
                "Put your final answer within \\boxed{}."
            )

        # If there are previous steps, update the last step's outcome
        if self.trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = formatted_observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
        
        if done:
            return
        
        self.messages.append({
            "role": "user",
            "content": formatted_observation
        })
        cur_step = Step(
            observation=formatted_observation,
            step=self.step
        )
        self.trajectory.steps.append(cur_step)

    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state based on the model's response.
        """        
        assert self.trajectory.steps, "Trajectory should not be empty when update_from_model is called."
        
        # Update the current step in the trajectory
        cur_step = self.get_current_state()
        cur_step.model_response = response

        if not self.accumulate_thinking:
            _, sep, after = response.partition("</think>")
            if sep:
                response = after

        self.messages.append({"role": "assistant", "content": response})
        
        self.step += 1

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Return conversation history for model interaction."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory
```

### MathEnv Implementation

The `MathEnv` demonstrates how to create an environment that supports self-correction through reward-based feedback:

```python
from typing import Any, Dict, Tuple
from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import math_reward_fn


class MathEnv(BaseEnv):
    """
    A math environment that presents mathematical problems and evaluates solutions.
    Supports self-correction by asking the agent to retry if the answer is incorrect.
    """
    
    def __init__(self, task: Dict, max_attempts: int = 2):
        """
        Initialize the MathEnv.
        
        Args:
            task: Dictionary containing problem info (question, ground_truth, etc.)
            max_attempts: Maximum number of attempts allowed (default: 2)
        """
        self.task = task
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self.is_correct = False
        
    def reset(self) -> Tuple[Dict[str, str], Dict]:
        """Reset environment and return initial observation and info."""
        self.current_attempt = 0
        self.is_correct = False
        
        observation = {"question": self.task["question"]}
        info = {"attempt": self.current_attempt, "max_attempts": self.max_attempts}
        
        return observation, info
    
    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute action (model response) and return (observation, reward, terminated, truncated, info).
        """
        self.current_attempt += 1
        
        # Use rllm's math reward function to evaluate the response
        reward_output = math_reward_fn(self.task, action)
        reward = reward_output.reward
        self.is_correct = reward > 0.0
        
        # Determine if episode is done
        terminated = self.is_correct or self.current_attempt >= self.max_attempts
        truncated = False
        
        observation = None  # Will be handled by agent's update_from_env
        
        info = {
            "attempt": self.current_attempt,
            "max_attempts": self.max_attempts,
            "is_correct": self.is_correct,
            "reward_metadata": reward_output.metadata
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    @staticmethod
    def from_dict(env_args: Dict) -> "MathEnv":
        """Create MathEnv instance from dictionary configuration."""
        return MathEnv(
            task=env_args["task"],
            max_attempts=env_args.get("max_attempts", 2)
          )
```

## Detailed Interaction Walkthrough

Let's trace through how the MathAgent and MathEnv handle a complete interaction cycle with self-correction:

### Step 0: Initial Problem Solving

1. **Environment Reset**: `env.reset()` returns initial observation containing the math question
2. **Agent Processing**: 
    - Agent calls `update_from_env()` with the question
    - Formats question with step-by-step instructions
    - Creates Step 0 in trajectory and prepares user message
3. **Model Generation**: Language model uses `agent.chat_completions` to generate response
4. **Response Integration**: 
    - Agent calls `update_from_model()` with model's solution attempt
    - Updates Step 0 with model response and adds to conversation history
5. **Environment Evaluation**: `env.step()` evaluates response using `math_reward_fn`

### Step 1: Self-Correction (if needed)

1. **Environment Feedback**: Environment returns reward (0.0 if incorrect) and episode status
2. **Agent Processing**:
    - Agent calls `update_from_env()` with feedback
    - Updates Step 0 with reward and outcome
    - If not done, creates Step 1 with correction prompt
3. **Model Generation**: Model attempts to correct previous answer with new context
4. **Response Integration**: Updates Step 1 with corrected response
5. **Final Evaluation**: Environment provides final reward and terminates episode

This walkthrough demonstrates how the standardized interface enables complex behaviors like self-correction while maintaining clean separation of concerns between agent and environment.


## Next Steps

Now that you understand how agents and environments work together, explore these related topics:

- **[Execution Engine](execution-engine.md)**: Learn how to orchestrate parallel agent-environment interactions for efficient batch processing
