"""
Gen Image Workflow: wraps the image generation task workflow.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from vision_deepresearch_async_workflow.gen_image_deepresearch_agent import GenImageDeepResearchAgent
from vision_deepresearch_async_workflow.gen_image_deepresearch_reward import save_trajectory_result
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationReason, Workflow


def _extract_action_from_response(response: str) -> Action:
    """Extract an Action from the model response."""
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})


def _map_termination_reason(termination: str) -> TerminationReason:
    """Map termination string to TerminationReason enum."""
    mapping = {
        "answer": TerminationReason.ENV_DONE,
        "max_response_length_exceeded": TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
        "repeated_response": TerminationReason.REPEATED_RESPONSE,
        "timeout": TerminationReason.UNKNOWN,
        "max_rounds_reached": TerminationReason.UNKNOWN,
        "error": TerminationReason.UNKNOWN,
        "answer_parse_failed": TerminationReason.UNKNOWN,
        "no_answer": TerminationReason.UNKNOWN,
    }
    return mapping.get(termination, TerminationReason.UNKNOWN)


def _get_next_observation(messages: list[dict], current_index: int) -> str:
    """Get the next observation after the current message."""
    if current_index + 1 < len(messages):
        next_msg = messages[current_index + 1]
        if next_msg["role"] == "user" and "<tool_response>" in next_msg["content"]:
            return next_msg["content"]
    return ""


# Dedicated reward thread pool, separated from the tool executor.
# Ensures reward computation (image generation + scoring) always has enough parallelism.
_REWARD_EXECUTOR: Optional[ThreadPoolExecutor] = None
_REWARD_EXECUTOR_SIZE = 32


def _get_reward_executor() -> ThreadPoolExecutor:
    global _REWARD_EXECUTOR
    if _REWARD_EXECUTOR is None:
        _REWARD_EXECUTOR = ThreadPoolExecutor(max_workers=_REWARD_EXECUTOR_SIZE)
    return _REWARD_EXECUTOR


def _should_mask_episode(result: dict) -> tuple[bool, str]:
    """Decide whether an episode should be masked."""
    termination = result.get("termination", "")
    
    # Mask if we did not produce a final answer
    if termination != "answer":
        return True, termination or "no_final_answer"
    
    prediction = result.get("prediction", {})
    
    # Mask if prediction contains an error
    if isinstance(prediction, dict) and "error" in prediction:
        return True, f"prediction_error: {prediction['error']}"
    
    return False, ""


class GenImageDeepResearchWorkflow(Workflow):
    """Workflow for the image generation task."""
    
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        tools: dict | None = None,
        system_prompt: str | None = None,
        reward_function: RewardFunction | None = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)
        
        print("[GenImageWorkflow] Initializing workflow")
        
        self.tools = tools or {}
        # Bind executor to tools
        for tool in self.tools.values():
            if hasattr(tool, "set_executor"):
                tool.set_executor(self.executor)
        
        self.system_prompt = system_prompt
        self.reward_function = reward_function
        
        # Create agent
        self.agent = GenImageDeepResearchAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
            **kwargs,
        )
        
        print(f"[GenImageWorkflow] Tools: {list(self.tools.keys())}")
    
    async def run(self, task: dict, uid: str = "", **kwargs) -> Episode:
        """
        Run a single task; uid is the unique identifier for this rollout.
        
        Args:
            task: dict containing fields like question, gt_image, etc.
            uid: unique identifier for this task/rollout
        
        Returns:
            Episode
        """
        self.task = task
        self.uid = uid
        question = task.get("question", "")
        
        print(f"\n[GenImageWorkflow] ===== Run Workflow =====")
        print(f"[GenImageWorkflow] Question: {question[:100]}...")
        
        # Run agent
        result = await self.agent.run(question, **kwargs)
        
        print(f"[GenImageWorkflow] Agent finished, termination: {result.get('termination')}")
        
        # Merge prediction into task so reward can read from episode.task
        self.task = {**task, "prediction": result.get("prediction", {})}
        
        # Convert to Episode
        episode = self._result_to_episode(result, task)
        episode.task = self.task
        episode.id = uid
        
        # Store global_step (if provided) into episode.info
        if "global_step" in kwargs:
            if episode.info is None:
                episode.info = {}
            episode.info["global_step"] = kwargs["global_step"]

        # For overlong or repeated responses: set reward=0 and stop early to skip image generation & scoring.
        if result.get("termination") in ("max_response_length_exceeded", "repeated_response") and episode.trajectories:
            episode.trajectories[0].reward = 0.0
            messages = (episode.info or {}).get("messages", [])
            prediction = (episode.info or {}).get("prediction", {})
            token_usage = (episode.info or {}).get("token_usage", {})
            timing = (episode.info or {}).get("timing", {})
            sample_id = task.get("id") or task.get("sample_id") or uid
            global_step = (episode.info or {}).get("global_step")
            termination = result.get("termination", "")
            save_trajectory_result(
                trajectory_id=f"traj_termination_{uid or 'unknown'}",
                question=task.get("question", ""),
                messages=messages,
                prediction=prediction if isinstance(prediction, dict) else {"error": str(prediction)},
                reward=0.0,
                generated_image_path=None,
                termination=termination,
                sample_id=sample_id,
                token_usage=token_usage if isinstance(token_usage, dict) else {},
                timing=timing if isinstance(timing, dict) else {},
                global_step=global_step,
            )
            print(
                f"[GenImageWorkflow] {result.get('termination', '').upper()}: reward=0; skip image generation and scoring"
            )
            print("[GenImageWorkflow] Episode constructed")
            return episode

        # Before returning, run reward (image generation + scoring) and write back to trajectory.reward.
        # The async reward internally blocks (HTTP calls). Running it directly in the event loop can stall other tasks.
        # Run it in a dedicated reward thread pool, separate from the tool executor, to keep reward computations parallel.
        if self.reward_function and episode.trajectories:
            try:
                loop = asyncio.get_running_loop()
                reward_executor = _get_reward_executor()
                if asyncio.iscoroutinefunction(self.reward_function):
                    _ep, _kw = episode, dict(kwargs)

                    def _run_async_reward():
                        return asyncio.run(self.reward_function([_ep], **_kw))

                    rewards = await loop.run_in_executor(reward_executor, _run_async_reward)
                else:
                    rewards = await loop.run_in_executor(
                        reward_executor,
                        lambda: self.reward_function([episode], **kwargs),
                    )
                if rewards and len(rewards) >= 1:
                    r = float(rewards[0])
                    # Ensure non-negative reward
                    episode.trajectories[0].reward = max(0.0, r)
            except Exception as e:
                print(f"[GenImageWorkflow] Reward function failed: {e}")
                import traceback
                traceback.print_exc()

        timing = (episode.info or {}).get("timing", {})
        if isinstance(timing, dict):
            model_total = float(timing.get("total_model_time", 0.0) or 0.0)
            tool_total = float(timing.get("total_tool_time", 0.0) or 0.0)
            image_gen_time = float(timing.get("image_gen_time", 0.0) or 0.0)
            score_time = float(timing.get("score_time", 0.0) or 0.0)
            model_each = [round(float(x), 1) for x in timing.get("model_call_durations", [])]
            tool_each = [round(float(x), 1) for x in timing.get("tool_call_durations", [])]
            timing_components_sum = round(model_total + tool_total + image_gen_time + score_time, 1)
            timing["model_call_durations"] = model_each
            timing["tool_call_durations"] = tool_each
            timing["total_model_time"] = round(model_total, 1)
            timing["total_tool_time"] = round(tool_total, 1)
            timing["image_gen_time"] = round(image_gen_time, 1)
            timing["score_time"] = round(score_time, 1)
            timing["timing_components_sum"] = timing_components_sum
            timing["trajectory_total_time"] = round(float(timing.get("trajectory_total_time", 0.0) or 0.0), 1)
            if episode.info is not None:
                episode.info["timing"] = timing
            print(
                "[GenImageWorkflow] Timing: "
                f"model_each={model_each}, model_total={timing['total_model_time']:.1f}s; "
                f"tool_each={tool_each}, tool_total={timing['total_tool_time']:.1f}s; "
                f"image_gen={timing['image_gen_time']:.1f}s; score={timing['score_time']:.1f}s; "
                f"components_sum={timing_components_sum:.1f}s; "
                f"trajectory_total={timing['trajectory_total_time']:.1f}s"
            )

        print("[GenImageWorkflow] Episode constructed")
        return episode
    
    def _result_to_episode(self, result: dict, data_item: dict) -> Episode:
        """Convert agent result to an Episode."""
        
        messages = result.get("messages", [])
        prediction = result.get("prediction", {})
        termination = result.get("termination", "unknown")
        
        print(f"[GenImageWorkflow] Converting to Episode, message count: {len(messages)}")
        
        # Build Steps
        steps: List[Step] = []
        
        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            
            content = msg["content"]
            # Keep cumulative chat context for verl tokenization.
            context = messages[: idx + 1]
            action = _extract_action_from_response(content)
            observation = _get_next_observation(messages, idx)
            
            step = Step(
                chat_completions=context.copy(),
                model_response=content,
                action=action,
                observation=observation,
                info={},
                reward=0.0,  # Computed later by reward function
            )
            steps.append(step)
        
        print(f"[GenImageWorkflow] Built {len(steps)} steps")
        
        # Create Trajectory
        trajectory = Trajectory(steps=steps)
        
        # Decide whether to mask
        should_mask, mask_reason = _should_mask_episode(result)
        
        if should_mask:
            print(f"[GenImageWorkflow] Episode masked: {mask_reason}")
        
        # Create Episode (only fields supported by Episode: trajectories, info, etc.
        # id/task/termination_reason are set by postprocess_episode)
        episode = Episode(
            trajectories=[trajectory],
            info={
                "question": data_item.get("question", ""),
                "prediction": prediction,
                "termination": termination,
                "messages": messages,
                "token_usage": result.get("token_usage", {}),
                "time_taken": result.get("time_taken", 0),
                "rounds": result.get("rounds", 0),
                "should_mask": should_mask,
                "mask_reason": mask_reason,
                "response_truncated_once": result.get("response_truncated_once", False),
                "format_error_once": result.get("format_error_once", False),
                "timing": result.get("timing", {}),
            },
        )
        episode.termination_reason = _map_termination_reason(termination)
        
        return episode
    
    def reset(self):
        """Reset workflow state."""
        if self.agent:
            self.agent.reset()
