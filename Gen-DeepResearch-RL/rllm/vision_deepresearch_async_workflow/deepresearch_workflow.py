from io import BytesIO
import asyncio
import re
from typing import Any, List, Optional

from PIL import Image

from vision_deepresearch_async_workflow.deepresearch_agent import DeepResearchAgent
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationReason, Workflow

import base64


def as_pil_image(image: Any) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, str) and image.startswith("data:image/"):
        try:
            _, encoded = image.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            try:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        data_str = None
        if "data" in image and isinstance(image["data"], str):
            data_str = image["data"]
        elif "path" in image and isinstance(image["path"], str):
            data_str = image["path"]
        elif "url" in image and isinstance(image["url"], str):
            data_str = image["url"]
        if data_str:
            if data_str.startswith("data:image/"):
                try:
                    _, encoded = data_str.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception:  # noqa: BLE001
                    return None
            try:
                return Image.open(data_str).convert("RGB")
            except Exception:  # noqa: BLE001
                return None

    if isinstance(image, str):
        try:
            return Image.open(image).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    return None


def _extract_action_from_response(response: str) -> Action:
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})


def _is_valid_format(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    pattern = (
        r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<answer>.*?</answer>)\s*$"
    )
    return re.match(pattern, content, re.DOTALL) is not None


def _format_reward_for_step(step: Step) -> float:
    if step.info.get("step_error"):
        return 0.0
    if _has_tool_error_observation(step.observation):
        return 0.0
    content = step.model_response if isinstance(step.model_response, str) else ""
    return 1.0 if _is_valid_format(content) else 0.0


def _has_tool_error_observation(observation: Any) -> bool:
    if not isinstance(observation, str):
        return False
    error_markers = (
        "[Json Parse Error]",
        "[Python Interpreter Error]",
        "Python execution error:",
        "PythonInterpreter tool not available",
        "PythonInterpreter tool is not callable",
    )
    return any(marker in observation for marker in error_markers)


def _is_step_error(step: Step) -> bool:
    if step.info.get("step_error"):
        return True
    return _has_tool_error_observation(step.observation)


def _get_next_observation(messages: list[dict], current_index: int) -> str:
    if current_index + 1 < len(messages):
        next_msg = messages[current_index + 1]
        if next_msg["role"] == "user" and "<tool_response>" in next_msg["content"]:
            return next_msg["content"]
    return ""


def _map_termination_reason(termination: str) -> TerminationReason:
    mapping = {
        "answer": TerminationReason.ENV_DONE,
        "timeout": TerminationReason.UNKNOWN,
        "max_rounds_reached": TerminationReason.UNKNOWN,
        "token_limit_no_answer": TerminationReason.UNKNOWN,
        "answer_token_limit": TerminationReason.UNKNOWN,
        "exceed available llm calls": TerminationReason.UNKNOWN,
        "prompt_budget_reached": TerminationReason.UNKNOWN,
        "max_rounds_reached_no_answer": TerminationReason.UNKNOWN,
        "repetition_detected": TerminationReason.UNKNOWN,  # Will be masked
        "tool_errors_too_many": TerminationReason.UNKNOWN,  # Will be masked
        "consecutive_step_errors": TerminationReason.UNKNOWN,  # Will be masked
        "error": TerminationReason.UNKNOWN,  # Will be masked
    }
    return mapping.get(termination, TerminationReason.UNKNOWN)


def _evaluate_answer(prediction: str, ground_truth: str) -> bool:
    if not prediction or not ground_truth:
        return False
    return prediction.strip().lower() == ground_truth.strip().lower()


def _should_mask_episode(result: dict, episode: Episode) -> tuple[bool, str]:
    """Determine if the entire episode should be masked based on answer/step error conditions."""
    steps = episode.trajectories[0].steps if episode.trajectories else []
    termination = result.get("termination", "")

    # Mask directly if no final answer is produced.
    if termination != "answer":
        return True, termination or "no_final_answer"

    # Check if there are too many step errors.
    total_steps = len(steps)

    if total_steps > 0:
        step_error_steps = sum(1 for step in steps if _is_step_error(step))
        if step_error_steps / total_steps > 0.5:
            return True, "tool_errors_too_many"

    return False, ""


def _to_pil_image(img: Any) -> Optional[Image.Image]:
    """Best-effort conversion to PIL.Image for downstream multi-modal pipeline."""
    if isinstance(img, Image.Image):
        return img
    pil = as_pil_image(img)
    if pil is not None:
        return pil
    if isinstance(img, dict) and "bytes" in img:
        try:
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(img, str):
        try:
            return Image.open(img).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    return None


class DeepResearchWorkflow(Workflow):
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

        self.tools = tools or {}
        for tool in self.tools.values():
            if hasattr(tool, "set_executor"):
                tool.set_executor(self.executor)
        self.system_prompt = system_prompt
        self.reward_function = reward_function

        self.agent = DeepResearchAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        question = task.get("question", task.get("query", "No question provided"))
        answer = task.get("answer", "")

        print(f"🚀 Starting DeepResearch workflow for task {uid}")
        print(f"   Question: {question}")

        try:
            raw_images = None
            if "images" in task:
                raw_images = task.get("images")

            pil_images: List[Image.Image] = []
            if raw_images is not None:
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                for img in raw_images:
                    pil = _to_pil_image(img)
                    if pil is not None:
                        pil_images.append(pil)

            if pil_images:
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    images=pil_images,
                    image_path=raw_images[0],
                    **kwargs,
                )
            else:
                result = await self.agent.run(
                    question=question, answer=answer, **kwargs
                )

            episode = self._convert_result_to_episode(result, task, uid)

            prediction = result.get("prediction", "")
            if self.reward_function is not None and prediction:
                try:
                    if asyncio.iscoroutinefunction(self.reward_function):
                        reward_out = await self.reward_function(task, prediction)
                    else:
                        loop = asyncio.get_running_loop()
                        reward_out = await loop.run_in_executor(
                            self.executor,
                            lambda: self.reward_function(task, prediction),
                        )
                except Exception as err:  # noqa: BLE001
                    print(f"Reward function failed: {err}")
                else:
                    if reward_out.is_correct is not None:
                        episode.is_correct = bool(reward_out.is_correct)
                    if isinstance(reward_out.metadata, dict):
                        reward_metadata = episode.info.setdefault("reward_metadata", {})
                        for key, value in reward_out.metadata.items():
                            if key not in reward_metadata:
                                reward_metadata[key] = value
                    if getattr(reward_out, "reward", None) is not None:
                        episode.info["reward_function_reward"] = float(
                            reward_out.reward
                        )

            # Check whether to mask the whole episode.
            should_mask_episode, mask_reason = _should_mask_episode(result, episode)

            if should_mask_episode:
                episode.termination_reason = TerminationReason.UNKNOWN
                episode.metrics = {
                    "reward/outcome": 0.0,
                    "masked": 1.0,
                }
                episode.info["mask_reason"] = mask_reason or result.get(
                    "termination", "unknown"
                )
            else:
                # No mask: use outcome_reward only, no format reward.
                outcome_reward = 1.0 if episode.is_correct else 0.0
                for trajectory in episode.trajectories:
                    if not trajectory.steps:
                        continue
                    trajectory.reward = outcome_reward

                    last_step = trajectory.steps[-1]
                    last_step.reward = trajectory.reward
                    trajectory.steps = [last_step]

                episode.metrics = {
                    "reward/outcome": outcome_reward,
                    "masked": 0.0,
                }

            print(f"✅ DeepResearch workflow completed for task {uid}")
            print(f"   Prediction: {result.get('prediction', 'No prediction')}")
            print(f"   True Answer: {answer}")
            print(f"   Metrics: {episode.metrics}")
            if episode.info.get("mask_reason"):
                print(f"   Mask Reason: {episode.info['mask_reason']}")
            return episode

        except Exception as exc:  # noqa: BLE001
            print(f"❌ DeepResearch workflow failed for task {uid}: {exc}")
            episode = Episode()
            episode.id = uid
            episode.task = task
            episode.termination_reason = TerminationReason.ERROR
            episode.is_correct = False
            episode.trajectories = []
            episode.metrics = {
                "reward/outcome": 0.0,
                "masked": 1.0,
            }
            episode.info = {"error": str(exc)}
            return episode

    def _convert_result_to_episode(self, result: dict, task: dict, uid: str) -> Episode:
        messages = result.get("messages", [])
        prediction = result.get("prediction", "")
        termination = result.get("termination", "unknown")
        rounds = result.get("rounds", 0)
        time_taken = result.get("time_taken", 0.0)

        trajectories: list[Trajectory] = []
        steps: list[Step] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant":
                context = messages[: i + 1]
                assistant_content = msg.get("content", "")
                action = _extract_action_from_response(assistant_content)
                observation = _get_next_observation(messages, i)
                step = Step(
                    chat_completions=context.copy(),
                    model_response=assistant_content,
                    action=action,
                    observation=observation,
                    reward=0.0,
                )
                step.model_output = None
                if msg.get("step_error"):
                    step.info["step_error"] = True
                if _has_tool_error_observation(observation):
                    step.info["step_error"] = True
                steps.append(step)
            i += 1

        trajectory = Trajectory(
            name="deepresearch_agent",
            task=task,
            steps=steps,
            reward=0.0,
            info={},
        )
        trajectories.append(trajectory)

        answer_text = task.get("answer", "")
        is_correct = _evaluate_answer(prediction, answer_text) if answer_text else False

        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = _map_termination_reason(termination)
        episode.is_correct = is_correct
        episode.trajectories = trajectories
        episode.metrics = {}
        episode.info = {
            "rounds": rounds,
            "time_taken": time_taken,
            "prediction": prediction,
            "answer": answer_text,
            "token_usage": result.get("token_usage", {}),
        }
        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        # MultiTurnReactAgent handles per-run state; nothing to reset here.
        return

    def is_multithread_safe(self) -> bool:
        return True


__all__ = ["DeepResearchWorkflow"]
