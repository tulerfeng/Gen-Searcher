"""
Gen 版 reward：Judge 使用公司 DeepSeek API（.env 中 DEEPSEEK_API_KEY / DEEPSEEK_API_BASE / JUDGE_MODEL）。
使用方式：from vision_deepresearch_async_workflow.gen_deepresearch_reward import gen_deepresearch_reward_fn_async
"""
import os

from rllm.agents.agent import Action
from rllm.rewards.deepresearch_reward import RewardDeepResearchFn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput


class GenRewardDeepResearchFn(RewardDeepResearchFn):
    """DeepResearch reward with Judge 走公司 DeepSeek API（DEEPSEEK_API_BASE / DEEPSEEK_API_KEY / JUDGE_MODEL）."""

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        if os.getenv("DEEPSEEK_API_BASE"):
            self.base_url = os.getenv("DEEPSEEK_API_BASE", "").rstrip("/")
            if not self.api_key:
                self.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.model or "gpt-" in (self.model or "").lower():
                self.model = os.getenv("JUDGE_MODEL", "deepseek-chat")


async def gen_deepresearch_reward_fn_async(task_info: dict, action: str) -> RewardOutput:
    """Gen 版 async reward：使用公司 DeepSeek 做 Judge."""
    reward_config = RewardConfig()
    reward_fn = GenRewardDeepResearchFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    reward_input = RewardInput(task_info=task_info, action=action)
    return await reward_fn.async_call(reward_input)


__all__ = ["GenRewardDeepResearchFn", "gen_deepresearch_reward_fn_async"]
