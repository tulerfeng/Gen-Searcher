"""
Gen 版训练入口：3 个公司工具 + Gen system prompt + Gen reward（公司 DeepSeek Judge）。
使用方式：python -m vision_deepresearch_async_workflow.train_deepresearch_workflow_fsdp_gen
运行前请 source .env.gen 或设置 DEEPSEEK_API_KEY / DEEPSEEK_API_BASE / JUDGE_MODEL 等。
"""
import hydra

from vision_deepresearch_async_workflow.gen_deepresearch_reward import (
    gen_deepresearch_reward_fn_async,
)
from vision_deepresearch_async_workflow.gen_deepresearch_tools_async_executor import (
    GEN_DEEPRESEARCH_SYSTEM_PROMPT,
    get_all_tools,
)
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    DeepResearchWorkflow,
)

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("Vision-DeepResearch-QA", "train")
    test_dataset = DatasetRegistry.load_dataset("Vision-DeepResearch-QA", "test")

    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={
            "reward_function": gen_deepresearch_reward_fn_async,
            "tools": get_all_tools(),
            "system_prompt": GEN_DEEPRESEARCH_SYSTEM_PROMPT,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
