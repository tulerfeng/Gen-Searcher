import hydra

from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import (
    get_all_tools,
)
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    DeepResearchWorkflow,
)

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import deepresearch_reward_fn_async
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer_megatron",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("Vision-DeepResearch-QA", "train")
    test_dataset = DatasetRegistry.load_dataset("Vision-DeepResearch-QA", "test")

    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={
            "reward_function": deepresearch_reward_fn_async,
            "tools": get_all_tools(),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
