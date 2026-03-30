import hydra

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data import DatasetRegistry
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("frozenlake", "train")
    val_dataset = DatasetRegistry.load_dataset("frozenlake", "test")

    trainer = AgentTrainer(
        agent_class=FrozenLakeAgent,
        env_class=FrozenLakeEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
