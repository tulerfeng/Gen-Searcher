import hydra

from rllm.agents.swe_agent import SWEAgent
from rllm.data import DatasetRegistry
from rllm.environments.swe.swe import SWEEnv
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Load SWE datasets - using names from prepare_swe_data.py
    train_dataset = DatasetRegistry.load_dataset("R2E_Gym_Subset", "train")
    val_dataset = DatasetRegistry.load_dataset("SWE_Bench_Verified", "test")

    trainer = AgentTrainer(
        agent_class=SWEAgent,
        env_class=SWEEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
