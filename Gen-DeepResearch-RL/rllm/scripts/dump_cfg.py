from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    yaml_str = OmegaConf.to_yaml(cfg, resolve=False)
    out_path = Path("rllm/trainer/config/_generated_agent_ppo_trainer.yaml")
    out_path.write_text(yaml_str, encoding="utf-8")
    print(f"Config dumped to {out_path}")


if __name__ == "__main__":
    main()
