"""
Gen Image training entry: image generation task.
Usage: python -m vision_deepresearch_async_workflow.train_image_deepresearch_workflow_fsdp_gen
"""
import os
import hydra

from vision_deepresearch_async_workflow.gen_image_deepresearch_reward import (
    gen_image_deepresearch_reward_fn_async,
)
from vision_deepresearch_async_workflow.gen_image_deepresearch_tools_executor import (
    create_gen_image_tools,
)
from vision_deepresearch_async_workflow.gen_image_deepresearch_workflow import (
    GenImageDeepResearchWorkflow,
)

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    """
    Main training function for the image generation task.
    
    Environment variables:
    - DEEPSEEK_API_KEY: DeepSeek API key
    - DEEPSEEK_API_BASE: DeepSeek API base URL
    - QWEN_EDIT_APP_URL / QWEN_EDIT_APP_PATH: Qwen Edit image generation service URLs (JSON array or comma-separated) and HTTP path (default /generate)
    - GEN_REWARD_API_KEY / GEN_REWARD_API_BASE_URL / GEN_REWARD_MODEL: GPT-4.1 scoring aligned with KnowGen eval (gpt_eval_knowgen)
      (same prompt, same overall formula; scoring uses the original question, not gen_prompt)
    - GEN_IMAGE_OUTPUT_DIR: output directory for generated images
    - GEN_IMAGE_TIMEOUT: image generation timeout (seconds)
    - IMAGE_SEARCH_PROXY_IPS: proxy IP list for image search
    - BROWSE_JINA_PROXY: proxy for read-proxy browsing
    - JINA_API_KEYS: read-proxy API keys
    - SERPER_KEY_ID: API key (X-API-KEY) for text + image search when using Serper
    - TEXT_SEARCH_API_BASE_URL: full POST URL for web search (e.g. https://google.serper.dev/search)
    - IMAGE_SEARCH_API_BASE_URL: full POST URL for image search (e.g. https://google.serper.dev/images)
    """
    print("\n[TrainGenImage] ===== Initialize Training =====")
    print(f"[TrainGenImage] Config: {config}")
    
    # Load dataset (name from env; default Vision-DeepResearch-Gen)
    dataset_name = os.environ.get("DATASET_NAME", "Vision-DeepResearch-Gen").strip()
    print(f"[TrainGenImage] Dataset name: {dataset_name}")
    print("[TrainGenImage] Loading dataset...")
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    
    print(f"[TrainGenImage] Train dataset size: {len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'}")
    print(f"[TrainGenImage] Test dataset size: {len(test_dataset) if hasattr(test_dataset, '__len__') else 'unknown'}")
    
    # Create tools
    print("[TrainGenImage] Creating tools...")
    tools = create_gen_image_tools()
    
    # Create trainer
    print("[TrainGenImage] Creating Trainer...")
    trainer = AgentTrainer(
        workflow_class=GenImageDeepResearchWorkflow,
        workflow_args={
            "reward_function": gen_image_deepresearch_reward_fn_async,
            "tools": tools,
            "system_prompt": None,  # Defined in GenImageDeepResearchAgent
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    print("[TrainGenImage] ===== Start Training =====\n")
    
    trainer.train()
    
    print("\n[TrainGenImage] ===== Training Completed =====")


if __name__ == "__main__":
    main()
