import argparse
import glob
import json
import os
import tempfile
from typing import Optional

import requests
from estimate_013 import estimate_from_config
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from megatron.core import parallel_state as mpu
from pydantic import BaseModel, field_validator

from mbridge import AutoBridge

# The directory of the current script (main.py)
WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# Mount static files from the webui directory
app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(WEBUI_DIR, "index.html"))


@app.get("/style.css")
async def read_css():
    return FileResponse(os.path.join(WEBUI_DIR, "style.css"))


@app.get("/script.js")
async def read_js():
    return FileResponse(os.path.join(WEBUI_DIR, "script.js"))


SUPPORTED_MODELS = [
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "moonshotai/Moonlight-16B-A3B",
    "moonshotai/Kimi-K2-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "XiaomiMiMo/MiMo-7B-RL",
]


@app.get("/local-hf-configs")
async def get_supported_models():
    """Return the list of HF model identifiers supported by the UI."""
    return SUPPORTED_MODELS


@app.get("/get-megatron-config/{model_path:path}")
async def get_remote_hf_config(model_path: str):
    """Fetch the HuggingFace config.json for the given model id."""
    url = f"https://huggingface.co/{model_path}/raw/main/config.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Failed to fetch config from {url}: {str(e)}"}


class MBridgeEstimateConfig(BaseModel):
    hf_model_path: str
    custom_hf_config: Optional[dict] = None  # Renamed for clarity

    # Hardware & Training
    num_gpus: int = 8
    mbs: int = 1
    seq_len: int = 4096
    use_distributed_optimizer: bool = True
    # Recompute settings are now part of the main config
    recompute_granularity: str = "selective"
    recompute_method: str = "uniform"
    recompute_num_layers: Optional[int] = 1

    # Selective recompute modules (optional list only used when granularity==selective)
    recompute_modules: Optional[list[str]] = None

    # 新增：Embedding/Loss PP Split 选项
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # Parallelism
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    vpp: Optional[int] = None
    etp: Optional[int] = None

    # Pipeline stage layer counts
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None

    # New field: custom pipeline-model-parallel layout
    pipeline_model_parallel_layout: Optional[str] = None  # Comma-separated ints

    @field_validator("num_gpus")
    def num_gpus_must_be_multiple_of_8(cls, v):
        if v <= 0 or v % 8 != 0:
            raise ValueError("must be a positive multiple of 8")
        return v


def patch_parallel_states(config: MBridgeEstimateConfig):
    from mbridge.core.parallel_states import ParallelStates

    ParallelStates.get_default_parallel_states = lambda: ParallelStates(
        tp_size=config.tp,
        pp_size=config.pp,
        ep_size=config.ep,
        cp_size=config.cp,
        vpp_size=config.vpp,
        etp_size=config.etp,
    )


@app.post("/estimate_with_mbridge")
async def estimate_with_mbridge(config: MBridgeEstimateConfig):
    # Validate Inputs
    if config.num_gpus <= 0 or config.num_gpus % 8 != 0:
        return {"error": "Total number of GPUs must be a positive multiple of 8."}

    parallel_product = config.tp * config.pp * config.cp
    if parallel_product == 0:  # Avoid division by zero
        return {"error": "Parallelism dimensions (TP, PP, CP) cannot be zero."}

    if config.num_gpus % parallel_product != 0:
        return {
            "error": f"Number of GPUs ({config.num_gpus}) must be divisible by the product of TP*PP*CP ({parallel_product})."
        }

    patch_parallel_states(config)

    # If the path is just a filename, assume it's in our local model-configs dir
    hf_model_path = config.hf_model_path
    # This logic needs to change. The custom config from the UI is an HF config, not a Megatron config.
    # We need to load it via a temporary file.
    if config.custom_hf_config:
        try:
            # Create a temporary file to save the custom HF config
            with tempfile.NamedTemporaryFile(
                mode="w+",
                delete=False,
                suffix=".json",
                dir=os.path.join("/dev/shm"),
            ) as tmp:
                json.dump(config.custom_hf_config, tmp)
                tmp_path = tmp.name

            # Load the bridge from the temporary config file
            from transformers import AutoConfig

            AutoConfig.trust_remote_code = True
            bridge = AutoBridge.from_pretrained(tmp_path)
            tf_config = bridge.config
            hf_config = bridge.hf_config

        finally:
            # Ensure the temporary file is deleted
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # If no custom config, load from the original path
        if not os.path.isabs(hf_model_path) and not hf_model_path.startswith(
            ("http", "./", "../")
        ):
            hf_model_path = os.path.join("/dev/shm", hf_model_path)
        bridge = AutoBridge.from_pretrained(hf_model_path)
        tf_config = bridge.config
        hf_config = bridge.hf_config

    # --- Configuration Unification ---
    # Update the tf_config with values from the form. This makes tf_config the single source of truth.
    tf_config.tensor_model_parallel_size = config.tp
    tf_config.pipeline_model_parallel_size = config.pp
    tf_config.expert_model_parallel_size = config.ep
    tf_config.context_parallel_size = config.cp
    tf_config.recompute_granularity = config.recompute_granularity
    tf_config.recompute_method = config.recompute_method
    tf_config.recompute_num_layers = config.recompute_num_layers
    # 新增：Selective 模式下的模块列表
    tf_config.recompute_modules = (
        config.recompute_modules if config.recompute_modules is not None else []
    )
    # 新增：Embedding/Loss PP Split
    tf_config.account_for_embedding_in_pipeline_split = (
        config.account_for_embedding_in_pipeline_split
    )
    tf_config.account_for_loss_in_pipeline_split = (
        config.account_for_loss_in_pipeline_split
    )
    tf_config.num_layers_per_virtual_pipeline_stage = (
        config.vpp if config.vpp and config.vpp > 1 else None
    )

    if config.num_layers_in_first_pipeline_stage is not None:
        tf_config.num_layers_in_first_pipeline_stage = (
            config.num_layers_in_first_pipeline_stage
        )
    if config.num_layers_in_last_pipeline_stage is not None:
        tf_config.num_layers_in_last_pipeline_stage = (
            config.num_layers_in_last_pipeline_stage
        )

    # Handle custom pipeline layout if provided
    if config.pipeline_model_parallel_layout:
        from megatron.core.transformer.pipeline_parallel_layer_layout import (
            PipelineParallelLayerLayout,
        )

        tf_config.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
            config.pipeline_model_parallel_layout, config.pp
        )
    # print(tf_config)

    # Create a minimal 'args' object with parameters not present in TransformerConfig
    args = argparse.Namespace()
    args.micro_batch_size = config.mbs
    args.seq_length = config.seq_len
    args.use_distributed_optimizer = config.use_distributed_optimizer
    args.data_parallel_size = config.num_gpus // parallel_product
    args.expert_tensor_parallel_size = config.etp if config.etp else 1

    # These are required by the estimator but can be derived or defaulted
    args.transformer_impl = "transformer_engine"
    args.fp8 = False
    args.num_experts = getattr(tf_config, "num_moe_experts", 1)  # Needed for layer spec
    args.moe_grouped_gemm = True  # Default
    args.qk_layernorm = tf_config.qk_layernorm
    args.multi_latent_attention = "deepseek" in getattr(hf_config, "model_type", "")
    args.padded_vocab_size = getattr(hf_config, "vocab_size")
    args.max_position_embeddings = getattr(hf_config, "max_position_embeddings")
    args.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
    args.world_size = config.num_gpus

    # This function now returns (aggregated_pp_reports, raw_chunk_reports)
    aggregated_reports, raw_chunk_reports = estimate_from_config(tf_config, args)

    processed_reports = []
    for rpt in aggregated_reports:
        p = rpt.copy()
        p.pop("details", None)
        processed_reports.append(p)

    return {"processed_report": processed_reports, "raw_report": raw_chunk_reports}
