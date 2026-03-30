# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Models module for MBridge package.

This module provides specific bridge implementations for various language models.
Each bridge class handles the conversion between Hugging Face model formats and
Megatron-Core, with model-specific adjustments.

Classes:
    Qwen2Bridge: Bridge implementation for Qwen2 models
    Qwen3Bridge: Bridge implementation for Qwen3 models
    LLaMA2Bridge: Bridge implementation for LLaMA2 models
    DeepseekV3Bridge: Bridge implementation for DeepseekV3 models
    MimoBridge: Bridge implementation for MIMO models
"""
# isort: off
# fmt: off

from .deepseek_v3 import DeepseekV3Bridge
from .llama import LLaMABridge
from .mimo import MimoBridge
from .mixtral import MixtralBridge
from .qwen2 import Qwen2Bridge
from .qwen2_5_vl import Qwen2_5VLBridge
from .qwen2moe import Qwen2MoEBridge
from .qwen3 import Qwen3Bridge
from .qwen3moe import Qwen3MoEBridge
from .glm4moe import GLM4MoEBridge
from .glm4_vl import Glm4VLBridgeMoe, Glm4VLBridgeDense
from .gemma3 import Gemma3Bridge
from .internvl3 import InternVL3Bridge
from .qwen3_vl import Qwen3VLBridge, Qwen3VLBridge
