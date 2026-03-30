# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Core module for MBridge package.

This module provides the core functionality for bridging between Hugging Face models
and Megatron-Core. It includes classes for different model types and utilities for
handling parallelism and model conversion.

Classes:
    Bridge: Base class for model bridges
    LLMBridge: Bridge implementation for language models
    VLMBridge: Bridge implementation for vision-language models
    AutoBridge: Automatic bridge selection based on model type

Functions:
    register_model: Decorator to register model classes
"""

from .auto_bridge import AutoBridge
from .bridge import Bridge, register_model
from .llm_bridge import LLMBridge
from .vlm_bridge import VLMBridge
