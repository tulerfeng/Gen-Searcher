from dataclasses import dataclass

import torch
from megatron.core.jit import jit_fuser
from megatron.core.transformer import TransformerConfig


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


@dataclass
class Gemma3TransformerConfig(TransformerConfig):
    transformer_impl: str = "transformer_engine"
    image_size: int = 896
    patch_size: int = 14
    mm_tokens_per_image: int = 256

    embed_scale: float = 1.0
    rope_local_base_freq: float = 10000.0
    sliding_window_pattern: int = 6
    sliding_window: int = 1024
    query_pre_attn_scalar: int = 256


def get_vision_model_config(config: TransformerConfig):
    config.num_layers = 27
    config.num_attention_heads = 16
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_size = 1152
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = 4304
    config.gated_linear_unit = False
    config.activation_func = fast_gelu
    config.kv_channels = 72
    config.num_query_groups = 16
    config.layernorm_zero_centered_gamma = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = "LayerNorm"
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    config.layernorm_epsilon = 1e-6
    return config


def get_vision_projection_config(config: Gemma3TransformerConfig):
    config.image_size = 896
    config.patch_size = 14
    config.mm_tokens_per_image = 256
    config.layernorm_zero_centered_gamma = True
    config.add_bias_linear = False
    return config
