from dataclasses import dataclass

import torch
from megatron.core.transformer import TransformerConfig


@dataclass
class InternvlTransformerConfig(TransformerConfig):
    initializer_factor: float = 0.1
    drop_path_rate: float = 0.1


def get_vision_model_config(
    config: InternvlTransformerConfig, apply_query_key_layer_scaling=False
):
    config.num_layers = 24
    config.num_attention_heads = 16
    config.num_query_groups = config.num_attention_heads
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_size = 1024
    config.kv_channels = 64
    config.hidden_dropout = 0.0
    config.ffn_hidden_size = 4096
    config.gated_linear_unit = False
    config.activation_func = torch.nn.functional.gelu
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = "LayerNorm"
    config.layernorm_epsilon = 1e-6
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    # internvl 内部的shape比较奇怪，有奇数
    config.sequence_parallel = False
    return config


def get_vision_projection_config(config: InternvlTransformerConfig):
    config.activation_func = torch.nn.functional.gelu
    config.ffn_hidden_size = config.hidden_size
    config.gated_linear_unit = False
    config.add_bias_linear = True
    config.normalization = "LayerNorm"
    config.layernorm_epsilon = 1e-5
    config.sequence_parallel = False
    return config
