# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from ..core import LLMBridge, register_model
from ..core.bridge import Bridge, register_model


@register_model("llama")
class LLaMABridge(LLMBridge):
    """
    Bridge implementation for LLaMA2 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for LLaMA2 models, handling the conversion between
    Hugging Face LLaMA2 format and Megatron-Core.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }
    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    def _build_config(self):
        """
        Build the configuration for LLaMA2 models.

        Configures LLaMA2-specific parameters such as attention bias settings.

        Returns:
            TransformerConfig: Configuration object for LLaMA2 models
        """
        qkv_bias = getattr(self.hf_config, "attention_bias", False)
        return self._build_base_config(add_qkv_bias=qkv_bias)

    def _get_gptmodel_args(self) -> dict:
        """
        Get GPT model arguments specific to LLaMA2.

        Handles LLaMA2-specific configurations such as RoPE scaling
        for extended context length models.

        Returns:
            dict: Dictionary of arguments for GPTModel initialization
        """
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                # assert (
                #     self.hf_config.rope_scaling["type"] == "linear"
                # ), "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = (
                    self.hf_config.rope_scaling["factor"]
                )
        ret = dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )
        ret.update(rope_scaling_args)
        return ret
