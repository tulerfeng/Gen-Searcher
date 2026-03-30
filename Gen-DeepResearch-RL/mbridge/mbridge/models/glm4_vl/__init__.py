from typing import Callable, Generator, Optional

import torch
from megatron.core.transformer.enums import AttnBackend

from ...core import register_model
from .base_bridge import Glm4VLBridgeBase
from .transformer_layer import Glm4TransformerLayer

"""
actually this only apply to glm-4.1v
"""


@register_model("glm4v")
class Glm4VLBridgeDense(Glm4VLBridgeBase):
    try:
        from transformers.models.glm4v.modeling_glm4v import Glm4vVisionModel
    except:
        Glm4vVisionModel = None

    HfVisionClass: type = Glm4vVisionModel

    _VISUAL_MAPPING = {
        # vision
        "vision_model.blocks.{layer_number}.attn.proj.weight": [
            "model.visual.blocks.{layer_number}.attn.proj.weight"
        ],
        "vision_model.blocks.{layer_number}.attn.qkv.weight": [
            "model.visual.blocks.{layer_number}.attn.qkv.weight"
        ],
        "vision_model.blocks.{layer_number}.norm1.weight": [
            "model.visual.blocks.{layer_number}.norm1.weight"
        ],
        "vision_model.blocks.{layer_number}.mlp.gate_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.gate_proj.weight",
        ],
        "vision_model.blocks.{layer_number}.mlp.up_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.up_proj.weight",
        ],
        "vision_model.blocks.{layer_number}.mlp.down_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.down_proj.weight",
        ],
        "vision_model.blocks.{layer_number}.norm2.weight": [
            "model.visual.blocks.{layer_number}.norm2.weight"
        ],
    }

    _DIRECT_MAPPING = {
        "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
        "vision_model.downsample.weight": "model.visual.downsample.weight",
        "vision_model.downsample.bias": "model.visual.downsample.bias",
        "vision_model.merger.gate_proj.weight": "model.visual.merger.gate_proj.weight",
        "vision_model.merger.up_proj.weight": "model.visual.merger.up_proj.weight",
        "vision_model.merger.down_proj.weight": "model.visual.merger.down_proj.weight",
        "vision_model.merger.proj.weight": "model.visual.merger.proj.weight",
        "vision_model.merger.post_projection_norm.weight": "model.visual.merger.post_projection_norm.weight",
        "vision_model.merger.post_projection_norm.bias": "model.visual.merger.post_projection_norm.bias",
        "vision_model.embeddings.position_embedding.weight": "model.visual.embeddings.position_embedding.weight",
        "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
        "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
        "vision_model.post_conv_layernorm.weight": "model.visual.post_conv_layernorm.weight",
        "vision_model.post_layernorm.weight": "model.visual.post_layernorm.weight",
    }
    _ATTENTION_MAPPING = {
        # language
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight"
        ],
        "language_model.decoder.layers.{layer_number}.post_self_attn_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_self_attn_layernorm.weight"
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_up_proj.weight"
        ],
        "mlp.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.down_proj.weight"
        ],
        "post_mlp_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_mlp_layernorm.weight"
        ],
    }

    # norm layer of glm4.1v is a little weird
    def _get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.transformer.transformer_block import (
            TransformerBlockSubmodules,
        )

        block_spec = super()._get_transformer_layer_spec(vp_stage=vp_stage)
        assert isinstance(block_spec, TransformerBlockSubmodules)
        for layer_spec in block_spec.layer_specs:
            layer_spec.module = Glm4TransformerLayer
            layer_spec.submodules.post_self_attn_layernorm = block_spec.layer_norm
            layer_spec.submodules.post_mlp_layernorm = block_spec.layer_norm
        # replace layer
        return block_spec

    def _build_config(self):
        kwargs = {}
        kwargs["image_start_token_id"] = self.hf_config.image_start_token_id
        kwargs["image_end_token_id"] = self.hf_config.image_end_token_id
        kwargs["video_start_token_id"] = self.hf_config.video_start_token_id
        kwargs["video_end_token_id"] = self.hf_config.video_end_token_id
        kwargs["image_token_id"] = self.hf_config.image_token_id
        kwargs["video_token_id"] = self.hf_config.video_token_id
        kwargs["spatial_merge_size"] = self.hf_config.vision_config.spatial_merge_size
        kwargs["mrope_section"] = self.hf_config.rope_scaling.get("mrope_section", None)
        kwargs["attention_backend"] = AttnBackend.fused
        kwargs["moe_router_dtype"] = "fp32"
        kwargs["disable_bf16_reduced_precision_matmul"] = True
        kwargs["persist_layer_norm"] = True
        kwargs["bias_activation_fusion"] = True
        kwargs["bias_dropout_fusion"] = True
        kwargs["add_qkv_bias"] = True

        return self._build_base_config(**kwargs)


# a hack for the origin code to work with Glm4vMoeConfig
def attr_getter(self, attr):
    if attr == "text_config":
        raise AttributeError()
    return getattr(self.text_config, attr)


try:
    from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig

    Glm4vMoeConfig.__getattr__ = attr_getter
except:
    pass


"""
actually this only apply to glm-4.5v
"""


@register_model("glm4v_moe")
class Glm4VLBridgeMoe(Glm4VLBridgeBase):
    try:
        from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeVisionModel
    except:
        Glm4vMoeVisionModel = None

    HfVisionClass: type = Glm4vMoeVisionModel

    _VISUAL_MAPPING = {
        # vision
        "vision_model.blocks.{layer_number}.attn.proj.weight": [
            "model.visual.blocks.{layer_number}.attn.proj.weight"
        ],
        "vision_model.blocks.{layer_number}.attn.qkv.weight": [
            "model.visual.blocks.{layer_number}.attn.qkv.weight"
        ],
        "vision_model.blocks.{layer_number}.norm1.weight": [
            "model.visual.blocks.{layer_number}.norm1.weight"
        ],
        "vision_model.blocks.{layer_number}.mlp.gate_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.gate_proj.weight",
        ],
        "vision_model.blocks.{layer_number}.mlp.up_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.up_proj.weight",
        ],
        "vision_model.blocks.{layer_number}.mlp.down_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.down_proj.weight",
        ],
        "vision_model.blocks.{layer_number}.norm2.weight": [
            "model.visual.blocks.{layer_number}.norm2.weight"
        ],
    }

    _DIRECT_MAPPING = {
        "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
        "vision_model.downsample.weight": "model.visual.downsample.weight",
        "vision_model.downsample.bias": "model.visual.downsample.bias",
        "vision_model.merger.gate_proj.weight": "model.visual.merger.gate_proj.weight",
        "vision_model.merger.up_proj.weight": "model.visual.merger.up_proj.weight",
        "vision_model.merger.down_proj.weight": "model.visual.merger.down_proj.weight",
        "vision_model.merger.proj.weight": "model.visual.merger.proj.weight",
        "vision_model.merger.post_projection_norm.weight": "model.visual.merger.post_projection_norm.weight",
        "vision_model.merger.post_projection_norm.bias": "model.visual.merger.post_projection_norm.bias",
        "vision_model.embeddings.position_embedding.weight": "model.visual.embeddings.position_embedding.weight",
        "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
        "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
        "vision_model.post_conv_layernorm.weight": "model.visual.post_conv_layernorm.weight",
        "vision_model.post_layernorm.weight": "model.visual.post_layernorm.weight",
    }
    _ATTENTION_MAPPING = {
        # language
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight"
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.down_proj.weight"
        ],
        "mlp.shared_experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"
        ],
        "mlp.shared_experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
        ],
        "pre_mlp_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.router.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate.weight"
        ],
        "mlp.router.expert_bias": [
            "model.language_model.layers.{layer_number}.mlp.gate.e_score_correction_bias"
        ],
        "mlp.experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }

    def _build_config(self):
        kwargs = {}
        kwargs["image_start_token_id"] = self.hf_config.image_start_token_id
        kwargs["image_end_token_id"] = self.hf_config.image_end_token_id
        kwargs["video_start_token_id"] = self.hf_config.video_start_token_id
        kwargs["video_end_token_id"] = self.hf_config.video_end_token_id
        kwargs["image_token_id"] = self.hf_config.image_token_id
        kwargs["video_token_id"] = self.hf_config.video_token_id
        kwargs["spatial_merge_size"] = self.hf_config.vision_config.spatial_merge_size
        kwargs["mrope_section"] = self.hf_config.rope_scaling.get("mrope_section", None)
        kwargs["attention_backend"] = AttnBackend.fused
        kwargs["disable_bf16_reduced_precision_matmul"] = True
        kwargs["persist_layer_norm"] = True
        kwargs["bias_activation_fusion"] = True
        kwargs["bias_dropout_fusion"] = True
        kwargs["add_qkv_bias"] = True

        # moe
        kwargs["moe_token_dispatcher_type"] = "alltoall"
        kwargs["moe_router_bias_update_rate"] = 0.001
        kwargs["moe_router_enable_expert_bias"] = True
        kwargs["moe_aux_loss_coeff"] = getattr(self.hf_config, "aux_loss_alpha", 0.001)
        kwargs["moe_router_dtype"] = "fp32"
        # kwargs["moe_router_load_balancing_type"] = "seq_aux_loss"
        kwargs["moe_router_load_balancing_type"] = "none"  # default None for RL
        kwargs["moe_shared_expert_overlap"] = True
        kwargs["moe_grouped_gemm"] = True
        kwargs["moe_ffn_hidden_size"] = self.hf_config.moe_intermediate_size
        kwargs["num_moe_experts"] = self.hf_config.n_routed_experts
        kwargs["moe_shared_expert_intermediate_size"] = int(
            self.hf_config.moe_intermediate_size
        ) * int(self.hf_config.n_shared_experts)
        kwargs["moe_router_topk"] = self.hf_config.text_config.num_experts_per_tok
        kwargs["moe_router_pre_softmax"] = True
        kwargs["moe_router_score_function"] = "sigmoid"

        kwargs["moe_router_topk_scaling_factor"] = self.hf_config.routed_scaling_factor
        first_k_dense_replace = getattr(self.hf_config, "first_k_dense_replace", 0)

        # first_k_dense_replace
        if first_k_dense_replace > 0:
            num_hidden_layers = self.hf_config.text_config.num_hidden_layers
            assert num_hidden_layers > first_k_dense_replace
            moe_layer_freq = [1] * self.hf_config.num_hidden_layers
            for i in range(first_k_dense_replace):
                moe_layer_freq[i] = 0
            kwargs["moe_layer_freq"] = moe_layer_freq

        return self._build_base_config(**kwargs)
