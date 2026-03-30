from copy import deepcopy

from mbridge.core import register_model
from mbridge.models.qwen3_vl.base_bridge import Qwen3VBaseBridge
from mbridge.models.qwen3_vl.transformer_config import Qwen3VLTransformerConfig

_QWEN3VIT_DIRECT_MAPPING = {
    "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
    "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
    "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
    "vision_model.merger.patch_norm.weight": "model.visual.merger.norm.weight",
    "vision_model.merger.patch_norm.bias": "model.visual.merger.norm.bias",
    "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
}

_QWEN3VIT_ATTENTION_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
        "model.visual.blocks.{layer_number}.attn.proj.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.bias": [
        "model.visual.blocks.{layer_number}.attn.proj.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
        "model.visual.blocks.{layer_number}.attn.qkv.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
        "model.visual.blocks.{layer_number}.attn.qkv.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
        "model.visual.blocks.{layer_number}.norm1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_bias": [
        "model.visual.blocks.{layer_number}.norm1.bias",
    ],
}

_QWEN3VIT_MLP_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc1.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": [
        "model.visual.blocks.{layer_number}.mlp.linear_fc2.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
        "model.visual.blocks.{layer_number}.norm2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_bias": [
        "model.visual.blocks.{layer_number}.norm2.bias",
    ],
}

_QWEN3VIT_OTHER_MAPPING = {
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.weight": [
        "model.visual.deepstack_merger_list.{layer_number}.norm.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.bias": [
        "model.visual.deepstack_merger_list.{layer_number}.norm.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.weight": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc1.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.bias": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc1.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.weight": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc2.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.bias": [
        "model.visual.deepstack_merger_list.{layer_number}.linear_fc2.bias",
    ],
}


@register_model("qwen3_vl")
class Qwen3VLBridge(Qwen3VBaseBridge):
    """
    Bridge implementation for Qwen3VL models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen3VL models, handling the conversion between
    Hugging Face Qwen3VL format and Megatron-Core.
    """

    TransformerConfigClass = Qwen3VLTransformerConfig
    _DIRECT_MAPPING = {
        **_QWEN3VIT_DIRECT_MAPPING,
        "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        **_QWEN3VIT_ATTENTION_MAPPING,
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    _MLP_MAPPING = {
        **_QWEN3VIT_MLP_MAPPING,
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.down_proj.weight",
        ],
    }

    _OTHER_MAPPING = {
        **_QWEN3VIT_OTHER_MAPPING,
    }

    def _build_config(self):
        """
        Build the configuration for LLaMA2 models.

        Configures LLaMA2-specific parameters such as attention bias settings.

        Returns:
            TransformerConfig: Configuration object for LLaMA2 models
        """
        return self._build_base_config(
            # qwen specific
            text_config_key="text_config",
            layernorm_epsilon=self.hf_config.text_config.rms_norm_eps,
            qk_layernorm=True,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
            async_tensor_model_parallel_allreduce=True,
            distribute_saved_activations=False,
            cp_comm_type="p2p",
            # qwen3vl specific
            mrope_section=self.hf_config.text_config.rope_scaling.get(
                "mrope_section",
                [24, 20, 20],
            ),
            patch_size=self.hf_config.vision_config.patch_size,
            temporal_patch_size=self.hf_config.vision_config.temporal_patch_size,
            in_channels=self.hf_config.vision_config.in_channels,
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            num_position_embeddings=self.hf_config.vision_config.num_position_embeddings,
            out_hidden_size=self.hf_config.vision_config.out_hidden_size,
            deepstack_visual_indexes=deepcopy(
                self.hf_config.vision_config.deepstack_visual_indexes
            ),
        )


@register_model("qwen3_vl_moe")
class Qwen3VLMoEBridge(Qwen3VBaseBridge):

    TransformerConfigClass = Qwen3VLTransformerConfig
    _DIRECT_MAPPING = {
        **_QWEN3VIT_DIRECT_MAPPING,
        "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        **_QWEN3VIT_ATTENTION_MAPPING,
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    _MLP_MAPPING = {
        **_QWEN3VIT_MLP_MAPPING,
        "language_model.decoder.layers.{layer_number}.pre_mlp_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.router.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.experts.gate_up_proj",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.experts.down_proj",
        ],
    }

    _OTHER_MAPPING = {
        **_QWEN3VIT_OTHER_MAPPING,
    }

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        if (
            name.startswith("vision_model.")
            or ".pre_mlp_layernorm.weight" in name
            or ".mlp.router.weight" in name
        ):
            return super()._weight_name_mapping_mlp(name)

        assert ".mlp.experts.linear_fc" in name
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        key = key.split(".weight")[0] + ".weight"
        convert_names = []
        mapping_names = self._MLP_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _build_config(self):
        return self._build_base_config(
            text_config_key="text_config",
            layernorm_epsilon=self.hf_config.text_config.rms_norm_eps,
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.text_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.text_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.text_config.num_experts,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_dtype="fp32",
            # moe_router_load_balancing_type="aux_loss",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
            async_tensor_model_parallel_allreduce=True,
            distribute_saved_activations=False,
            cp_comm_type="p2p",
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            # qwen3vl specific
            mrope_section=self.hf_config.text_config.rope_scaling.get(
                "mrope_section",
                [24, 20, 20],
            ),
            patch_size=self.hf_config.vision_config.patch_size,
            temporal_patch_size=self.hf_config.vision_config.temporal_patch_size,
            in_channels=self.hf_config.vision_config.in_channels,
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            num_position_embeddings=self.hf_config.vision_config.num_position_embeddings,
            out_hidden_size=self.hf_config.vision_config.out_hidden_size,
            deepstack_visual_indexes=deepcopy(
                self.hf_config.vision_config.deepstack_visual_indexes
            ),
        )
