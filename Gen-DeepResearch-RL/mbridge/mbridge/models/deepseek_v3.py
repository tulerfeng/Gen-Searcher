# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


from typing import Callable, Generator, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend

from ..core import LLMBridge, register_model


@register_model("deepseek_v3")
class DeepseekV3Bridge(LLMBridge):
    """
    Specific bridge implementation for DeepseekV3 models.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _MLP_MAPPING = {
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        "mlp.shared_experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"
        ],
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
        ],
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
        "mlp.router.expert_bias": [
            "model.layers.{layer_number}.mlp.gate.e_score_correction_bias"
        ],
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }

    _ATTENTION_MAPPING = {
        "input_layernorm.weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_q_proj.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight"
        ],
        "self_attention.linear_kv_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.kv_a_proj_with_mqa.weight"
        ],
        "self_attention.linear_kv_up_proj.layer_norm_weight": [
            "model.layers.{layer_number}.self_attn.kv_a_layernorm.weight"
        ],
        "self_attention.linear_kv_up_proj.weight": [
            "model.layers.{layer_number}.self_attn.kv_b_proj.weight"
        ],
        "self_attention.linear_q_down_proj.weight": [
            "model.layers.{layer_number}.self_attn.q_a_proj.weight"
        ],
        "self_attention.linear_q_up_proj.weight": [
            "model.layers.{layer_number}.self_attn.q_b_proj.weight"
        ],
        "self_attention.linear_q_up_proj.layer_norm_weight": [
            "model.layers.{layer_number}.self_attn.q_a_layernorm.weight"
        ],
    }

    _SHARED_STATE_DICT_MAPPING = {
        "embedding.word_embeddings.weight": [
            "model.embed_tokens.weight",
            "model.layers.61.embed_tokens.weight",
        ],
        "output_layer.weight": [
            "lm_head.weight",
            "model.layers.61.shared_head.head.weight",
        ],
    }

    TransformerConfigClass = MLATransformerConfig

    def _build_config(self):
        hf_config = self.hf_config
        mla_rope_config = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "rope",
        }
        if "rope_scaling" in hf_config and hf_config.rope_scaling is not None:
            mla_rope_config.update(hf_config.rope_scaling)
        moe_layer_freq = [1] * hf_config.num_hidden_layers
        for i in range(
            min(hf_config.first_k_dense_replace, hf_config.num_hidden_layers)
        ):
            moe_layer_freq[i] = 0

        mtp_args = {}
        if "num_nextn_predict_layers" in hf_config:
            mtp_args["mtp_num_layers"] = hf_config.num_nextn_predict_layers
            mtp_args["mtp_loss_scaling_factor"] = 0.1

        base_config = {
            "attention_backend": AttnBackend.fused,
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "ffn_hidden_size": hf_config.intermediate_size,
            "qk_layernorm": True,
            # moe specific
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_bias_update_rate": 0.001,
            "moe_router_enable_expert_bias": True,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "num_moe_experts": hf_config.n_routed_experts,
            "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size
            * hf_config.n_shared_experts,
            "moe_aux_loss_coeff": getattr(hf_config, "aux_loss_alpha", 0.001),
            # moe_router_load_balancing_type="seq_aux_loss",
            "moe_router_load_balancing_type": "none",  # default None for RL
            "moe_shared_expert_overlap": True,
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_pre_softmax": True,
            "moe_router_topk_scaling_factor": hf_config.routed_scaling_factor,
            "moe_layer_freq": moe_layer_freq,
            # MLA
            "q_lora_rank": hf_config.q_lora_rank,
            "kv_lora_rank": hf_config.kv_lora_rank,
            "qk_head_dim": hf_config.qk_nope_head_dim,
            "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
            "v_head_dim": hf_config.v_head_dim,
            "rotary_base": hf_config.rope_theta,
            "rotary_scaling_factor": mla_rope_config["factor"],
            "rope_type": mla_rope_config["type"],
            "mscale": mla_rope_config["mscale"],
            "mscale_all_dim": mla_rope_config["mscale_all_dim"],
            "beta_fast": mla_rope_config["beta_fast"],
            "beta_slow": mla_rope_config["beta_slow"],
            # mcore 0.12 moe
            "moe_router_dtype": "fp32",
            "disable_bf16_reduced_precision_matmul": True,
            # other
            "persist_layer_norm": True,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
        }

        import megatron.core

        megatron_version = getattr(megatron.core, "__version__")
        if megatron_version >= "0.14":
            base_config["original_max_position_embeddings"] = mla_rope_config[
                "original_max_position_embeddings"
            ]
        else:
            base_config["max_position_embeddings"] = mla_rope_config[
                "original_max_position_embeddings"
            ]

        base_config.update(mtp_args)
        return self._build_base_config(**base_config)

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.

        Constructs a dictionary of arguments required to initialize a GPTModel
        based on the configuration.

        Returns:
            dict: A dictionary of arguments for GPTModel initialization
        """
        ret = dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )

        return ret

    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        """
        Creates and returns a model provider function.

        The returned function creates a GPTModel with the specified configuration
        when called with pre_process and post_process parameters.

        Args:
            post_model_creation_callbacks: List of callbacks to be called after model creation

        Returns:
            function: A provider function that creates and returns a GPTModel instance
        """

        share_embeddings_and_output_weights = getattr(
            self.hf_config, "tie_word_embeddings", False
        )

        def provider(pre_process, post_process, vp_stage: Optional[int] = None):
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            gptmodel_args = self._get_gptmodel_args()
            if vp_stage is not None and self.has_vp_stage:
                gptmodel_args["vp_stage"] = vp_stage

            if (
                self.config.mtp_num_layers is not None
                and self.config.mtp_num_layers > 0
            ):
                if (
                    hasattr(transformer_layer_spec, "layer_specs")
                    and len(transformer_layer_spec.layer_specs) == 0
                ):
                    # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                    # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                    # transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config) # from copy https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0rc4/pretrain_gpt.py#L173
                    transformer_layer_spec_for_mtp = (
                        get_gpt_layer_with_transformer_engine_spec(
                            num_experts=self.config.num_moe_experts,
                            moe_grouped_gemm=self.config.moe_grouped_gemm,
                            qk_layernorm=self.config.qk_layernorm,
                            multi_latent_attention=True,
                        )
                    )
                else:
                    transformer_layer_spec_for_mtp = transformer_layer_spec
                mtp_block_spec = get_gpt_mtp_block_spec(
                    self.config,
                    transformer_layer_spec_for_mtp,
                    use_transformer_engine=True,
                    vp_stage=vp_stage,
                )
                gptmodel_args["mtp_block_spec"] = mtp_block_spec
                print(f"mtp_block_spec: {mtp_block_spec}")
            model = GPTModel(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                **gptmodel_args,
            )
            for callback in post_model_creation_callbacks:
                callback(
                    model,
                    pre_process=pre_process,
                    post_process=post_process,
                    config=self.config,
                    hf_config=self.hf_config,
                )

            return model

        return provider

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"
        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)
        elif (
            "self_attention" in mcore_weights_name
            or "input_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:

        # note: only support one mtp layer for now
        if (
            self.config.mtp_num_layers == 1
            and self.config.num_layers == 61
            and mcore_weights_name in self._SHARED_STATE_DICT_MAPPING
        ):
            hf_names = self._SHARED_STATE_DICT_MAPPING[mcore_weights_name]
            return hf_names, [mcore_weights] * len(hf_names)
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _get_safetensor_io(self, weights_path: str):
        if self.dtype == torch.bfloat16:
            from .ext.deepseek_v3.dequant_fp8_safetensor_io import (
                DequantFP8SafeTensorIO,
            )

            return DequantFP8SafeTensorIO(self._get_actual_hf_path(weights_path))
        else:
            raise NotImplemented("only support bfloat16 for now")

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [
                            x.format(layer_number=layer_number, expert_id=expert_id)
                            for x in mapping_names
                        ]
                    )
                else:
                    convert_names.extend(
                        [x.format(layer_number=layer_number) for x in mapping_names]
                    )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _convert_mtp_param(self, name: str) -> tuple[list[str]]:
        assert self.config.mtp_num_layers == 1, "only support one mtp layer for now"
        assert self.config.num_layers == 61, "only support 61 layers for now"
        direct_name_mapping = {
            "mtp.layers.0.enorm.weight": "model.layers.61.enorm.weight",
            "mtp.layers.0.hnorm.weight": "model.layers.61.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": "model.layers.61.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight": "model.layers.61.shared_head.norm.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]]
        assert "mtp.layers.0.transformer_layer" in name, "mtp not found"
        # use proxy name to convert
        proxy_name = name.replace("mtp.layers.0.transformer_layer", "decoder.layers.61")
        if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
            convert_names = self._weight_name_mapping_attention(proxy_name)
        elif "mlp" in proxy_name:
            convert_names = self._weight_name_mapping_mlp(proxy_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names
