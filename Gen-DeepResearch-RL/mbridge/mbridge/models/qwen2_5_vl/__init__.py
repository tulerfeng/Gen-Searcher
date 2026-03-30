import re
from copy import deepcopy
from typing import Callable, Generator, Optional

import torch
from transformers import AutoConfig

from mbridge.core.safetensor_io import SafeTensorIO

from ...core import VLMBridge, register_model
from ...core.util import unwrap_model
from .model import Qwen2_5VLModel
from .transformer_config import get_vision_model_config, get_vision_projection_config


class Qwen2_5VLSafeTensorIO(SafeTensorIO):

    def _mapping_weight_names_old2new(
        self,
        hf_weight_names: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        ret_hf_weight_names = []
        mapping_hf_weight_names = {}
        for hf_weight_name in hf_weight_names:
            new_hf_weight_name = hf_weight_name
            if hf_weight_name not in self.index:
                if new_hf_weight_name.startswith("model."):
                    new_hf_weight_name = new_hf_weight_name.replace(
                        "model.", "model.language_model."
                    )
                elif new_hf_weight_name.startswith("visual."):
                    new_hf_weight_name = new_hf_weight_name.replace(
                        "visual.", "model.visual."
                    )
            ret_hf_weight_names.append(new_hf_weight_name)
            mapping_hf_weight_names[new_hf_weight_name] = hf_weight_name
        return ret_hf_weight_names, mapping_hf_weight_names

    def _mapping_weight_names_new2old(
        self,
        hf_weight_names: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        ret_hf_weight_names = []
        mapping_hf_weight_names = {}
        for hf_weight_name in hf_weight_names:
            new_hf_weight_name = hf_weight_name
            if hf_weight_name in self.index:
                if new_hf_weight_name.startswith("model.language_model."):
                    new_hf_weight_name = new_hf_weight_name.replace(
                        "model.language_model.", "model."
                    )
                elif new_hf_weight_name.startswith("model.visual."):
                    new_hf_weight_name = new_hf_weight_name.replace(
                        "model.visual.", "visual."
                    )
            ret_hf_weight_names.append(new_hf_weight_name)
            mapping_hf_weight_names[new_hf_weight_name] = hf_weight_name
        return ret_hf_weight_names, mapping_hf_weight_names


@register_model("qwen2_5_vl")
class Qwen2_5VLBridge(VLMBridge):
    """
    Bridge implementation for Qwen2.5VL models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen2.5VL models, handling the conversion between
    Hugging Face Qwen2.5VL format and Megatron-Core.
    """

    _DIRECT_MAPPING = {
        "language_model.embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
        "vision_model.patch_embed.proj.weight": "visual.patch_embed.proj.weight",
        "vision_model.decoder.final_layernorm.weight": "visual.merger.ln_q.weight",
        "vision_model.projection.encoder.linear_fc1.weight": "visual.merger.mlp.0.weight",
        "vision_model.projection.encoder.linear_fc1.bias": "visual.merger.mlp.0.bias",
        "vision_model.projection.encoder.linear_fc2.weight": "visual.merger.mlp.2.weight",
        "vision_model.projection.encoder.linear_fc2.bias": "visual.merger.mlp.2.bias",
    }
    _ATTENTION_MAPPING = {
        # language
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        # vision
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "visual.blocks.{layer_number}.attn.proj.weight"
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.bias": [
            "visual.blocks.{layer_number}.attn.proj.bias"
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "visual.blocks.{layer_number}.attn.qkv.bias"
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "visual.blocks.{layer_number}.attn.qkv.weight"
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "visual.blocks.{layer_number}.norm1.weight"
        ],
    }

    _MLP_MAPPING = {
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
            "model.layers.{layer_number}.mlp.gate_proj.bias",
            "model.layers.{layer_number}.mlp.up_proj.bias",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.down_proj.weight"
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": [
            "model.layers.{layer_number}.mlp.down_proj.bias"
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        # vision
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "visual.blocks.{layer_number}.mlp.gate_proj.weight",
            "visual.blocks.{layer_number}.mlp.up_proj.weight",
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
            "visual.blocks.{layer_number}.mlp.gate_proj.bias",
            "visual.blocks.{layer_number}.mlp.up_proj.bias",
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
            "visual.blocks.{layer_number}.mlp.down_proj.weight"
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": [
            "visual.blocks.{layer_number}.mlp.down_proj.bias"
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
            "visual.blocks.{layer_number}.norm2.weight"
        ],
    }

    def _get_safetensor_io(self, weights_path: str):
        return Qwen2_5VLSafeTensorIO(self._get_actual_hf_path(weights_path))

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["language_model.output_layer.weight"] = (
                "model.embed_tokens.weight"
            )

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            return ["model.embed_tokens.weight"]
        return []

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._ATTENTION_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._MLP_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        Export MCore weights to Hugging Face format.

        Takes MCore weight names and tensor, outputs Hugging Face weight names and tensors.
        Due to MCore's runtime optimizations involving weight merging, output can be a list.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor

        Returns:
            tuple: (hf_names, hf_weights) - lists of Hugging Face weight names and tensors

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        if len(hf_names) == 1:
            if re.match(
                r"vision_model.decoder.layers.\d+.self_attention.linear_qkv.[weight|bias]",
                mcore_weights_name,
            ):
                vision_num_query_groups = self.hf_config.vision_config.num_heads
                vision_attention_heads = vision_num_query_groups
                vision_hidden_size = self.hf_config.vision_config.hidden_size
                vision_head_dim = vision_hidden_size // vision_attention_heads
                out_shape = (
                    [vision_num_query_groups, -1, vision_hidden_size]
                    if ".bias" not in mcore_weights_name
                    else [vision_num_query_groups, -1]
                )
                qkv = mcore_weights.view(*out_shape)
                q_len = (
                    vision_head_dim * vision_attention_heads // vision_num_query_groups
                )
                k_len = vision_head_dim
                v_len = vision_head_dim
                single_out_shape = (
                    [-1, vision_hidden_size]
                    if ".bias" not in mcore_weights_name
                    else [-1]
                )
                q = qkv[:, :q_len].reshape(*single_out_shape)
                k = qkv[:, q_len : q_len + k_len].reshape(*single_out_shape)
                v = qkv[:, q_len + k_len :].reshape(*single_out_shape)
                in_shape = (
                    [
                        3,
                        vision_num_query_groups,
                        -1,
                        vision_head_dim,
                        vision_hidden_size,
                    ]
                    if "bias" not in mcore_weights_name
                    else [3, vision_num_query_groups, -1]
                )
                qkv = torch.cat([q, k, v], dim=0).view(*in_shape).contiguous()
                original_shape = (
                    [-1, vision_hidden_size]
                    if ".bias" not in mcore_weights_name
                    else [-1]
                )
                qkv = qkv.view(*original_shape)
                return [hf_names[0]], [qkv]
            return [hf_names[0]], [mcore_weights]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # split qkv
            assert len(hf_names) == 3
            # split qkv
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads

            if "vision_model" in mcore_weights_name:
                num_attention_heads = self.hf_config.vision_config.num_heads
                num_key_value_heads = self.hf_config.vision_config.num_heads
            head_dim = getattr(
                self.hf_config, "head_dim", hidden_dim // num_attention_heads
            )
            out_shape = (
                [num_key_value_heads, -1, hidden_dim]
                if ".bias" not in mcore_weights_name
                else [num_key_value_heads, -1]
            )
            qkv = mcore_weights.view(*out_shape)
            q_len = head_dim * num_attention_heads // num_key_value_heads
            k_len = head_dim
            v_len = head_dim
            single_out_shape = (
                [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
            )
            q = qkv[:, :q_len].reshape(*single_out_shape)
            k = qkv[:, q_len : q_len + k_len].reshape(*single_out_shape)
            v = qkv[:, q_len + k_len :].reshape(*single_out_shape)
            return hf_names, [q, k, v]

        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            # split gate_proj and up_proj
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Import Hugging Face weights to MCore format.

        Takes Hugging Face weight names and tensors, outputs MCore weight tensor.
        Due to MCore's runtime optimizations involving weight merging, input is a list.

        Args:
            mcore_weights_name: MCore weight name
            hf_weights: List of Hugging Face weight tensors

        Returns:
            torch.Tensor: MCore weight tensor

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        if len(hf_weights) == 1:
            if re.match(
                r"vision_model.decoder.layers.\d+.self_attention.linear_qkv.[weight|bias]",
                mcore_weights_name,
            ):
                vision_num_query_groups = self.hf_config.vision_config.num_heads
                vision_attention_heads = vision_num_query_groups
                vision_hidden_size = self.hf_config.vision_config.hidden_size
                vision_head_dim = vision_hidden_size // vision_attention_heads
                in_shape = (
                    [
                        3,
                        vision_num_query_groups,
                        -1,
                        vision_head_dim,
                        vision_hidden_size,
                    ]
                    if "bias" not in mcore_weights_name
                    else [3, vision_num_query_groups, -1]
                )
                q, k, v = hf_weights[0].view(*in_shape)
                q = q.view([vision_num_query_groups, vision_head_dim, -1])
                k = k.view([vision_num_query_groups, vision_head_dim, -1])
                v = v.view([vision_num_query_groups, vision_head_dim, -1])

                out_shape = (
                    [-1, vision_hidden_size]
                    if ".bias" not in mcore_weights_name
                    else [-1]
                )
                qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
                return qkv

            return hf_weights[0]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            if "vision_model" in mcore_weights_name:
                num_attention_heads = self.hf_config.vision_config.num_heads
                num_key_value_heads = self.hf_config.vision_config.num_heads
            head_dim = getattr(
                self.hf_config, "head_dim", hidden_dim // num_attention_heads
            )
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // group_dim
            q = q.view(
                [
                    real_num_key_value_heads,
                    group_dim,
                    -1,
                ]
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qkv
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            # merge gate_proj and up_proj
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _get_layer_number(self, vpp_rank: int, local_layer_number: int, models) -> int:
        # map vpp layer number to global layer number
        unwrapped_model = unwrap_model(models[vpp_rank])
        global_layer_number = (
            unwrapped_model.language_model.decoder.layers[
                local_layer_number
            ].layer_number
            - 1
        )
        return global_layer_number

    def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module, consider_ep: bool = True
    ) -> dict[str, str]:
        """
        Map local weight names to global weight names, supporting VPP and EP.

        Args:
            model: The model instance

        Returns:
            dict: Mapping from local weight names to global weight names
        """

        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "language_model") and hasattr(
            model.language_model, "decoder"
        ):
            for idx, layer in enumerate(model.language_model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [
            k for k in model.state_dict().keys() if "_extra_state" not in k
        ]
        ret = {}
        for param_name in all_param_names:
            keyword = "language_model.decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(
                    f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
                )
            else:
                ret[param_name] = param_name

        return ret

    def _model_provider(
        self,
        post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]],
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

            from megatron.core.extensions.transformer_engine import (
                TEColumnParallelLinear,
                TERowParallelLinear,
            )
            from megatron.core.models.gpt.moe_module_specs import MLPSubmodules
            from megatron.core.models.vision.vit_layer_specs import (
                get_vit_layer_with_transformer_engine_spec,
            )

            vision_transformer_config = get_vision_model_config(deepcopy(self.config))
            vision_transformer_config.pipeline_model_parallel_size = 1
            vision_transformer_config.first_pipeline_num_layers = None

            vision_projection_config = get_vision_projection_config(
                deepcopy(self.config),
                vision_transformer_config.hidden_size,
                spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            )
            vision_projection_layer_spec = MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

            model = Qwen2_5VLModel(
                language_transformer_config=self.config,
                language_transformer_layer_spec=transformer_layer_spec,
                language_vocab_size=self.hf_config.vocab_size,
                language_max_sequence_length=self.hf_config.max_position_embeddings,
                vision_transformer_config=vision_transformer_config,
                vision_transformer_layer_spec=vision_transformer_layer_spec,
                vision_projection_config=vision_projection_config,
                vision_projection_layer_spec=vision_projection_layer_spec,
                vision_projection_type="mlp",
                language_rotary_base=self.hf_config.rope_theta,
                pre_process=pre_process,
                post_process=post_process,
                add_decoder=True,
                add_encoder=True,
                parallel_output=True,
                language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                vp_stage=vp_stage,
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

    def _build_config(self):
        """
        Build the configuration for LLaMA2 models.

        Configures LLaMA2-specific parameters such as attention bias settings.

        Returns:
            TransformerConfig: Configuration object for LLaMA2 models
        """
        return self._build_base_config(
            # qwen specific
            add_qkv_bias=True,
            mrope_section=self.hf_config.rope_scaling["mrope_section"],
        )
