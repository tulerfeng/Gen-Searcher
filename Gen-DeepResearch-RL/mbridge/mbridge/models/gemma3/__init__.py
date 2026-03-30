import re
from copy import deepcopy
from typing import Callable, Generator, Optional

import torch
from transformers import AutoConfig

from mbridge.core import VLMBridge, register_model
from mbridge.core.util import unwrap_model
from mbridge.models.gemma3.model import Gemma3Model
from mbridge.models.gemma3.projector import get_projector_module_spec_te
from mbridge.models.gemma3.transformer_config import (
    Gemma3TransformerConfig,
    get_vision_model_config,
    get_vision_projection_config,
)
from mbridge.models.gemma3.transformer_layer import (
    get_gemma3_layer_spec_te,
    get_layer_spec_te,
)


@register_model("gemma3")
class Gemma3Bridge(VLMBridge):
    """
    Bridge implementation for Gemma3 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Gemma3 models, handling the conversion between
    Hugging Face Gemma3 format and Megatron-Core.
    """

    TransformerConfigClass = Gemma3TransformerConfig

    _DIRECT_MAPPING = {
        "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
        "vision_model.position_embeddings.weight": "vision_tower.vision_model.embeddings.position_embedding.weight",
        "vision_model.conv1.weight": "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_model.conv1.bias": "vision_tower.vision_model.embeddings.patch_embedding.bias",
        "vision_model.ln_post.weight": "vision_tower.vision_model.post_layernorm.weight",
        "vision_model.ln_post.bias": "vision_tower.vision_model.post_layernorm.bias",
        "vision_projection.mm_input_projection.layer_norm_weight": "multi_modal_projector.mm_soft_emb_norm.weight",
        "vision_projection.mm_input_projection.weight": "multi_modal_projector.mm_input_projection_weight",
    }

    _ATTENTION_MAPPING = {
        # language
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "language_model.model.layers.{layer_number}.self_attn.q_proj.weight",
            "language_model.model.layers.{layer_number}.self_attn.k_proj.weight",
            "language_model.model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "language_model.model.layers.{layer_number}.input_layernorm.weight"
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
            "language_model.model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
            "language_model.model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        # vision
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.k_proj.weight",
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.q_proj.bias",
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.k_proj.bias",
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.out_proj.weight",
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.bias": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.self_attn.out_proj.bias",
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.layer_norm1.weight",
        ],
        "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_bias": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.layer_norm1.bias",
        ],
    }

    _MLP_MAPPING = {
        # language
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.mlp.gate_proj.weight",
            "language_model.model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
            "language_model.model.layers.{layer_number}.mlp.down_proj.weight"
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
            "language_model.model.layers.{layer_number}.pre_feedforward_layernorm.weight"
        ],
        # vision
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.mlp.fc2.weight",
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.mlp.fc2.bias",
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.mlp.fc1.weight",
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.mlp.fc1.bias",
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.layer_norm2.weight",
        ],
        "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_bias": [
            "vision_tower.vision_model.encoder.layers.{layer_number}.layer_norm2.bias",
        ],
    }

    _OTHER_MAPPING = {
        "language_model.decoder.layers.{layer_number}.post_attention_layernorm.weight": [
            "language_model.model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "language_model.decoder.layers.{layer_number}.post_feedforward_layernorm.weight": [
            "language_model.model.layers.{layer_number}.post_feedforward_layernorm.weight"
        ],
    }

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["language_model.output_layer.weight"] = (
                "language_model.model.embed_tokens.weight"
            )

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            return ["language_model.model.embed_tokens.weight"]
        return []

    def _get_mcore_config_by_name(self, mcore_weights_name: str):
        if "vision_projection." in mcore_weights_name:
            assert hasattr(self, "vision_projection_config")
            return self.vision_projection_config
        if "vision_model." in mcore_weights_name:
            assert hasattr(self, "vision_config")
            return self.vision_config
        return self.config

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

    def _weight_name_mapping_other(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._OTHER_MAPPING[key]
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
            if "vision_projection.mm_input_projection.weight" == mcore_weights_name:
                return [hf_names[0]], [mcore_weights.T.contiguous()]
            return [hf_names[0]], [mcore_weights]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # split qkv
            assert len(hf_names) == 3
            if "language_model." in mcore_weights_name:
                tmp_config = self.hf_config.text_config
            else:
                assert "vision_model." in mcore_weights_name
                tmp_config = self.hf_config.vision_config

            num_attention_heads = tmp_config.num_attention_heads
            num_key_value_heads = getattr(
                tmp_config, "num_key_value_heads", num_attention_heads
            )
            hidden_dim = tmp_config.hidden_size
            head_dim = getattr(
                tmp_config, "head_dim", hidden_dim // num_attention_heads
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
            if "vision_projection.mm_input_projection.weight" == mcore_weights_name:
                return hf_weights[0].T.contiguous()
            return hf_weights[0]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            if "language_model." in mcore_weights_name:
                tmp_config = self.hf_config.text_config
            else:
                assert "vision_model." in mcore_weights_name
                tmp_config = self.hf_config.vision_config

            num_attention_heads = tmp_config.num_attention_heads
            num_key_value_heads = getattr(
                tmp_config, "num_key_value_heads", num_attention_heads
            )
            hidden_dim = tmp_config.hidden_size
            head_dim = getattr(
                tmp_config, "head_dim", hidden_dim // num_attention_heads
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
            assert vp_stage is None, "not support vpp now"
            transformer_layer_spec = get_gemma3_layer_spec_te(is_vit=False)
            self.config.activation_func = torch.nn.functional.gelu
            self.config.embed_scale = self.config.hidden_size**0.5
            self.config.sliding_window = self.hf_config.text_config.sliding_window

            vision_transformer_layer_spec = get_layer_spec_te(is_vit=True)
            vision_config = deepcopy(self.config)
            vision_config = get_vision_model_config(vision_config)
            vision_config.recompute_granularity = None
            vision_config.recompute_method = None
            vision_config.recompute_num_layers = None
            vision_config.pipeline_model_parallel_size = 1
            vision_config.num_layers_in_first_pipeline_stage = None
            vision_config.num_layers_in_last_pipeline_stage = None

            vision_projection_layer_spec = get_projector_module_spec_te()
            vision_projection_config = deepcopy(self.config)
            vision_projection_config = get_vision_projection_config(
                vision_projection_config
            )
            vision_projection_config.pipeline_model_parallel_size = 1
            vision_projection_config.recompute_granularity = None
            vision_projection_config.recompute_method = None
            vision_projection_config.recompute_num_layers = None

            setattr(self, "vision_config", vision_config)
            setattr(self, "vision_projection_config", vision_projection_config)

            model = Gemma3Model(
                language_transformer_config=self.config,
                language_transformer_layer_spec=transformer_layer_spec,
                language_vocab_size=self.hf_config.text_config.vocab_size,
                language_max_sequence_length=self.hf_config.text_config.max_position_embeddings,
                vision_transformer_config=vision_config,
                vision_transformer_layer_spec=vision_transformer_layer_spec,
                vision_projection_config=vision_projection_config,
                vision_projection_layer_spec=vision_projection_layer_spec,
                parallel_output=True,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                language_position_embedding_type="rope",
                language_rotary_percent=1.0,
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=pre_process,
                add_decoder=True,
                img_h=self.config.image_size,
                img_w=self.config.image_size,
                patch_dim=self.config.patch_size,
                language_rotary_base=self.hf_config.text_config.rope_theta,
                language_rope_scaling=False,
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
            text_config_key="text_config",
            add_qkv_bias=False,
            qk_layernorm=True,
            image_size=896,
            patch_size=14,
            mm_tokens_per_image=256,
            layernorm_zero_centered_gamma=True,
            normalization="RMSNorm",
            rope_local_base_freq=10000.0,
        )
