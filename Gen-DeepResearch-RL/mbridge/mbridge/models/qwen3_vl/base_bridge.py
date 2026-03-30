from copy import deepcopy
from typing import Callable, Optional

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)

from mbridge.core import VLMBridge
from mbridge.core.util import unwrap_model
from mbridge.models.qwen3_vl.model import Qwen3VLModel
from mbridge.models.qwen3_vl.transformer_config import get_vision_model_config
from mbridge.models.qwen3_vl.utils import PatchMergerSubmodules


class Qwen3VBaseBridge(VLMBridge):

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config.text_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["language_model.output_layer.weight"] = (
                "model.language_model.embed_tokens.weight"
            )

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config.text_config, "tie_word_embeddings", False):
            return ["model.language_model.embed_tokens.weight"]
        return []

    def _get_mcore_config_by_name(self, mcore_weights_name: str):
        if "vision_model." in mcore_weights_name:
            assert hasattr(self, "vision_config")
            return self.vision_config
        return self.config

    def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module, consider_ep: bool = True
    ) -> dict[str, str]:
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

        # ep
        if self.mpu.ep_size > 1 and consider_ep:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            local_expert_to_global_expert = {
                i: i + num_experts_per_rank * self.mpu.ep_rank
                for i in range(num_experts_per_rank)
            }
            for k in ret.keys():
                v = ret[k]
                if ".mlp.experts.linear_fc" in v:
                    name_prefix, local_expert_id = v.split(".weight")
                    global_expert_idx = local_expert_to_global_expert[
                        int(local_expert_id)
                    ]
                    ret[k] = f"{name_prefix}.weight{global_expert_idx}"

        return ret

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
            # vision model
            tmp_config = self.hf_config.vision_config
            vision_hidden_size = tmp_config.hidden_size
            vision_num_query_groups = tmp_config.num_heads
            vision_head_dim = vision_hidden_size // tmp_config.num_heads
            if ".attn.qkv.weight" in hf_names[0]:
                mcore_weights = (
                    mcore_weights.view(
                        vision_num_query_groups,
                        3,
                        -1,
                        vision_head_dim,
                        vision_hidden_size,
                    )
                    .transpose(0, 1)
                    .reshape(-1, vision_hidden_size)
                    .contiguous()
                )

            if ".attn.qkv.bias" in hf_names[0]:
                mcore_weights = (
                    mcore_weights.view(
                        vision_num_query_groups,
                        3,
                        -1,
                    )
                    .transpose(0, 1)
                    .reshape(-1)
                    .contiguous()
                )

            # moe
            if ".mlp.experts.linear_fc" in mcore_weights_name:
                # get export index
                experts_key = hf_names[0]
                experts_idx = int(mcore_weights_name.split(".weight")[-1])

                if experts_key not in self.export_weights_buff:
                    self.export_weights_buff[experts_key] = {}
                assert experts_idx not in self.export_weights_buff
                self.export_weights_buff[experts_key][experts_idx] = mcore_weights.T

                if (
                    len(self.export_weights_buff[experts_key])
                    < self.config.num_moe_experts
                ):
                    return [], []

                mcore_weights_list = []
                for idx in range(self.config.num_moe_experts):
                    mcore_weights_list.append(
                        self.export_weights_buff[experts_key].pop(idx)
                    )
                self.export_weights_buff.pop(experts_key)
                return [hf_names[0]], [torch.stack(mcore_weights_list)]

            return [hf_names[0]], [mcore_weights]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            assert "vision_model" not in mcore_weights_name
            # split qkv
            assert len(hf_names) == 3
            # split qkv
            num_key_value_heads = self.hf_config.text_config.num_key_value_heads
            hidden_dim = self.hf_config.text_config.hidden_size
            num_attention_heads = self.hf_config.text_config.num_attention_heads

            head_dim = getattr(
                self.hf_config.text_config,
                "head_dim",
                hidden_dim // num_attention_heads,
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
            # vision model
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            tmp_config = self.hf_config.vision_config
            vision_hidden_size = tmp_config.hidden_size
            vision_num_query_groups = tmp_config.num_heads
            vision_head_dim = vision_hidden_size // tmp_config.num_heads
            if ".attn.qkv.weight" in hf_names[0]:
                return (
                    hf_weights[0]
                    .view(
                        3,
                        vision_num_query_groups,
                        -1,
                        vision_head_dim,
                        vision_hidden_size,
                    )
                    .transpose(0, 1)
                    .flatten(1, 2)
                    .reshape(-1, vision_hidden_size)
                    .contiguous()
                )

            if ".attn.qkv.bias" in hf_names[0]:
                return (
                    hf_weights[0]
                    .view(
                        3,
                        vision_num_query_groups,
                        -1,
                    )
                    .transpose(0, 1)
                    .flatten(1, 2)
                    .view(-1)
                    .contiguous()
                )
            # pad embeding and output layer
            if self.make_vocab_size_divisible_by is not None and (
                "embedding.word_embeddings.weight" in mcore_weights_name
                or "output_layer.weight" in mcore_weights_name
            ):
                assert hf_weights[0].shape[0] == self.vocab_size
                assert self.padded_vocab_size is not None

                embed_dim = hf_weights[0].shape[1]
                extra_zeros = torch.zeros(
                    (self.padded_vocab_size - self.vocab_size, embed_dim),
                    device=hf_weights[0].device,
                    dtype=hf_weights[0].dtype,
                )
                return torch.cat((hf_weights[0], extra_zeros), dim=0)

            # moe
            if ".mlp.experts.linear_fc" in mcore_weights_name:
                # get export index
                local_experts_idx = int(mcore_weights_name.split(".weight")[-1])
                num_experts = self.config.num_moe_experts
                num_experts_per_rank = num_experts // self.mpu.ep_size
                experts_idx = (
                    local_experts_idx + num_experts_per_rank * self.mpu.ep_rank
                )
                return hf_weights[0][experts_idx].T.clone().contiguous()

            return hf_weights[0]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.text_config.num_key_value_heads
            hidden_dim = self.hf_config.text_config.hidden_size
            num_attention_heads = self.hf_config.text_config.num_attention_heads
            if "vision_model" in mcore_weights_name:
                num_attention_heads = self.hf_config.text_config.vision_config.num_heads
                num_key_value_heads = self.hf_config.text_config.vision_config.num_heads
            head_dim = getattr(
                self.hf_config.text_config,
                "head_dim",
                hidden_dim // num_attention_heads,
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

        def provider(
            pre_process,
            post_process,
            add_decoder=True,
            add_encoder=True,
            vp_stage: Optional[int] = None,
        ):
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            vision_transformer_config = get_vision_model_config(
                deepcopy(self.config), self.hf_config.vision_config
            )
            vision_transformer_config.pipeline_model_parallel_size = 1
            vision_transformer_config.first_pipeline_num_layers = None

            vision_patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

            setattr(self, "vision_config", vision_transformer_config)

            model = Qwen3VLModel(
                language_transformer_config=self.config,
                language_transformer_layer_spec=transformer_layer_spec,
                language_vocab_size=self.hf_config.text_config.vocab_size,
                language_max_sequence_length=self.hf_config.text_config.max_position_embeddings,
                vision_transformer_config=vision_transformer_config,
                vision_transformer_layer_spec=vision_transformer_layer_spec,
                vision_patch_merger_spec=vision_patch_merger_spec,
                language_rotary_base=self.hf_config.text_config.rope_theta,
                pre_process=pre_process,
                post_process=post_process,
                add_decoder=add_decoder,
                add_encoder=add_encoder,
                parallel_output=True,
                language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
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
