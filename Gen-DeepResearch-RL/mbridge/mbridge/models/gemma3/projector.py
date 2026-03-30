from dataclasses import dataclass, field
from typing import Dict, Union

import torch
from einops import rearrange
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TENorm,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import make_viewless_tensor
from torch import nn

from mbridge.models.gemma3.transformer_config import Gemma3TransformerConfig


@dataclass
class Gemma3MultiModalProjectorSubmodules:
    mm_soft_emb_norm: Union[ModuleSpec, type] = IdentityOp
    mm_input_projection: Union[ModuleSpec, type] = IdentityOp
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


def get_projector_module_spec():
    return Gemma3MultiModalProjectorSubmodules(
        mm_soft_emb_norm=TENorm,  # 这里有一些问题，因为TENorm用了te
        mm_input_projection=ColumnParallelLinear,
    )


def get_projector_module_spec_te():
    return Gemma3MultiModalProjectorSubmodules(
        mm_input_projection=TELayerNormColumnParallelLinear,
        # NOTE(guanyouhe): 没有测试过
        sharded_state_dict_keys_map={
            "mm_soft_emb_norm.": "mm_input_projection.layer_norm_",
        },
    )


class Gemma3MultiModalProjector(MegatronModule):

    def __init__(
        self,
        config: Gemma3TransformerConfig,
        submodules: Gemma3MultiModalProjectorSubmodules,
        input_size: int,
    ):
        super().__init__(config=config)
        assert submodules is not None, "MLPSubmodules must be provided"
        assert (
            config.layernorm_zero_centered_gamma
        ), "config.layernorm_zero_centered_gamma is True"

        self.mm_soft_emb_norm = build_module(
            submodules.mm_soft_emb_norm,
            config=self.config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )
        self.mm_input_projection = build_module(
            submodules.mm_input_projection,
            input_size,
            config.hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name=None,
        )

        patches_per_image = int(config.image_size // config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

        if self.config.sequence_parallel:
            assert patches_per_image % self.config.tensor_model_parallel_size == 0
            self.patches_per_image_h = (
                patches_per_image // self.config.tensor_model_parallel_size
            )
            self.patches_per_image_w = patches_per_image
        else:
            self.patches_per_image_h = patches_per_image
            self.patches_per_image_w = patches_per_image
        assert self.patches_per_image_h % self.config.context_parallel_size == 0
        self.patches_per_image_h = (
            self.patches_per_image_h // self.config.context_parallel_size
        )

    # hidden_states: seq_len x batch_size x hidden_size
    def forward(self, hidden_states: torch.Tensor):
        if self.config.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        _, batch_size, hidden_size = hidden_states.shape

        hidden_states = rearrange(hidden_states, "s b h -> b h s")
        hidden_states = hidden_states.reshape(
            batch_size, hidden_size, self.patches_per_image_h, self.patches_per_image_w
        )
        hidden_states = hidden_states.contiguous()

        pooled_vision_outputs = self.avg_pool(hidden_states)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = rearrange(pooled_vision_outputs, "b h s -> s b h")

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)
        output, output_bias = self.mm_input_projection(normed_vision_outputs)
        assert output_bias is None
        output = rearrange(output, "s b h -> b s h")
        output = gather_from_tensor_model_parallel_region(output)

        if output_bias is not None:
            output = output + output_bias
        # the encoder produces "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)
        return output
