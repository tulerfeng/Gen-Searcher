from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core import __version__
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from packaging import version

try:
    from megatron.core.process_groups_config import ModelCommProcessGroups
except:
    ModelCommProcessGroups = None
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from mbridge.models.ext.llama3_cp.llama3_cp_memory_efficient_attention import (
    MemoryEfficientAttention,
)
from mbridge.models.gemma3.transformer_config import Gemma3TransformerConfig


@dataclass
class Gemma3TransformerLayerSubmodules(TransformerLayerSubmodules):
    post_attention_layernorm: Union[ModuleSpec, type] = IdentityOp
    post_feedforward_layernorm: Union[ModuleSpec, type] = IdentityOp


class Gemma3TransformerLayer(TransformerLayer):

    def __init__(
        self,
        config: Gemma3TransformerConfig,
        submodules: Gemma3TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
        vp_stage: Optional[int] = None,
    ):
        if model_comm_pgs is None and vp_stage is None:
            super(Gemma3TransformerLayer, self).__init__(
                config=config,
                submodules=submodules,
                layer_number=layer_number,
                hidden_dropout=hidden_dropout,
            )
        else:
            super(Gemma3TransformerLayer, self).__init__(
                config=config,
                submodules=submodules,
                layer_number=layer_number,
                hidden_dropout=hidden_dropout,
                model_comm_pgs=model_comm_pgs,
                vp_stage=vp_stage,
            )

        self.post_attention_layernorm = build_module(
            submodules.post_attention_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm = build_module(
            submodules.post_feedforward_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.is_sliding = bool(self.layer_number % config.sliding_window_pattern)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        # Residual connection.
        if isinstance(rotary_pos_emb, tuple) and isinstance(attention_mask, tuple):
            if self.is_sliding:
                # using local position embeddings
                rotary_pos_emb = rotary_pos_emb[1]
                attention_mask = attention_mask[1]
            else:
                # using global position embeddings
                rotary_pos_emb = rotary_pos_emb[0]
                attention_mask = attention_mask[0]
        residual = hidden_states

        extra_kwargs = {}
        if version.parse(__version__) >= version.parse("0.12.0"):
            extra_kwargs["inference_context"] = inference_context
        else:
            extra_kwargs["inference_params"] = inference_params

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        hidden_states, hidden_states_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **extra_kwargs,
        )

        if hidden_states_bias is not None:
            hidden_states = hidden_states + hidden_states_bias
        else:
            hidden_states = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            **extra_kwargs,
        )

        if (
            isinstance(attention_output_with_bias, dict)
            and "context" in attention_output_with_bias
        ):
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        hidden_states, hidden_states_bias = self.mlp(pre_mlp_layernorm_output)
        if hidden_states_bias is not None:
            hidden_states = hidden_states + hidden_states_bias
        else:
            hidden_states = hidden_states
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output
        return output, context


def get_norm_mlp_module_spec_te() -> ModuleSpec:
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )


def get_gemma3_layer_spec_te(is_vit=False) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal

    mlp = get_norm_mlp_module_spec_te()
    return ModuleSpec(
        module=Gemma3TransformerLayer,
        submodules=Gemma3TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=MemoryEfficientAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            post_attention_layernorm=TENorm,
            post_feedforward_layernorm=TENorm,
        ),
    )


def get_layer_spec_te(is_vit=False) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal

    mlp = get_norm_mlp_module_spec_te()
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
