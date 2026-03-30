# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Dict, Literal, Optional, Union

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    _get_block_submodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from .base import (
    MemEstimator,
    cum_mul,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    set_global_config,
)
from .layers import ColumnParallelLinear, LanguageModelEmbedding, TransformerBlock


class GPTModel(MemEstimator):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        config.use_cpu_initialization = True

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        # remove RotaryEmbedding

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
        )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

    def num_parameter(self):
        ret = 0
        if self.pre_process:
            ret += self.embedding.num_parameter()
        ret += self.decoder.num_parameter()
        if self.post_process:
            ret += self.output_layer.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        self._inited = True
        ret = 0

        self.num_act_pre = 0
        self.num_act_post = 0
        self.num_act_per_layer = 0
        self.num_act_between_layers = 0
        self.num_layers = self.decoder.layers.modules.__len__()

        if self.pre_process:
            self.num_act_pre = self.embedding.num_activation(input_shape)
            ret += self.num_act_pre
            input_shape = self.embedding.mock_forward(input_shape)
        ret += self.decoder.num_activation(input_shape)
        if self.decoder.layers.modules.__len__() > 0:
            self.num_act_per_layer = self.decoder.layers.modules[0].num_activation()
        input_shape = self.decoder.mock_forward(input_shape)
        self.num_act_between_layers = cum_mul(input_shape)

        if self.post_process:
            self.num_act_post = self.output_layer.num_activation(input_shape)
            softmax_activation = (
                self.output_layer.num_activation(input_shape) * 2
            )  # due to softmax is calculate in fp32
            self.num_act_post += softmax_activation
            ret += self.num_act_post
        return ret

    def mock_forward(self, input_shape: list[int]):
        if self.pre_process:
            input_shape = self.embedding.mock_forward(input_shape)
        input_shape = self.decoder.mock_forward(input_shape)
        if self.post_process:
            input_shape = self.output_layer.mock_forward(input_shape)
        return input_shape
