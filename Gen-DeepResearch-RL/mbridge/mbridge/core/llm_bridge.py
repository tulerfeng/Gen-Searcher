# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import inspect
import math
from typing import Callable, Generator, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from torch.nn import functional as F

from .bridge import Bridge
from .util import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    unwrap_model,
)


class LLMBridge(Bridge):
    """
    Bridge implementation for Large Language Models.

    This class extends the base Bridge class to provide specific functionality
    for handling Large Language Models (LLMs) like GPT models.
    """

    TransformerConfigClass = TransformerConfig
    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": "attention_dropout",
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _build_base_config(self, text_config_key=None, **kwargs):
        """
        Build the base configuration for the model.

        Args:
            **kwargs: Additional configuration overrides

        Returns:
            TransformerConfig: The constructed transformer configuration
        """
        if text_config_key is None:
            hf_config = self.hf_config
        else:
            assert hasattr(self.hf_config, text_config_key)
            hf_config = getattr(self.hf_config, text_config_key)
        dtype = self.dtype
        overlap_p2p_comm = self.mpu.vpp_size is not None and self.mpu.pp_size > 1
        batch_p2p_comm = False

        base_config = {
            # Activation and normalization
            "activation_func": F.silu,
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
            # Data types
            "pipeline_dtype": dtype,
            "params_dtype": dtype,
            "bf16": dtype is torch.bfloat16,
            # Parallel configuration
            "tensor_model_parallel_size": self.mpu.tp_size,
            "pipeline_model_parallel_size": self.mpu.pp_size,
            "expert_model_parallel_size": self.mpu.ep_size,
            "expert_tensor_parallel_size": self.mpu.etp_size,
            "virtual_pipeline_model_parallel_size": self.mpu.vpp_size,
            "context_parallel_size": self.mpu.cp_size,
            "sequence_parallel": self.mpu.tp_size > 1,
            # Common settings
            "variable_seq_lengths": True,
            "masked_softmax_fusion": True,
            "moe_token_dispatcher_type": "alltoall",
            "add_bias_linear": False,
            "use_cpu_initialization": False,
            "overlap_p2p_comm": overlap_p2p_comm,
            "batch_p2p_comm": batch_p2p_comm,
        }
        # Model architecture parameters
        config_mapped = {}
        for mcore_key, hf_key in self._CONFIG_MAPPING.items():
            if isinstance(hf_key, tuple):
                hf_key, default_val = hf_key
                config_mapped[mcore_key] = getattr(hf_config, hf_key, default_val)
            else:
                config_mapped[mcore_key] = getattr(hf_config, hf_key)
        base_config.update(config_mapped)

        # Update with any provided overrides
        base_config.update(kwargs)
        base_config.update(self.extra_args)

        if "attention_backend" in base_config:
            from megatron.core.transformer.enums import AttnBackend

            if isinstance(base_config["attention_backend"], str):
                base_config["attention_backend"] = AttnBackend[
                    base_config["attention_backend"]
                ]

        return self.TransformerConfigClass(**base_config)

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.

        Constructs a dictionary of arguments required to initialize a GPTModel
        based on the configuration.

        Returns:
            dict: A dictionary of arguments for GPTModel initialization
        """
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )

    def _get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        """
        Gets the transformer layer specification.

        Creates and returns a specification for the transformer layers based on
        the current configuration.

        Returns:
            TransformerLayerSpec: Specification for transformer layers

        Raises:
            AssertionError: If normalization is not RMSNorm
        """
        assert (
            self.config.normalization == "RMSNorm"
        ), "only RMSNorm is supported for now"
        # check if get_gpt_decoder_block_spec has vp_stage parameter
        sig = inspect.signature(get_gpt_decoder_block_spec)
        self.has_vp_stage = "vp_stage" in sig.parameters  # for mcore 0.12 compatibility
        extra_args = {}
        if self.has_vp_stage:
            extra_args["vp_stage"] = vp_stage
        transformer_layer_spec = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True, **extra_args
        )
        return transformer_layer_spec

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
            # add pad vocab_size
            self.vocab_size = gptmodel_args["vocab_size"]
            self.padded_vocab_size = self.vocab_size
            if self.make_vocab_size_divisible_by is not None:
                self.padded_vocab_size = int(
                    math.ceil(self.vocab_size / self.make_vocab_size_divisible_by)
                    * self.make_vocab_size_divisible_by
                )
            gptmodel_args["vocab_size"] = self.padded_vocab_size

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
