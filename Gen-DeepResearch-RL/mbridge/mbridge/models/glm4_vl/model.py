import logging
from collections import namedtuple
from typing import List

import torch
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PreTrainedModel

from .transformer_config import GLM4VLTransformerConfig
from .vl_mixin import VLMixin


class Glm4VLModel(MegatronModule, VLMixin):
    """Glm4 VL multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
            language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for
            positional embedding.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline
            parallelism). Defaults to True.
    """

    def __init__(
        self,
        config: GLM4VLTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: str,
        hf_config: "hf_config",
        hf_vision_cls: type,
        parallel_output: bool = True,
        rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        rotary_base: int = 10000,
        fp16_lm_cross_entropy: bool = False,
        share_embeddings_and_output_weights: bool = False,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vision_model = None

        if self.pre_process:
            self.vision_model = hf_vision_cls._from_config(hf_config.vision_config)
        else:
            self.vision_model = None

        self.language_model = GPTModel(
            config=config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=rotary_base,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )

    @property
    def share_embeddings_and_output_weights(self):
        return self.language_model.share_embeddings_and_output_weights

    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        if freeze_language_model and self.language_model is not None:
            for param in self.language_model.parameters():
                param.requires_grad = False

        if (
            freeze_vision_model or freeze_vision_projection
        ) and self.vision_model is not None:
            for name, param in self.vision_model.named_parameters():
                # vision projection
                if any(e in name for e in ["downsample", "merger"]):
                    if freeze_vision_projection:
                        param.requires_grad = False
                else:
                    if freeze_vision_model:
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        runtime_gather_output=False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the Qwen2VL model.

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len,
                combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

            video_start_index:
                0 -- all video
                len(video_seq) -- all image
                others -- mixture
            *_input_mask: should not be None in the first PP stage
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """

        # TODO(liuzhenhai93@outlook.com): support context parallel

        use_inference_image_cache = (
            inference_context and inference_context.sequence_len_offset > 0
        )
        combined_embeddings = None
        if self.pre_process:
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            )  # [text_seq_len, b, h_language]
            if not use_inference_image_cache:
                combined_embeddings = self.merge_image_or_video(
                    combined_embeddings,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )

            if self.config.sequence_parallel:
                # naive sequence_parallel support
                combined_embeddings = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        combined_embeddings
                    )
                )

        if position_ids is None:
            # if position_ids is not prepared in dataloader
            position_ids, _ = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            # inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            **(extra_block_kwargs or {}),
            **kwargs,
        )

        return output
