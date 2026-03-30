from typing import List, Optional

import torch
from einops import rearrange
from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params

from mbridge.core.util import split_data_cp_rank
from mbridge.models.gemma3.gpt_model import Gemma3GPTModel
from mbridge.models.gemma3.projector import Gemma3MultiModalProjector
from mbridge.utils.mappings import all_gather_to_context_parallel_region


# Note: This is under development and may be missing features.
class Gemma3Model(MegatronModule):
    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "learned_absolute",
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        img_h: int = 336,
        img_w: int = 336,
        patch_dim: int = 14,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
    ) -> None:
        super().__init__(config=language_transformer_config)
        assert (
            language_transformer_config.context_parallel_size == 1
        ), "not support context parallel now"

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.tensor_model_parallel_size_lm = (
            language_transformer_config.tensor_model_parallel_size
        )

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        if self.add_decoder:
            self.language_model = Gemma3GPTModel(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
                rope_scaling=language_rope_scaling,
                rope_scaling_factor=language_rope_scaling_factor,
                scatter_embedding_sequence_parallel=False,
            )
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )
            self._language_max_sequence_length = language_max_sequence_length
            self._language_is_pipeline_parallel = (
                language_transformer_config.pipeline_model_parallel_size > 1
            )

        if self.add_encoder:
            self.vision_model = CLIPViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                img_h=img_h,
                img_w=img_w,
                class_token_len=0,
                patch_dim=patch_dim,
                model_subtype="siglip",
                add_class_token=False,
            )

            vision_projection_input_size = vision_transformer_config.hidden_size
            # Map (intermediate) vision model outputs to the language model input dimension.
            self.vision_projection = Gemma3MultiModalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_input_size,
            )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _gather_image_embedings(self, image_embeddings):
        if self.context_parallel_lm < 2:
            return image_embeddings

        return all_gather_to_context_parallel_region(
            image_embeddings, 1, torch.distributed.ReduceOp.SUM
        )

    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        use_inference_kv_cache,
        image_token_index,
    ):
        assert (
            self.add_decoder
        ), "input text preprocessing is only needed for the language model"
        assert input_ids is not None

        if image_embeddings is None:
            return language_embeddings

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return None, None, None

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings

        special_image_mask = (input_ids == image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(language_embeddings)

        # batch x seq_len x hidden
        image_embeddings = self._gather_image_embedings(image_embeddings)
        language_embeddings = language_embeddings.masked_scatter(
            special_image_mask, image_embeddings
        )
        # seq_len x batch x hidden
        language_embeddings = rearrange(language_embeddings, "b s h -> s b h")
        return language_embeddings

    def _process_embedding_token_parallel(self, combined_embeddings):
        # combined_embeddings: seq-len x batch-size x hidden-dim
        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings

        shard_factor = seq_dim = None
        if self.pre_process:
            if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
                shard_factor = (
                    self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
                )
                seq_dim = 0
            elif self.context_parallel_lm > 1:
                shard_factor = self.context_parallel_lm * 2
                seq_dim = 0
            elif self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm
                seq_dim = 0

            assert (
                combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length should be divisible by {shard_factor} for \
                Sequence/Context parallelism"

        if self.context_parallel_lm > 1 and self.pre_process:
            combined_embeddings = split_data_cp_rank(
                combined_embeddings, self.context_parallel_lm, 0
            )

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        image_token_index: Optional[int] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input images of shape [num_tiles, img_h, img_w].
                num_tiles means the number of image tiles in this batch.
                num_tiles = 0 if the batch doesn't contain images.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Language model attention mask
                [batch, 1, 1, combined_seq_len]. NOTE: attention_mask is typically None and
                attn_mask_type in layer specs determines the attention mask used.
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_context (BaseInferenceContext): Inference-time parameters including KV cache.
            image_token_index (int): ID for input images. Default None means `image_token_index`
                arg in the constructor will be used.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            packed_seq_params (PackedSeqParams): 1) If using sequence packing, must contain
                subsample length information. 2) If using SP/CP with padding mask type,
                must contain padded token information.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
        """

        inference_context = deprecate_inference_params(
            inference_context, inference_params
        )

        use_inference_kv_cache = (
            inference_context is not None
            and "image_tokens_count" in inference_context.key_value_memory_dict
        )
        has_images = images is not None and images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor(
                [], dtype=images.dtype, device=images.device
            ).reshape(0, 0, 0)
        elif self.add_encoder and has_images:
            image_embeddings = self.vision_model(
                images
            )  # [num_tiles, img_seq_len, h_vision]

            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(
                1, 0, 2
            ).contiguous()  # [img_seq_len, num_tiles, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(
                image_embeddings
            )  # [img_seq_len, num_tiles, h_language]

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_context is not None:
                inference_context.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0] * image_embeddings.shape[1]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings

        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            input_ids_text[input_ids_text == image_token_index] = 0
            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]

            language_embeddings = language_embeddings.transpose(
                1, 0
            ).contiguous()  # [b, text_seq_len, h_language]

        combined_embeddings = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            use_inference_kv_cache,
            image_token_index,
        )  # [combined_seq_len, b, h_language]

        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            combined_embeddings = self._process_embedding_token_parallel(
                combined_embeddings
            )

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

        return output
