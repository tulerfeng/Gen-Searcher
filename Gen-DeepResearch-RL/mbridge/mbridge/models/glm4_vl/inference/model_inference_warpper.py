from typing import Any, Dict

import torch
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)

from mbridge.core.util import unwrap_model


class Glm4vInferenceWrapper(GPTInferenceWrapper):

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ):
        assert (
            not self.inference_params.is_decode_only()
        ), "`prep_inference_input` should only be called in prefill mode"

        # build mrope position ids
        attention_mask, position_ids = self._build_attention_mask_and_position_ids(
            prompts_tokens, image_grid_thw, video_grid_thw
        )
        return {
            "tokens": prompts_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
        }

    def _build_attention_mask_and_position_ids(
        self, prompts_tokens, image_grid_thw, video_grid_thw
    ):
        unwrapped = unwrap_model(self.model)
        position_ids, _ = unwrapped.get_rope_index(
            input_ids=prompts_tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        return None, position_ids

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:

        batch = {}
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        tokens2use = tokens[:, context_start_position:context_end_position]
        # flash_decode is false, full position_ids is required
        positions2use = position_ids  # position_ids[..., context_start_position:context_end_position]
        if attention_mask is not None:
            attention_mask2use = attention_mask[
                ..., context_start_position:context_end_position, :context_end_position
            ]
        else:
            attention_mask2use = None

        batch["tokens"] = tokens2use
        batch["position_ids"] = positions2use
        batch["attention_mask"] = attention_mask
        batch["pixel_values"] = inference_input["pixel_values"]
        batch["pixel_values_videos"] = inference_input["pixel_values_videos"]
        batch["image_grid_thw"] = inference_input["image_grid_thw"]
        batch["video_grid_thw"] = inference_input["video_grid_thw"]
        return batch

    def _forward(self, inference_input: Dict[str, Any]):
        """Runs a forward pass of the model.

        Args:
            inference_input(Dict[str, Any]): The input data.

        Returns:
            The model output logits.
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        pixel_values = inference_input["pixel_values"]
        pixel_values_videos = inference_input["pixel_values_videos"]
        image_grid_thw = inference_input["image_grid_thw"]
        video_grid_thw = inference_input["video_grid_thw"]

        output = self.model(
            tokens,
            position_ids=position_ids,
            attention_mask=None,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inference_context=self.inference_context,
            runtime_gather_output=False,
        )
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        return logits
