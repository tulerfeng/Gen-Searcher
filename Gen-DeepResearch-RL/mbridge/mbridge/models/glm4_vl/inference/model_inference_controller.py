from collections import OrderedDict

import torch
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

from .inference_request import Glm4vInferenceRequest


class Glm4vGenerationController(TextGenerationController):

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[str, InferenceRequest],
    ):
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Currently only supports batch size 1 inference.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests
        """
        assert (
            len(active_requests) == 1
        ), f"Glm4v inference currently only supports batch size 1"

        request = list(active_requests.values())[0]

        assert isinstance(
            request, Glm4vInferenceRequest
        ), f"Found inference request of type {type(request)}, expected Glm4vInferenceRequest"

        return self.inference_wrapped_model.prep_inference_input(
            prompts_tokens,
            request.pixel_values,
            request.pixel_values_videos,
            request.image_grid_thw,
            request.video_grid_thw,
        )
