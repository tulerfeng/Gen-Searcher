from dataclasses import dataclass

import torch
from megatron.core.inference.inference_request import InferenceRequest


@dataclass(kw_only=True)
class Glm4vInferenceRequest(InferenceRequest):
    """Class for a Glm4v inference request"""

    pixel_values: torch.Tensor = None
    pixel_values_videos: torch.Tensor = None
    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None
