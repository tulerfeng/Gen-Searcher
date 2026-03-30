from dataclasses import dataclass

from megatron.core.transformer import TransformerConfig


@dataclass
class GLM4VLTransformerConfig(TransformerConfig):
    spatial_merge_size: int = None
    image_token_id: int = None
    video_token_id: int = None
    video_start_token_id: int = None
    video_end_token_id: int = None
    image_start_token_id: int = None
    image_end_token_id: int = None
