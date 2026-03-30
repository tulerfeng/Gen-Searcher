from typing import Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
)

try:
    from megatron.core.transformer.custom_layers.transformer_engine import TENorm

    NORM_IMPL = TENorm
except:
    NORM_IMPL = torch.nn.LayerNorm

from mbridge.models.internvl3.transformer_config import InternvlTransformerConfig


class Internvl2bVitTransformerBlock(TransformerBlock):

    def __init__(
        self,
        config: InternvlTransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        pre_process: bool = True,
        post_process: bool = True,
    ):
        self.dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_layers)
        ]
        super().__init__(
            config=config, spec=spec, pre_process=pre_process, post_process=post_process
        )

    def _build_layers(self):

        def build_layer(layer_spec, layer_number, drop_path_rate):
            return build_module(
                layer_spec,
                config=self.config,
                layer_number=layer_number,
                drop_path_rate=drop_path_rate,
            )

        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1, self.dpr[i])
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        if self.post_process:
            self.final_layernorm = NORM_IMPL(
                self.config.hidden_size, eps=self.config.layernorm_epsilon
            )
        else:
            self.final_layernorm = None


class InternvlVitModel(VisionModule):

    def __init__(
        self,
        config: InternvlTransformerConfig,
        layer_spec: ModuleSpec,
        patch_dim: int = 14,
        img_h: int = 448,
        img_w: int = 448,
        model_subtype: str = "internvit",
    ) -> None:

        error_msg = f"InternvlViTModel model subtype {model_subtype} is not supported."
        assert model_subtype in ["internvit", "internvit300M"], error_msg

        super().__init__(config=config)
        self.downsample_ratio = 0.5  # TODO(guanyouhe): 写入到 config 内

        self.visual_hidden_size = config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w

        assert self.img_h % self.patch_dim == 0 and self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.num_positions = self.num_patches + 1
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
        )

        self.position_embedding = torch.nn.Parameter(
            torch.randn(1, self.num_positions, self.visual_hidden_size)
        )
        self.class_token = torch.nn.Parameter(
            torch.zeros(1, 1, self.visual_hidden_size)
        )
        self.decoder = Internvl2bVitTransformerBlock(
            config=config, spec=layer_spec, pre_process=True, post_process=False
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """
        Set the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.img_h // self.patch_dim, self.img_w // self.patch_dim, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(2, 1, 0, 3).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        x = self.conv1(x)  # shape: [batch_size, out_channel, h, w]
        _, _, height, width = x.shape

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # x.shape: [batch_size, h * w, out_channel],
        # h * w as seqlen, out_channel as hidden_size

        class_token = self.class_token.expand(x.shape[0], 1, -1)
        x = torch.cat(
            [class_token, x], dim=1
        )  # x.shape: [batch_size, h * w + 1, out_channel],

        assert x.shape[1] == self.num_positions, f"{x.shape[1]} != {self.num_positions}"
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        x = x + position_embedding
        x = x.permute(
            1, 0, 2
        ).contiguous()  # x.shape: [h * w + 1, batch_size, out_channel],

        x = self.decoder(hidden_states=x, attention_mask=None)

        x = x[1:, :, :]  # x.shape: [h * w, batch_size, out_channel],
        x = x.permute(
            1, 0, 2
        ).contiguous()  # x.shape: [batch_size, h * w, out_channel],

        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, -1)
        x = self.pixel_shuffle(x, scale_factor=self.downsample_ratio)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

        return x
