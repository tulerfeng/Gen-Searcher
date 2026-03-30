import os

import torch


def test_dequant_fp8_safetensor_io():
    return
    if "DEEPSEEK_V3_HF_DIR" not in os.environ:
        print("DEEPSEEK_V3_HF_DIR is not set")
        return
    from mbridge.models.ext.deepseek_v3.dequant_fp8_safetensor_io import (
        DequantFP8SafeTensorIO,
    )

    hf_dir = os.environ["DEEPSEEK_V3_HF_DIR"]
    io = DequantFP8SafeTensorIO(hf_dir)
    weight_names = [
        "model.layers.61.mlp.experts.197.up_proj.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight_scale_inv",
    ]
    weights = io.load_some_hf_weight(weight_names)
    for name, weight in weights.items():
        print(f"{name} {weight.dtype=} {weight.shape=} {weight}")
        assert weight.dtype == torch.bfloat16
