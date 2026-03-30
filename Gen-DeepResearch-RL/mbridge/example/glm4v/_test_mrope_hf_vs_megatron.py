# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 load_model_and_export.py --model_path /path/to/model


import argparse
import json
import os
from typing import List

import requests
import torch
from megatron.core import parallel_state
from megatron.core import parallel_state as mpu
from megatron.core.models.common.embeddings import MultimodalRotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from PIL import Image
from transformers import AutoProcessor
from transformers.models.glm4v.modeling_glm4v import (
    Glm4vTextRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)

from mbridge import AutoBridge
from mbridge.models.glm4_vl.vl_mixin import VLMixin

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]


def get_sample_for_forward(image, hf_model_path):
    processor = AutoProcessor.from_pretrained(hf_model_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image])
    ret = {}
    ret["prompt_tokens"] = torch.LongTensor(inputs.input_ids).to(
        torch.cuda.current_device()
    )
    ret["pixel_values"] = torch.from_numpy(inputs.pixel_values).to(
        torch.cuda.current_device()
    )
    ret["image_grid_thw"] = torch.from_numpy(inputs.image_grid_thw).to(
        torch.cuda.current_device()
    )
    return ret


def apply_rope_hf(q, k, bridge, position_ids):
    rotary_emb = Glm4vTextRotaryEmbedding(bridge.hf_config).cuda()
    position_embeddings = rotary_emb(q, position_ids)
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(  # diff with Llama
        q, k, cos, sin, bridge.hf_config.rope_scaling["mrope_section"]
    )
    return query_states, key_states


def apply_rope_megatron(q, k, bridge, position_ids):
    rotary_emb = MultimodalRotaryEmbedding(
        kv_channels=bridge.config.kv_channels,
        rotary_interleaved=bridge.config.rotary_interleaved,
        rotary_base=bridge.hf_config.rope_theta,
        rotary_percent=bridge.hf_config.partial_rotary_factor,
    ).cuda()
    rotary_pos_emb = rotary_emb(position_ids, bridge.config.mrope_section)
    q_pos_emb, k_pos_emb = rotary_pos_emb, rotary_pos_emb
    q = apply_rotary_pos_emb(q, q_pos_emb, config=bridge.config, cu_seqlens=None)
    k = apply_rotary_pos_emb(k, k_pos_emb, config=bridge.config, cu_seqlens=None)
    return q, k


def init_q_k(bridge, token_ids):
    head_num = bridge.config.num_attention_heads
    dim = bridge.config.hidden_size // head_num
    bsz = token_ids.shape[0]
    q_len = token_ids.shape[1]
    q = torch.randn(q_len, bsz, head_num, dim, device="cuda")
    k = torch.randn(q_len, bsz, head_num, dim, device="cuda")
    return q, k


def test_diff(a, b, a_text="", b_text=""):
    diff = a - b
    print(f"{a_text} {a}")
    print(f"{b_text} {b}")
    print(
        f"{a_text} vs {b_text}: diff max {diff.max()}; min {diff.min()}; mean {diff.abs().mean()}"
    )


def init_distributed(tp=2, pp=1, cp=1, vpp=1, ep=1, etp=None):
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
    if pp <= 1:
        vpp = None
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor model parallel size")
    parser.add_argument(
        "--pp", type=int, default=1, help="Pipeline model parallel size"
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=1, help="Virtual pipeline model parallel size"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert model parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save weights"
    )
    args = parser.parse_args()

    # Initialize distributed environment
    init_distributed(
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        vpp=args.vpp,
        ep=args.ep,
        etp=args.etp,
    )

    # Load megatron model
    hf_model_path = args.model_path
    image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    sampls = get_sample_for_forward(image, hf_model_path)
    bridge = AutoBridge.from_pretrained(hf_model_path)
    vl_mixin = VLMixin()
    vl_mixin.config = bridge.config
    position_ids, _ = vl_mixin.get_rope_index(
        sampls["prompt_tokens"], sampls["image_grid_thw"]
    )
    # set sequence_parallel = False for forward
    bridge.config.sequence_parallel = False
    # q_len, bsz, -1, dim
    q, k = init_q_k(bridge, sampls["prompt_tokens"])
    # bsz, -1, q_len, dim
    q_emb_hf, k_emb_hf = apply_rope_hf(
        q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), bridge, position_ids
    )
    q_emb_hf = q_emb_hf.permute(2, 0, 1, 3)
    k_emb_hf = k_emb_hf.permute(2, 0, 1, 3)
    # q_len, bsz, -1, dim
    q_emb_mg, k_emb_mg = apply_rope_megatron(q, k, bridge, position_ids)
    test_diff(q_emb_hf, q_emb_mg, "q_emb_hf", "q_emb_mg")
    test_diff(k_emb_hf, k_emb_mg, "k_emb_hf", "k_emb_mg")
    torch.distributed.barrier()
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
