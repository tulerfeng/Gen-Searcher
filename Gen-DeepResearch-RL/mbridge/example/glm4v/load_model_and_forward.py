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
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router

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


# hf logits vs megatron logits
def cos_similarity(a, b):
    print(f"a {a.shape} b {b.shape}")
    a = a.float()
    # a = a / a.norm(dim=-1, keepdim=True)
    a = torch.exp(a)
    a = a / a.norm(dim=-1, keepdim=True)
    """
    a = (a - a.mean(dim=-1, keepdim=True)) 
    a = a / a.norm(dim=-1, keepdim=True)
    """
    b = b.float()
    # b =  b / b.norm(dim=-1, keepdim=True)
    b = torch.exp(b)
    b = b / b.norm(dim=-1, keepdim=True)
    """
    b = (b - b.mean(dim=-1, keepdim=True)) 
    b =  b / b.norm(dim=-1, keepdim=True)
    """
    sim = (a * b).sum(dim=-1)
    print(
        f"hf vs megatron cos_similarity min: {sim.min()}; max: {sim.max()}; mean: {sim.mean()}"
    )


def is_first_rank():
    """First tensor and pipeline parallel rank."""
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        and parallel_state.get_tensor_model_parallel_rank() == 0
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
    print(f"rank{torch.distributed.get_rank()}: start loading model ...")
    bridge = AutoBridge.from_pretrained(hf_model_path)
    # set sequence_parallel = False for forward
    bridge.config.sequence_parallel = False
    model = bridge.get_model(post_model_creation_callbacks=[freeze_moe_router])
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    # load hf model
    hf_model = (
        AutoModelForImageTextToText.from_pretrained(hf_model_path).to("cuda").bfloat16()
    )

    print(
        f"rank{torch.distributed.get_rank()} {hf_model.dtype}: end hf load weight, start forward ..."
    )

    sample = get_sample_for_forward(
        Image.open("./example/glm4v/australia.jpg"), hf_model_path
    )

    with torch.no_grad():
        hf_output = hf_model(
            input_ids=sample["prompt_tokens"],
            pixel_values=sample["pixel_values"],
            image_grid_thw=sample["image_grid_thw"],
        )

        megatron_output = model[0](
            input_ids=sample["prompt_tokens"],
            pixel_values=sample["pixel_values"],
            image_grid_thw=sample["image_grid_thw"],
        )
        cos_similarity(hf_output.logits, megatron_output)

    torch.distributed.barrier()
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
