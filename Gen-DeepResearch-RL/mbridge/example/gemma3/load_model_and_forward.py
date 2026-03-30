# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 example/gemma3/load_model_and_forward.py --model_path /path/to/model

import argparse
import os

import requests
import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from mbridge import AutoBridge

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]


def gen_attn_mask(token_type_ids, sliding_window, attn_head_size_per_tp):
    seq_length = len(token_type_ids)
    mask_pair = []
    last_val = 0
    for i in range(len(token_type_ids)):
        if token_type_ids[i] != last_val:
            if token_type_ids[i] == 1:
                mask_pair.append([i, -1])
            else:
                mask_pair[-1][-1] = i
            last_val = token_type_ids[i]

    token_type_mask = torch.zeros(
        (seq_length, seq_length), dtype=torch.long, device="cpu"
    )
    for pair in mask_pair:
        token_type_mask[pair[0] : pair[1], pair[0] : pair[1]] = 1

    attn_mask_src = torch.tril(
        torch.ones(
            (seq_length, seq_length),
            dtype=torch.long,
            device="cpu",
        )
    )
    attn_mask = (attn_mask_src | token_type_mask).to(torch.bool)

    slice_mask = torch.tril(
        torch.ones(
            (seq_length, seq_length),
            dtype=torch.long,
            device="cpu",
        ),
        diagonal=-sliding_window,
    )

    sliding_window_attention_mask = ((attn_mask_src - slice_mask) | token_type_mask).to(
        torch.bool
    )
    attn_mask = (
        torch.zeros(attn_mask.shape, dtype=torch.bfloat16)
        .masked_fill_(attn_mask.logical_not(), float("-inf"))
        .unsqueeze(0)
    )
    attn_mask = attn_mask.expand(attn_head_size_per_tp, *attn_mask.shape[1:])
    sliding_window_attention_mask = (
        torch.zeros(sliding_window_attention_mask.shape, dtype=torch.bfloat16)
        .masked_fill_(sliding_window_attention_mask.logical_not(), float("-inf"))
        .unsqueeze(0)
    )
    sliding_window_attention_mask = sliding_window_attention_mask.expand(
        attn_head_size_per_tp,
        *sliding_window_attention_mask.shape[1:],
    )
    # cp by split
    # if cp_size > 1:
    #     attn_mask = split_data_cp_rank(attn_mask, cp_size, 1, cp_rank)
    #     sliding_window_attention_mask = split_data_cp_rank(
    #         sliding_window_attention_mask, cp_size, 1, cp_rank
    #     )

    return attn_mask.unsqueeze(0), sliding_window_attention_mask.unsqueeze(0)


def get_sample_for_forward(image, hf_model_path, hf_config, tp):
    processor = AutoProcessor.from_pretrained(hf_model_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image])

    # padding for xformers
    pad_token_id = processor.tokenizer.pad_token_id
    prompt_len = len(inputs.input_ids[0])
    pad_multi_of = 8
    pad_len = (
        prompt_len + pad_multi_of - 1
    ) // pad_multi_of * pad_multi_of - prompt_len
    inputs.input_ids[0] += [pad_token_id] * pad_len
    inputs.token_type_ids[0] += [0] * pad_len
    inputs.attention_mask[0] += [1] * pad_len

    ret = {}
    ret["input_ids"] = torch.LongTensor(inputs.input_ids).to(
        torch.cuda.current_device()
    )
    ret["attention_mask"] = torch.LongTensor(inputs.attention_mask).to(
        torch.cuda.current_device()
    )
    ret["token_type_ids"] = torch.LongTensor(inputs.token_type_ids).to(
        torch.cuda.current_device()
    )
    ret["pixel_values"] = (
        torch.from_numpy(inputs.pixel_values[0])
        .unsqueeze(0)
        .to(torch.cuda.current_device())
    )

    position_ids = torch.arange(
        ret["input_ids"].size()[1], dtype=torch.long, device=torch.cuda.current_device()
    )
    ret["position_ids"] = position_ids.unsqueeze(0).expand_as(ret["input_ids"])

    attn_mask, sliding_window_attention_mask = gen_attn_mask(
        inputs.token_type_ids[0],  # only one image
        hf_config.text_config.sliding_window,
        hf_config.text_config.num_attention_heads // tp,
    )
    ret["attn_mask"] = attn_mask.to(torch.cuda.current_device())
    ret["sliding_window_attention_mask"] = sliding_window_attention_mask.to(
        torch.cuda.current_device()
    )

    return ret


# hf logits vs megatron logits
def cos_similarity(a, b):
    print(f"a {a.shape} b {b.shape}")
    a = a.float()
    # a = a / a.norm(dim=-1, keepdim=True)
    a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
    a = a / a.norm(dim=-1, keepdim=True)
    """
    a = (a - a.mean(dim=-1, keepdim=True)) 
    a = a / a.norm(dim=-1, keepdim=True)
    """
    b = b.float()
    # b =  b / b.norm(dim=-1, keepdim=True)
    b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
    b = b / b.norm(dim=-1, keepdim=True)
    """
    b = (b - b.mean(dim=-1, keepdim=True)) 
    b =  b / b.norm(dim=-1, keepdim=True)
    """
    sim = (a * b).sum(dim=-1)
    print(
        f"hf vs megatron cos_similarity min: {sim.min()}; max: {sim.max()}; mean: {sim.mean()}"
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
    # if sample["input_ids"].shape[-1] % 2 == 0,
    # bridge.config.sequence_parallel can be Trueu
    bridge.config.sequence_parallel = False
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    # load hf model
    hf_model = (
        AutoModelForImageTextToText.from_pretrained(hf_model_path).to("cuda").bfloat16()
    )

    print(
        f"rank{torch.distributed.get_rank()} {hf_model.dtype}: end hf load weight, start forward ..."
    )

    try:
        image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(image_url, stream=True, timeout=5).raw)
    except:
        if os.path.exists("../australia.jpg"):
            image = Image.open("../australia.jpg")
        else:
            print(
                "Your machine needs to be able to download images"
                + "or download the images to your machine first"
            )
            raise FileNotFoundError
    sample = get_sample_for_forward(image, hf_model_path, bridge.hf_config, args.tp)

    with torch.no_grad():
        hf_output = hf_model(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            token_type_ids=sample["token_type_ids"],
            pixel_values=sample["pixel_values"],
            position_ids=sample["position_ids"],
        )

        image_token_id = bridge.hf_config.image_token_index
        megatron_output = model[0](
            images=sample["pixel_values"].to(torch.bfloat16),
            input_ids=sample["input_ids"],
            position_ids=sample["position_ids"],
            attention_mask=(
                sample["attn_mask"],
                sample["sliding_window_attention_mask"],
            ),
            image_token_index=image_token_id,
        )
        if mpu.get_tensor_model_parallel_world_size() > 1:
            megatron_output = gather_from_tensor_model_parallel_region(megatron_output)
        cos_similarity(hf_output.logits, megatron_output)

    torch.distributed.barrier()
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
