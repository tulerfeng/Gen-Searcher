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
from transformers import AutoProcessor

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


def get_sample_for_forward(input_ids, token_type_ids, pad_token_id, hf_config, tp):
    prompt_len = len(input_ids)
    pad_multi_of = 8
    pad_len = (
        prompt_len + pad_multi_of - 1
    ) // pad_multi_of * pad_multi_of - prompt_len
    input_ids = input_ids + [pad_token_id] * pad_len
    token_type_ids = token_type_ids + [0] * pad_len

    ret = {}
    ret["input_ids"] = torch.LongTensor([input_ids]).to(torch.cuda.current_device())
    ret["token_type_ids"] = torch.LongTensor([token_type_ids]).to(
        torch.cuda.current_device()
    )

    position_ids = torch.arange(
        ret["input_ids"].size()[1], dtype=torch.long, device=torch.cuda.current_device()
    )
    ret["position_ids"] = position_ids.unsqueeze(0).expand_as(ret["input_ids"])

    attn_mask, sliding_window_attention_mask = gen_attn_mask(
        token_type_ids,
        hf_config.text_config.sliding_window,
        hf_config.text_config.num_attention_heads // tp,
    )
    ret["attn_mask"] = attn_mask.to(torch.cuda.current_device())
    ret["sliding_window_attention_mask"] = sliding_window_attention_mask.to(
        torch.cuda.current_device()
    )

    return ret


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
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    # generate
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
    processor = AutoProcessor.from_pretrained(hf_model_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image])
    input_ids = inputs.input_ids[0]
    token_type_ids = inputs.token_type_ids[0]
    pixel_values = (
        torch.from_numpy(inputs.pixel_values[0])
        .unsqueeze(0)
        .to(torch.cuda.current_device())
        .to(torch.bfloat16)
    )
    pad_token_id = processor.tokenizer.pad_token_id
    image_token_id = bridge.hf_config.image_token_index
    eos_token_id = processor.tokenizer.encode("<end_of_turn>")[-1]

    generated_tokens = []
    max_new_tokens = 1000
    from tqdm import trange

    for _ in trange(max_new_tokens):
        sample = get_sample_for_forward(
            input_ids, token_type_ids, pad_token_id, bridge.hf_config, args.tp
        )
        with torch.no_grad():
            megatron_output = model[0](
                images=pixel_values,
                input_ids=sample["input_ids"],
                position_ids=sample["position_ids"],
                attention_mask=(
                    sample["attn_mask"],
                    sample["sliding_window_attention_mask"],
                ),
                image_token_index=image_token_id,
            )
            if mpu.get_tensor_model_parallel_world_size() > 1:
                megatron_output = gather_from_tensor_model_parallel_region(
                    megatron_output
                )
        # Get the next token
        next_token = megatron_output[:, len(input_ids) - 1, :].argmax(dim=-1)[0].item()
        generated_tokens.append(next_token)
        input_ids.append(next_token)
        token_type_ids.append(0)
        if next_token == eos_token_id:
            break

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        generated_text = processor.decode(generated_tokens, skip_special_tokens=False)
        print(f"Generated text:\n{generated_text}")
    torch.distributed.barrier()
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
