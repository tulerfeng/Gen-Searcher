# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 2.load_model_and_export_multiple_gpus.py --model_path /path/to/model


import argparse
import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model


def init_distributed(tp=1, pp=1, cp=1, vpp=1, ep=1, etp=None):
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


def generate_sequence(
    prompt, model, hf_model_path, max_new_tokens=100, trust_remote_code=False
):
    try:
        assert mpu.get_tensor_model_parallel_world_size() == 1
        assert mpu.get_pipeline_model_parallel_world_size() == 1
        assert mpu.get_context_parallel_world_size() == 1
    except Exception as e:
        print(e)
        print("only EP is supported in example generate, skip")
        return
    """Generate text sequence"""
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path, trust_remote_code=trust_remote_code
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.cuda()
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
        0
    )
    attention_mask = torch.ones_like(input_ids).to(input_ids.device)

    generated_tokens = []
    cur_input_ids = input_ids
    cur_position_ids = position_ids
    cur_attention_mask = attention_mask
    from tqdm import trange

    for _ in trange(max_new_tokens):
        # Move inputs to GPU
        cur_input_ids = cur_input_ids.cuda()
        cur_position_ids = cur_position_ids.cuda()
        cur_attention_mask = cur_attention_mask.cuda()

        # Forward inference with the model
        with torch.no_grad():
            model[0].cuda()
            output = model[0].module(
                cur_input_ids, cur_position_ids, cur_attention_mask
            )

        # Get the next token
        next_token = output.argmax(dim=-1)[:, -1]
        generated_tokens.append(next_token.item())

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Update input sequence
        cur_input_ids = torch.cat([cur_input_ids, next_token.unsqueeze(0)], dim=1)
        cur_position_ids = torch.arange(
            cur_input_ids.shape[1], device=cur_input_ids.device
        ).unsqueeze(0)
        cur_attention_mask = torch.ones_like(cur_input_ids)

    # Decode the generated token sequence
    generated_text = tokenizer.decode(generated_tokens)
    if torch.distributed.get_rank() == 0:
        print(f"Generated text:\n{generated_text}")

    return generated_text


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor model parallel size")
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
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code"
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

    # Load model
    hf_model_path = args.model_path
    print(f"rank{torch.distributed.get_rank()}: start loading model")
    bridge = AutoBridge.from_pretrained(hf_model_path)
    model = bridge.get_model(post_model_creation_callbacks=[])
    print(
        f"rank{torch.distributed.get_rank()}: start loading weights from {hf_model_path}"
    )
    bridge.load_weights(model, hf_model_path)

    prompt = "A bubble sort in python is "
    generate_sequence(
        prompt, model, args.model_path, args.max_tokens, args.trust_remote_code
    )
    # return
    # export weights
    keys = bridge.safetensor_io.load_hf_weight_names()
    loaded_keys = set()
    not_matched_keys = set()
    for k, v in bridge.export_weights(model):
        if torch.distributed.get_rank() != 0:
            continue
        gt = bridge.safetensor_io.load_one_hf_weight(k).to(v.device)
        if k != "lm_head.weight":
            assert v.shape == gt.shape, f"mismatch of {k} {v.shape=} {gt.shape=}"
            if not torch.allclose(v.sum(), gt.sum(), atol=1e-5):
                not_matched_keys.add(k)
        else:
            if v.shape[0] == 1:
                print(f"this is a value model, {k} {v.shape=} {gt.shape=}")
        loaded_keys.add(k)
        print(k, "export ok")
    if args.save_path:
        bridge.save_weights(model, args.save_path, memory_efficient=False)

    missing_keys = set(keys) - loaded_keys
    missing_keys = sorted(list(missing_keys))
    if torch.distributed.get_rank() == 0:
        print(f"missing keys: {missing_keys}")
        print(f"not_matched_keys: {not_matched_keys}")

    # wait for save finish
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
