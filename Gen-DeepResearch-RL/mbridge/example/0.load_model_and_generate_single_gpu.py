import argparse
import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer

from mbridge import AutoBridge


def init_distributed():
    """Initialize distributed environment"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)


def load_model(hf_model_path, trust_remote_code=False):
    """Load model"""
    bridge = AutoBridge.from_pretrained(
        hf_model_path, trust_remote_code=trust_remote_code
    )
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path)
    return model


def generate_sequence(
    prompt, model, hf_model_path, max_new_tokens=100, trust_remote_code=False
):
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
    print(f"Generated text:\n{generated_text}")
    return generated_text


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
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
    init_distributed()

    # Load model
    model = load_model(args.model_path, args.trust_remote_code)
    print(f"Model loaded: {args.model_path}")

    # Generate text
    prompt = "A quick sort in python for me is \n```python\n"
    generate_sequence(
        prompt, model, args.model_path, args.max_tokens, args.trust_remote_code
    )


if __name__ == "__main__":
    main()
