# torchrun --nproc_per_node=8 example/internvl3/load_model_and_forward.py --model_path /path/to/model

import argparse

import torch
from data_proc import get_infer_data
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoModel

from mbridge import AutoBridge


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
    bridge = AutoBridge.from_pretrained(
        hf_model_path, trust_remote_code=True, make_vocab_size_divisible_by=256
    )
    # set sequence_parallel = False for forward
    bridge.config.sequence_parallel = False
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    hf_model = AutoModel.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).to(torch.cuda.current_device())
    sample = get_infer_data(hf_model, hf_model_path)

    # megatron-lm generate
    input_ids = sample["input_ids"].tolist()
    generated_tokens = []
    max_new_tokens = 1000
    from tqdm import trange

    for _ in trange(
        max_new_tokens, disable=(mpu.get_tensor_model_parallel_rank() == 0)
    ):
        with torch.no_grad():
            megatron_output = model[0](
                images=sample["pixel_values"],
                input_ids=torch.LongTensor(input_ids).to(torch.cuda.current_device()),
                position_ids=sample["position_ids"],
                attention_mask=None,
                image_token_index=sample["img_context_token_id"],
            )
            if mpu.get_tensor_model_parallel_world_size() > 1:
                megatron_output = gather_from_tensor_model_parallel_region(
                    megatron_output
                )
        # Get the next token
        next_token = megatron_output[:, -1, :].argmax(dim=-1)[0].item()
        generated_tokens.append(next_token)
        input_ids[0].append(next_token)
        if next_token == sample["eos_token_id"]:
            break

    mlm_text = sample["tokenizer"].decode(generated_tokens, skip_special_tokens=True)

    # hf generate
    hf_model.img_context_token_id = sample["img_context_token_id"]
    hf_output = hf_model.generate(
        sample["pixel_values"],
        input_ids=sample["input_ids"],
        attention_mask=sample["attention_mask"],
        **sample["generation_config"],
    )
    hf_text = sample["tokenizer"].decode(hf_output[0], skip_special_tokens=True)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print("*" * 10 + " megatron-lm generate text" + "*" * 10)
        print(f"{mlm_text}")
        print(f"\n ------ vs ------ \n")
        print("*" * 10 + " transformers generate text" + "*" * 10)
        print(f"{hf_text}")
        print("*" * 100)

    torch.distributed.barrier()
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
