# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 example/qwen3vl/load_model_and_forward.py --model_path /path/to/model

import argparse
from typing import Any

from tqdm import trange

try:
    from transformers import Qwen3VLProcessor
except:
    print(f"your install the tranformers>=4.57.0 or install from source")

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from example.qwen3vl.load_model_and_forward import (
    gather_output_from_cp,
    get_args,
    get_sample_for_forward,
    mcore_fwd_fn,
)
from mbridge import AutoBridge


def init_distributed(tp=2, pp=1, cp=1, vpp=1, ep=1, etp=None):
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % 8)
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


def broadcast_object_within_pp(obj: Any) -> Any:
    group = mpu.get_pipeline_model_parallel_group()

    if torch.distributed.get_world_size(group) > 1:
        obj_list = [obj]
        torch.distributed.broadcast_object_list(
            obj_list,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=group,
        )
        return obj_list[0]
    else:
        return obj


def main():
    args = get_args()
    print(f"{args=}")

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
    if args.pp > 1:
        num_layer = bridge.hf_config.text_config.num_hidden_layers
        first_last_layer = num_layer - (num_layer + args.pp - 1) // args.pp * (
            args.pp - 2
        )
        assert first_last_layer > 1
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=first_last_layer // 2,
            num_layers_in_last_pipeline_stage=(first_last_layer + 1) // 2,
        )
    bridge.config.sequence_parallel = True
    model = bridge.get_model(model_type=ModelType.encoder_and_decoder)
    assert len(model) == 1
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    eos_token_id = bridge.hf_config.text_config.eos_token_id
    sample = get_sample_for_forward(hf_model_path, args.sample_type)
    input_ids = sample["input_ids"].tolist()
    generated_tokens = []
    max_new_tokens = 1000
    torch.distributed.barrier()
    seq_length_factor = args.tp
    if args.cp > 1:
        seq_length_factor *= args.cp * 2
    with torch.no_grad():
        fwd_bwd_function = get_forward_backward_func()

        for i in trange(
            max_new_tokens, disable=(mpu.get_tensor_model_parallel_rank() == 0)
        ):
            real_seq_length = sample["input_ids"].shape[-1]
            seq_length = real_seq_length
            if real_seq_length % seq_length_factor != 0:
                seq_length = (
                    (real_seq_length + seq_length_factor - 1)
                    // seq_length_factor
                    * seq_length_factor
                )
                sample["input_ids"] = F.pad(
                    sample["input_ids"],
                    (0, seq_length - real_seq_length, 0, 0),
                    value=0,
                )

            mcore_output = fwd_bwd_function(
                forward_step_func=mcore_fwd_fn,
                data_iterator=iter([sample]),
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=seq_length,
                decoder_seq_length=seq_length,
                micro_batch_size=1,
            )

            next_token = -1
            if mpu.is_pipeline_last_stage():
                megatron_output = mcore_output[0]["logits"]
                if mpu.get_context_parallel_world_size() > 1:
                    megatron_output = gather_output_from_cp(
                        megatron_output,
                        1,
                        mpu.get_context_parallel_world_size(),
                        mpu.get_context_parallel_group(),
                    )
                if mpu.get_tensor_model_parallel_world_size() > 1:
                    megatron_output = gather_from_tensor_model_parallel_region(
                        megatron_output
                    )

                megatron_output = megatron_output[:, :real_seq_length, :]
                next_token = megatron_output[:, -1, :].argmax(dim=-1)[0].item()
                if (
                    torch.distributed.get_rank()
                    == torch.distributed.get_world_size() - 1
                ):
                    print(f"{i=} {next_token=}")

            next_token = broadcast_object_within_pp(next_token)
            generated_tokens.append(next_token)
            input_ids[0].append(next_token)
            sample["input_ids"] = torch.tensor(
                input_ids, device=torch.cuda.current_device()
            )
            if next_token == eos_token_id:
                break

    if torch.distributed.get_rank() == 0:
        print(f"{generated_tokens=}")
        processor = Qwen3VLProcessor.from_pretrained(hf_model_path)
        output_text = processor.batch_decode(
            [generated_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"{output_text=}")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
