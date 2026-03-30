# Example to use tp/pp/cp/vpp to test dense model
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
    bridge = AutoBridge.from_pretrained(
        hf_model_path, trust_remote_code=True, make_vocab_size_divisible_by=256
    )
    # set sequence_parallel = False for forward
    bridge.config.sequence_parallel = False
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    # check the export
    keys = bridge.safetensor_io.load_hf_weight_names()
    loaded_keys = set()
    # export weights
    for k, v in bridge.export_weights(model):
        gt = bridge.safetensor_io.load_one_hf_weight(k).cuda()
        assert v.shape == gt.shape, f"mismatch of {k}"
        assert torch.equal(v, gt), f"mismatch of {k}"
        loaded_keys.add(k)

    missing_keys = set(keys) - loaded_keys
    missing_keys = sorted(list(missing_keys))
    assert len(missing_keys) == 0
    print(f"missing keys: {missing_keys}")

    # load hf model
    hf_model = AutoModel.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).to("cuda")
    print(
        f"rank{torch.distributed.get_rank()} {hf_model.dtype}: end hf load weight, start forward ..."
    )
    sample = get_infer_data(hf_model, hf_model_path)

    with torch.no_grad():
        hf_model.img_context_token_id = sample["img_context_token_id"]
        hf_output = hf_model(
            sample["pixel_values"],
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            image_flags=sample["image_flags"],
            position_ids=sample["position_ids"],
        )

        megatron_output = model[0](
            images=sample["pixel_values"],
            input_ids=sample["input_ids"],
            position_ids=sample["position_ids"],
            attention_mask=None,
            image_token_index=sample["img_context_token_id"],
        )
        if mpu.get_tensor_model_parallel_world_size() > 1:
            megatron_output = gather_from_tensor_model_parallel_region(megatron_output)

        cos_similarity(hf_output.logits, megatron_output[:, :, : bridge.vocab_size])

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
