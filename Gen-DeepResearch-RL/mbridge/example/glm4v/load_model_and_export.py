# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 load_model_and_export.py --model_path /path/to/model


import argparse
import json
import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model


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


def compare_parameter_list(parameter_list, hf_model_path):
    # gather all parameters
    list_of_parameter_list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(list_of_parameter_list, parameter_list)
    full_parameter_list = set(e for p_list in list_of_parameter_list for e in p_list)
    # load list from index file
    index_map_file = os.path.join(hf_model_path, "model.safetensors.index.json")
    assert os.path.exists(index_map_file)
    with open(index_map_file) as f:
        file_mapping = json.load(f)
        hf_parameter_list = set(file_mapping["weight_map"].keys())

    # check equal
    diff1 = full_parameter_list - hf_parameter_list
    diff2 = hf_parameter_list - full_parameter_list

    assert not diff1, f"megatron_parameter_list - hf_parameter_list {diff1} "
    assert not diff2, f"hf_parameter_list - megatron_parameter_list {diff2} "


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

    # Load model
    hf_model_path = args.model_path
    print(f"rank{torch.distributed.get_rank()}: start loading model")
    bridge = AutoBridge.from_pretrained(hf_model_path)
    model = bridge.get_model(post_model_creation_callbacks=[freeze_moe_router])
    for k, v in model[0].named_parameters():
        print(f"{k} => {v.shape}")
    print(
        f"rank{torch.distributed.get_rank()}: start loading weights from {hf_model_path}"
    )
    bridge.load_weights(model, hf_model_path, memory_efficient=True)

    print("end load weight")
    # export weights and compare value
    parameter_list = []
    for k, v in bridge.export_weights(model):
        gt = bridge.safetensor_io.load_one_hf_weight(k).to(v.device)
        if k != "lm_head.weight":
            assert v.shape == gt.shape, f"mismatch of {k} {v.shape=} {gt.shape=}"
            v_sum = v.sum()
            gt_sum = gt.sum()
            if v_sum.item() != gt_sum.item():
                print(
                    f"mismatch of {k}, {v_sum} vs {gt_sum}, {v.device} vs {gt.device}, {v.dtype} vs {gt.dtype}"
                )
            # assert v_sum.item() == gt_sum.item(), f"mismatch of {k}, {v_sum} vs {gt_sum}"
        else:
            if v.shape[0] == 1:
                print(f"this is a value model, {k} {v.shape=} {gt.shape=}")
        if torch.distributed.get_rank() == 0:
            print(k, "export ok")
        parameter_list.append(k)

    # compare parameter list
    compare_parameter_list(parameter_list, hf_model_path)
    if args.save_path:
        bridge.save_weights(model, args.save_path, memory_efficient=False)

    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
