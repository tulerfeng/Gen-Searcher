#!/usr/bin/env python
"""
Example script to launch the DeepSeek-V3 model with Megatron across multiple machines using Ray.

Prerequisites: Ray, NCCL and CUDA are properly installed and configured on every machine.

Start the Ray cluster:
1. On the head node (node-0):
    ray start --head --num-gpus=<GPUs on head> --port=6379
2. On every additional node (node-1 … node-N):
    ray start --address=\"node-0-ip:6379\" --num-gpus=<GPUs on that node>

Run this script from any machine (typically the head):
    python 4.launch_deepseeekv3_with_ray.py \
        --model_path /path/to/hf_model 

The script creates world_size=num_nodes*gpus_per_node Ray workers, binds one GPU to each, sets the necessary distributed environment variables and launches Megatron with the DeepSeek-V3 model.
"""

import argparse
import os
import socket
from typing import Optional

import ray
import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model

# ---------- Distributed initialization ----------


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp: int,
    pp: int,
    cp: int,
    vpp: Optional[int],
    ep: int,
    etp: Optional[int],
):
    """Initialize torch.distributed according to Megatron's requirements.

    Reuses the helper logic from example/2.load_model_and_export_multiple_gpus.py.
    """
    # Common NCCL/FSDP environment variables
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Ray controls the visible GPU list via CUDA_VISIBLE_DEVICES.
    # Inside the process those GPUs are re-indexed to a contiguous range 0..N-1.
    # Each Ray actor gets exactly one GPU, therefore logical index 0 is safe to use.
    vis_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    local_rank = 0  # Logical index within the restricted visible range
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(backend="nccl")

    # Disable virtual pipeline parallelism when pipeline parallel size <= 1
    if pp <= 1:
        vpp = None

    # Megatron internal parallel initialization
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


# ---------- Ray worker ----------


@ray.remote(num_gpus=1)
def worker_fn(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    hf_model_path: str,
    tp: int,
    pp: int,
    cp: int,
    vpp: int,
    ep: int,
    etp: Optional[int],
    num_layers_in_first_pipeline_stage: Optional[int] = None,
    num_layers_in_last_pipeline_stage: Optional[int] = None,
):
    """Worker that runs on a single GPU.

    Loads the model and weights (optionally saves them) and demonstrates how to
    correctly configure Megatron's distributed environment inside a Ray actor.
    """
    # 1. Initialize distributed environment
    init_distributed(
        rank, world_size, master_addr, master_port, tp, pp, cp, vpp, ep, etp
    )

    # 2. Load model & weights
    bridge = AutoBridge.from_pretrained(hf_model_path)
    bridge.set_extra_args(
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
    )
    # bridge.config.mtp_num_layers = 0
    model = bridge.get_model(post_model_creation_callbacks=[], wrap_with_ddp=False)

    # maintain router bias dtype
    for m in model:
        from mbridge.core.util import unwrap_model

        m = unwrap_model(m)
        if hasattr(m, "decoder"):
            for l in m.decoder.layers:
                if (
                    hasattr(l, "mlp")
                    and hasattr(l.mlp, "router")
                    and hasattr(l.mlp.router, "_maintain_float32_expert_bias")
                ):
                    print(f"maintain router bias dtype for {l.mlp.router}")
                    l.mlp.router._maintain_float32_expert_bias()

    # bridge.load_weights(model, hf_model_path, memory_efficient=True)

    print(f"[rank {rank}] Model loaded, proceeding with post-processing ...")

    #########################################################
    ## if you want to save distributed_checkpoint, you need to save it here
    ## note: it is not verified
    #########################################################
    # save_distributed_checkpoint = False
    # if save_distributed_checkpoint:
    #     from megatron.training.checkpointing import save_checkpoint
    #     save_checkpoint(
    #         iteration=0,
    #         model=model,
    #         optimizer=None,
    #         opt_param_scheduler=None,
    #         num_floating_point_operations_so_far=0
    #     )

    # verify if the weights are loaded correctly
    for k, v in bridge.export_weights(model):
        if torch.distributed.get_rank() != 0:
            continue
        gt = bridge.safetensor_io.load_one_hf_weight(k).to(v.device)
        if k != "lm_head.weight":
            assert v.shape == gt.shape, f"mismatch of {k} {v.shape=} {gt.shape=}"
            assert v.sum().item() == gt.sum().item(), f"mismatch of {k} {v=} {gt=}"
        else:
            if v.shape[0] == 1:
                print(f"this is a value model, {k} {v.shape=} {gt.shape=}")
        if torch.distributed.get_rank() == 0:
            print(k, "export ok")

    # Synchronize all processes to ensure workflow completes
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return f"rank {rank} done"


# ---------- Main entry ----------


def main():
    parser = argparse.ArgumentParser(
        description="Using Ray to launch Megatron across machines"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=4,
        help="Number of physical nodes in the Ray cluster",
    )
    parser.add_argument(
        "--gpus_per_node", type=int, default=8, help="Number of GPUs per node"
    )

    # Megatron parallelism parameters
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=4, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=1, help="Virtual pipeline parallel size"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
    )
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=14,
        help="Number of layers in the first pipeline stage",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=15,
        help="Number of layers in the last pipeline stage",
    )

    parser.add_argument(
        "--master_port", type=int, default=12355, help="NCCL master port"
    )
    args = parser.parse_args()

    # Connect to the running Ray cluster
    ray.init()

    world_size = args.num_nodes * args.gpus_per_node

    master_addr = socket.gethostbyname(ray.util.get_node_ip_address())

    futures = []
    rank = 0
    for _node_idx in range(args.num_nodes):
        for _local_gpu in range(args.gpus_per_node):
            futures.append(
                worker_fn.remote(
                    rank,
                    world_size,
                    master_addr,
                    args.master_port,
                    args.model_path,
                    args.tp,
                    args.pp,
                    args.cp,
                    args.vpp,
                    args.ep,
                    args.etp,
                    args.num_layers_in_first_pipeline_stage,
                    args.num_layers_in_last_pipeline_stage,
                )
            )
            rank += 1

    for res in ray.get(futures):
        print(res)

    print("all done")


if __name__ == "__main__":
    main()
