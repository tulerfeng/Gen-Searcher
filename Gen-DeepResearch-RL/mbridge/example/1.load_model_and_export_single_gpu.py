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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code"
    )
    args = parser.parse_args()

    # Initialize distributed environment
    init_distributed()

    # Load model
    hf_model_path = args.model_path
    bridge = AutoBridge.from_pretrained(
        hf_model_path, trust_remote_code=args.trust_remote_code
    )
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path)
    print(f"Model loaded: {args.model_path}")

    keys = bridge.safetensor_io.load_hf_weight_names()
    loaded_keys = set()
    # export weights
    for k, v in bridge.export_weights(model):
        gt = bridge.safetensor_io.load_one_hf_weight(k).cuda()
        assert v.shape == gt.shape, f"mismatch of {k}"
        assert torch.equal(v, gt), f"mismatch of {k}"
        loaded_keys.add(k)
        print(k, "export ok")

    missing_keys = set(keys) - loaded_keys
    missing_keys = sorted(list(missing_keys))
    print(f"missing keys: {missing_keys}")


if __name__ == "__main__":
    main()
