# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 load_model_and_export.py --model_path /path/to/model


import argparse
import json
import os
from typing import List

import requests
import torch
from megatron.core import parallel_state
from megatron.core import parallel_state as mpu
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from PIL import Image
from transformers import AutoProcessor

from mbridge import AutoBridge
from mbridge.models.glm4_vl.inference import Glm4vInferenceRequest, get_inference_engine
from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model


def is_first_rank():
    """First tensor and pipeline parallel rank."""
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


class GenerateHelper:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    def __init__(self, model_path, model, vocab_size, sampling_params=None):
        self.processor = AutoProcessor.from_pretrained(model_path)
        # tokenizer no eos token id
        self.processor.tokenizer.eod = self.processor.tokenizer.pad_token_id
        # hack detokenize => _decode
        self.processor.tokenizer.detokenize = self.processor.tokenizer._decode
        self.engine = get_inference_engine(model, self.processor.tokenizer, vocab_size)
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.2, top_k=100, top_p=0.2, num_tokens_to_generate=512
            )
        self.sampling_params = sampling_params

    def generate(self, image):
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image])
        # assert False, f"pixel_values {type(inputs.pixel_values)} {type(inputs.image_grid_thw)} {inputs.pixel_values.shape} {inputs.image_grid_thw.shape}"
        inference_request = Glm4vInferenceRequest(
            request_id=self.engine.get_new_request_id(),
            prompt=text,
            prompt_tokens=inputs.input_ids[0],
            sampling_params=self.sampling_params,
            pixel_values=torch.from_numpy(inputs.pixel_values).to(
                torch.cuda.current_device()
            ),
            image_grid_thw=torch.from_numpy(inputs.image_grid_thw).to(
                torch.cuda.current_device()
            ),
        )
        results: List[InferenceRequest] = self.engine.generate(
            inference_requests=[inference_request]
        )

        return results[0].generated_text


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

    # Load model
    hf_model_path = args.model_path
    print(f"rank{torch.distributed.get_rank()}: start loading model ...")
    bridge = AutoBridge.from_pretrained(hf_model_path)
    # set sequence_parallel = False for generate
    bridge.config.sequence_parallel = False
    model = bridge.get_model(post_model_creation_callbacks=[freeze_moe_router])
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start generate ...")

    image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    generator = GenerateHelper(hf_model_path, model[0], bridge.hf_config.vocab_size)
    result = generator.generate(image)
    if is_first_rank():
        print(f"generated {result}")
    torch.distributed.barrier()
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
