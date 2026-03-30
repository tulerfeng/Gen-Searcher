import argparse

import torch

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except:
    print(f"your install the tranformers>=4.57.0 or install from source")

from example.qwen3vl.load_model_and_forward import get_sample_for_forward

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="image",
        choices=["image", "video", "mix"],
        help="sample type",
    )
    args = parser.parse_args()

    # default: Load the model on the available device(s)
    torch.set_grad_enabled(False)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Preparation for inference
    inputs = get_sample_for_forward(args.model_path, args.sample_type)

    # Inference: Generation of the output
    hf_output = model.forward(**inputs)
    # print(hf_output)
    torch.save(hf_output.logits, "/tmp/hf_qwen3vl.pt")
