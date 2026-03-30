# GLM4V Adaptation Testing
This work is not only to test whether the adaptation of GLM4V is correct, but also to accumulate some experience and tools, so that the model adaptation process from mbridge hf to megatron can follow this test in the future.
## GLM4V Test Environment Setup
### Megatron 1.2.0
### transformers 
* Main branch (commit id: 4f9b4e62bc52a52b19a6a4a1a6bfc61a3f5b65b1), installed from source code
## Testing
### 1.Load Export Test
* A1(hf) - load -> B(megatron) - export -> A2(hf), check A1 vs A2
### 2.Inspect Megatron Inference Generation Results
* Using [Megatron's native code](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.12.0/examples/inference) for inference, check if the generated results is gibberish
### 3.Compare Output Logits (hf vs megatron)
*For the same input, the logit distribution of the same token from hf and megatron should be quite similar, and we use cosine similarity as the similarity metric.
