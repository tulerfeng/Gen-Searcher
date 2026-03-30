# MoE LLM Memory Estimator

Accurate, Configurable, Modularized memory estimator for (not only) MoE LLMs

## Design

1. Reuse the megatron-lm training argument parser.

2. Simulate the model construction and forward/backward/optimizer.step procedures to calculate accurate memory consumption.

See [slides](./MoEMemoryEstimator.pdf) for more details.

## Dependencies
```
pip install sentencepiece tokenizers transformers ipdb termcolor
```

## Quick Start
### Use WebUI

```
bash run_webui.sh
```
or open https://huggingface.co/spaces/ISEEKYAN/megatron_memory_estimator

### User your own mcore scripts
Bascially you will need to replace the `pretrain_gpt.py` with `estimate.py`, everything should be the same.
```
bash example_mixtral_8x7b.sh
```
Just replace all the arguments after `estimate.py` with your own training args.


## Modularized Memory Estimator

To accurately estimate the memory consumption of the model parameters and activation, we design a mock of `torch.nn.Module` named [`MemEstimator`](./moe_mem_estimator/base.py).

### Module Completion for LLM

We have implemented all the necessary Modules required during the construction of a large language model (LLM). When using the estimator, the GPTModel can be constructed in the same way as during the training process.

### Custom Module Support

For other newly defined custom modules, only the corresponding MemEstimator needs to be implemented to enable memory consumption estimation. This design makes it highly flexible and applicable to a wide range of model architectures and customizations within the LLM domain.

run `bash run_estimator.sh` to see modularized estimated memory consumption of the above Mixtral-8x7B, note that existing version may estimate more memory that the real situation:

```
Number of parameters in every GPU in billions:  3.75
Number of activation in every GPU in billions:  11.56
num_bytes_per_parameter=6.75
Theoretical memory footprints: weight and optimizer=24151.92 MB, activation=22050.01 MB, total=46201.93 MB
```

## Some result comparison

I conducted some experiments on a single node with 8 GPUs machine.
The model config is a [Mixtral-8x2B with reduced layers](./model_configs/test_small.yaml).

| nlayers | TP | MBS | SEQ_LEN | Allocated/GB | Real | Real Model | Real act. | est.0 | est.0 model | est.0 act. | est.1 | est.1 model | est.1 act. |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 4 | 2 | 1 | 4096 | 7.61 | 6.6 | 5.3 | 1.3 | 6.63 | 5.24 | 1.39 | 6.1 | 5.4 | 0.7 |
| 4 | 2 | 2 | 4096 | 7.96 | 8 | 5.3 | 2.7 | 8.02 | 5.24 | 2.78 | 6.9 | 5.4 | 1.5 |
| 4 | 1 | 1 | 4096 | 13.41 | 11.2 | 8.8 | 2.4 | 11.18 | 8.74 | 2.44 | 10.4 | 8.9 | 1.5 |
| 4 | 1 | 2 | 4096 | 13.81 | 13.8 | 8.8 | 5 | 13.62 | 8.74 | 4.88 | 11.8 | 8.9 | 2.9 |

`est.0` is the modularized estimator and the `est.1` is the naive estimator, where `act.` means activation, and `MEM` is the real memory consumption. Some findings are:

1. `est.0 model` is larger the real `MEM`.
2. `est.0 act.` is far larger than the real situation, the probable reason is that the current version does not consider recomputation/checkpointing.

Some estimator results are like:
```
input_shape=[1, 4096]
Number of parameters in every GPU in billions:  1.25
Number of activation in every GPU in billions:  1.31
num_bytes_per_parameter=7.5
GPTModel        /* n_params=1,250,961,408       n_act=1,308,100,608 */ (
  (embedding): LanguageModelEmbedding   /* n_params=65,536,000  n_act=8,390,656 */ (
    (word_embeddings): VocabParallelEmbedding   /* n_params=65,536,000  n_act=2,048 */ ()
    (embedding_dropout): Dropout        /* n_params=0   n_act=8,388,608 */ ()
  )
  (decoder): TransformerBlock   /* n_params=1,119,889,408       n_act=1,168,637,952 */ (
    (layers): ModuleList        /* n_params=1,119,887,360       n_act=1,160,249,344 */ (
      (0-3): 4 x TransformerLayer       /* n_params=279,971,840 n_act=290,062,336 */ (
        (input_layernorm): IdentityOp   /* n_params=0   n_act=0 */ ()
        (self_attention): SelfAttention /* n_params=12,582,912  n_act=25,296,896 */ (
          (core_attention): TEDotProductAttention       /* n_params=0   n_act=131,072 */ ()
          (linear_qkv): ColumnParallelLinear    /* n_params=8,388,608   n_act=16,777,216 */ ()
          (q_layernorm): IdentityOp     /* n_params=0   n_act=0 */ ()
          (k_layernorm): IdentityOp     /* n_params=0   n_act=0 */ ()
          (linear_proj): RowParallelLinear      /* n_params=4,194,304   n_act=8,388,608 */ ()
        )
        (pre_cross_attn_layernorm): IdentityOp  /* n_params=0   n_act=0 */ ()
        (cross_attention): IdentityOp   /* n_params=0   n_act=0 */ ()
        (cross_attn_bda): IdentityOp    /* n_params=0   n_act=0 */ ()
        (pre_mlp_layernorm): RMSNorm    /* n_params=2,048       n_act=8,388,608 */ ()
        (mlp): MoELayer /* n_params=267,386,880 n_act=256,376,832 */ (
          (router): TopKRouter  /* n_params=0   n_act=16,777,216 */ ()
          (experts): SequentialMLP      /* n_params=267,386,880 n_act=111,411,200 */ (
            (local_experts): ModuleList /* n_params=267,386,880 n_act=0 */ (
              (0-7): 8 x MLP    /* n_params=33,423,360  n_act=13,926,400 */ (
                (linear_fc1): ColumnParallelLinear      /* n_params=22,282,240  n_act=5,570,560 */ ()
                (linear_fc2): RowParallelLinear /* n_params=11,141,120  n_act=2,785,280 */ ()
              )
            )
          )
        )
      )
    )
    (final_layernorm): RMSNorm  /* n_params=2,048       n_act=8,388,608 */ ()
  )
  (output_layer): ColumnParallelLinear  /* n_params=65,536,000  n_act=131,072,000 */ ()
)
Theoretical memory footprints: weight and optimizer=8947.57 MB, activation=2495.00 MB, total=11442.58 MB
Theoretical memory footprints: weight and optimizer=8.74 GB, activation=2.44 GB, total=11.17 GB
```


