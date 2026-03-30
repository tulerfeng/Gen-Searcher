# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Pretrain GPT."""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
import inspect
import os
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    MockGPTDataset,
)
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from moe_mem_estimator.base import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_global_config,
    set_pipeline_model_parallel_rank,
)
from moe_mem_estimator.gpt_model import GPTModel
from moe_mem_estimator.layers import MLASelfAttention, MoELayer

torch.distributed.get_rank = lambda: 0
torch.cuda.get_device_capability = lambda: [8]


def estimate_from_config(config, args):
    """
    Estimate memory usage from a given config and args, instead of global state.
    Now supports virtual pipeline model parallelism for more accurate results.
    """

    args.moe_grouped_gemm = True
    patch_parallel_states()
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    input_shape = [args.micro_batch_size, args.seq_length]

    set_global_config(config)
    print(config)
    # return
    cli_reports = []

    if config.pipeline_model_parallel_size > 1:
        for pp_rank in range(config.pipeline_model_parallel_size):
            set_pipeline_model_parallel_rank(pp_rank)
            print(
                f"\n------------------------------[Pipeline_Parallelism_Rank={pp_rank}]------------------------------"
            )
            input_shape, rpt = report_memory_usage_one_pp_rank(
                input_shape, args, config, pp_rank, config.pipeline_model_parallel_size
            )
            cli_reports.append(rpt)
    else:
        set_pipeline_model_parallel_rank(0)
        _, rpt = report_memory_usage_one_pp_rank(input_shape, args, config)
        cli_reports.append(rpt)

    aggregated_reports: list[dict] = cli_reports

    # 返回 (聚合后的 pp 报告列表, 全量 raw chunk 列表)
    return aggregated_reports, cli_reports


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.

    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.fp8,
        )
    else:
        return get_gpt_layer_local_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
        )


def model_provider(
    args, config, pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> GPTModel:
    use_te = True
    if args.num_experts:
        # Define the decoder block spec
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=use_te,
            normalization="LayerNorm",
            qk_l2_norm=False,
            vp_stage=vp_stage,
        )
    else:
        # Define the decoder layer spec
        transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
    mtp_block_spec = None
    # TODO fp8
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=getattr(config, "fp16_lm_cross_entropy", False),
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=getattr(args, "rotary_percent", 1.0),
        rotary_base=getattr(args, "rotary_base", 10000),
        rope_scaling=getattr(config, "use_rope_scaling", False),
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )

    return model


def get_model(
    model_provider_func, args, config, model_type=ModelType.encoder_or_decoder
):
    """Build the model."""
    # args = get_args()
    # args.model_type = model_type

    # Build model.
    if not getattr(args, "virtual_pipeline_model_parallel_size", None):
        args.virtual_pipeline_model_parallel_size = None
    if config.pipeline_model_parallel_layout:
        args.virtual_pipeline_model_parallel_size = (
            config.pipeline_model_parallel_layout.virtual_pipeline_model_parallel_size
        )
        config.virtual_pipeline_model_parallel_size = (
            config.pipeline_model_parallel_layout.virtual_pipeline_model_parallel_size
        )

    def build_model():
        if (
            get_pipeline_model_parallel_world_size() > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            if model_type == ModelType.encoder_and_decoder:
                assert (
                    config.encoder_pipeline_model_parallel_size == 0
                ), "Interleaved schedule not supported for model with encoder on separate PP rank"
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
                post_process = is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)

                this_model = model_provider_func(
                    args,
                    config,
                    pre_process=pre_process,
                    post_process=post_process,
                    vp_stage=i,
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = is_pipeline_first_stage()
            post_process = is_pipeline_last_stage()
            if model_type == ModelType.encoder_and_decoder:
                if get_pipeline_model_parallel_world_size() > 1:
                    rank = get_pipeline_model_parallel_rank()
                    first_decoder_rank = config.encoder_pipeline_model_parallel_size
                    world_size = get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == first_decoder_rank
                    post_process = (rank == (first_decoder_rank - 1)) or (
                        rank == (world_size - 1)
                    )
                model = model_provider_func(
                    args,
                    config,
                    pre_process=pre_process,
                    post_process=post_process,
                )
            else:
                model = model_provider_func(
                    args, config, pre_process=pre_process, post_process=post_process
                )
            model.model_type = model_type
        return model

    model = build_model()

    if not isinstance(model, list):
        model = [model]
    return model


NUM_BYTES_IN_MEGABYTE = 1024 * 1024
NUM_BYTES_IN_GIGABYTE = 1024 * 1024 * 1024


def patch_parallel_states():
    from megatron.core import parallel_state

    parallel_state.is_pipeline_first_stage = is_pipeline_first_stage
    parallel_state.is_pipeline_last_stage = is_pipeline_last_stage
    parallel_state.get_pipeline_model_parallel_rank = get_pipeline_model_parallel_rank
    parallel_state.get_pipeline_model_parallel_world_size = (
        get_pipeline_model_parallel_world_size
    )
    parallel_state.get_virtual_pipeline_model_parallel_world_size = (
        get_virtual_pipeline_model_parallel_world_size
    )
    parallel_state.is_inside_encoder = lambda: False
    parallel_state.get_pipeline_model_parallel_decoder_start = lambda: 0


def report_memory_usage(args, config=None):
    args.moe_grouped_gemm = True
    patch_parallel_states()
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    input_shape = [args.micro_batch_size, args.seq_length]

    set_global_config(config)

    cli_reports = []

    if config.pipeline_model_parallel_size > 1:
        for pp_rank in range(config.pipeline_model_parallel_size):
            set_pipeline_model_parallel_rank(pp_rank)
            print(
                f"\n------------------------------[Pipeline_Parallelism_Rank={pp_rank}]------------------------------"
            )
            input_shape, rpt = report_memory_usage_one_pp_rank(
                input_shape, args, config, pp_rank, config.pipeline_model_parallel_size
            )
            cli_reports.append(rpt)
    else:
        set_pipeline_model_parallel_rank(0)
        _, rpt = report_memory_usage_one_pp_rank(input_shape, args, config)
        cli_reports.append(rpt)

    # Optionally pretty print summary
    print("\n===== Summary (per PP rank) =====")
    for r in cli_reports:
        print(
            f"PP{r['pp_rank']}  total {r['total_gb']} GB  (weight_grad {r['weight_grad_gb']} GB weight_grad_optim {r['weight_grad_optim_gb']} GB  act {r['activation_gb']} GB)"
        )


def report_memory_usage_one_pp_rank(
    input_shape: list[int], args, config, pp_rank=0, pp_size=1
) -> tuple[list[int], dict]:
    print(f"{input_shape=}")
    model: list[GPTModel] = get_model(model_provider, args, config)
    num_parameter_this_shard_all = 0
    num_parameter_this_shard_sparse_all = 0
    num_activation_all = 0
    output_shape = input_shape
    for vpp_rank, one_chunk in enumerate(model):
        num_parameter_this_shard = one_chunk.num_parameter()
        num_activation = one_chunk.num_activation(output_shape)
        output_shape = one_chunk.mock_forward(output_shape)
        print(f"{output_shape=}")
        num_parameter_this_shard_sparse = 0
        for layer in one_chunk.decoder.layers.modules:
            if isinstance(layer.mlp, MoELayer):
                num_parameter_this_shard_sparse += layer.mlp.num_parameter()
                if (
                    "shared_experts" in layer.mlp.__dir__()
                    and layer.mlp.shared_experts is not None
                ):
                    num_parameter_this_shard_sparse -= (
                        layer.mlp.shared_experts.num_parameter()
                    )
        num_activation_this_shard_mlp = sum(
            [m.mlp.num_activation() for m in one_chunk.decoder.layers.modules]
        )
        if len(model) > 1:
            if vpp_rank >= 1 and vpp_rank < len(model) - 1:
                num_microbatch_this_pp_rank = pp_size
            elif vpp_rank == 0:
                num_microbatch_this_pp_rank = pp_size + max(
                    (pp_size - pp_rank) * 2 - 1 - pp_size, 0
                )
            elif vpp_rank == len(model) - 1:
                num_microbatch_this_pp_rank = min((pp_size - pp_rank) * 2 + 1, pp_size)
        else:
            num_microbatch_this_pp_rank = pp_size - pp_rank

        num_parameter_this_shard_sparse = 0
        for layer in one_chunk.decoder.layers.modules:
            if isinstance(layer.mlp, MoELayer):
                num_parameter_this_shard_sparse += layer.mlp.num_parameter()
                if (
                    "shared_experts" in layer.mlp.__dir__()
                    and layer.mlp.shared_experts is not None
                ):
                    num_parameter_this_shard_sparse -= (
                        layer.mlp.shared_experts.num_parameter()
                    )

        one_chunk.__repr__()
        print(one_chunk)
        print(
            f"Number of parameters in every GPU in billions: "
            f"{num_parameter_this_shard / 10**9: .2f} where mlp part is {num_parameter_this_shard_sparse / 10**9: .2f}"
        )
        num_parameter_this_shard_all += num_parameter_this_shard
        num_parameter_this_shard_sparse_all += num_parameter_this_shard_sparse
        # recompute
        if config.recompute_granularity == "full":
            recompute_num_layers = config.recompute_num_layers
            num_layers = one_chunk.num_layers
            common_act = (
                one_chunk.num_act_pre
                + one_chunk.num_act_between_layers
                * num_layers
                * num_microbatch_this_pp_rank
            )  # recompute with pipeline parallel
            info = "With this recomputing setting, the number of activation achieve peak when "
            if config.recompute_method == "block":
                num_layers_with_loss = num_layers - recompute_num_layers
                if num_layers_with_loss == 0:
                    peak1 = common_act + one_chunk.num_act_post
                    peak2 = common_act + one_chunk.num_act_per_layer
                    if peak1 > peak2:
                        info += "calculating loss"
                    else:
                        info += "back-propogating loss"
                    num_activation = max(peak1, peak2)
                else:
                    info += f"calculating loss with {num_layers_with_loss} non-recompute layers"
                    num_activation = (
                        common_act
                        + one_chunk.num_act_post
                        + one_chunk.num_act_per_layer
                        * num_layers_with_loss
                        * num_microbatch_this_pp_rank
                    )
            elif config.recompute_method == "uniform":
                peak1 = common_act + one_chunk.num_act_post
                peak2 = (
                    (common_act + one_chunk.num_act_per_layer)
                    if vpp_rank == 0
                    else (common_act)
                )
                if peak1 > peak2:
                    info += "calculating loss"
                else:
                    info += f"back-propogating loss recomputing every {recompute_num_layers} layers"
                num_activation = max(peak1, peak2)
            if len(one_chunk.decoder.layers.modules) > 0 and isinstance(
                one_chunk.decoder.layers.modules[0].self_attention, MLASelfAttention
            ):  # MLA recompute achieve peak at backward
                num_activation += one_chunk.decoder.layers.modules[
                    0
                ].self_attention.core_attention.num_activation()
            print(info)

        else:
            num_activation = (
                num_activation - one_chunk.num_act_post
            ) * num_microbatch_this_pp_rank + one_chunk.num_act_post

        # CP
        num_activation = num_activation / config.context_parallel_size
        if pp_size == 1:
            print(
                f"Number of activation in every GPU in billions: "
                f"{num_activation / 10**9: .2f} where mlp part is {num_activation_this_shard_mlp / 10**9: .2f}"
            )
        else:
            print(
                f"Number of activation per microbatch in every GPU in billions: "
                f"{num_activation / 10**9: .2f} where mlp part is {num_activation_this_shard_mlp / 10**9: .2f}"
                f", {num_microbatch_this_pp_rank=} {vpp_rank=}"
            )
        num_activation_all += num_activation
    num_bytes_per_parameter = (
        18
        if not args.use_distributed_optimizer
        else 6 + (12 / args.data_parallel_size / config.context_parallel_size)
    )
    if config.expert_model_parallel_size * config.expert_tensor_parallel_size > 1:
        num_bytes_per_parameter_dense = num_bytes_per_parameter
        num_bytes_per_parameter_moe = (
            18
            if not args.use_distributed_optimizer
            else 6
            + (
                12
                / (
                    args.world_size
                    / config.pipeline_model_parallel_size
                    / config.expert_model_parallel_size
                    / config.expert_tensor_parallel_size
                )
            )
        )
        print(f"{num_bytes_per_parameter_dense=} {num_bytes_per_parameter_moe=}")
        weight_grad_memory = num_parameter_this_shard_all * 6 / NUM_BYTES_IN_GIGABYTE
        weight_grad_optim_memory = (
            (num_parameter_this_shard_all - num_parameter_this_shard_sparse_all)
            * num_bytes_per_parameter_dense
            + num_parameter_this_shard_sparse_all * num_bytes_per_parameter_moe
        ) / NUM_BYTES_IN_GIGABYTE
    else:
        print(f"{num_bytes_per_parameter=}")
        weight_grad_memory = num_parameter_this_shard_all * 6 / NUM_BYTES_IN_GIGABYTE
        weight_grad_optim_memory = (
            num_parameter_this_shard_all
            * num_bytes_per_parameter
            / NUM_BYTES_IN_GIGABYTE
        )

    activation_memory = (
        num_activation_all * 2 / NUM_BYTES_IN_GIGABYTE
    )  # only support fp16
    total_memory = weight_grad_optim_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_grad_optim_memory:.2f} GB, "
        f"activation={activation_memory:.2f} GB, total={total_memory:.2f} GB\n"
    )

    # 生成与 estimate_from_config 相同格式的聚合报告
    model_breakdown_concat = "\n\n".join(
        [f"--- vpp_chunk {i} ---\n{str(m)}" for i, m in enumerate(model)]
    )

    report = {
        "pp_rank": pp_rank,
        "parameters_b": num_parameter_this_shard_all / 1e9,
        "activation_b": num_activation_all / 1e9,
        "weight_grad_gb": round(weight_grad_memory, 2),
        "weight_grad_optim_gb": round(weight_grad_optim_memory, 2),
        "activation_gb": round(activation_memory, 2),
        "total_gb": round(total_memory, 2),
        "model_breakdown": model_breakdown_concat,
        "details": None,
    }

    return output_shape, report


if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    import ipdb

    with ipdb.launch_ipdb_on_exception():
        args = get_args()
        report_memory_usage(args)
