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
from typing import Union

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
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
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
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_global_config,
    set_pipeline_model_parallel_rank,
)
from moe_mem_estimator.gpt_model import GPTModel
from moe_mem_estimator.layers import MLASelfAttention, MoELayer


def _calculate_rank_memory(config, args, input_shape, pp_rank=0, pp_size=1):
    """
    Calculates the memory for a single pipeline parallel rank, containing the detailed logic.
    """
    # Build the model for the current rank
    set_global_config(config)
    pre_process = pp_rank == 0
    post_process = pp_rank == pp_size - 1

    use_te = True
    if hasattr(config, "spec") and config.spec is not None:
        transformer_layer_spec = import_module(config.spec)
    else:
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                config.num_moe_experts,
                config.moe_grouped_gemm,
                config.qk_layernorm,
                config.multi_latent_attention,
                config.fp8,
            )
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                config.num_moe_experts,
                config.moe_grouped_gemm,
                config.qk_layernorm,
                config.multi_latent_attention,
            )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=getattr(config, "fp16_lm_cross_entropy", False),
        parallel_output=True,
        share_embeddings_and_output_weights=args.tie_word_embeddings,
        position_embedding_type="rope",
        rotary_percent=getattr(args, "rotary_percent", 1.0),
        rotary_base=getattr(args, "rotary_base", 10000),
        rope_scaling=getattr(config, "use_rope_scaling", False),
    )

    # --- Start of detailed memory calculation logic ---
    num_parameter_this_shard = model.num_parameter()
    num_activation = model.num_activation(input_shape)
    output_shape = model.mock_forward(input_shape)

    num_parameter_this_shard_sparse = sum(
        layer.mlp.num_parameter()
        for layer in model.decoder.layers.modules
        if isinstance(layer.mlp, MoELayer)
    )
    num_activation_this_shard_mlp = sum(
        m.mlp.num_activation() for m in model.decoder.layers.modules
    )

    num_microbatch_this_pp_rank = pp_size - pp_rank
    if config.num_layers_per_virtual_pipeline_stage is not None:
        layers_this_pprank = len(model.decoder.layers.modules)
        vpp_size = layers_this_pprank // config.num_layers_per_virtual_pipeline_stage
        if vpp_size > 0:
            num_microbatch_this_pp_rank = (
                pp_size * (vpp_size - 1) + (pp_size - pp_rank) * 2 - 1
            ) / vpp_size

    # Activation Recomputation
    # The base activation number is for one microbatch. With pipeline parallelism,
    # the total activation is multiplied by the number of microbatches in flight.
    # Recomputation reduces this by re-calculating activations during the backward pass
    # instead of storing them.

    # This is the activation memory without any recomputation.
    num_activation = (
        num_activation - model.num_act_post
    ) * num_microbatch_this_pp_rank + model.num_act_post

    if config.recompute_granularity == "full":
        # This logic is transplanted from the more detailed `report_memory_usage_one_pp_rank`
        recompute_num_layers = config.recompute_num_layers
        num_layers = model.num_layers
        # Activations of a model with recompute enabled.
        # The activation of a layer is an input to the next layer.
        # So, the total activation is the sum of the activations of all layers,
        # plus the activation of the embedding layer.
        # The activation of a layer is stored only if it is not recomputed.
        common_act = (
            model.num_act_pre
            + model.num_act_between_layers * num_layers * num_microbatch_this_pp_rank
        )
        if config.recompute_method == "block":
            num_layers_with_loss = num_layers - recompute_num_layers
            if num_layers_with_loss == 0:
                peak1 = common_act + model.num_act_post
                peak2 = common_act + model.num_act_per_layer
                recomputed_activation = max(peak1, peak2)
            else:
                recomputed_activation = (
                    common_act
                    + model.num_act_post
                    + model.num_act_per_layer
                    * num_layers_with_loss
                    * num_microbatch_this_pp_rank
                )
        elif config.recompute_method == "uniform":
            peak1 = common_act + model.num_act_post
            peak2 = (
                common_act
                + model.num_act_per_layer
                * recompute_num_layers
                * num_microbatch_this_pp_rank
            )
            recomputed_activation = max(peak1, peak2)

        if isinstance(model.decoder.layers.modules[0].self_attention, MLASelfAttention):
            recomputed_activation += model.decoder.layers.modules[
                0
            ].self_attention.core_attention.num_activation()

        num_activation = recomputed_activation

    elif config.recompute_granularity == "selective":
        # Selective recomputation is the default in Megatron-LM and is handled
        # by Transformer Engine. The base `num_activation` calculation from `GPTModel`
        # already reflects this. We just need to scale it by the number of in-flight microbatches.
        # This is already the case, so we do nothing here.
        pass

    # Context Parallelism
    if config.context_parallel_size > 1:
        num_activation = (
            num_activation - num_activation_this_shard_mlp
        ) / config.context_parallel_size + num_activation_this_shard_mlp

    # Calculate bytes per parameter for optimizer states
    if args.use_distributed_optimizer:
        base_optim_bytes = 6  # FP16 weight, FP32 master weight
        world_optim_bytes = 12  # FP32 grad, FP32 momentum, FP32 variance
    else:
        base_optim_bytes = 18  # All states on each GPU
        world_optim_bytes = 0

    num_bytes_per_parameter = base_optim_bytes + (
        world_optim_bytes / (args.data_parallel_size * config.context_parallel_size)
    )

    # Handle MoE optimizer state sharding if applicable
    if num_parameter_this_shard_sparse > 0 and config.expert_model_parallel_size > 1:
        moe_dp_size = (
            args.data_parallel_size
            * config.tensor_model_parallel_size
            // (config.expert_model_parallel_size * args.expert_tensor_parallel_size)
        )
        num_bytes_per_parameter_moe = base_optim_bytes + (
            world_optim_bytes / moe_dp_size
        )

        weight_and_optimizer_memory = (
            (num_parameter_this_shard - num_parameter_this_shard_sparse)
            * num_bytes_per_parameter
            + num_parameter_this_shard_sparse * num_bytes_per_parameter_moe
        ) / NUM_BYTES_IN_GIGABYTE
    else:
        weight_and_optimizer_memory = (
            num_parameter_this_shard * num_bytes_per_parameter
        ) / NUM_BYTES_IN_GIGABYTE

    activation_memory = num_activation * 2 / NUM_BYTES_IN_GIGABYTE  # Use GIGABYTE
    total_memory = weight_and_optimizer_memory + activation_memory

    report = {
        "pp_rank": pp_rank,
        "parameters_b": num_parameter_this_shard / 1e9,
        "activation_b": num_activation / 1e9,  # Renamed from _gb to _b
        "weight_optimizer_gb": round(weight_and_optimizer_memory, 2),
        "activation_gb": round(activation_memory, 2),
        "total_gb": round(total_memory, 2),
        "details": model.dump(),
        "model_breakdown": str(model),
    }
    print(model)

    return report, output_shape


def estimate_from_config(config, args):
    """
    Estimate memory usage from a given config and args, instead of global state.
    This version iterates over pipeline parallel ranks for accurate estimation.
    """
    reports = []
    input_shape = [args.micro_batch_size, args.seq_length]
    pp_size = config.pipeline_model_parallel_size

    if pp_size > 1:
        for pp_rank in range(pp_size):
            set_pipeline_model_parallel_rank(pp_rank)
            report_for_rank, new_input_shape = _calculate_rank_memory(
                config, args, input_shape, pp_rank, pp_size
            )
            reports.append(report_for_rank)
            input_shape = new_input_shape  # Pass output shape to the next stage
    else:
        report_for_rank, _ = _calculate_rank_memory(config, args, input_shape, 0, 1)
        reports.append(report_for_rank)

    return reports


def model_provider() -> GPTModel:
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)
    assert not args.use_legacy_models

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts,
                args.moe_grouped_gemm,
                args.qk_layernorm,
                args.multi_latent_attention,
                args.fp8,
            )
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts,
                args.moe_grouped_gemm,
                args.qk_layernorm,
                args.multi_latent_attention,
            )
    set_global_config(config)
    pre_process = is_pipeline_first_stage()
    post_process = is_pipeline_last_stage()
    # TODO fp8
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
    )

    return model


NUM_BYTES_IN_MEGABYTE = 1024 * 1024
NUM_BYTES_IN_GIGABYTE = 1024 * 1024 * 1024


def report_memory_usage():
    args = get_args()
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    input_shape = [args.micro_batch_size, args.seq_length]

    if config.pipeline_model_parallel_size > 1:
        for pp_rank in range(config.pipeline_model_parallel_size):
            set_pipeline_model_parallel_rank(pp_rank)
            print(f"\n----------[Pipeline_Parallelism_Rank={pp_rank}]----------")
            input_shape = report_memory_usage_one_pp_rank(
                input_shape, pp_rank, config.pipeline_model_parallel_size
            )
    else:
        report_memory_usage_one_pp_rank(input_shape)


def report_memory_usage_one_pp_rank(
    input_shape: list[int], pp_rank=0, pp_size=1
) -> list[int]:
    args = get_args()

    print(f"{input_shape=}")
    model: GPTModel = model_provider()
    num_parameter_this_shard = model.num_parameter()
    num_activation = model.num_activation(input_shape)
    output_shape = model.mock_forward(input_shape)

    num_parameter_this_shard_sparse = 0
    for layer in model.decoder.layers.modules:
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
        [m.mlp.num_activation() for m in model.decoder.layers.modules]
    )
    num_microbatch_this_pp_rank = pp_size - pp_rank
    # vpp
    if args.num_layers_per_virtual_pipeline_stage is not None:
        layers_this_pprank = model.decoder.layers.modules.__len__()
        vpp_size = layers_this_pprank // args.num_layers_per_virtual_pipeline_stage
        num_microbatch_this_pp_rank = (
            pp_size * (vpp_size - 1) + (pp_size - pp_rank) * 2 - 1
        ) / vpp_size

    num_parameter_this_shard_sparse = 0
    for layer in model.decoder.layers.modules:
        if isinstance(layer.mlp, MoELayer):
            num_parameter_this_shard_sparse += layer.mlp.num_parameter()
            if (
                "shared_experts" in layer.mlp.__dir__()
                and layer.mlp.shared_experts is not None
            ):
                num_parameter_this_shard_sparse -= (
                    layer.mlp.shared_experts.num_parameter()
                )
    num_microbatch_this_pp_rank = pp_size - pp_rank
    # vpp
    if args.num_layers_per_virtual_pipeline_stage is not None:
        layers_this_pprank = model.decoder.layers.modules.__len__()
        vpp_size = layers_this_pprank // args.num_layers_per_virtual_pipeline_stage
        num_microbatch_this_pp_rank = (
            pp_size * (vpp_size - 1) + (pp_size - pp_rank) * 2 - 1
        ) / vpp_size
    model.__repr__()
    print(model)
    print(
        f"Number of parameters in every GPU in billions: "
        f"{num_parameter_this_shard / 10**9: .2f} where mlp part is {num_parameter_this_shard_sparse / 10**9: .2f}"
    )
    # recompute
    if args.recompute_granularity == "full":
        recompute_num_layers = args.recompute_num_layers
        num_layers = model.num_layers
        common_act = (
            model.num_act_pre
            + model.num_act_between_layers * num_layers * num_microbatch_this_pp_rank
        )  # recompute with pipeline parallel
        info = (
            "With this recomputing setting, the number of activation achieve peak when "
        )
        if args.recompute_method == "block":
            num_layers_with_loss = num_layers - recompute_num_layers
            if num_layers_with_loss == 0:
                peak1 = common_act + model.num_act_post
                peak2 = common_act + model.num_act_per_layer
                if peak1 > peak2:
                    info += "calculating loss"
                else:
                    info += "back-propogating loss"
                num_activation = max(peak1, peak2)
            else:
                info += (
                    f"calculating loss with {num_layers_with_loss} non-recompute layers"
                )
                num_activation = (
                    common_act
                    + model.num_act_post
                    + model.num_act_per_layer
                    * num_layers_with_loss
                    * num_microbatch_this_pp_rank
                )
        elif args.recompute_method == "uniform":
            peak1 = common_act + model.num_act_post
            peak2 = (
                common_act
                + model.num_act_per_layer
                * recompute_num_layers
                * num_microbatch_this_pp_rank
            )
            if peak1 > peak2:
                info += "calculating loss"
            else:
                info += f"back-propogating loss recomputing every {recompute_num_layers} layers"
            num_activation = max(peak1, peak2)
        if isinstance(
            model.decoder.layers.modules[0].self_attention, MLASelfAttention
        ):  # MLA recompute achieve peak at backward
            num_activation += model.decoder.layers.modules[
                0
            ].self_attention.core_attention.num_activation()
        print(info)

    else:
        num_activation = (
            num_activation - model.num_act_post
        ) * num_microbatch_this_pp_rank + model.num_act_post

    # CP
    num_activation = (
        num_activation - num_activation_this_shard_mlp
    ) / args.context_parallel_size + num_activation_this_shard_mlp
    if pp_size == 1:
        print(
            f"Number of activation in every GPU in billions: "
            f"{num_activation / 10**9: .2f} where mlp part is {num_activation_this_shard_mlp / 10**9: .2f}"
        )
    else:
        print(
            f"Number of activation per microbatch in every GPU in billions: "
            f"{num_activation / 10**9: .2f} where mlp part is {num_activation_this_shard_mlp / 10**9: .2f}"
            f", {num_microbatch_this_pp_rank=}"
        )
    num_bytes_per_parameter = (
        18
        if not args.use_distributed_optimizer
        else 6 + (12 / args.data_parallel_size / args.context_parallel_size)
    )
    if args.expert_model_parallel_size * args.expert_tensor_parallel_size > 1:
        num_bytes_per_parameter_dense = num_bytes_per_parameter
        num_bytes_per_parameter_moe = (
            18
            if not args.use_distributed_optimizer
            else 6
            + (
                12
                / (
                    args.data_parallel_size
                    * args.context_parallel_size
                    * args.tensor_model_parallel_size
                    / args.expert_model_parallel_size
                    / args.expert_tensor_parallel_size
                )
            )
        )
        print(f"{num_bytes_per_parameter_dense=} {num_bytes_per_parameter_moe=}")

        weight_and_optimizer_memory = (
            (num_parameter_this_shard - num_parameter_this_shard_sparse)
            * num_bytes_per_parameter_dense
            + num_parameter_this_shard_sparse * num_bytes_per_parameter_moe
        ) / NUM_BYTES_IN_GIGABYTE
    else:
        print(f"{num_bytes_per_parameter=}")
        weight_and_optimizer_memory = (
            num_parameter_this_shard * num_bytes_per_parameter / NUM_BYTES_IN_GIGABYTE
        )

    activation_memory = num_activation * 2 / NUM_BYTES_IN_GIGABYTE  # only support fp16
    total_memory = weight_and_optimizer_memory + activation_memory
    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory/1024:.2f} GB, "
        f"activation={activation_memory/1024:.2f} GB, total={total_memory/1024:.2f} GB\n"
    )

    # import ipdb

    # ipdb.set_trace()
    return output_shape
    pass


if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    import ipdb

    with ipdb.launch_ipdb_on_exception():
        report_memory_usage()
