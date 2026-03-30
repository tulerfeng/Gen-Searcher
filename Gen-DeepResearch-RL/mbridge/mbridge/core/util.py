# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import inspect
import json
import os
from collections import defaultdict
from functools import lru_cache

import torch
from megatron.core import mpu
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import (
    StragglerDetector,
    check_param_hashes_across_dp_replicas,
    get_model_config,
    is_te_min_version,
)


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    fp16: bool = False,
    bf16: bool = True,
    virtual_pipeline_model_parallel_size: int = None,
    encoder_pipeline_model_parallel_size: int = 0,
    use_torch_fsdp2: bool = False,
    use_custom_fsdp: bool = False,
    use_precision_aware_optimizer: bool = False,
    use_cpu_initialization: bool = False,
    init_model_with_meta_device: bool = False,
    overlap_param_gather_with_optimizer_step: bool = False,
    data_parallel_random_init: bool = True,
    ddp_config: dict = None,
    optimizer_config: dict = None,
):
    """Build the model.
    copied from megatron/training/training.py but remove args
    """

    # Build model.
    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and virtual_pipeline_model_parallel_size is not None
        ):
            if model_type == ModelType.encoder_and_decoder:
                assert (
                    encoder_pipeline_model_parallel_size == 0
                ), "Interleaved schedule not supported for model with encoder on separate PP rank"
            model = []
            for i in range(virtual_pipeline_model_parallel_size):
                # Set pre_process and post_process only after virtual rank is set.
                if (
                    "vp_stage"
                    in inspect.signature(mpu.is_pipeline_first_stage).parameters
                ):
                    pre_process = mpu.is_pipeline_first_stage(
                        ignore_virtual=False, vp_stage=i
                    )
                    post_process = mpu.is_pipeline_last_stage(
                        ignore_virtual=False, vp_stage=i
                    )
                else:
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                this_model = model_provider_func(
                    pre_process=pre_process, post_process=post_process, vp_stage=i
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            add_encoder = True
            add_decoder = True
            if model_type == ModelType.encoder_and_decoder:
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    rank = mpu.get_pipeline_model_parallel_rank()
                    first_decoder_rank = encoder_pipeline_model_parallel_size
                    world_size = mpu.get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == first_decoder_rank
                    post_process = (rank == (first_decoder_rank - 1)) or (
                        rank == (world_size - 1)
                    )
                    add_encoder = mpu.is_inside_encoder(rank)
                    add_decoder = mpu.is_inside_decoder(rank)
                model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder,
                )
            else:
                model = model_provider_func(
                    pre_process=pre_process, post_process=post_process
                )
            model.model_type = model_type
        return model

    if init_model_with_meta_device:
        with torch.device("meta"):
            model = build_model()
    else:
        model = build_model()
    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )

    # Print number of parameters.
    num_parameters = sum(
        [
            sum([p.nelement() for p in model_module.parameters()])
            for model_module in model
        ]
    )
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                num_parameters,
            ),
            flush=True,
        )

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not (use_torch_fsdp2 and use_cpu_initialization)
        and not init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if fp16 or bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        from megatron.core.distributed import DistributedDataParallelConfig

        if use_torch_fsdp2:
            try:
                from megatron.core.distributed import (
                    TorchFullyShardedDataParallel as torch_FSDP,
                )

                HAVE_FSDP2 = True
            except ImportError:
                HAVE_FSDP2 = False
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        elif use_custom_fsdp:
            from megatron.core.distributed.custom_fsdp import (
                FullyShardedDataParallel as custom_FSDP,
            )

            DP = custom_FSDP
        else:
            from megatron.core.distributed import DistributedDataParallel as DDP

            DP = DDP

        config = get_model_config(model[0])

        # default
        kwargs = {"grad_reduce_in_fp32": True, "use_distributed_optimizer": True}
        if ddp_config is not None:
            kwargs.update(ddp_config)
        if optimizer_config is not None:
            import warnings

            warnings.warn(
                "optimizer_config is deprecated to set DistributedDataParallelConfig, use ddp_config instead",
                DeprecationWarning,
            )
            kwargs.update(optimizer_config)
        if use_custom_fsdp and use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False

        ddp_config = DistributedDataParallelConfig(**kwargs)

        if not use_torch_fsdp2:
            # In the custom FSDP and DDP use path, we need to initialize the bucket size.

            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000,
                    1000000
                    * mpu.get_data_parallel_world_size(with_context_parallel=True),
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [
            DP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0)
                or overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()
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
                    # print(f"maintain router bias dtype for {l.mlp.router}")
                    l.mlp.router._maintain_float32_expert_bias()
    return model


from megatron.core import DistributedDataParallel as DDP

try:
    from megatron.core.distributed.custom_fsdp import (
        FullyShardedDataParallel as custom_FSDP,
    )
except ImportError:
    from megatron.core.distributed.fsdp import FullyShardedDataParallel as custom_FSDP

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, custom_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, custom_FSDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def broadcast_from_megatron_pp(tensor: torch.Tensor):
    # tensor is not None only in one of the pp ranks
    if tensor is not None:
        shape = tensor.shape
        dtype = tensor.dtype
        tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
        partition_dim = getattr(tensor, "partition_dim", None)
        tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
    else:
        tensor_spec = None
    tensor_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=tensor_spec_output,
        obj=tensor_spec,
        group=mpu.get_pipeline_model_parallel_group(),
    )
    # find the src rank
    target_tensor_spec = None
    src_rank = None
    for rank, tensor_spec in enumerate(tensor_spec_output):
        if tensor_spec is not None:
            if target_tensor_spec is None:
                target_tensor_spec = tensor_spec
            else:
                raise ValueError("A tensor exists on two pp ranks")
            src_rank = rank
    assert target_tensor_spec is not None
    if tensor is None:
        tensor = torch.empty(
            size=target_tensor_spec[0],
            dtype=target_tensor_spec[1],
            device=torch.cuda.current_device(),
        )
        if target_tensor_spec[2] is not None:
            tensor.tensor_model_parallel = target_tensor_spec[2]
        if target_tensor_spec[3] is not None:
            tensor.partition_dim = target_tensor_spec[3]

    global_rank = torch.distributed.get_global_rank(
        group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank
    )
    torch.distributed.broadcast(
        tensor=tensor, src=global_rank, group=mpu.get_pipeline_model_parallel_group()
    )
    return tensor


def broadcast_str_from_megatron_pp(obj: any) -> any:
    obj_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=obj_output, obj=obj, group=mpu.get_pipeline_model_parallel_group()
    )

    src_rank = None
    target_obj = None
    for rank, item in enumerate(obj_output):
        if item is not None:
            if target_obj is not None:
                raise ValueError("An object exists on two pp ranks")
            target_obj = item
            src_rank = rank

    assert target_obj is not None, "No valid object found to broadcast."

    global_rank = torch.distributed.get_global_rank(
        group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank
    )

    obj_output = [None] * torch.distributed.get_world_size(
        group=mpu.get_pipeline_model_parallel_group()
    )
    obj_output[0] = target_obj
    torch.distributed.broadcast_object_list(
        object_list=obj_output,
        src=global_rank,
        group=mpu.get_pipeline_model_parallel_group(),
    )

    return obj_output[0]


# reference: megatron/training/utils.py get_batch_on_this_cp_rank
def split_data_cp_rank(
    val: torch.Tensor, cp_size: int, seq_dim: int, cp_rank: int = None
):
    assert cp_size > 1
    assert 0 == val.shape[seq_dim] % (2 * cp_size), f"{val.shape=} {cp_size=}"
    if cp_rank is None:
        cp_rank = mpu.get_context_parallel_rank()
    if val is None:
        return val

    val = val.view(
        *val.shape[0:seq_dim],
        2 * cp_size,
        val.shape[seq_dim] // (2 * cp_size),
        *val.shape[(seq_dim + 1) :],
    )

    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
    val = val.index_select(seq_dim, index)
    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])

    return val


def expand_thw(thw: torch.Tensor) -> torch.Tensor:
    assert thw.dim() == 2
    repeats = thw[:, 0].to(torch.long)
    assert torch.all(repeats > 0), "thw[:,0] must be > 0"

    idx = torch.arange(thw.size(0), device=thw.device).repeat_interleave(repeats)
    out = thw[idx].clone()
    out[:, 0] = 1
    return out


def collapse_thw(expanded: torch.Tensor) -> torch.Tensor:
    assert expanded.dim() == 2
    assert expanded.size(1) >= 2
    if expanded.shape[0] < 2:
        return expanded

    # find the diff
    other = expanded[:, 1:]
    prev = torch.cat([other[:1], other[:-1]], dim=0)
    change = (other != prev).any(dim=1)
    # the index0 must be now row
    change[0] = True

    # find the diff
    starts = torch.nonzero(change, as_tuple=False).squeeze(1)
    ends = (
        torch.cat([starts[1:], torch.tensor([other.size(0)], device=other.device)]) - 1
    )
    counts = ends - starts + 1

    rows_other = other[starts]
    result_first_col = counts.to(expanded.dtype).unsqueeze(1)
    result = torch.cat([result_first_col, rows_other], dim=1)
    return result


# also can use in qwen2vl/qwen2.5vl
def qwen2vl_pad_and_split(
    cp_size: int,
    hw_factor: int,
    pixel_values: list[torch.Tensor],
    image_grid_thws: list[torch.Tensor],
):
    assert len(pixel_values) == len(image_grid_thws)
    # split the pixel_values
    split_pixel_values = []
    split_image_grid_thws = []
    for pixel_value, image_grid_thw in zip(pixel_values, image_grid_thws):
        split_image_grid_thw = list(torch.split(image_grid_thw, 1, dim=0))
        split_image_grid_thws.extend(split_image_grid_thw)
        slice_begin = 0
        for ele in split_image_grid_thw:
            slice_end = slice_begin + ele.prod().item()
            split_pixel_values.append(pixel_value[slice_begin:slice_end].clone())
            slice_begin = slice_end

    pixel_values = split_pixel_values
    image_grid_thws = split_image_grid_thws
    img_num = len(image_grid_thws)

    img_num_per_rank = img_num // cp_size
    img_num_remain = img_num % cp_size
    cp_img_num = []
    for i in range(cp_size):
        cp_img_num.append(img_num_per_rank)
        if i < img_num_remain:
            cp_img_num[i] += 1

    img_idx = 0
    new_pixel_values = []
    new_image_grid_thws = []
    images_padded = []
    for i in range(cp_size):
        seq_len = 0
        img_begin_idx = img_idx
        img_end_idx = img_begin_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        for j in range(img_begin_idx, img_end_idx):
            seq_len += pixel_values[j].size(0)
            new_pixel_values.append(pixel_values[j])
            new_image_grid_thws.append(image_grid_thws[j])

        image_padded = 0 != seq_len % hw_factor
        if image_padded:
            padded_seqlen = (seq_len + hw_factor - 1) // hw_factor * hw_factor - seq_len
            assert padded_seqlen > 0 and padded_seqlen % 4 == 0
            new_pixel_values.append(
                torch.zeros(
                    [padded_seqlen, pixel_values[0].size(-1)],
                    dtype=pixel_values[0].dtype,
                    device=pixel_values[0].device,
                )
            )
            new_image_grid_thws.append(
                torch.tensor(
                    [[1, 2, padded_seqlen // 2]],
                    dtype=image_grid_thws[0].dtype,
                    device=image_grid_thws[0].device,
                )
            )
            cp_img_num[i] += 1
        images_padded.append(int(image_padded))

    return new_pixel_values, new_image_grid_thws, cp_img_num, images_padded


@torch.no_grad
def qwen3vl_cp_split(
    cp_size: int,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
):
    assert cp_size > 1
    if pixel_values is None:
        assert image_grid_thw is None
        return None, None, None, None

    assert not pixel_values.requires_grad
    assert not image_grid_thw.requires_grad
    # expand video thw
    image_grid_thw = expand_thw(image_grid_thw)

    hw_factor = 4
    new_pixel_values, new_image_grid_thws, cp_img_num, images_padded = (
        qwen2vl_pad_and_split(
            cp_size,
            hw_factor,
            [pixel_values],
            [image_grid_thw],
        )
    )
    for image_padded in images_padded:
        assert not image_padded, "qwen3vl vit not support sp now, no need to paded"

    pixel_values = torch.cat(new_pixel_values, dim=0)
    image_grid_thw = torch.cat(new_image_grid_thws, dim=0)
    return pixel_values, image_grid_thw, cp_img_num, images_padded


def get_vision_cp_data(
    vision_data: torch.Tensor,
    vision_grid_thw: torch.Tensor,
    square_merge_size: int,
    cp_img_num: list[int],
    images_padded: list[bool],
):
    """Get vision data and grid_thw for context parallelism.
    Returns:
        vision_data (torch.Tensor): Vision data of shape [total_thw_size, n_features].
        vision_grid_thw (torch.Tensor): Vision grid_thw of shape [total_thw_size, 3].
        seqlens_list (list of torch.Tensor): List of seqlens of the vision data in each context parallel rank,
                                             for the all gather after vision encoder.
    """
    # we use the context parallelism size and context parallel group of LLM for vision model.
    # we only divide the number of images in each context parallel rank.
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    assert cp_size == len(cp_img_num)

    seqlens = torch.repeat_interleave(
        vision_grid_thw[:, 1] * vision_grid_thw[:, 2], vision_grid_thw[:, 0]
    )
    vision_grid_thw_list = []
    vision_data_list = []
    seqlens_list = []
    img_idx = 0
    for i in range(cp_size):
        start_idx = img_idx
        end_idx = start_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        vision_grid_thw_list.append(vision_grid_thw[start_idx:end_idx])
        if images_padded[i]:
            seqlens_list.append(seqlens[start_idx : end_idx - 1])
        else:
            seqlens_list.append(seqlens[start_idx:end_idx])
        data_start_idx = seqlens[:start_idx].sum()
        data_end_idx = seqlens[:end_idx].sum()
        vision_data_list.append(vision_data[data_start_idx:data_end_idx])
    new_vision_grid_thw = vision_grid_thw_list[cp_rank]
    new_vision_data = vision_data_list[cp_rank]
    new_seqlens_list = [t // square_merge_size for t in seqlens_list]
    return new_vision_data, new_vision_grid_thw, new_seqlens_list


class AllGatherVisionEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, seqlens_on_cp_ranks):
        outputs = []
        for i in range(len(seqlens_on_cp_ranks)):
            o = torch.zeros(
                (seqlens_on_cp_ranks[i].sum(), *input.shape[1:]),
                device=input.device,
                dtype=input.dtype,
                layout=input.layout,
            )
            outputs.append(o)
        torch.distributed.all_gather(
            outputs, input, group=mpu.get_context_parallel_group()
        )
        cp_rank = mpu.get_context_parallel_rank()
        ctx.cp_rank = cp_rank
        ctx.save_for_backward(*seqlens_on_cp_ranks)

        output = torch.cat(outputs, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cp_rank = ctx.cp_rank
        seqlens_on_cp_ranks = ctx.saved_tensors
        start_idx = (
            torch.cat(seqlens_on_cp_ranks[:cp_rank]).sum() if cp_rank != 0 else 0
        )
        end_idx = start_idx + seqlens_on_cp_ranks[cp_rank].sum()
        grad_output = grad_output[start_idx:end_idx]
        return grad_output, None


def preprocess_packed_seqs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=input_ids.device
    )
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = (
        seqlens_in_batch.tolist()
    )  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = (
        seqlens_in_batch_padded.tolist()
    )  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = (
        cu_seqlens_padded.tolist()
    )  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(
            shape, dtype=input_ids.dtype, device=input_ids.device
        )
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[
                    i, attention_mask[i]
                ]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i, attention_mask[i]]
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[
                    start_idx + half_seqlen : start_idx + half_seqlen + remain_len
                ] = d[remain_start:remain_end]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )

    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    seq_lens_cpu: list[int] = (
        attention_mask.sum(dim=1, dtype=torch.int32).cpu().tolist()
    )

    shape = [batch_size, seq_len] + list(
        output.shape[2:]
    )  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(
            output_list, output.detach(), group=mpu.get_context_parallel_group()
        )
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]
    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new[i, attention_mask[i]] = output[0][start_idx : start_idx + s]
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[
                    packed_start_idx
                    + half_seqlen : packed_start_idx
                    + s_len_padded_chunk
                ],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[
                s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen
            ] = o1
        output_new[i, attention_mask[i]] = tmp[:s_len]

    return output_new
