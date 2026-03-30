# coding=utf-8
# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com

from functools import partial

import torch
import torch.distributed

try:
    import xformers
except:
    pass


# ref from: https://github.com/zhuzilin/ring-flash-attention
def llama3_prepare_cu_kv_slice(
    cu_seqlens: torch.Tensor, causal: bool, rank: int, world_size: int
):
    """
    Args:
        cu_seqlens: torch.Tensor, the cu_seqlens of all the sequences across the ring process group.

    Returns:
        local_k_slice: slice, the slice of the k that the local q need. Note
            that this may be longer than `total_seq_len // world_size`.
    """
    total_length = cu_seqlens[-1]
    assert total_length % world_size == 0
    length_per_rank = total_length // world_size
    left = torch.searchsorted(cu_seqlens, rank * length_per_rank)
    right = torch.searchsorted(cu_seqlens, (rank + 1) * length_per_rank)
    length_per_rank = length_per_rank.item()

    # after this, cu_seqlens[left:right + 1] contains all the sequence for this rank
    if cu_seqlens[left] != rank * length_per_rank:
        left -= 1
    left = left.item()
    right = right.item()

    if causal:
        # when causal, we hope
        # - the last k seq is of the same length as the last q seq
        slice_right = (rank + 1) * length_per_rank
    else:
        # when not causal, we hope
        # - the last k is full seq
        slice_right = cu_seqlens[right].item()

    slice_left = cu_seqlens[left].item()

    local_kv_slice = slice(slice_left, slice_right)
    return local_kv_slice


# ref from: https://github.com/zhuzilin/ring-flash-attention
def llama3_prepare_cu_seqlens(
    cu_seqlens: torch.Tensor, causal: bool, rank: int, world_size: int
):
    """
    Args:
        cu_seqlens: torch.Tensor, the cu_seqlens of all the sequences across the ring process group.

    Returns:
        cu_seqlens_q: torch.Tensor, the cu_seqlens of the q slice for this rank.
        cu_seqlens_k: torch.Tensor, the cu_seqlens of the k slice that the local q need. Note
            that this may be longer than `total_seq_len // world_size`.
    """
    if world_size == 1:
        return cu_seqlens.clone(), cu_seqlens.clone()
    total_length = cu_seqlens[-1]
    assert total_length % world_size == 0
    length_per_rank = total_length // world_size
    left = torch.searchsorted(cu_seqlens, rank * length_per_rank)
    right = torch.searchsorted(cu_seqlens, (rank + 1) * length_per_rank)
    length_per_rank = length_per_rank.item()

    # after this, cu_seqlens[left:right + 1] contains all the sequence for this rank
    if cu_seqlens[left] != rank * length_per_rank:
        left -= 1
    left = left.item()
    right = right.item()

    # q is always the same. just calculate the cu_seqlens for the local slice
    cu_seqlens_q = cu_seqlens[left : right + 1].clone()
    cu_seqlens_q -= rank * length_per_rank
    cu_seqlens_q[0] = 0
    cu_seqlens_q[-1] = length_per_rank

    cu_seqlens_k = cu_seqlens[left : right + 1].clone()
    if causal:
        # when causal, we hope
        # - the last k seq is of the same length as the last q seq
        slice_right = (rank + 1) * length_per_rank
        cu_seqlens_k[-1] = slice_right

    slice_left = cu_seqlens[left].item()
    cu_seqlens_k -= slice_left

    return cu_seqlens_q, cu_seqlens_k


def llama3_prepare_cu_seqlens_zigzag(
    cu_seqlens: torch.Tensor, causal: bool, rank: int, world_size: int
):
    cu_seqlens_q, cu_seqlens_kv = llama3_prepare_cu_seqlens(
        cu_seqlens, causal, rank, world_size * 2
    )
    local_seqlens_q = [0]
    local_seqlens_kv = [0]
    local_seqlens_q.extend((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist())
    local_seqlens_kv.extend((cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).tolist())
    cu_seqlens_q, cu_seqlens_kv = llama3_prepare_cu_seqlens(
        cu_seqlens, causal, world_size * 2 - 1 - rank, world_size * 2
    )
    local_seqlens_q.extend((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist())
    local_seqlens_kv.extend((cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).tolist())

    local_seqlens_q = torch.cumsum(
        torch.tensor(local_seqlens_q, dtype=torch.int32), 0, dtype=torch.int32
    )
    local_seqlens_kv = torch.cumsum(
        torch.tensor(local_seqlens_kv, dtype=torch.int32), 0, dtype=torch.int32
    )
    return local_seqlens_q, local_seqlens_kv


def llama3_prepare_memory_efficient_mask(
    cu_seqlens: torch.Tensor, causal: bool, rank: int, world_size: int, device=None
):
    cu_seqlens_q, cu_seqlens_kv = llama3_prepare_cu_seqlens(
        cu_seqlens, causal, rank, world_size
    )
    local_seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    local_seqlens_kv = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).tolist()
    local_attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(
        q_seqlen=local_seqlens_q,
        kv_seqlen=local_seqlens_kv,
        device=device,
    )
    if causal:
        local_attn_bias = local_attn_bias.make_causal_from_bottomright()

    return local_attn_bias


def forward_cat_func_zigzag(gathered_logits, cp_size, seq_dim):
    reorded_logits = [None for _ in range(2 * cp_size)]
    for rank in range(cp_size):
        gathered_logits_chunks = gathered_logits[rank].chunk(2, dim=seq_dim)
        reorded_logits[rank] = gathered_logits_chunks[0]
        reorded_logits[2 * cp_size - rank - 1] = gathered_logits_chunks[1]
    gathered_logits = torch.cat(reorded_logits, dim=seq_dim)
    gathered_logits = gathered_logits.view(
        *gathered_logits.shape[0:seq_dim], -1, *gathered_logits.shape[(seq_dim + 2) :]
    )
    return gathered_logits


def return_self(input):
    return input


@torch.no_grad
def gather_cp_forward_zigzag(
    input,
    seq_dim,
    async_op=False,
    process_group: torch.distributed.ProcessGroup = None,
):
    cp_size = torch.distributed.get_world_size(process_group)
    if cp_size == 1:
        return partial(return_self, input), None

    # Split local_logits into two parts
    input = input.view(
        *input.shape[0:seq_dim],
        2,
        input.shape[seq_dim] // 2,
        *input.shape[(seq_dim + 1) :],
    )

    gathered_logits = [torch.zeros_like(input) for _ in range(cp_size)]
    handle = torch.distributed.all_gather(
        gathered_logits, input, async_op=async_op, group=process_group
    )

    return partial(forward_cat_func_zigzag, gathered_logits, cp_size, seq_dim), handle


def backward_process(local_grad, shape):
    return local_grad.view(shape)


@torch.no_grad
def gather_cp_backward_zigzag(
    grad_output,
    seq_dim,
    async_op=False,
    reduce_op=torch.distributed.ReduceOp.AVG,
    process_group: torch.distributed.ProcessGroup = None,
):
    cp_size = torch.distributed.get_world_size(process_group)
    if cp_size == 1:
        return partial(return_self, grad_output), None

    # Reshape grad_output to match the forward pass
    grad_output = grad_output.view(
        *grad_output.shape[0:seq_dim],
        2 * cp_size,
        grad_output.shape[seq_dim] // (2 * cp_size),
        *grad_output.shape[(seq_dim + 1) :],
    )

    reordered_indices = []
    for rank in range(cp_size):
        reordered_indices.append(rank)
        reordered_indices.append(2 * cp_size - rank - 1)
    reordered_indices = torch.tensor(reordered_indices, device=grad_output.device)
    grad_output = torch.index_select(grad_output, dim=seq_dim, index=reordered_indices)

    split_tensors = torch.split(
        grad_output, grad_output.size(seq_dim) // cp_size, dim=seq_dim
    )
    grad_list = [t.contiguous() for t in split_tensors]
    # TODO(guanyouhe): 这里可能要做一下处理
    grad_list = [t.squeeze(seq_dim + 1) for t in grad_list]
    local_shape = grad_list[0].shape

    local_grad = torch.empty(
        local_shape,
        dtype=grad_output.dtype,
        device=torch.cuda.current_device(),
    )
    handle = torch.distributed.reduce_scatter(
        local_grad,
        grad_list,
        op=reduce_op,
        async_op=async_op,
        group=process_group,
    )

    new_shape = (
        *local_grad.shape[0:seq_dim],
        -1,
        *local_grad.shape[(seq_dim + 2) :],
    )
    return partial(backward_process, local_grad, new_shape), handle


def forward_cat_func(tensor_list, seq_dim):
    return torch.cat(tensor_list, dim=seq_dim).contiguous()


@torch.no_grad
def gather_cp_forward(
    input,
    seq_dim,
    async_op=False,
    process_group: torch.distributed.ProcessGroup = None,
):
    """Gather tensors and concatinate along the last dimension."""
    cp_size = torch.distributed.get_world_size(group=process_group)
    if cp_size == 1:
        return partial(return_self, input), None

    # Size and dimension.
    assert seq_dim < input.dim() and seq_dim >= 0, "Invalid dimension to gather along."
    rank = torch.distributed.get_rank(group=process_group)

    tensor_list = [torch.empty_like(input) for _ in range(cp_size)]
    tensor_list[rank] = input
    handle = torch.distributed.all_gather(
        tensor_list, input, async_op=async_op, group=process_group
    )

    return partial(forward_cat_func, tensor_list, seq_dim), handle


@torch.no_grad
def gather_cp_backward(
    grad_output,
    seq_dim,
    async_op=False,
    reduce_op=torch.distributed.ReduceOp.AVG,
    process_group: torch.distributed.ProcessGroup = None,
):
    assert (
        seq_dim < grad_output.dim() and seq_dim >= 0
    ), "Invalid dimension to reduce-scatter along."
    cp_size = torch.distributed.get_world_size(group=process_group)
    if cp_size == 1:
        return partial(return_self, grad_output), None

    dim_size = list(grad_output.size())
    assert (
        dim_size[seq_dim] % cp_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    dim_size[seq_dim] = dim_size[seq_dim] // cp_size

    output = torch.empty(
        dim_size, dtype=grad_output.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output,
        grad_output.contiguous(),
        op=reduce_op,
        async_op=async_op,
        group=process_group,
    )
    return partial(return_self, output), handle
