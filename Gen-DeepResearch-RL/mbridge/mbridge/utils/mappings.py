import torch
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
)


class _AllGatherToContextParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, seq_dim, bwd_op):
        assert seq_dim in [0, 1] and input_.dim() > seq_dim
        cp_size = get_context_parallel_world_size()
        # Split local_logits into two parts
        input_ = input_.view(
            *input_.shape[0:seq_dim],
            2,
            input_.shape[seq_dim] // 2,
            *input_.shape[(seq_dim + 1) :],
        )

        gathered_logits = [torch.zeros_like(input_) for _ in range(cp_size)]
        torch.distributed.all_gather(
            gathered_logits, input_, group=get_context_parallel_group()
        )

        reorded_logits = [None for _ in range(2 * cp_size)]
        if seq_dim == 1:
            for rank in range(cp_size):
                reorded_logits[rank] = gathered_logits[rank][:, 0]
                reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][:, 1]
        elif seq_dim == 0:
            for rank in range(cp_size):
                reorded_logits[rank] = gathered_logits[rank][0]
                reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][1]
        else:
            assert False
        gathered_logits = torch.cat(reorded_logits, dim=seq_dim)

        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        ctx.cp_group = get_context_parallel_group()
        ctx.local_shape = input_.shape
        ctx.bwd_op = bwd_op

        return gathered_logits

    @staticmethod
    def backward(ctx, grad_output):
        seq_dim = ctx.seq_dim
        assert seq_dim in [0, 1]
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        local_shape = ctx.local_shape
        bwd_op = ctx.bwd_op

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
        if seq_dim == 1:
            grad_output = grad_output[:, reordered_indices, :]
        elif seq_dim == 0:
            grad_output = grad_output[reordered_indices, :]
        else:
            assert False

        split_tensors = torch.split(
            grad_output, grad_output.size(seq_dim) // cp_size, dim=seq_dim
        )
        grad_list = [t.contiguous() for t in split_tensors]
        assert split_tensors[0].shape == local_shape

        local_grad = torch.empty(
            local_shape, dtype=grad_output.dtype, device=torch.cuda.current_device()
        )
        torch.distributed.reduce_scatter(
            local_grad, grad_list, op=bwd_op, group=cp_group
        )

        local_grad = local_grad.view(
            *local_grad.shape[0:seq_dim],
            -1,
            *local_grad.shape[(seq_dim + 2) :],
        )
        return local_grad, None, None


def all_gather_to_context_parallel_region(
    local_tensor, gather_dim=1, bwd_op=torch.distributed.ReduceOp.AVG
):
    return _AllGatherToContextParallelRegion.apply(local_tensor, gather_dim, bwd_op)
