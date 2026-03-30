# coding=utf-8
# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com
# ref from: https://github.com/zhuzilin/ring-flash-attention

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed
from einops import rearrange
from torch import Tensor, nn

try:
    import xformers
    from xformers.ops.fmha import (
        memory_efficient_attention_backward,
        memory_efficient_attention_forward_requires_grad,
    )
    from xformers.ops.fmha.attn_bias import AttentionBias
    from xformers.ops.fmha.common import AttentionOp
except:

    class AttentionBias:
        pass

    class AttentionOp:
        pass


from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType

from mbridge.models.ext.llama3_cp.cp_utils import (
    gather_cp_backward,
    gather_cp_backward_zigzag,
    gather_cp_forward,
    gather_cp_forward_zigzag,
)


@dataclass
class GpatchPackedSeqParams(PackedSeqParams):
    """
    MemoryEfficientAttention params
    """

    use_zigzag: bool = True
    kv_slice: slice = None
    heads_k_stride: int = None


def get_bwd_op(fwd_op):
    if fwd_op is None:
        return None
    op_map = {
        xformers.ops.fmha.cutlass.FwOp: xformers.ops.fmha.cutlass.BwOp,
        xformers.ops.fmha.flash.FwOp: xformers.ops.fmha.flash.BwOp,
        xformers.ops.fmha.flash3.FwOp: xformers.ops.fmha.flash3.BwOp,
        # xformers.ops.fmha.triton.FwOp: xformers.ops.fmha.triton.BwOp,
    }
    assert fwd_op in op_map
    return op_map[fwd_op]


class Llama3MemoryEfficientAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Union[Tensor, AttentionBias]] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
        op: Optional[AttentionOp] = None,
        output_dtype: Optional[torch.dtype] = None,
        process_group: torch.distributed.ProcessGroup = None,
        heads_k_stride: int = 1,
        use_zigzag: bool = True,
        kv_slice=None,
    ):
        out_list = []
        lse_list = []
        q_head_nums = query.shape[2]
        _, _, k_head_nums, _ = key.shape
        assert k_head_nums % heads_k_stride == 0
        assert q_head_nums % heads_k_stride == 0
        if use_zigzag:
            gather_forward_func = gather_cp_forward_zigzag
        else:
            gather_forward_func = gather_cp_forward

        if isinstance(attn_bias, Tensor) and attn_bias.shape[1] > heads_k_stride:
            attn_bias = attn_bias[:, :heads_k_stride, :, :]

        k_0 = key[:, :, 0:heads_k_stride, :].contiguous()
        v_0 = value[:, :, 0:heads_k_stride, :].contiguous()
        key_func, key_handle = gather_forward_func(
            k_0,
            seq_dim=1,
            async_op=True,
            process_group=process_group,
        )
        value_func, value_handle = gather_forward_func(
            v_0,
            seq_dim=1,
            async_op=True,
            process_group=process_group,
        )
        wait_handles = [key_handle, value_handle]
        for i in range(0, k_head_nums, heads_k_stride):
            for handle in wait_handles:
                if handle is not None:
                    handle.wait()
            wait_handles = []
            key_0, value_0 = key_func(), value_func()
            if kv_slice is not None:
                key_0 = key_0[:, kv_slice]
                value_0 = value_0[:, kv_slice]

            if i + heads_k_stride < k_head_nums:
                k_0 = key[:, :, (i + 1) : (i + 1 + heads_k_stride), :].contiguous()
                v_0 = value[:, :, (i + 1) : (i + 1 + heads_k_stride), :].contiguous()
                key_func, key_handle = gather_forward_func(
                    k_0,
                    seq_dim=1,
                    async_op=True,
                    process_group=process_group,
                )
                value_func, value_handle = gather_forward_func(
                    v_0,
                    seq_dim=1,
                    async_op=True,
                    process_group=process_group,
                )
                wait_handles = [key_handle, value_handle]

            query_0 = query[
                :,
                :,
                i
                * q_head_nums
                // k_head_nums : (i + heads_k_stride)
                * q_head_nums
                // k_head_nums,
                :,
            ]

            out, lse = memory_efficient_attention_forward_requires_grad(
                query_0,
                key_0,
                value_0,
                attn_bias=attn_bias,
                p=p,
                scale=scale,
                op=op,
                output_dtype=output_dtype,
            )
            out_list.append(out)
            lse_list.append(lse)

        out = torch.cat(out_list, dim=2)
        ctx.save_for_backward(query, key, value)
        ctx.out_list = out_list
        ctx.lse_list = lse_list

        ctx.attn_bias = attn_bias
        ctx.p = p
        ctx.scale = scale
        ctx.op = op
        ctx.output_dtype = output_dtype
        ctx.process_group = process_group
        ctx.heads_k_stride = heads_k_stride
        ctx.use_zigzag = use_zigzag
        ctx.kv_slice = kv_slice

        return out

    @staticmethod
    def backward(ctx, dout, *args):
        query, key, value = ctx.saved_tensors
        out_list = ctx.out_list
        lse_list = ctx.lse_list

        attn_bias = ctx.attn_bias
        p = ctx.p
        scale = ctx.scale
        op = get_bwd_op(ctx.op)
        output_dtype = ctx.output_dtype
        process_group = ctx.process_group
        heads_k_stride = ctx.heads_k_stride
        use_zigzag = ctx.use_zigzag
        kv_slice = ctx.kv_slice
        if use_zigzag:
            gather_forward_func = gather_cp_forward_zigzag
            gather_backward_func = gather_cp_backward_zigzag
        else:
            gather_forward_func = gather_cp_forward
            gather_backward_func = gather_cp_backward

        q_head_nums = query.shape[2]
        _, _, k_head_nums, _ = key.shape
        split_dout = torch.split(
            dout, dout.shape[2] // (k_head_nums // heads_k_stride), dim=2
        )
        assert len(split_dout) == len(out_list) == len(lse_list)

        dq_list = []
        dk_bwd_list = []
        dv_bwd_list = []
        k_0 = key[:, :, 0:heads_k_stride, :].contiguous()
        v_0 = value[:, :, 0:heads_k_stride, :].contiguous()
        key_func, key_handle = gather_forward_func(
            k_0,
            seq_dim=1,
            async_op=True,
            process_group=process_group,
        )
        value_func, value_handle = gather_forward_func(
            v_0,
            seq_dim=1,
            async_op=True,
            process_group=process_group,
        )
        wait_handles = [key_handle, value_handle]
        for i in range(0, k_head_nums, heads_k_stride):
            for handle in wait_handles:
                if handle is not None:
                    handle.wait()
            wait_handles = []
            key_0, value_0 = key_func(), value_func()
            if kv_slice is not None:
                # for reuse memory
                dk_buffer, dv_buffer = key_0, value_0
                key_0 = key_0[:, kv_slice]
                value_0 = value_0[:, kv_slice]

            if i + heads_k_stride < k_head_nums:
                k_0 = key[:, :, (i + 1) : (i + 1 + heads_k_stride), :].contiguous()
                v_0 = value[:, :, (i + 1) : (i + 1 + heads_k_stride), :].contiguous()
                key_func, key_handle = gather_forward_func(
                    k_0,
                    seq_dim=1,
                    async_op=True,
                    process_group=process_group,
                )
                value_func, value_handle = gather_forward_func(
                    v_0,
                    seq_dim=1,
                    async_op=True,
                    process_group=process_group,
                )
                wait_handles = [key_handle, value_handle]

            query_0 = query[
                :,
                :,
                i
                * q_head_nums
                // k_head_nums : (i + heads_k_stride)
                * q_head_nums
                // k_head_nums,
                :,
            ]

            dq, dk, dv = memory_efficient_attention_backward(
                split_dout[i],
                out_list[i],
                lse_list[i],
                query_0,
                key_0,
                value_0,
                attn_bias=attn_bias,
                p=p,
                scale=scale,
                op=op,
            )
            if kv_slice is not None:
                dk_buffer.zero_()
                dv_buffer.zero_()
                dk_buffer[:, kv_slice] = dk
                dv_buffer[:, kv_slice] = dv
                dk = dk_buffer
                dv = dv_buffer

            dk_bwd_list.append(
                gather_backward_func(
                    dk,
                    seq_dim=1,
                    async_op=True,
                    reduce_op=torch.distributed.ReduceOp.SUM,
                    process_group=process_group,
                )
            )
            dv_bwd_list.append(
                gather_backward_func(
                    dv,
                    seq_dim=1,
                    async_op=True,
                    reduce_op=torch.distributed.ReduceOp.SUM,
                    process_group=process_group,
                )
            )

            dq_list.append(dq)

        dk_list = []
        dv_list = []
        for i in range(len(dk_bwd_list)):
            dk_func, dk_handle = dk_bwd_list[i]
            dv_func, dv_handle = dv_bwd_list[i]
            if dk_handle is not None:
                dk_handle.wait()
            if dv_handle is not None:
                dv_handle.wait()
            dk_list.append(dk_func())
            dv_list.append(dv_func())

        d_query = torch.cat(dq_list, dim=2)
        d_key = torch.cat(dk_list, dim=2)
        d_value = torch.cat(dv_list, dim=2)
        return (d_query, d_key, d_value) + (None,) * 9


# q/k/v: batch_size x seqlen x head_cnt x head_dim
def llama3_memory_efficient_attention_func(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Optional[Union[Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[AttentionOp] = None,
    output_dtype: Optional[torch.dtype] = None,
    process_group: torch.distributed.ProcessGroup = None,
    heads_k_stride: int = 1,
    use_zigzag: bool = True,
    kv_slice=None,
) -> torch.Tensor:
    return Llama3MemoryEfficientAttnFunc.apply(
        query,
        key,
        value,
        attn_bias,
        p,
        scale,
        op,
        output_dtype,
        process_group,
        heads_k_stride,
        use_zigzag,
        kv_slice,
    )


class MemoryEfficientAttention(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        # TODO(guanyouhe): 这里 mcore0.13 多了一个 model_comm_pgs，看看你是否需要修改
        model_comm_pgs=None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attention_dropout = attention_dropout
        self.softmax_scale = softmax_scale

        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_query_groups
        )
        self.scaling = None
        if hasattr(self.config, "query_pre_attn_scalar"):
            self.scaling = self.config.query_pre_attn_scalar**-0.5

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_query_groups, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, slen, num_query_groups, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, slen, num_query_groups, n_rep, head_dim
        )
        return hidden_states.reshape(batch, slen, num_query_groups * n_rep, head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: GpatchPackedSeqParams = None,
    ):
        query = rearrange(query, "s b q d -> b s q d")
        key = rearrange(key, "s b h d -> b s h d")
        value = rearrange(value, "s b h d -> b s h d")
        if self.num_key_value_groups > 1:
            key = self.repeat_kv(key, self.num_key_value_groups)
            value = self.repeat_kv(value, self.num_key_value_groups)

        use_zigzag = (
            packed_seq_params.use_zigzag if packed_seq_params is not None else True
        )
        kv_slice = packed_seq_params.kv_slice if packed_seq_params is not None else None
        heads_k_stride = (
            packed_seq_params.heads_k_stride
            if packed_seq_params is not None
            else key.shape[2]
        )
        if isinstance(attention_mask, AttentionBias):
            if attn_mask_type in [AttnMaskType.no_mask]:
                assert not use_zigzag, "no mask should not use zigzag"
            # if packed_seq_params is not None:
            # kv_slice = packed_seq_params.cu_seqlens_kv_padded
        elif attention_mask is not None:
            last_dim = key.shape[1] * self.config.context_parallel_size
            attention_mask = attention_mask[:, :, :, :last_dim]

        assert key.shape[2] % heads_k_stride == 0, f"{key.shape=} {heads_k_stride=}"
        if attention_mask is not None and not isinstance(attention_mask, AttentionBias):
            assert (
                heads_k_stride == attention_mask.shape[1]
            ), "memory_efficient_attention need heads dim"

        if attention_mask is None and attn_mask_type in [AttnMaskType.causal]:
            assert False, "可以支持，但要设置 mask"

        attn_output = llama3_memory_efficient_attention_func(
            query,
            key,
            value,
            attn_bias=attention_mask,
            p=self.attention_dropout if self.attention_dropout else 0.0,
            scale=self.scaling,
            process_group=get_context_parallel_group(),
            heads_k_stride=heads_k_stride,
            use_zigzag=use_zigzag,
            kv_slice=kv_slice,
        )
        attn_output = rearrange(attn_output, "b s h d -> s b (h d)").contiguous()

        return attn_output
