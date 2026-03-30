# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import math
import types
import warnings
from copy import deepcopy
from typing import Dict, Literal, Optional, Union

from megatron.core.extensions.transformer_engine import (
    _get_extra_te_kwargs,
    condition_init_method,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.common.embeddings import (
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, import_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import (
    MLATransformerConfig,
    TransformerConfig,
)
from megatron.core.utils import divide

from .base import (
    MemEstimator,
    _addindent,
    colored,
    cum_mul,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_global_config,
)


class LanguageModelEmbedding(MemEstimator):
    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        num_tokentypes: int = 0,
    ):
        super().__init__()

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = (
            position_embedding_type == "learned_absolute"
        )
        self.num_tokentypes = num_tokentypes
        self.reduce_scatter_embeddings = (
            (not self.add_position_embedding)
            and self.num_tokentypes <= 0
            and self.config.sequence_parallel
        )
        # Word embeddings (parallel).
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
        )

        # TODO if self.add_position_embedding:

        # TODO if self.num_tokentypes > 0:

        self.embedding_dropout = Dropout(self.config.hidden_dropout)

    def num_parameter(self):
        ret = self.word_embeddings.num_parameter()
        ret += self.embedding_dropout.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        ret = self.word_embeddings.num_activation(input_shape)
        input_shape = self.word_embeddings.mock_forward(input_shape)
        ret += self.embedding_dropout.num_activation(input_shape)
        return ret

    def mock_forward(self, input_shape: list[int]):
        input_shape = self.word_embeddings.mock_forward(input_shape)
        return input_shape


class VocabParallelEmbedding(MemEstimator):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size,
            )
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.deterministic_mode = config.deterministic_mode
        self.weight = (self.num_embeddings_per_partition, self.embedding_dim)

    def num_parameter(self):
        return self.weight[0] * self.weight[1]

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape) * self.weight[1]

    def mock_forward(self, input_shape: list[int]):
        return input_shape + [self.weight[1]]


class Dropout(MemEstimator):
    def __init__(self, p=0, *args, **kwargs):
        super().__init__()
        self.p = p

    def num_parameter(self):
        return 0

    def num_activation(self, input_shape: list[int]):
        if self.p == 0:
            return 0
        return cum_mul(input_shape[:])

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class ColumnParallelLinear(MemEstimator):
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer=None,
        grad_output_buffer=None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
        is_mla: bool = False,
    ):
        super().__init__()

        if is_mla and config.sequence_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            output_size = divide(output_size, tp_size)
            parallel_mode = None
            tp_size = 1
            tp_group = None
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self.disable_grad_reduce = disable_grad_reduce

        if is_expert:
            world_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()

        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:
            self.weight = (self.output_size_per_partition, self.input_size)
        else:
            self.weight = (self.output_size_per_partition, self.input_size)

        if bias:
            self.bias = [self.output_size_per_partition]
        else:
            self.bias = None

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and world_size <= 1:
            warnings.warn(
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {world_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = (
            world_size > 1
            and not self.sequence_parallel
            and not self.disable_grad_reduce
        )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

    def num_parameter(self):
        ret = cum_mul(self.weight)
        if self.bias is not None:
            ret += self.bias[0]
        return ret

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:-1]) * self.weight[0]

    def mock_forward(self, input_shape: list[int]):
        try:
            assert self.weight[-1] == input_shape[-1]
        except:

            print(f"{self.weight=} {input_shape=}")
            __import__("ipdb").set_trace()
            raise
        return input_shape[:-1] + [self.weight[0]]


class RowParallelLinear(MemEstimator):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_parallel`, `input_is_parallel` must be `True`"
            )

        # Divide the weight matrix along the last dimension.
        if self.is_expert:
            world_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()

        self.input_size_per_partition = divide(input_size, world_size)

        self.weight = (self.output_size, self.input_size_per_partition)
        if bias:
            self.bias = [self.output_size]
        else:
            self.bias = None

    def num_parameter(self):
        ret = cum_mul(self.weight)
        if self.bias is not None:
            ret += self.bias[0]
        return ret

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:-1]) * self.weight[1]

    def mock_forward(self, input_shape: list[int]):
        assert self.weight[0] == input_shape[-1]
        return input_shape[:-1] + [self.weight[1]]


class RMSNorm(MemEstimator):
    def __init__(self, hidden_size: int, *args, **kwargs):
        super().__init__()
        self.weight = hidden_size

    def num_parameter(self):
        return self.weight

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:])

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class GetBiasDropoutAdd(MemEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def num_parameter(self):
        return 0

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:])

    def mock_forward(self, input_shape: list[int]):
        return input_shape


get_bias_dropout_add = GetBiasDropoutAdd()


class MLP(MemEstimator):

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__()

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
        )

    def num_parameter(self):
        return self.linear_fc1.num_parameter() + self.linear_fc2.num_parameter()

    def num_activation(self, input_shape: list[int]):
        result = 0
        result += self.linear_fc1.num_activation(input_shape)
        intermediate_shape = self.linear_fc1.mock_forward(input_shape)
        result += cum_mul(intermediate_shape) / 2  # activation layer
        self.linear_fc2.num_activation(intermediate_shape)

        return result

    def mock_forward(self, input_shape: list[int]):
        intermediate_shape = self.linear_fc1.mock_forward(input_shape)
        output_shape = self.linear_fc2.mock_forward(intermediate_shape)
        return output_shape


class ModuleList(MemEstimator):
    def __init__(self, modules: list[MemEstimator] = None):
        super().__init__()
        if modules is None:
            modules = []
        self.modules = modules

    def __repr__(self):
        """Return a custom repr for ModuleList that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self.modules]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        stat = (
            "\t/* n_params="
            + colored(f"{self.num_parameter()/1024/1024:.2f}M", "red")
            + "\tn_act="
            + colored(f"{self.num_activation()/1024/1024:.2f}M", "green")
            + " */"
        )
        main_str = self._get_name() + stat + " ("
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def dump(self):
        list_of_reprs = [repr(item) for item in self.modules]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"
        list_of_dumps = [item.dump() for item in self.modules]

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        repeated_blocks_dump = [list_of_dumps[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)
            repeated_blocks_dump(list_of_dumps[i])
        modules = {}
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks_dump):
            key = f"({start_id})"
            if start_id != end_id:
                n = end_id - start_id + 1
                key = f"({start_id}-{end_id}) {n} layers"
            modules[key] = b

        ret = {}
        ret["name"] = self._get_name()
        ret["n_params"] = self.num_parameter()
        ret["n_act"] = self.num_activation()
        if len(modules) > 0:
            ret["modules"] = modules
        return ret

    def append(self, m: MemEstimator):
        self.modules.append(m)

    def __len__(
        self,
    ):
        return self.modules.__len__()

    def num_parameter(self):
        return sum([x.num_parameter() for x in self.modules])

    def num_activation(self, input_shape: list[int]):
        result = 0
        for m in self.modules:
            result += m.num_activation(input_shape)
            input_shape = m.mock_forward(input_shape)

        return result

    def mock_forward(self, input_shape: list[int]):
        for m in self.modules:
            result += m.num_activation(input_shape)
            input_shape = m.mock_forward(input_shape)
        return input_shape


class SequentialMLP(MemEstimator):
    def __init__(self, num_local_experts, config: TransformerConfig, submodules):
        super().__init__()
        self.config = config
        self.add_bias = config.add_bias_linear
        self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts
        self.local_experts = ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def num_parameter(self):
        return self.local_experts.num_parameter()

    def num_activation(self, input_shape: list[int], tokens_per_expert=None):
        # assume all the inputs are routed equally
        all_tokens = input_shape[1]
        result = 0
        for m in self.local_experts.modules:
            result += m.num_activation(
                input_shape[:1]
                + [all_tokens // self.num_local_experts]
                + input_shape[2:]
            )
        return result

    def mock_forward(self, input_shape: list[int], tokens_per_expert=None):
        # assume all the inputs are routed to the first expert
        input_shape = self.local_experts.modules[0].mock_forward(input_shape)
        return input_shape


class TEGroupedMLP(MemEstimator):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules):
        super().__init__()
        self.config = config
        self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.moe_ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.num_local_experts,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name="fc1",
        )

        self.activation_func = self.config.activation_func

        self.activation_recompute = (
            self.config.recompute_granularity == "selective"
            and "moe_act" in self.config.recompute_modules
        )
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.num_local_experts,
            self.config.moe_ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name="fc2",
        )
        # TODO if self.config.fp8:

    def num_parameter(self):
        ret = self.linear_fc1.num_parameter()
        ret += self.linear_fc2.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int], tokens_per_expert=None):
        ret = 0
        if not self.activation_recompute:
            ret += self.linear_fc1.num_activation(input_shape)
        input_shape = self.linear_fc1.mock_forward(input_shape)

        # activation
        if not self.activation_recompute:
            ret += cum_mul(input_shape) / 2  # swiglu or gelu
        input_shape = deepcopy(input_shape)
        input_shape[-1] //= 2

        self.linear_fc2.num_activation(input_shape)
        return ret

    def mock_forward(self, input_shape: list[int], tokens_per_expert=None):
        # assume all the inputs are routed to the first expert
        input_shape = self.local_experts.modules[0].mock_forward(input_shape)
        return input_shape


class TEGroupedLinear(MemEstimator):
    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: str,
        config: ModelParallelConfig,
        init_method,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        super().__init__()
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = (
            self.config.disable_parameter_transpose_cache
        )

        extra_kwargs = _get_extra_te_kwargs(config)
        extra_kwargs["ub_name"] = tp_comm_buffer_name

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if self.expert_parallel:
            extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

        # For MoE models, the comms between TP and EP group is explicitly handled by
        # MoE token dispatcher. So we disable comms by making TE agnostic of model parallel.
        self.explicit_expert_comm = is_expert and (
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if is_expert:
            tp_size = get_expert_tensor_parallel_world_size()
        else:
            tp_size = get_tensor_model_parallel_world_size()
        if self.explicit_expert_comm:
            if parallel_mode == "column":
                output_size = divide(output_size, tp_size)
            elif parallel_mode == "row":
                input_size = divide(input_size, tp_size)
            parallel_mode = None
            tp_size = 1
        assert not bias, "bias is not considered for now"

        self.num_gemms = num_gemms
        self.input_size = input_size
        self.output_size = output_size

    def num_parameter(self):
        ret = self.num_gemms * self.input_size * self.output_size
        return ret

    def num_activation(self, input_shape: list[int], tokens_per_expert=None):
        ret = cum_mul(self.mock_forward(input_shape))
        return ret

    def mock_forward(self, input_shape: list[int], tokens_per_expert=None):
        return input_shape[:-1] + [self.output_size]


class TEColumnParallelGroupedLinear(TEGroupedLinear):
    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )


class TERowParallelGroupedLinear(TEGroupedLinear):
    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
    ):

        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )


class SharedExpertMLP(MLP):
    """
    MLP layer for Shared Experts.
    """

    def __init__(self, config: TransformerConfig, spec: ModuleSpec):
        config = deepcopy(config)
        assert (
            config.add_bias_linear == False
        ), "bias is not supported in the shared experts, "
        "please set '--disable-bias-linear' instead."

        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config=config, submodules=spec.submodules)

        self.use_shared_expert_gate = spec.params.get("gate", False)
        if self.use_shared_expert_gate:
            assert False, "use_shared_expert_gate is not Implemented"
            # self.gate_weight = torch.nn.Parameter(torch.empty((1, self.config.hidden_size)))
            # if config.perform_initialization:
            #     if get_cuda_rng_tracker().is_initialized():
            #         with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
            #             config.init_method(self.gate_weight)
            # else:
            #     config.init_method(self.gate_weight)
            # self.gate_weight.data = self.gate_weight.data.to(dtype=config.params_dtype)
            # setattr(self.gate_weight, 'sequence_parallel', self.config.sequence_parallel)
        else:
            self.gate_weight = None


class TransformerBlock(MemEstimator):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ):
        super().__init__()
        self.config = config

        self.submodules = _get_block_submodules(config, spec, vp_stage)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        self.cuda_graphs = {}
        self.current_microbatch = -1
        self.input_tensor = None
        self.checkpoint_core_attention = (
            self.config.recompute_granularity == "selective"
            and "core_attn" in self.config.recompute_modules
        )

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)
        self.tp_only_amax_red = config.tp_only_amax_red

    def _build_layers(self):
        def build_layer(layer_spec, layer_number):
            return build_module(
                layer_spec,
                config=self.config,
                layer_number=layer_number,
                vp_stage=self.vp_stage,
            )

        # offset is implicit in TransformerLayer
        self.layers = ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

    def num_parameter(self):
        ret = self.layers.num_parameter()
        if self.final_layernorm is not None:
            ret += self.final_layernorm.num_parameter()

        return ret

    def num_activation(self, input_shape: list[int]):
        result = self.layers.num_activation(input_shape)
        if self.final_layernorm is not None:
            result += self.final_layernorm.num_activation(input_shape)
        return result

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class TopKRouter(MemEstimator):

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None

    def num_parameter(self):
        return 0

    def num_activation(self, input_shape: list[int]):
        result = cum_mul(input_shape) * 2  # sinkhorn and sinkhorn activation
        return result

    def mock_forward(self, input_shape: list[int]):
        return input_shape[:-1] + [self.topk]


class MoELayer(MemEstimator):

    def __init__(
        self, config: TransformerConfig, submodules=None, layer_number: int = None
    ):
        super().__init__()
        self.config = config
        self.submodules = submodules
        self.moe_layer_recompute = config.moe_layer_recompute

        self.expert_parallel_size = get_expert_model_parallel_world_size()
        assert (
            self.expert_parallel_size > 0
        ), "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = (
            self.config.num_moe_experts // self.expert_parallel_size
        )
        local_expert_indices_offset = (
            get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.moe_layer_recompute = (
            config.recompute_granularity == "selective"
            and "moe" in config.recompute_modules
        )

        self.router = TopKRouter(config=self.config)
        self.use_shared_expert = (
            self.config.moe_shared_expert_intermediate_size is not None
        )
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(
            map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices)
        )

        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number
        # Initialize experts
        self.experts = build_module(
            self.submodules.experts, self.num_local_experts, self.config
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(
                self.config, self.submodules.shared_experts
            )
            # if self.shared_expert_overlap:
            #     self.token_dispatcher.set_shared_experts(self.shared_experts)

    def num_parameter(self):
        ret = self.experts.num_parameter() + self.router.num_parameter()
        if self.use_shared_expert:
            ret += self.shared_experts.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        if self.moe_layer_recompute:
            return 0
        tp_size = get_tensor_model_parallel_world_size()
        etp_size = get_expert_tensor_parallel_world_size()
        new_input_shape = deepcopy(input_shape)
        new_input_shape[1] = input_shape[1] // tp_size * etp_size
        input_shape = new_input_shape

        result = self.router.num_activation(input_shape)
        result += cum_mul(input_shape) * self.router.topk  # token dispatcher
        moe_input_shape_average = deepcopy(input_shape)
        moe_input_shape_average[1] = int(moe_input_shape_average[1] * self.router.topk)

        result += self.experts.num_activation(moe_input_shape_average)
        if self.use_shared_expert:
            result += self.shared_experts.num_activation(input_shape)

        if self.config.moe_layer_recompute:
            result = cum_mul(input_shape) * 2
        return result

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class IdentityOp(MemEstimator):
    def num_parameter(self):
        return 0

    def num_activation(self, input_shape: list[int]):
        return 0

    def mock_forward(self, input_shape: list[int]):
        return input_shape


IdentityFuncOp = IdentityOp
TERowParallelLinear = RowParallelLinear
TEColumnParallelLinear = ColumnParallelLinear
TELayerNormColumnParallelLinear = ColumnParallelLinear


class TEDotProductAttention(MemEstimator):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__()
        self.config = config

    def num_parameter(self):
        return 0

    def num_activation(
        self, q_shape: list[int], k_shape: list[int], v_shape: list[int]
    ):
        bs, seqs, heads, dim = q_shape
        if self.config.multi_latent_attention and False:
            result = bs * seqs * seqs * heads
        else:
            bs, seqs, heads, dim = k_shape
            result = (
                bs * seqs * dim * heads * 2  # * self.config.tensor_model_parallel_size
            )  # flash attention
            if self.config.context_parallel_size > 1:
                result *= 2
        return result

    def mock_forward(
        self,
        hidden_size: int,
        q_shape: list[int],
        k_shape: list[int],
        v_shape: list[int],
    ):
        seqs, bs, heads, dim = q_shape
        return [seqs, bs, hidden_size]


class TransformerLayer(MemEstimator):
    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__()
        self.config = config

        if config.enable_cuda_graph and self.training:
            assert (
                not config.cpu_offloading and config.recompute_granularity is None
            ), "Cudagraphs not supported"
            self.cudagraph_manager = CudaGraphManager()

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage
        )
        self.hidden_dropout = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention, config=self.config, layer_number=layer_number
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention, config=self.config, layer_number=layer_number
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(
            submodules.cross_attn_bda, config=self.config
        )

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 8: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == "selective":
            if "layernorm" in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp):
                    self.recompute_input_layernorm = True
                if not isinstance(self.pre_mlp_layernorm, IdentityOp):
                    self.recompute_pre_mlp_layernorm = True
            if "mlp" in self.config.recompute_modules:

                if not isinstance(self.mlp, MoELayer):
                    self.recompute_mlp = True

    def num_parameter(self):
        result = self.input_layernorm.num_parameter()
        result += self.self_attention.num_parameter()
        result += self.pre_cross_attn_layernorm.num_parameter()
        result += self.cross_attention.num_parameter()
        result += self.cross_attn_bda.num_parameter()
        result += self.pre_mlp_layernorm.num_parameter()
        result += self.mlp.num_parameter()

        return result

    def num_activation(self, input_shape: list[int]):
        result = 0
        result += self.self_attention.num_activation(input_shape)
        if not self.recompute_mlp:
            result += self.mlp.num_activation(input_shape)
        # __import__('ipdb').set_trace()
        # sequence parallel
        if self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            input_shape = deepcopy(input_shape)
            input_shape[1] /= self.config.tensor_model_parallel_size
        if not self.recompute_input_layernorm:
            result += self.input_layernorm.num_activation(input_shape)
        if not self.recompute_pre_mlp_layernorm:
            result += self.pre_mlp_layernorm.num_activation(input_shape)
        result += self.self_attn_bda.num_activation(input_shape)
        result += self.mlp_bda.num_activation(input_shape)
        return result

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class SelfAttention(MemEstimator):

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        layer_number: int,
        attn_mask_type,
    ):
        super().__init__()

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = ""

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = (
            self.config.kv_channels * self.config.num_attention_heads
        )
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        world_size = get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.config.num_attention_heads, world_size
        )
        self.num_query_groups_per_partition = divide(
            self.config.num_query_groups, world_size
        )
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
        )
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )
        self.checkpoint_core_attention = (
            self.config.recompute_granularity == "selective"
        )

    def num_parameter(self):
        result = 0
        result += self.core_attention.num_parameter()
        result += self.linear_proj.num_parameter()
        result += self.linear_qkv.num_parameter()
        if self.q_layernorm is not None:
            result += self.q_layernorm.num_parameter()
        if self.k_layernorm is not None:
            result += self.k_layernorm.num_parameter()

        return result

    def num_activation(self, input_shape: list[int]):
        ret = 0
        ## in estimator: act(linear) = 1.5*cum_mul(input_shape)
        ## in reality: act(linear) = cum_mul(input_shape), act(rotary) = cum_mul(input_shape), act(attn_forward_func_with_cp) = cum_mul(input_shape)
        # ret += self.linear_qkv.num_activation(input_shape)
        mixed_qkv_shape = self.linear_qkv.mock_forward(input_shape)
        new_tensor_shape = mixed_qkv_shape[:-1] + [
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        ]
        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        # [sq, b, ng, (np/ng + 2) * hn]
        # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        q_shape = new_tensor_shape[:-1] + [split_arg_list[0]]
        k_shape = new_tensor_shape[:-1] + [split_arg_list[1]]
        v_shape = new_tensor_shape[:-1] + [split_arg_list[2]]
        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        q_shape = (
            q_shape[:2]
            + [cum_mul(q_shape[-2:]) // self.hidden_size_per_attention_head]
            + [self.hidden_size_per_attention_head]
        )

        if not self.checkpoint_core_attention:
            ret += self.core_attention.num_activation(q_shape, k_shape, v_shape)
        ret += self.linear_proj.num_activation(input_shape)
        ## in reality: act(linear) = cum_mul(input_shape), act(rotary) = cum_mul(input_shape), act(attn_forward_func_with_cp) = cum_mul(input_shape)
        ret += self.linear_proj.num_activation(input_shape) * 3

        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class Linear(MemEstimator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:

        super().__init__()
        self.weight = (in_features, out_features)

    def num_parameter(self):
        return self.weight[0] * self.weight[1]

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:-1]) * self.weight[1]

    def mock_forward(self, input_shape: list[int]):
        return input_shape[:-1] + [self.weight[1]]


class MLASelfAttention(MemEstimator):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ) -> None:

        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = "self"
        self.world_size = get_tensor_model_parallel_world_size()
        # assert (
        #     world_size == 1
        # ), "MLA is not supported with Tensor Parallelism yet, \
        # use Expert Parallelism and Pipeline Parallelism for better performance."

        self.query_projection_size = (
            self.config.v_head_dim * self.config.num_attention_heads
        )

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        # Per attention head and per partition values.
        world_size = get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.config.num_attention_heads, world_size
        )
        self.num_query_groups_per_partition = divide(
            self.config.num_query_groups, world_size
        )
        # TODO Rotary Embedding
        # self.rotary_pos_emb = YarnRotaryEmbedding(
        #     self.config.qk_pos_emb_head_dim,
        #     rotary_base=self.config.rotary_base,
        #     scaling_factor=self.config.rotary_scaling_factor,
        #     original_max_position_embeddings=self.config.max_position_embeddings,
        #     beta_fast=self.config.beta_fast,
        #     beta_slow=self.config.beta_slow,
        #     mscale=self.config.mscale,
        #     mscale_all_dim=self.config.mscale_all_dim,
        # )

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
        )

        if self.config.q_lora_rank is None:
            # Not projectiing query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                is_mla=True,
            )

        else:
            self.linear_q_down_proj = Linear(
                self.config.hidden_size, self.config.q_lora_rank, bias=False
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                is_mla=True,
            )
        self.linear_kv_down_proj = Linear(
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            bias=False,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads
            * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            is_mla=True,
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == "selective"
        )

    def num_parameter(self):
        result = 0
        result += self.core_attention.num_parameter()
        result += self.linear_proj.num_parameter()
        if self.config.q_lora_rank is None:
            result += self.linear_q_proj.num_parameter()
        else:
            result += self.linear_q_down_proj.num_parameter()
            result += self.linear_q_up_proj.num_parameter()
        result += self.linear_kv_down_proj.num_parameter()
        result += self.linear_kv_up_proj.num_parameter()
        result += self.kv_layernorm.num_parameter()
        if self.config.q_lora_rank is not None:
            result += self.q_layernorm.num_parameter()

        return result

    def num_activation(self, input_shape: list[int]):
        q_len, bsz, _ = input_shape
        ret = 0
        if self.config.q_lora_rank is not None:
            ret += self.linear_q_down_proj.num_activation(input_shape)
            q_compressed_shape = self.linear_q_down_proj.mock_forward(input_shape)
            ret += self.q_layernorm.num_activation(q_compressed_shape)
            ret += self.linear_q_up_proj.num_activation(q_compressed_shape)
            q_shape = self.linear_q_up_proj.mock_forward(q_compressed_shape)
        else:
            # hidden_states:[s, b, 2048], q: [s, b, n * 192]
            ret += self.linear_q_proj.num_activation(input_shape)
            q_shape = self.linear_q_proj.mock_forward(input_shape)

        # kv_combined: [s, b, 576]
        ret += self.linear_kv_down_proj.num_activation(input_shape)
        kv_combined_shape = self.linear_kv_down_proj.mock_forward(input_shape)
        # kv_compressed:[s, b, 512], k_pos_emb: [s, b, 64]
        kv_compressed_shape = kv_combined_shape[:-1] + [self.config.kv_lora_rank]

        # kv: [s, b, 2048]
        ret += self.kv_layernorm.num_activation(kv_compressed_shape)
        ret += self.linear_kv_up_proj.num_activation(kv_compressed_shape)

        q_shape = [q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim]
        k_shape = [q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim]
        v_shape = [
            q_len,
            bsz,
            self.num_attention_heads_per_partition,
            self.config.v_head_dim,
        ]

        if not self.checkpoint_core_attention:
            ret += self.core_attention.num_activation(q_shape, k_shape, v_shape)

        ret += self.linear_proj.num_activation(input_shape)

        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class TENorm:
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs, te

        if config.normalization == "LayerNorm":
            # TODO layernorm
            pass
        elif config.normalization == "RMSNorm":
            assert hasattr(
                te.pytorch, "RMSNorm"
            ), "Transformer-Engine >= v0.11 required to use this feature"
            instance = RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        else:
            raise Exception("Only LayerNorm and RMSNorm are curently supported")

        return instance


def build_module(
    spec_or_module: Union[ModuleSpec, type], *args, **kwargs
) -> MemEstimator:
    """replace module with MemEstimators"""
    if isinstance(spec_or_module, types.FunctionType):
        return globals()[spec_or_module.__name__]

    if isinstance(spec_or_module, ModuleSpec) and isinstance(
        spec_or_module.module, types.FunctionType
    ):
        assert False
        return spec_or_module.module

    if isinstance(spec_or_module, type):
        module = spec_or_module
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):
        module = spec_or_module.module
    else:
        module = import_module(spec_or_module.module)

    if isinstance(module, types.FunctionType):
        assert False
        return module

    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:
        kwargs["submodules"] = spec_or_module.submodules

    try:
        module = globals()[module.__name__]
        return module(
            *args,
            **spec_or_module.params if hasattr(spec_or_module, "params") else {},
            **kwargs,
        )
    except Exception as e:
        # import ipdb

        # ipdb.set_trace()
        # improve the error message since we hide the module name in the line above
        import sys

        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )


from megatron.core.transformer.transformer_block import (
    BaseTransformerLayer,
    LayerNormImpl,
    TransformerBlockSubmodules,
)


def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


from megatron.core.transformer.transformer_block import get_num_layers_to_build


def ___get_num_layers_to_build(config: TransformerConfig) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):

        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = get_pipeline_model_parallel_world_size()

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        assert (
            layers_to_distribute % pipeline_stages_left == 0
        ), "With uneven pipelineing the left over layers must be divisible by left over stages"
        num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.
        if (
            is_pipeline_first_stage(ignore_virtual=True)
            and config.num_layers_in_first_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if (
            is_pipeline_last_stage(ignore_virtual=True)
            and config.num_layers_in_last_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert (
            num_layers % config.pipeline_model_parallel_size == 0
        ), "num_layers should be divisible by pipeline_model_parallel_size"
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    # if get_virtual_pipeline_model_parallel_world_size() is not None:
    #     # Interleaved pipeline parallelism:
    #     # Number of layers in each model chunk is the number of layers in the stage,
    #     # divided by the number of model chunks in a stage.
    #     # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
    #     # layers to stages like (each list is a model chunk):
    #     # Stage 0: [0]  [2]  [4]  [6]
    #     # Stage 1: [1]  [3]  [5]  [7]
    #     # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
    #     # layers to stages like (each list is a model chunk):
    #     # Stage 0: [0, 1]  [4, 5]
    #     # Stage 1: [2, 3]  [6, 7]
    #     vp_size = get_virtual_pipeline_model_parallel_world_size()

    #     assert (
    #         num_layers_per_pipeline_rank % vp_size == 0
    #     ), "num_layers_per_pipeline_rank should be divisible by vp_size"
    #     num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

    #     num_layers_to_build = num_layers_per_virtual_rank

    # else:
    #     # Non-interleaved pipeline parallelism:
    #     # Each stage gets a contiguous set of layers.
    #     num_layers_to_build = num_layers_per_pipeline_rank
    num_layers_to_build = num_layers_per_pipeline_rank
    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if is_pipeline_first_stage() and config.account_for_embedding_in_pipeline_split:
        num_layers_to_build -= 1
        assert (
            num_layers_to_build >= 0
        ), "Not enough layers in the first virtual pipeline stage"

    if is_pipeline_last_stage() and config.account_for_loss_in_pipeline_split:
        num_layers_to_build -= 1
        assert (
            num_layers_to_build >= 0
        ), "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


from megatron.core.transformer.enums import LayerType


def get_transformer_layer_offset(
    config: TransformerConfig, vp_stage: Optional[int] = None
):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    pipeline_rank = get_pipeline_model_parallel_rank()

    if config.pipeline_model_parallel_size > 1:

        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        elif (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = (
                    middle_num_layers // vp_size
                )

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pipeline_rank == 0:
                    offset = vp_stage * total_virtual_chunks
                else:
                    offset = (
                        vp_stage * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + (pipeline_rank - 1)
                        * (
                            num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                            // middle_pipeline_stages
                        )
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = (
                        middle_num_layers // middle_pipeline_stages
                    )
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank
                    if config.num_layers_in_first_pipeline_stage is None
                    else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = (
                num_layers // config.pipeline_model_parallel_size
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (
                    pipeline_rank * num_layers_per_virtual_rank
                )

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not is_pipeline_first_stage(
                        ignore_virtual=False, vp_stage=vp_stage
                    )
                ):
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not is_pipeline_first_stage(
                        ignore_virtual=False, vp_stage=vp_stage
                    )
                ):
                    offset -= 1
    else:
        offset = 0
    return offset
