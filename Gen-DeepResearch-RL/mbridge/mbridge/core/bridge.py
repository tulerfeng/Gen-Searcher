# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import os
from abc import ABC
from typing import Callable, Generator

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import ModelType
from transformers import AutoConfig
from transformers.utils.hub import cached_file

from .parallel_states import ParallelStates
from .safetensor_io import SafeTensorIO
from .util import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    get_model,
    unwrap_model,
)


class Bridge(ABC):
    """
    Base model bridge class.

    This class implements the core functionality to bridge between
    Hugging Face models and Megatron-Core optimized implementations.
    """

    def __init__(
        self,
        hf_config: AutoConfig,
        dtype: torch.dtype = torch.bfloat16,
        parallel_states: ParallelStates = None,
        make_vocab_size_divisible_by: int = None,
    ):
        """
        Initialize a bridge instance.

        Args:
            hf_config: Hugging Face model configuration
            dtype: Data type for model parameters
            parallel_states: Parallel processing states, or None to use default
        """
        self.hf_config = hf_config
        self.extra_args = {}
        self.dtype = dtype
        self.mpu = parallel_states
        if self.mpu is None:
            self.mpu = ParallelStates.get_parallel_state()
        self.config = self._build_config()
        self.safetensor_io = None

        self._adjust_mapping_for_shared_weights()
        # Pad the vocab size to be divisible by this value.
        # This is added for computational efficieny reasons.
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.vocab_size = None
        self.padded_vocab_size = None

        # Some moe models require multiple weights to be combined into one,
        # such as qwen3vl. It will cache it into this buff until all weights are collected.
        self.export_weights_buff = {}

    def get_model(
        self,
        weight_path: str = None,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        fp16: bool = False,
        bf16: bool = True,
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
        post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]] = [],
        extra_provider_args: dict = {},
        **kwargs,
    ):
        """
        Get a model instance.

        Args:
            weight_path: Path to model weights or Hugging Face model identifier
            model_type: Type of model to create
            wrap_with_ddp: Whether to wrap with DDP
            fp16: Whether to use FP16 precision
            bf16: Whether to use BF16 precision
            encoder_pipeline_model_parallel_size: Size of encoder pipeline parallelism
            use_torch_fsdp2: Whether to use PyTorch FSDP 2.0
            use_custom_fsdp: Whether to use custom FSDP
            use_precision_aware_optimizer: Whether to use precision-aware optimizer
            use_cpu_initialization: Whether to initialize on CPU
            init_model_with_meta_device: Whether to initialize with meta device
            overlap_param_gather_with_optimizer_step: Whether to overlap parameter gathering
            data_parallel_random_init: Whether to use random initialization in data parallel
            optimizer_config: Optimizer configuration
            post_model_creation_callbacks: List of callbacks to be called after model creation
            extra_provider_args: Additional arguments for the model provider
            **kwargs: Additional arguments

        Returns:
            Model instance
        """
        # share_embeddings_and_output_weights = getattr(
        #     self.hf_config, "tie_word_embeddings", False
        # )
        # if (
        #     share_embeddings_and_output_weights
        #     and self.mpu.vpp_size
        #     and self.mpu.vpp_size > 1
        # ):
        #     raise ValueError("tie_word_embeddings is not supported for VPP > 1")
        model = get_model(
            self._model_provider(
                post_model_creation_callbacks,
                **extra_provider_args,
            ),
            model_type=model_type,
            wrap_with_ddp=wrap_with_ddp,
            fp16=fp16,
            bf16=bf16,
            virtual_pipeline_model_parallel_size=self.mpu.vpp_size,
            encoder_pipeline_model_parallel_size=encoder_pipeline_model_parallel_size,
            use_torch_fsdp2=use_torch_fsdp2,
            use_custom_fsdp=use_custom_fsdp,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
            use_cpu_initialization=use_cpu_initialization,
            init_model_with_meta_device=init_model_with_meta_device,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            data_parallel_random_init=data_parallel_random_init,
            ddp_config=ddp_config,
            optimizer_config=optimizer_config,
            **kwargs,
        )
        if weight_path:
            self.load_weights(model, self._get_actual_hf_path(weight_path))
        return model

    def _get_safetensor_io(self, weights_path: str):
        return SafeTensorIO(self._get_actual_hf_path(weights_path))

    def _get_mcore_config_by_name(self, mcore_weights_name: str):
        return self.config

    def load_weights(
        self,
        models: list[torch.nn.Module],
        weights_path: str,
        memory_efficient: bool = False,
    ) -> None:
        """
        Load weights from a Hugging Face model into a Megatron-Core model.

        Args:
            models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)
            weights_path: Path to the weights file or Hugging Face model identifier
        """
        self.safetensor_io = self._get_safetensor_io(weights_path)

        for i, model in enumerate(models):
            # map local weight names to global weight names
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            # map local weight names to huggingface weight names
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }
            # only tp_rank0/etp_rank0 load from disk, others load from tp_rank0/etp_rank0
            to_load_from_disk = []
            for local_name, hf_names in local_to_hf_map.items():
                if ".mlp.experts.linear_fc" in local_name:
                    if self.mpu.etp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                    else:
                        # special case for lm_head.weight
                        # if make value model, every tp rank will load lm_head.weight
                        if "lm_head.weight" in hf_names:
                            to_load_from_disk.extend(hf_names)

            # load huggingface weights
            if not memory_efficient:
                hf_weights_map = self.safetensor_io.load_some_hf_weight(
                    to_load_from_disk
                )

            # import mcore weights
            for local_name, hf_names in local_to_hf_map.items():
                param = model.state_dict()[local_name]
                # hf format to mcore format
                if set(to_load_from_disk) & set(hf_names):
                    if not memory_efficient:
                        hf_weights = [hf_weights_map[x] for x in hf_names]
                    else:
                        hf_weights = [
                            self.safetensor_io.load_one_hf_weight(x) for x in hf_names
                        ]
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None
                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (
                        mcore_weight is None or mcore_weight.shape[0] != 1
                    ):
                        # skip lm_head.weight when the model is a value model
                        continue

                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in local_name:
                    # split mcore weights across etp
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous()
                            for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    # split mcore weights across tp
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.tp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous()
                            for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                # load
                param.copy_(param_to_load)

    def _save_weights_fast(
        self,
        per_tensor_generator: Generator[tuple[str, torch.Tensor], None, None],
        weights_path: str,
    ) -> None:
        dp_rank = parallel_state.get_data_parallel_rank()
        save_rank = dp_rank * self.mpu.cp_size + self.mpu.cp_rank
        dp_cp_size = self.mpu.cp_size * parallel_state.get_data_parallel_world_size()

        # save dense weight will faster
        param_cnt = 0
        for hf_weight_name, tensor in per_tensor_generator:
            # only tp_rank 0 should write
            if param_cnt % dp_cp_size == save_rank and self.mpu.tp_rank == 0:
                self.safetensor_io.save_tmp_hf_weight(
                    hf_weight_name, tensor, weights_path
                )
            param_cnt += 1
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.safetensor_io.save_hf_weight_merge(
            weights_path,
            rank,
            world_size,
        )

        if 0 == rank:
            self.safetensor_io.save_index(weights_path)
            self.hf_config.save_pretrained(weights_path)
        torch.distributed.barrier()
        return

    def save_weights(
        self,
        models: list,
        weights_path: str,
        memory_efficient: bool = False,
        distributed_filesystem: bool = False,
    ) -> None:
        """
        Save weights from a Megatron-Core model into a Hugging Face model.
        when `distributed_filesystem` is ture, `weights_path` should be distributed file system
        """
        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        rank = torch.distributed.get_rank() if is_distributed else 0
        if not os.path.exists(weights_path):
            os.makedirs(weights_path, exist_ok=True)
        per_tensor_generator = self.export_weights(models, distributed_filesystem)

        if distributed_filesystem:
            assert (
                memory_efficient
            ), f"distributed_filesystem should use with memory_efficient"
            assert is_distributed, f"distributed_filesystem should use in distributed"
            return self._save_weights_fast(per_tensor_generator, weights_path)

        if rank != 0:
            for _, _ in per_tensor_generator:
                pass
            return
        if rank == 0:
            if memory_efficient:
                self.safetensor_io.save_hf_weight_memory_efficient(
                    per_tensor_generator, weights_path
                )
            else:
                self.safetensor_io.save_hf_weight(
                    per_tensor_generator,
                    weights_path,
                    self._get_hf_shared_weight_keys(),
                )
            self.safetensor_io.save_index(weights_path)
            self.hf_config.save_pretrained(weights_path)
        return

    def set_extra_args(self, **kwargs):
        """
        Set additional configuration arguments.

        Args:
            **kwargs: Key-value pairs of additional arguments
        """
        self.extra_args.update(kwargs)
        self.config = self._build_config()

    def export_weights(
        self, models: list[torch.nn.Module], distributed_filesystem: bool = False
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        assert (
            len(self.export_weights_buff) == 0
        ), f"should be empty {self.export_weights_buff=}"
        models = [unwrap_model(model) for model in models]

        def get_model_chunk_generator():
            for model in models:
                existing_keys = set()
                for name, param in model.named_parameters():
                    existing_keys.add(name)
                    yield name, param

                # note
                # there is a bug in megatron GPTModel
                # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
                # for now we patch it by adding those keys to extra_keys.
                extra_keys = [
                    x
                    for x in model.state_dict().keys()
                    if "_extra_state" not in x
                    and "expert_bias" in x
                    and x not in existing_keys
                ]
                for name in extra_keys:
                    yield name, model.state_dict()[name].to(torch.cuda.current_device())

        weights_names = []
        for vpp_rank, model in enumerate(models):
            existing_keys = set()
            for name, param in model.named_parameters():
                existing_keys.add(name)
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))
            extra_keys = [
                x
                for x in model.state_dict().keys()
                if "_extra_state" not in x
                and "expert_bias" in x
                and x not in existing_keys
            ]
            for name in extra_keys:
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))

        weights_names_all_pp = [None] * self.mpu.pp_size
        torch.distributed.all_gather_object(
            object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group
        )
        weights_names_all_pp = sum(weights_names_all_pp, [])
        model_chunk_generator = get_model_chunk_generator()
        local_to_global_maps = [
            self._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)
            for model in models
        ]
        for iter_pp_rank, iter_vpp_rank, iter_name in weights_names_all_pp:
            local_to_global_map = local_to_global_maps[iter_vpp_rank]
            if iter_pp_rank == self.mpu.pp_rank:
                try:
                    name, param = next(model_chunk_generator)
                except StopIteration:
                    name, param = None, None
                name = local_to_global_map[iter_name]
            else:
                name, param = None, None
                if distributed_filesystem:
                    continue

            if distributed_filesystem:
                assert iter_pp_rank == self.mpu.pp_rank
                broad_pp_param = param
            else:
                name = broadcast_str_from_megatron_pp(name)
                broad_pp_param = broadcast_from_megatron_pp(param)

            # EP
            if ".mlp.experts.linear_fc" in name and self.mpu.ep_size > 1:
                num_experts = self.config.num_moe_experts
                num_experts_per_rank = num_experts // self.mpu.ep_size
                infer_params = [
                    torch.empty_like(broad_pp_param) for _ in range(self.mpu.ep_size)
                ]
                torch.distributed.all_gather(
                    infer_params, broad_pp_param, group=self.mpu.ep_group
                )

                name_prefix, local_expert_id = name.split(".weight")
                local_expert_id = int(local_expert_id)
                global_expert_ids = [
                    num_experts_per_rank * ep_rank + local_expert_id
                    for ep_rank in range(self.mpu.ep_size)
                ]
                global_expert_names = [
                    f"{name_prefix}.weight{expert_id}"
                    for expert_id in global_expert_ids
                ]

                for name, param in zip(global_expert_names, infer_params):
                    if self.mpu.etp_size > 1:
                        # gather etp
                        etp_params = [
                            torch.empty_like(param) for _ in range(self.mpu.etp_size)
                        ]
                        torch.distributed.all_gather(
                            etp_params, param, group=self.mpu.etp_group
                        )
                        params = etp_params
                    else:
                        params = [param]

                    merge_params = self._weight_merge_across_tp(
                        name, params, broad_pp_param
                    )
                    converted_names, converted_params = self._weight_to_hf_format(
                        name, merge_params
                    )
                    # Some moe models require multiple weights to be merge into one, such as qwen3vl
                    if len(converted_names) == 0:
                        continue

                    yield from zip(converted_names, converted_params)
                continue

            # TP
            if (
                hasattr(broad_pp_param, "tensor_model_parallel")
                and broad_pp_param.tensor_model_parallel
            ):
                # allocate a new tensor with proper size
                if self.mpu.tp_size <= 1:
                    infer_params = [broad_pp_param]
                else:
                    infer_params = [
                        torch.empty_like(broad_pp_param)
                        for _ in range(self.mpu.tp_size)
                    ]
                    torch.distributed.all_gather(
                        infer_params, broad_pp_param, group=self.mpu.tp_group
                    )
                infer_params = self._weight_merge_across_tp(
                    name, infer_params, broad_pp_param
                )
            else:
                infer_params = broad_pp_param

            converted_names, converted_params = self._weight_to_hf_format(
                name, infer_params
            )
            # Some moe models require multiple weights to be merge into one, such as qwen3vl
            if len(converted_names) == 0:
                continue

            yield from zip(converted_names, converted_params)

    def _build_config(self):
        """
        Build the configuration for the model.
        This method must be implemented by subclasses.

        Returns:
            Configuration object for the model

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        """
        Create a model provider function.
        This method must be implemented by subclasses.

        Args:
            post_model_creation_callbacks: List of callbacks to be called after model creation

        Returns:
            Function that provides a model

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """

        def provider(pre_process, post_process):
            raise NotImplementedError("Subclasses must implement this method")
            model = None
            return model

        return provider

    def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module, consider_ep: bool = True
    ) -> dict[str, str]:
        """
        Map local weight names to global weight names, supporting VPP and EP.

        Args:
            model: The model instance

        Returns:
            dict: Mapping from local weight names to global weight names
        """
        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "decoder"):
            for idx, layer in enumerate(model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [
            k for k in model.state_dict().keys() if "_extra_state" not in k
        ]
        ret = {}
        for param_name in all_param_names:
            keyword = "decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(
                    f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
                )
            else:
                ret[param_name] = param_name

        # ep
        if self.mpu.ep_size > 1 and consider_ep:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            local_expert_to_global_expert = {
                i: i + num_experts_per_rank * self.mpu.ep_rank
                for i in range(num_experts_per_rank)
            }
            for k in ret.keys():
                v = ret[k]
                if ".mlp.experts.linear_fc" in v:
                    name_prefix, local_expert_id = v.split(".weight")
                    global_expert_idx = local_expert_to_global_expert[
                        int(local_expert_id)
                    ]
                    ret[k] = f"{name_prefix}.weight{global_expert_idx}"

        return ret

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        """
        Map attention weight names from MCore to Hugging Face.

        Args:
            name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._ATTENTION_MAPPING.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number) for x in mapping_names]
                )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _OTHER_MAPPING = {}

    def _adjust_mapping_for_shared_weights(self):
        pass

    def _get_hf_shared_weight_keys(self) -> list[str]:
        return []

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """
        Map MLP weight names from MCore to Hugging Face.

        Args:
            name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number) for x in mapping_names]
                )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_other(self, name: str) -> list[str]:
        """
        Map OTHER(In addition to attention/mlp/direct) weight names from MCore to Hugging Face.

        Args:
            name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._OTHER_MAPPING.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number) for x in mapping_names]
                )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if ".self_attention." in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            return self._weight_name_mapping_other(mcore_weights_name)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        Export MCore weights to Hugging Face format.

        Takes MCore weight names and tensor, outputs Hugging Face weight names and tensors.
        Due to MCore's runtime optimizations involving weight merging, output can be a list.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor

        Returns:
            tuple: (hf_names, hf_weights) - lists of Hugging Face weight names and tensors

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        if len(hf_names) == 1:
            # pad embeding and output layer
            if self.make_vocab_size_divisible_by is not None and (
                "embedding.word_embeddings.weight" in mcore_weights_name
                or "output_layer.weight" in mcore_weights_name
            ):
                assert mcore_weights.shape[0] == self.padded_vocab_size
                assert self.vocab_size is not None

                return [hf_names[0]], [mcore_weights[: self.vocab_size]]

            return [hf_names[0]], [mcore_weights]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # split qkv
            assert len(hf_names) == 3
            # split qkv
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            head_dim = (
                getattr(self.hf_config, "head_dim", hidden_dim // num_attention_heads)
                or hidden_dim // num_attention_heads
            )
            out_shape = (
                [num_key_value_heads, -1, hidden_dim]
                if ".bias" not in mcore_weights_name
                else [num_key_value_heads, -1]
            )
            qkv = mcore_weights.view(*out_shape)
            q_len = head_dim * num_attention_heads // num_key_value_heads
            k_len = head_dim
            v_len = head_dim
            single_out_shape = (
                [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
            )
            q = qkv[:, :q_len].reshape(*single_out_shape)
            k = qkv[:, q_len : q_len + k_len].reshape(*single_out_shape)
            v = qkv[:, q_len + k_len :].reshape(*single_out_shape)
            return hf_names, [q, k, v]

        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            # split gate_proj and up_proj
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Import Hugging Face weights to MCore format.

        Takes Hugging Face weight names and tensors, outputs MCore weight tensor.
        Due to MCore's runtime optimizations involving weight merging, input is a list.

        Args:
            mcore_weights_name: MCore weight name
            hf_weights: List of Hugging Face weight tensors

        Returns:
            torch.Tensor: MCore weight tensor

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        # Convert weights to the target dtype if needed
        # This handles cases where HF weights are FP32 but model expects BF16/FP16
        if (
            hasattr(self, "dtype")
            and self.dtype is not None
            and "expert_bias" not in mcore_weights_name
        ):
            hf_weights = [
                w.to(self.dtype) if w.dtype != self.dtype else w for w in hf_weights
            ]

        if len(hf_weights) == 1:
            # pad embeding and output layer
            if self.make_vocab_size_divisible_by is not None and (
                "embedding.word_embeddings.weight" in mcore_weights_name
                or "output_layer.weight" in mcore_weights_name
            ):
                assert hf_weights[0].shape[0] == self.vocab_size
                assert self.padded_vocab_size is not None

                embed_dim = hf_weights[0].shape[1]
                extra_zeros = torch.zeros(
                    (self.padded_vocab_size - self.vocab_size, embed_dim),
                    device=hf_weights[0].device,
                    dtype=hf_weights[0].dtype,
                )
                return torch.cat((hf_weights[0], extra_zeros), dim=0)

            return hf_weights[0]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            head_dim = (
                getattr(self.hf_config, "head_dim", hidden_dim // num_attention_heads)
                or hidden_dim // num_attention_heads
            )
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // group_dim
            q = q.view(
                [
                    real_num_key_value_heads,
                    group_dim,
                    -1,
                ]
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qkv
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            # merge gate_proj and up_proj
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_merge_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: list[torch.Tensor],
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge weights across tensor parallel ranks.
        In mcore format

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: List of MCore weight tensors from different TP ranks
            param: Parameter tensor

        Returns:
            torch.Tensor: Merged weight tensor
        """
        if self.mpu.tp_size == 1:
            assert len(mcore_weights) == 1
            return mcore_weights[0]
        if "mlp.experts.linear_fc" in mcore_weights_name:
            assert len(mcore_weights) == self.mpu.etp_size
        else:
            assert len(mcore_weights) == self.mpu.tp_size
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            return torch.cat(mcore_weights, dim=0)
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
            if not mcore_config.gated_linear_unit:
                return torch.cat(mcore_weights, dim=0)

            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for mcore_weight in mcore_weights:
                gate, up = mcore_weight.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            ret = torch.cat((gate, up), dim=0)

        elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
            ret = torch.cat(mcore_weights, dim=1)
        elif (
            "self_attention.linear_kv_down_proj.weight" in mcore_weights_name
            or "self_attention.linear_q_down_proj.weight" in mcore_weights_name
        ):
            # self_attention.linear_kv_down_proj.weight and self_attention.linear_q_down_proj.weight are copied
            return mcore_weights[0]
        else:
            assert (
                hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
            )
            ret = torch.cat(mcore_weights, dim=param.partition_dim)

        return ret

    def _weight_split_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: torch.Tensor,
        param: torch.Tensor,
        tp_split_size: int,
    ) -> list[torch.Tensor]:
        """
        Split weight tensor across tensor parallel ranks.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor
            param: Parameter tensor

        Returns:
            list: List of weight tensors split for each TP rank
        """
        if tp_split_size == 1:
            return [mcore_weights]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            return mcore_weights.chunk(tp_split_size)
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
            if not mcore_config.gated_linear_unit:
                return mcore_weights.chunk(tp_split_size)

            gate, up = mcore_weights.chunk(2)
            gates = gate.chunk(tp_split_size)
            ups = up.chunk(tp_split_size)
            ret = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
        elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
            ret = mcore_weights.chunk(tp_split_size, dim=1)
        else:
            if param.shape == mcore_weights.shape:
                return [mcore_weights for _ in range(tp_split_size)]
            assert len(param.shape) == len(mcore_weights.shape)
            for partition_dim, (s1, s2) in enumerate(
                zip(param.shape, mcore_weights.shape)
            ):
                if s1 != s2:
                    break

            ret = mcore_weights.chunk(tp_split_size, dim=partition_dim)
        return ret

    def _get_actual_hf_path(self, weight_path: str) -> str:
        """
        Get the actual Hugging Face path for the model weights.

        Args:
            weight_path: Path to the model weights or Hugging Face model identifier

        Returns:
            str: Actual path to the Hugging Face model weights
        """

        return os.path.dirname(cached_file(weight_path, "config.json"))


# Model registry
_MODEL_REGISTRY = {}


def register_model(model_types):
    """
    Model registration decorator.

    Args:
        model_types: String or list of strings representing model type identifiers

    Returns:
        Decorator function
    """
    if isinstance(model_types, str):
        model_types = [model_types]

    def decorator(cls):
        for model_type in model_types:
            _MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator
