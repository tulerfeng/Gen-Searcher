# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from abc import ABC

from megatron.core.transformer.transformer_config import TransformerConfig
from termcolor import colored
from torch.nn.modules.module import _addindent


def prehook_save_input_shape(func):
    def wrapper(self, *input_shapes, **kw_input_shapes):
        if len(input_shapes) + len(kw_input_shapes) == 0:
            if "_input_shape" in self.__dict__:
                return func(self, *self._input_shape, **self._kw_input_shapes)
            else:
                return 0
        self._input_shape = input_shapes
        self._kw_input_shapes = kw_input_shapes
        return func(self, *self._input_shape, **self._kw_input_shapes)

    return wrapper


class MetaBase(type):
    def __new__(cls, name, bases, attrs):
        if "num_activation" in attrs:
            attrs["num_activation"] = prehook_save_input_shape(attrs["num_activation"])

        return super().__new__(cls, name, bases, attrs)


class MemEstimator(metaclass=MetaBase):
    def __init__(self, *args, **kwargs):
        self._modules = {}
        pass

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        # extra_repr = self.extra_repr()
        # # empty string will be split into list ['']
        # if extra_repr:
        #     extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        stat = (
            "\t/* n_params="
            + colored(f"{self.num_parameter()/1024/1024:.2f}M", "red")
            + "\tn_act="
            + colored(f"{self.num_activation()/1024/1024:.2f}M", "green")
            + " */"
        )
        main_str = self._get_name() + stat + " ("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
        return f"{self.__class__.__name__} n_param={self.num_parameter()}"

    def dump(self):
        ret = {}
        ret["name"] = self._get_name()
        ret["n_params"] = self.num_parameter()
        ret["n_act"] = self.num_activation()
        modules = {}
        for key, module in self._modules.items():
            modules[key] = module.dump()
        if len(modules) > 0:
            ret["modules"] = modules
        return ret

    def _get_name(self):
        return self.__class__.__name__

    def num_parameter(self):
        """
        Calculate number of the model parameters
        """
        raise NotImplemented

    def num_activation(self, input_shape: list[int]):
        """
        Calculate number of the activation with given input_shape.
        Args:
            input shape
        """
        raise NotImplemented

    def mock_forward(self, input_shape: list[int]):
        """
        Mock the forward.
        Args:
            input shape
        return:
            output shape
        """
        raise NotImplemented

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, MemEstimator):
            modules = self.__dict__.get("_modules")
            modules[name] = value
        else:
            pass
        return super().__setattr__(name, value)

    def __delattr__(self, name):
        modules = self.__dict__.get("_modules")
        if name in modules:
            del modules[name]
        return super().__delattr__(name)


_global_config: TransformerConfig = None


def set_global_config(cfg):
    global _global_config
    _global_config = cfg


def get_tensor_model_parallel_world_size():
    global _global_config
    return _global_config.tensor_model_parallel_size


def get_tensor_model_parallel_rank():
    return 0


def get_expert_tensor_parallel_world_size():
    global _global_config
    return _global_config.expert_tensor_parallel_size


def get_expert_tensor_parallel_rank():
    return 0


_pp_rank = 0


def set_pipeline_model_parallel_rank(rank):
    global _pp_rank
    _pp_rank = rank


def get_pipeline_model_parallel_rank():
    global _pp_rank
    return _pp_rank


def get_virtual_pipeline_model_parallel_rank():
    return 0


def get_pipeline_model_parallel_world_size():
    global _global_config
    return _global_config.pipeline_model_parallel_size


def get_expert_model_parallel_rank():
    return 0


def get_expert_model_parallel_world_size():
    global _global_config
    return _global_config.expert_model_parallel_size


def get_virtual_pipeline_model_parallel_world_size():
    global _global_config
    return _global_config.virtual_pipeline_model_parallel_size


def is_pipeline_first_stage(ignore_virtual=False, vp_stage=None):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if (
        not ignore_virtual
        and get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        if vp_stage != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False, vp_stage=None):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    if (
        not ignore_virtual
        and get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        if vp_stage != (get_virtual_pipeline_model_parallel_world_size() - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1
    )


def cum_mul(l: list):
    try:
        ret = 1
        for one in l:
            ret *= one
        return ret
    except:
        return 0
        __import__("ipdb").set_trace()
