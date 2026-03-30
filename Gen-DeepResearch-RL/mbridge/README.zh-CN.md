# MBridge: 连接Megatron-Core与Hugging Face/强化学习

MBridge提供了Hugging Face模型和Megatron-Core优化实现之间的无缝桥接，用于高效的分布式训练和推理。同时，MBridge还提供了强化学习（RL）接入Megatron所需的必要工具和流程。
MBridge 是一个原型项目，其理念已被采纳为NVIDIA官方维护的 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)。如需训练循环、混合精度（FP8、BF16、FP4 等）、PEFT 等更高级功能，请参见 Megatron-Bridge。

[English Documentation](README.md)
## 202508更新
- 支持以bf16训练DeepSeekV3模型时，直接加载FP8 HF格式权重，无需保存额外的Megatron-Core格式权重（包含MTP支持，基于DeepSeek官方提供的反量化kernel）用法查看example/4

## 概述

MBridge允许您将流行的Hugging Face模型转换为Megatron-Core格式，使您能够利用先进的并行策略进行大规模训练和推理。该库支持各种模型架构，并简化了这些框架之间的转换过程。对于强化学习工作流，MBridge提供了连接RL算法与Megatron优化模型所需的接口和工具。

## 功能亮点

- **全面的模型支持**：支持多种模型架构，包括MoE（混合专家）模型
- **在线权重导入**：支持在线加载HF权重，支持各种并行策略，自动分片权重，无需保存额外的Megatron-Core格式权重
- **在线权重导出**：支持在线导出权重到HF格式用于推理引擎，支持TP/PP/CP/VPP/EP/ETP等并行策略
- **内存友好**：采用按张量策略，最小化加载/导出HF格式权重时的内存峰值
- **简洁API**：直观的模型转换和权重管理接口
- **支持Transformer Engine**：使用强大的Transformer Engine加速Megatron-Core模型，获得更好的性能（use_te=False目前不支持）

## 安装

```bash
pip install mbridge
```

## 快速开始

```python
from megatron.core import parallel_state as mpu
from mbridge import AutoBridge

# 初始化分布式环境
mpu.initialize_model_parallel(
    tensor_model_parallel_size=tp,
    pipeline_model_parallel_size=pp,
    virtual_pipeline_model_parallel_size=vpp,
    context_parallel_size=cp,
    expert_model_parallel_size=ep,
)

# 从Hugging Face加载模型
HF_MODEL_PATH = "/path/to/Qwen/Qwen2.5-7B-Instruct"
# or llama model
HF_MODEL_PATH = "/path/to/llama/llama3-8b-instruct"
bridge = AutoBridge.from_pretrained(HF_MODEL_PATH)

# 获取Megatron-Core模型并从Hugging Face加载权重
model = bridge.get_model(weight_path=HF_MODEL_PATH)

# 导出权重回Hugging Face格式用于推理引擎
for key, weight in bridge.export_weights(model):
    # 处理或保存导出的权重
    print(f"已导出: {key}")

# 保存模型到HF格式
bridge.save_weights(model, "path/to/save/model", memory_efficient=False) # 如果模型很大，设置memory_efficient=True
```

## 支持的模型

当前支持的模型：
- [x] Qwen2
- [x] Qwen2-MoE
- [x] Qwen3
- [x] Qwen3-MoE
- [x] LLaMA
- [x] DeepseekV3
- [x] Mixtral
- [x] Qwen2.5-VL
- [x] Mimo

## 示例

`example`目录包含展示常见用例的脚本：

- `0.load_model_and_generate_single_gpu.py`：在单GPU上加载模型并生成文本
- `1.load_model_and_export_single_gpu.py`：在单GPU上加载模型并导出权重
- `2.load_model_and_export_multiple_gpus.py`：使用多个GPU（TP/PP/CP/VPP并行）加载模型并导出权重

### 模型创建后回调

MBridge提供了一些后模型创建回调来定制化模型。

- `make_value_model`: 使模型变为 value model
- `freeze_moe_router`: 冻结模型中的MoE路由器

```python
from mbridge.utils.post_creation_callbacks import make_value_model, freeze_moe_router

bridge = AutoBridge.from_pretrained(HF_MODEL_PATH)
model = bridge.get_model(weight_path=HF_MODEL_PATH, post_model_creation_callbacks=[make_value_model, freeze_moe_router])
```
## 开发路线图

MBridge 将持续维护对热门模型的支持，但不会开发更多高级功能。更多高级功能请参见 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)。


## 致谢
- [veRL](https://github.com/volcengine/verl) 已将 MBridge 作为连接 Megatron-Core 的组件。
- [slime](https://github.com/THUDM/slime) 已将 MBridge 作为 Megatron-Core 权重转换器。
- [Nemo-RL](https://github.com/NVIDIA-NeMo/RL) 已将 Megatron-Bridge 作为连接 Megatron-Core 的组件。

## 许可证

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. 