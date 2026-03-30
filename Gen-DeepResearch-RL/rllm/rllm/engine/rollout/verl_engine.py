import asyncio
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


def _get_image_token_id(processor) -> int | None:
    """从 processor 获取「图像占位 token」的 id（Qwen 为 <|image_pad|>），用于在 prompt_ids 中统计 vision token 数。"""
    if processor is None:
        return None
    try:
        if hasattr(processor, "image_token_id"):
            return int(processor.image_token_id)
        if hasattr(processor, "tokenizer"):
            return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    except (TypeError, ValueError, KeyError):
        pass
    return None


def _get_vision_start_token_id(processor) -> int | None:
    """Qwen: <|vision_start|>"""
    if processor is None:
        return None
    try:
        if hasattr(processor, "vision_start_token_id"):
            return int(processor.vision_start_token_id)
        if hasattr(processor, "tokenizer"):
            return processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    except (TypeError, ValueError, KeyError):
        pass
    return None


def _get_vision_end_token_id(processor) -> int | None:
    """Qwen: <|vision_end|>"""
    if processor is None:
        return None
    try:
        if hasattr(processor, "tokenizer"):
            return processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    except (TypeError, ValueError, KeyError):
        pass
    return None


def _vision_token_count_from_prompt_ids(prompt_ids: list[int], processor) -> int:
    """从 processor 输出的 prompt_ids（input_ids[0]）中统计 vision token 数量：即 <|image_pad|> 出现次数，与 LLM 实际输入一致。"""
    image_token_id = _get_image_token_id(processor)
    if image_token_id is None:
        return 0
    try:
        return sum(1 for tid in prompt_ids if tid == image_token_id)
    except TypeError:
        return 0


def _per_image_vision_token_counts(prompt_ids: list[int], processor) -> list[int]:
    """从 prompt_ids 中按「每张图」统计 vision token 数：每段 vision_start..vision_end 内 <|image_pad|> 的个数，与 LLM 实际一致。"""
    image_token_id = _get_image_token_id(processor)
    vision_start_id = _get_vision_start_token_id(processor)
    vision_end_id = _get_vision_end_token_id(processor)
    if image_token_id is None or vision_start_id is None:
        return []
    try:
        counts: list[int] = []
        i = 0
        while i < len(prompt_ids):
            if prompt_ids[i] == vision_start_id:
                i += 1
                n = 0
                while i < len(prompt_ids) and prompt_ids[i] == image_token_id:
                    n += 1
                    i += 1
                if n > 0:
                    counts.append(n)
                if vision_end_id is not None:
                    while i < len(prompt_ids) and prompt_ids[i] != vision_end_id:
                        i += 1
                    if i < len(prompt_ids):
                        i += 1
            else:
                i += 1
        return counts
    except (TypeError, IndexError):
        return []


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=rollout_manager.server_handles)
        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
        sampling_params["max_new_tokens"] = max_tokens  # 传给后端，限制单轮生成长度

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]

        if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:
            image_data = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
            model_inputs = self.processor(text=[prompt], images=image_data)
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)
        else:
            image_data = None
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        prompt_length = len(prompt_ids)
        vision_prompt_length = _vision_token_count_from_prompt_ids(prompt_ids, self.processor)
        text_prompt_length = max(0, prompt_length - vision_prompt_length)
        per_image_vision_token_counts = _per_image_vision_token_counts(prompt_ids, self.processor)

        # 累积上下文（prompt）超限检查：使用 max_response_length + 4096 作为 rollout 轨迹总长度上限
        # 允许更长 rollout（多 4096），但 PPO 优化仍用 data.max_response_length 做 padding/truncation
        rollout_limit = self.max_response_length + 6000
        if enforce_max_prompt_length and prompt_length > rollout_limit:
            raise TerminationEvent(
                TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
                prompt_length=prompt_length,
                text_prompt_length=text_prompt_length,
                vision_prompt_length=vision_prompt_length,
                per_image_vision_token_counts=per_image_vision_token_counts,
            )

        token_output: TokenOutput = await self.server_manager.generate(request_id=application_id, prompt_ids=request_prompt_ids, image_data=image_data, sampling_params=sampling_params)  # type: ignore
        completion_ids: list[int] = token_output.token_ids

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=[],
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
            text_prompt_length=text_prompt_length,
            vision_prompt_length=vision_prompt_length,
            per_image_vision_token_counts=per_image_vision_token_counts,
        )

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])

    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])
