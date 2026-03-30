from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)

from mbridge.core.util import get_model_config

from .inference_request import Glm4vInferenceRequest
from .model_inference_controller import Glm4vGenerationController
from .model_inference_warpper import Glm4vInferenceWrapper


def get_inference_engine(model, tokenizer, vocab_size):
    config = get_model_config(model)
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=config.hidden_size,
        inference_batch_times_seqlen_threshold=32 * 1024,
        fp32_residual_connection=config.fp32_residual_connection,
        params_dtype=config.params_dtype,
        padded_vocab_size=vocab_size,
    )
    inference_wrapped_model = Glm4vInferenceWrapper(model, inference_wrapper_config)
    controller = Glm4vGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    inference_engine = StaticInferenceEngine(
        controller, max_batch_size=1, random_seed=0
    )

    return inference_engine

    """

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        num_tokens_to_generate=config.out_seq_length,
    )
    pass
    """
