import asyncio

from transformers import AutoTokenizer

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.integrations.smolagents import AsyncCodeAgent, RLLMOpenAIModel


async def run_code_agent(rollout_engine):
    model = RLLMOpenAIModel(rollout_engine=rollout_engine, sampling_params={"max_tokens": 1000})

    # Create async code agent
    async_code_agent = AsyncCodeAgent(
        tools=[],
        model=model,
        max_steps=5,
        additional_authorized_imports=["math", "random", "json"],  # Only include commonly available modules
        add_base_tools=False,  # Disable base tools to avoid dependency issues
    )

    print("📝 Task: Calculate the area of a circle with radius 5 using math module")

    await async_code_agent.arun("Calculate the area of a circle with radius 5 using the math module and return the result")


async def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 2048},
    )

    await run_code_agent(rollout_engine)


if __name__ == "__main__":
    asyncio.run(main())
