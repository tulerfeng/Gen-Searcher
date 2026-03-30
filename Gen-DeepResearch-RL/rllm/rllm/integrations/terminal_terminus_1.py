from __future__ import annotations

import json
import time
from typing import Any

from tenacity import retry, retry_if_not_exception_type, stop_after_attempt
from terminal_bench.agents.terminus_1 import (
    Command,
    CommandBatchResponse,
    Terminus,
)
from terminal_bench.llms.base_llm import (
    ContextLengthExceededError,
    OutputLengthExceededError,
    ParseError,
)
from terminal_bench.terminal.tmux_session import TmuxSession

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import TerminationReason


class RLLMModel(Terminus):
    """Terminus adapter that replaces LiteLLM+Chat with a generic RolloutEngine.

    Notes:
        - Reuses Terminus internals (prompt templates and command execution)
        - Maintains persistent message history identical to Terminal-Bench `Chat`.
        - Mirrors `_run_agent_loop` logic while swapping the LLM interaction path.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        model_name: str,
        max_episodes: int = 50,
        global_agent_timeout_sec: float = 600.0,
        api_base: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """Initialize adapter with an arbitrary rollout engine and TB settings."""
        super().__init__(
            model_name=model_name,
            max_episodes=max_episodes,
            api_base=api_base,
            temperature=temperature,
            **kwargs,
        )
        self._engine = rollout_engine
        self._messages: list[dict[str, str]] = []
        self._trajectory: Trajectory = Trajectory()
        self._global_agent_timeout_sec = global_agent_timeout_sec

    def build_initial_prompt(self, instruction: str, terminal_state: str) -> str:
        """Format the initial prompt using TB's template and JSON schema."""
        return self._prompt_template.format(
            response_schema=self._response_schema,
            instruction=instruction,
            history="",
            terminal_state=terminal_state,
        )

    def execute_commands(self, commands: list[Command], session: TmuxSession) -> tuple[bool, str]:
        """Execute a batch of commands in the tmux session and capture output."""
        return self._execute_commands(commands, session)

    def _format_prompt_for_schema(
        self,
        prompt: str,
        response_format: dict | type[CommandBatchResponse] | None,
    ) -> tuple[str, dict | None]:
        """Build OpenAI response_format from the pydantic response schema."""
        schema_dict = response_format.model_json_schema()  # type: ignore[attr-defined]
        return prompt, {"type": "json_schema", "json_schema": {"name": "response", "schema": schema_dict}}

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type((ContextLengthExceededError, OutputLengthExceededError)),
    )
    async def handle_llm_interaction_with_engine(
        self,
        prompt: str,
        response_format: dict | type[CommandBatchResponse] | None = CommandBatchResponse,
    ) -> CommandBatchResponse:
        """Call the rollout engine, update message history, and parse JSON output."""
        _, engine_response_format = self._format_prompt_for_schema(prompt, response_format)
        messages = self._messages + [{"role": "user", "content": prompt}]
        output: ModelOutput = await self._engine.get_model_response(messages, response_format=engine_response_format)

        assistant_text = output.text or ""

        # Update message history exactly like TB Chat
        self._messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_text},
            ]
        )

        # Truncation parity: raise if response was cut due to length
        if output.finish_reason == "length":
            raise OutputLengthExceededError(
                f"Response truncated (finish_reason=length) for model {getattr(self._engine, 'model', 'unknown')}",
                truncated_response=assistant_text,
            )

        # Validate and return
        try:
            return CommandBatchResponse.model_validate_json(assistant_text)
        except (json.JSONDecodeError, Exception) as e:
            # Wrap parse errors into TB's ParseError
            raise ParseError(f"Failed to parse LLM response: {e}") from e

    async def run_agent_loop_with_engine(self, initial_prompt: str, session: TmuxSession) -> tuple[Trajectory, TerminationReason]:
        """Run the TB control loop and build a trajectory with termination reason."""
        # Reset per-run state
        self._messages = []
        self._trajectory = Trajectory()
        prompt = initial_prompt
        deadline = time.time() + float(self._global_agent_timeout_sec)

        for episode in range(self._max_episodes):
            if time.time() > deadline:
                return self._trajectory, TerminationReason.TIMEOUT

            parsed_response = await self.handle_llm_interaction_with_engine(prompt=prompt, response_format=CommandBatchResponse)
            self._record_asciinema_marker(parsed_response.model_dump_json(), session)

            timeout_occurred, terminal_output = self._execute_commands(parsed_response.commands, session)

            # Last two messages are the user prompt and assistant reply
            step_messages = self._messages[-2:]
            step = Step(
                chat_completions=list(step_messages),
                observation=prompt,
                model_response=step_messages[-1]["content"],
                info={
                    "is_task_complete": parsed_response.is_task_complete,
                    "timeout_occurred": timeout_occurred,
                    "num_commands": len(parsed_response.commands),
                },
            )
            self._trajectory.steps.append(step)

            if parsed_response.is_task_complete:
                return self._trajectory, TerminationReason.ENV_DONE

            prompt = terminal_output

        return self._trajectory, TerminationReason.MAX_TURNS_EXCEEDED

    def get_trajectory(self) -> Trajectory:
        """Return the accumulated trajectory from the most recent run."""
        return self._trajectory
