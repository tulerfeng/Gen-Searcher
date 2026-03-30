import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import PIL.Image

# Optional SmolAgent imports - only if available
try:
    from smolagents import CodeAgent, Model, MultiStepAgent, Tool, ToolCallingAgent
    from smolagents.models import ChatMessage, ChatMessageStreamDelta, TokenUsage

    # Use MultiStepAgent as the base class for type hints
    SmolAgentBase = MultiStepAgent
    SmolModel = Model
    SmolTool = Tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False

    # Create dummy classes for type hints
    class CodeAgent:
        pass

    class ToolCallingAgent:
        pass

    class SmolAgentBase:
        pass

    class SmolModel:
        pass

    class SmolTool:
        pass


# Import BaseAgent from rLLM for wrapper classes
from rllm.agents.agent import Step, Trajectory
from rllm.engine import ModelOutput

logger = logging.getLogger(__name__)


class AsyncAgentMixin:
    """
    Mixin class providing common async run functionality for AsyncCodeAgent and AsyncToolCallingAgent.
    """

    async def arun(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Async version of run method. Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns an async generator that yields each step as it is executed.
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run.
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task.

        Returns:
            The final answer if stream=False, or an async generator if stream=True.
        """
        import time

        from smolagents.memory import SystemPromptStep, TaskStep
        from smolagents.monitoring import LogLevel

        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # Return the async generator for streaming mode
            return self._arun_stream(task=self.task, max_steps=max_steps, images=images)

        run_start_time = time.time()
        # Outputs are returned only at the end. We collect all steps.
        steps = []
        async for step in self._arun_stream(task=self.task, max_steps=max_steps, images=images):
            steps.append(step)

        from smolagents.memory import FinalAnswerStep

        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        if self.return_full_result:
            from smolagents.agents import RunResult
            from smolagents.memory import ActionStep, PlanningStep, Timing, TokenUsage
            from smolagents.utils import AgentMaxStepsError

            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, ActionStep) or isinstance(step, PlanningStep):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
                state = "max_steps_error"
            else:
                state = "success"

            messages = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                messages=messages,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output

    async def _arun_stream(self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None):
        """
        Async version of _run_stream. Generator that runs the agent step by step.
        """
        import time

        from rich.text import Text
        from smolagents.agent_types import handle_agent_output_types
        from smolagents.agents import ActionOutput
        from smolagents.memory import ActionStep, FinalAnswerStep, Timing
        from smolagents.monitoring import LogLevel
        from smolagents.utils import AgentError, AgentGenerationError

        self.step_number = 1
        returned_final_answer = False
        final_answer = None

        while not returned_final_answer and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)

            # For now, skip planning steps in async version to keep it simple
            # Planning step support can be added later if needed

            # Start action step!
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                async for output in self._step_stream_async(action_step):
                    # Yield all outputs
                    yield output

                    if isinstance(output, ActionOutput) and output.is_final_answer:
                        final_answer = output.output
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style="bold yellow"),
                            level=LogLevel.INFO,
                        )

                        if self.final_answer_checks:
                            self._validate_final_answer(final_answer)
                        returned_final_answer = True
                        action_step.is_final_answer = True

            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error
                raise e
            except Exception as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if not returned_final_answer and self.step_number == max_steps + 1:
            final_answer = await self._ahandle_max_steps_reached(task, images)
            yield action_step
        yield FinalAnswerStep(handle_agent_output_types(final_answer))

    async def _ahandle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"]) -> any:
        """Async version of _handle_max_steps_reached."""
        import time

        from smolagents.memory import ActionStep, Timing
        from smolagents.utils import AgentMaxStepsError

        action_step_start_time = time.time()
        final_answer = await self._aprovide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=getattr(final_answer, "token_usage", None),
        )
        final_memory_step.action_output = getattr(final_answer, "content", final_answer)
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return getattr(final_answer, "content", final_answer)

    async def _aprovide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None):
        """Async version of provide_final_answer."""
        from smolagents.agents import populate_template
        from smolagents.models import ChatMessage, MessageRole

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        if images:
            messages[0].content += [{"type": "image", "image": image} for image in images]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}),
                    }
                ],
            )
        )
        try:
            if hasattr(self.model, "generate_async"):
                chat_message = await self.model.generate_async(messages)
            else:
                # Fallback to sync wrapped in async
                import asyncio

                chat_message = await asyncio.get_event_loop().run_in_executor(None, lambda: self.model.generate(messages))

            # Record LLM call in trajectory for final answer (if agent has trajectory tracking)
            if hasattr(self, "_record_llm_call"):
                self._record_llm_call(messages, chat_message)
                # Mark as done since this is the final answer
                if hasattr(self, "_current_step") and self._current_step:
                    self._current_step.done = True
            return chat_message
        except Exception as e:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Error in generating final LLM output: {e}"}],
            )


class RLLMOpenAIModel(SmolModel):
    """
    A wrapper over SmolAgent's OpenAIServerModel that uses rLLM's execution engine
    for API calls instead of making direct calls to OpenAI.

    This ensures that when training with rLLM, all model calls go through rLLM's
    infrastructure (rate limiting, retry logic, distributed execution, etc.)
    while maintaining compatibility with SmolAgent's Model interface.

    Key Features:
    - Converts SmolAgent message format to OpenAI chat completion format
    - Properly handles tool responses: MessageRole.TOOL_RESPONSE → "role": "tool"
    - Includes tool_call_id for tool responses as required by OpenAI format
    - Skips MessageRole.TOOL_CALL messages (handled as part of assistant messages)
    """

    def __init__(self, rollout_engine=None, **kwargs):
        """
        Initialize the RLLM-integrated OpenAI model.

        Args:
            rollout_engine: rLLM's RolloutEngine instance
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        self.rollout_engine = rollout_engine
        self._last_input_token_count = 0
        self._last_output_token_count = 0

        # Store kwargs for potential future use
        self.kwargs = kwargs

        if not rollout_engine:
            raise ValueError("rollout_engine is required for RLLMOpenAIModel. Pass an instance of rLLM's RolloutEngine.")

    async def generate_async(self, messages: list[dict[str, Any]], stop_sequences: list[str] | None = None, response_format: dict[str, str] | None = None, tools_to_call_from: list | None = None, **kwargs) -> Any:
        """
        Async version of generate that can be called from async contexts.
        """
        try:
            # Convert SmolAgent messages to rLLM format if needed
            if isinstance(messages, list) and all(isinstance(msg, dict) for msg in messages):
                prompt = messages
            else:
                # Handle ChatMessage objects from SmolAgent using the helper method
                prompt = self._convert_smolagent_messages_to_openai(messages)

            model_output: ModelOutput = await self.rollout_engine.get_model_response(prompt, max_tokens=kwargs.pop("max_tokens", 4096), **kwargs)

            # Extract text and token usage from ModelOutput
            response_text = model_output.text
            completion_tokens = model_output.completion_length
            prompt_tokens = model_output.prompt_length

            # Create a ChatMessage-like response object
            if SMOLAGENTS_AVAILABLE:
                # Use actual token counts from ModelOutput
                input_tokens = prompt_tokens
                output_tokens = completion_tokens

                self._last_input_token_count = input_tokens
                self._last_output_token_count = output_tokens

                return ChatMessage(role="assistant", content=response_text, token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens))
            else:
                # Fallback for when SmolAgent is not available
                return {"role": "assistant", "content": response_text, "token_usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}}

        except Exception as e:
            logger.error(f"Error in RLLMOpenAIModel.generate_async: {e}")
            logger.error(f"Messages type: {type(messages)}")
            logger.error(f"Messages content: {messages}")

            # Import AgentError properly to avoid missing logger issue
            if SMOLAGENTS_AVAILABLE:
                from smolagents.monitoring import AgentLogger, LogLevel
                from smolagents.utils import AgentError

                # Create a proper SmolAgent logger
                agent_logger = AgentLogger(level=LogLevel.INFO)
                raise AgentError(f"Model generation failed: {e}", agent_logger) from e
            else:
                raise RuntimeError(f"Model generation failed: {e}") from e

    def generate(self, messages: list[dict[str, Any]], stop_sequences: list[str] | None = None, response_format: dict[str, str] | None = None, tools_to_call_from: list | None = None, **kwargs) -> Any:
        """
        Synchronous version of generate that wraps the async method.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                # This is a limitation - sync method called from async context
                raise RuntimeError("Cannot call sync generate() from async context. Use generate_async() instead.")
            else:
                # We're not in an async context, safe to run async method
                return loop.run_until_complete(self.generate_async(messages, stop_sequences, response_format, tools_to_call_from, **kwargs))
        except RuntimeError as e:
            if "no current event loop" in str(e).lower() or "no running event loop" in str(e).lower():
                # No event loop exists, create one
                return asyncio.run(self.generate_async(messages, stop_sequences, response_format, tools_to_call_from, **kwargs))
            else:
                # Re-raise other RuntimeErrors
                raise e

    def parse_tool_calls(self, message):
        """Parse tool calls from message - placeholder implementation."""
        # For now, return the message as-is since tool parsing is typically done by the model
        # This would need to be implemented based on your specific model's tool call format
        return message

    def generate_stream(self, messages, **kwargs):
        """
        Streaming generation - not supported in this implementation.
        Falls back to regular generation.
        """
        logger.warning("Streaming not supported in RLLMOpenAIModel, falling back to regular generation")
        response = self.generate(messages, **kwargs)

        # Yield the response as a single chunk
        if SMOLAGENTS_AVAILABLE:
            yield ChatMessageStreamDelta(content=response.content, token_usage=response.token_usage)
        else:
            yield {"content": response.get("content", ""), "token_usage": response.get("token_usage", {})}

    @property
    def last_input_token_count(self) -> int:
        """Get the token count from the last input."""
        return self._last_input_token_count

    @property
    def last_output_token_count(self) -> int:
        """Get the token count from the last output."""
        return self._last_output_token_count

    def _convert_smolagent_messages_to_openai(self, messages):
        """
        Helper method to convert SmolAgent messages to OpenAI chat completion format.
        This method is extracted for easier testing and debugging.
        """
        prompt = []
        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # Extract text content from SmolAgent's format
                if isinstance(msg.content, list):
                    # SmolAgent uses [{'type': 'text', 'text': '...'}] format
                    content_text = ""
                    for content_part in msg.content:
                        if isinstance(content_part, dict) and content_part.get("type") == "text":
                            content_text += content_part.get("text", "")
                        elif isinstance(content_part, dict) and "text" in content_part:
                            content_text += content_part["text"]
                        else:
                            content_text += str(content_part)
                else:
                    content_text = str(msg.content)

                # Convert SmolAgent roles to OpenAI format
                role_str = str(msg.role).replace("MessageRole.", "").lower()

                # Handle special role conversions for OpenAI chat completion format
                if role_str == "tool_response":
                    # SmolAgent's TOOL_RESPONSE should become OpenAI's "tool" role
                    openai_role = "tool"

                    # For tool responses, we need to include tool_call_id if available
                    message_dict = {"role": openai_role, "content": content_text}

                    # Check if the message has tool_call_id (required for tool responses in OpenAI format)
                    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                        message_dict["tool_call_id"] = msg.tool_call_id
                    elif hasattr(msg, "id") and msg.id:
                        # Fallback to using 'id' field if tool_call_id is not available
                        message_dict["tool_call_id"] = msg.id

                    prompt.append(message_dict)
                elif role_str == "tool_call":
                    # TOOL_CALL messages don't exist in OpenAI format - tool calls are part of assistant messages
                    # Skip these as they should be handled as part of assistant message tool_calls
                    continue
                else:
                    # Handle standard roles (system, user, assistant)
                    prompt.append({"role": role_str, "content": content_text})
            else:
                prompt.append(msg)
        return prompt


class AsyncCodeAgent(AsyncAgentMixin, CodeAgent):
    """
    A wrapper over CodeAgent that uses async model calls and implements rLLM trajectory interface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rllm_trajectory = Trajectory()
        self._current_step = None

    def _create_new_step(self):
        """Create a new Step object and append it to the trajectory."""
        self._current_step = Step()
        self._rllm_trajectory.steps.append(self._current_step)
        return self._current_step

    def _record_llm_call(self, input_messages, response):
        """Record an LLM call in a new step."""
        # Always create a new step for each LLM call
        step = self._create_new_step()

        # Convert messages to OpenAI format
        if hasattr(self.model, "_convert_smolagent_messages_to_openai"):
            openai_messages = self.model._convert_smolagent_messages_to_openai(input_messages)
        else:
            # Fallback - assume already in correct format
            openai_messages = input_messages if isinstance(input_messages, list) else [input_messages]

        # Add response
        response_content = response.content if hasattr(response, "content") else str(response)
        openai_messages.append({"role": "assistant", "content": response_content})

        # Update the new step
        step.chat_completions = openai_messages
        step.model_response = response_content

    @property
    def trajectory(self) -> Trajectory:
        """Return the rLLM trajectory object."""
        return self._rllm_trajectory

    async def astep(self, memory_step=None, **kwargs):
        """
        Async version of step method. Perform one step in the ReAct framework.
        Returns either None if the step is not final, or the final answer.
        """
        if memory_step is None:
            import time

            from smolagents.memory import ActionStep, Timing

            memory_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=time.time()),
            )

        result = None
        async for output in self._step_stream_async(memory_step):
            result = output
        return result

    async def _step_stream_async(self, memory_step):
        """
        Async version of _step_stream. Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("SmolAgents library is required for async operations")

        from rich.console import Group
        from rich.text import Text
        from smolagents.agents import ActionOutput
        from smolagents.local_python_executor import fix_final_answer_code
        from smolagents.memory import ToolCall
        from smolagents.models import CODEAGENT_RESPONSE_FORMAT
        from smolagents.monitoring import LogLevel
        from smolagents.utils import AgentExecutionError, AgentGenerationError, AgentParsingError, extract_code_from_text, parse_code_blobs, truncate_content

        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()

        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            stop_sequences.append(self.code_block_tags[1])

        try:
            additional_args = {}
            if self.grammar:
                additional_args["grammar"] = self.grammar
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT

            # Check if model has async generate method
            if hasattr(self.model, "generate_async"):
                chat_message = await self.model.generate_async(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content if hasattr(chat_message, "content") else str(chat_message)

                # Record LLM call in trajectory
                self._record_llm_call(input_messages, chat_message)

                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )
            else:
                # Fallback to sync method wrapped in async
                import asyncio

                chat_message = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_messages,
                        stop_sequences=stop_sequences,
                        **additional_args,
                    ),
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content

                # Record LLM call in trajectory
                self._record_llm_call(input_messages, chat_message)

                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # Add end code sequence if missing
            if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
                output_text += self.code_block_tags[1]
                if hasattr(memory_step.model_output_message, "content"):
                    memory_step.model_output_message.content = output_text

            if hasattr(chat_message, "token_usage"):
                memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text

        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        try:
            if self._use_structured_outputs_internally:
                code_action = json.loads(output_text)["code"]
                code_action = extract_code_from_text(code_action, self.code_block_tags) or code_action
            else:
                code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action

            # Record the code action in trajectory
            self._current_step.action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger) from e

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        ### Execute action ###
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        try:
            # Execute code synchronously (python execution is typically not async)
            code_output = self.python_executor(code_action)
            execution_outputs_console = []
            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]
            observation = "Execution logs:\n" + code_output.logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger) from e

        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        # Record the observation in trajectory
        self._current_step.observation = observation

        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(f"Out: {truncated_output}"),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output
        # Mark step as done if this is the final answer
        if code_output.is_final_answer:
            self._current_step.done = True

        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)


class AsyncToolCallingAgent(AsyncAgentMixin, ToolCallingAgent):
    """
    A wrapper over ToolCallingAgent that uses async model calls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def astep(self, memory_step=None, **kwargs):
        """
        Async version of step method. Perform one step in the ReAct framework.
        Returns either None if the step is not final, or the final answer.
        """
        if memory_step is None:
            import time

            from smolagents.memory import ActionStep, Timing

            memory_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=time.time()),
            )

        result = None
        async for output in self._step_stream_async(memory_step):
            result = output
        return result

    async def _step_stream_async(self, memory_step):
        """
        Async version of _step_stream. Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("SmolAgents library is required for async operations")

        from smolagents.agents import ActionOutput, ToolOutput
        from smolagents.models import parse_json_if_needed
        from smolagents.monitoring import LogLevel
        from smolagents.utils import AgentGenerationError, AgentParsingError, AgentToolExecutionError

        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            # Check if model has async generate method
            if hasattr(self.model, "generate_async"):
                chat_message = await self.model.generate_async(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                if hasattr(chat_message, "content") and chat_message.content is None and hasattr(chat_message, "raw") and chat_message.raw is not None:
                    log_content = str(chat_message.raw)
                else:
                    log_content = str(getattr(chat_message, "content", "")) or ""

                self.logger.log_markdown(
                    content=log_content,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )
            else:
                # Fallback to sync method wrapped in async
                import asyncio

                chat_message = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_messages,
                        stop_sequences=["Observation:", "Calling tools:"],
                        tools_to_call_from=self.tools_and_managed_agents,
                    ),
                )

                if chat_message.content is None and chat_message.raw is not None:
                    log_content = str(chat_message.raw)
                else:
                    log_content = str(chat_message.content) or ""

                self.logger.log_markdown(
                    content=log_content,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # Record model output
            memory_step.model_output_message = chat_message
            memory_step.model_output = getattr(chat_message, "content", None)
            if hasattr(chat_message, "token_usage"):
                memory_step.token_usage = chat_message.token_usage

        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        # Parse tool calls
        if not hasattr(chat_message, "tool_calls") or chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger) from e
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)

        final_answer, got_final_answer = None, False
        async for output in self._process_tool_calls_async(chat_message, memory_step):
            yield output
            if isinstance(output, ToolOutput):
                if output.is_final_answer:
                    if got_final_answer:
                        raise AgentToolExecutionError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True

                    # Manage state variables
                    if isinstance(final_answer, str) and final_answer in self.state.keys():
                        final_answer = self.state[final_answer]

        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

    async def _process_tool_calls_async(self, chat_message, memory_step):
        """Process tool calls asynchronously."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from rich.panel import Panel
        from rich.text import Text
        from smolagents.agent_types import AgentAudio, AgentImage
        from smolagents.agents import ToolOutput
        from smolagents.memory import ToolCall
        from smolagents.monitoring import LogLevel

        parallel_calls = {}
        assert hasattr(chat_message, "tool_calls") and chat_message.tool_calls is not None

        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id)
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        # Helper function to process a single tool call
        def process_single_tool_call(tool_call):
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # Process tool calls in parallel
        outputs = {}
        if len(parallel_calls) == 1:
            # If there's only one call, process it directly
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # If multiple tool calls, process them in parallel
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = [executor.submit(process_single_tool_call, tool_call) for tool_call in parallel_calls.values()]
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        # Update memory step
        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.model_output = memory_step.model_output or ""
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            message = f"Tool call {tool_output.id}: calling '{tool_output.tool_call.name}' with arguments: {tool_output.tool_call.arguments}\n"
            memory_step.model_output += message
            memory_step.observations += tool_output.observation + "\n"
        memory_step.model_output = memory_step.model_output.rstrip("\n")
        memory_step.observations = memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
