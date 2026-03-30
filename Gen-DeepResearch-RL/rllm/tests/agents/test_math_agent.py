from rllm.agents.agent import Step, Trajectory
from rllm.agents.math_agent import MathAgent


class TestMathAgent:
    """Simplified test suite for MathAgent core functionality."""

    def test_init_default(self):
        """Test MathAgent initialization with default parameters."""
        agent = MathAgent()
        assert agent.accumulate_thinking is True
        assert agent.instruction == "Let's think step by step, and put your final answer within \\boxed{}."
        assert isinstance(agent._trajectory, Trajectory)
        assert agent.messages == []

    def test_init_custom_accumulate_thinking(self):
        """Test MathAgent initialization with custom accumulate_thinking parameter."""
        agent = MathAgent(accumulate_thinking=False)
        assert agent.accumulate_thinking is False

    def test_reset(self):
        """Test the reset method."""
        agent = MathAgent()
        # Add some state to reset
        agent.messages = [{"role": "user", "content": "test"}]
        agent._trajectory.steps.append(Step())

        agent.reset()

        assert isinstance(agent._trajectory, Trajectory)
        assert agent._trajectory.steps == []
        assert agent.messages == []

    def test_properties(self):
        """Test key properties."""
        agent = MathAgent()
        test_messages = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
        agent.messages = test_messages

        assert agent.chat_completions == test_messages
        assert agent.trajectory == agent._trajectory
        assert isinstance(agent.trajectory, Trajectory)

    def test_update_from_env_initial_question(self):
        """Test update_from_env with initial question observation."""
        agent = MathAgent()
        observation = {"question": "What is the square root of 16?"}
        agent.update_from_env(observation, 0.0, False, {})

        # Check that message was added correctly
        assert len(agent.messages) == 1
        expected_content = f"What is the square root of 16? {agent.instruction}"
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == expected_content

    def test_update_from_env_follow_up_correction(self):
        """Test update_from_env with follow-up correction."""
        agent = MathAgent()

        # First, add an initial step to simulate prior interaction
        agent._trajectory.steps.append(Step())

        agent.update_from_env("correction needed", 0.5, False, {"attempt": 2})

        # Check that correction message was added
        assert len(agent.messages) == 1
        expected_correction = "Your previous answer may contain a mistake. Please review it carefully and answer again. Put your final answer within \\boxed{}."
        assert agent.messages[0]["content"] == expected_correction

    def test_update_from_model_basic(self):
        """Test basic update_from_model functionality."""
        agent = MathAgent()

        # First provide a question
        agent.update_from_env({"question": "What is 2+2?"}, 0.0, False, {})

        response = "<think>2 + 2 = 4</think> \\boxed{4}"
        action = agent.update_from_model(response)

        # Check that step was created
        assert len(agent._trajectory.steps) == 1

        # Check that message was added
        assert len(agent.messages) == 2  # User question + assistant response
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == response

        # Check return value
        assert action.action == response

    def test_update_from_model_without_accumulate_thinking(self):
        """Test update_from_model with accumulate_thinking=False."""
        agent = MathAgent(accumulate_thinking=False)

        # First provide a question and get first response
        agent.update_from_env({"question": "What is 2+2?"}, 0.0, False, {})
        response1 = "<think>Let me calculate this</think> The answer is 4"
        agent.update_from_model(response1)

        # Add another question/response to test thinking removal (since only non-last messages get processed)
        agent.update_from_env({"question": "What is 3+3?"}, 0.0, False, {})
        response2 = "<think>3 + 3 = 6</think> The answer is 6"
        agent.update_from_model(response2)

        # Check that thinking portion was removed from the first assistant message but not the last
        chat_completions = agent.chat_completions
        first_assistant_message = chat_completions[1]  # First assistant message
        last_assistant_message = chat_completions[-1]  # Last assistant message

        # First message should have thinking removed
        assert first_assistant_message["content"] == " The answer is 4"
        # Last message should keep thinking (as per the implementation)
        assert last_assistant_message["content"] == response2

    def test_update_from_model_no_thinking_tags(self):
        """Test update_from_model when response has no thinking tags."""
        agent = MathAgent(accumulate_thinking=False)

        # First provide a question
        agent.update_from_env({"question": "What is 2+2?"}, 0.0, False, {})

        response = "The answer is 4"
        agent.update_from_model(response)

        # Should keep original response since no thinking tags
        assert agent.messages[-1]["content"] == response

    def test_basic_interaction_flow(self):
        """Test a basic complete interaction flow."""
        agent = MathAgent()

        # Step 1: Initial environment update
        observation = {"question": "What is 5 + 3?"}
        agent.update_from_env(observation, 0.0, False, {})

        assert len(agent._trajectory.steps) == 0  # No step until model responds
        assert len(agent.messages) == 1

        # Step 2: Model response
        response = "<think>5 + 3 = 8</think> \\boxed{8}"
        action = agent.update_from_model(response)

        assert len(agent._trajectory.steps) == 1
        assert len(agent.messages) == 2
        assert action.action == response

    def test_trajectory_to_dict(self):
        """Test that trajectory can be converted to dict."""
        agent = MathAgent()

        # Add basic interaction
        agent.update_from_env({"question": "test"}, 0.0, False, {})
        agent.update_from_model("test response")

        trajectory_dict = agent.trajectory.to_dict()
        assert isinstance(trajectory_dict, dict)
        assert "steps" in trajectory_dict
        assert isinstance(trajectory_dict["steps"], list)
