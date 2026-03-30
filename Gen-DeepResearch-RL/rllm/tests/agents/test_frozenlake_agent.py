from rllm.agents.agent import Step, Trajectory
from rllm.agents.frozenlake_agent import FrozenLakeAgent


class TestFrozenLakeAgent:
    """Simplified test suite for FrozenLakeAgent core functionality."""

    def test_init_default(self):
        """Test FrozenLakeAgent initialization with default parameters."""
        agent = FrozenLakeAgent()
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1  # System message
        assert agent.step == 0
        assert agent.accumulate_thinking is True
        assert agent.multistep_prompt is False
        assert agent.max_steps is None
        assert agent.messages[0]["role"] == "system"
        assert "FrozenLake Quick Guide" in agent.messages[0]["content"]

    def test_init_with_params(self):
        """Test FrozenLakeAgent initialization with custom parameters."""
        agent = FrozenLakeAgent(max_steps=10, use_accumulate_thinking=False, use_multistep_prompt=True)
        assert agent.max_steps == 10
        assert agent.accumulate_thinking is False
        assert agent.multistep_prompt is True

    def test_reset(self):
        """Test the reset method."""
        agent = FrozenLakeAgent()
        initial_message_content = agent.messages[0]["content"]

        # Add some state to reset
        agent.messages.append({"role": "user", "content": "test"})
        agent.step = 5
        agent._trajectory.steps.append(Step())

        agent.reset()

        assert isinstance(agent._trajectory, Trajectory)
        assert agent._trajectory.steps == []
        assert len(agent.messages) == 1  # Should have system message
        assert agent.step == 0
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == initial_message_content

    def test_properties(self):
        """Test key properties."""
        agent = FrozenLakeAgent()
        assert agent.chat_completions == agent.messages
        assert agent.trajectory == agent._trajectory
        assert isinstance(agent.chat_completions, list)
        assert isinstance(agent.trajectory, Trajectory)

    def test_update_from_env_basic(self):
        """Test basic update_from_env functionality."""
        agent = FrozenLakeAgent()
        initial_message_count = len(agent.messages)

        observation = "P _ _ G\n_ O _ _\n_ _ _ _\n_ _ _ _"
        agent.update_from_env(observation, 0.0, False, {})

        # Check that message was added correctly
        assert len(agent.messages) == initial_message_count + 1
        assert agent.messages[-1]["role"] == "user"
        assert "Current Observation (0):" in agent.messages[-1]["content"]
        assert observation in agent.messages[-1]["content"]

    def test_update_from_model_basic(self):
        """Test basic update_from_model functionality."""
        agent = FrozenLakeAgent()

        # First need to have an observation
        agent.update_from_env("P _ G", 0.0, False, {})

        response = "I need to move right. ```Right```"
        action = agent.update_from_model(response)

        # Check that step was created
        assert len(agent._trajectory.steps) == 1
        current_step = agent._trajectory.steps[0]
        assert current_step.action == "3"  # Right = 3
        assert current_step.model_response == response

        # Check that message was added
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == response

        # Check step counter
        assert agent.step == 1

        # Check return value
        assert action.action == "3"

    def test_parse_model_response(self):
        """Test model response parsing for different directions."""
        agent = FrozenLakeAgent()

        test_cases = [
            ("I'll move ```Right```", "I'll move", "3"),
            ("Going up. ```Up```", "Going up.", "4"),
            ("Moving ```Down```", "Moving", "2"),
            ("Turn ```Left```", "Turn", "1"),
            ("Invalid move ```diagonal```", "Invalid move", "0"),  # Invalid action
            ("No action here", "No action here", "0"),  # No code block
        ]

        for response, expected_thought, expected_action in test_cases:
            thought, action = agent._parse_model_response(response)
            assert thought == expected_thought
            assert action == expected_action

    def test_basic_interaction_flow(self):
        """Test a basic complete interaction flow."""
        agent = FrozenLakeAgent()

        # Step 1: Environment provides observation
        observation = "P _ G"
        agent.update_from_env(observation, 0.0, False, {})
        assert len(agent._trajectory.steps) == 0  # No step until model responds
        assert len(agent.messages) == 2  # System + user message

        # Step 2: Model responds with action
        response = "I need to move right. ```Right```"
        action = agent.update_from_model(response)

        assert len(agent._trajectory.steps) == 1
        assert agent.step == 1
        assert action.action == "3"  # Right

        # Step 3: Environment provides new observation (could be goal reached)
        new_observation = "_ P G"
        agent.update_from_env(new_observation, 1.0, True, {"success": True})

        # Should have updated the trajectory
        assert len(agent.messages) == 4  # System + user + assistant + user

    def test_trajectory_to_dict(self):
        """Test that trajectory can be converted to dict."""
        agent = FrozenLakeAgent()

        # Add basic interaction
        agent.update_from_env("P _ G", 0.0, False, {})
        agent.update_from_model("Moving right. ```Right```")

        trajectory_dict = agent.trajectory.to_dict()
        assert isinstance(trajectory_dict, dict)
        assert "steps" in trajectory_dict
        assert isinstance(trajectory_dict["steps"], list)
