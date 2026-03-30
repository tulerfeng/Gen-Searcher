from unittest.mock import Mock, patch

import pytest

from rllm.agents.agent import Step, Trajectory
from rllm.agents.tool_agent import ToolAgent
from rllm.tools.tool_base import Tool


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name="mock_tool"):
        self.name = name
        self.description = "A mock tool for testing"
        self.parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "Test query"}}, "required": ["query"]}

    @property
    def json(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def call(self, **kwargs):
        return f"Mock tool called with {kwargs}"


class TestToolAgent:
    """Simplified test suite for ToolAgent core functionality."""

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_init_default(self, mock_get_parser):
        """Test ToolAgent initialization with default parameters."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        assert agent.system_prompt is not None
        assert agent.tools is not None
        assert agent.tool_parser is not None
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1  # System message
        assert agent.messages[0]["role"] == "system"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_init_with_tool_map(self, mock_get_parser):
        """Test ToolAgent initialization with tool_map."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        tool_map = {"mock_tool": MockTool}
        agent = ToolAgent(tool_map=tool_map)
        assert agent.tools is not None

    def test_init_with_both_tools_and_tool_map_raises_error(self):
        """Test that providing both tools and tool_map raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both 'tools' and 'tool_map' parameters"):
            ToolAgent(tools=["calculator"], tool_map={"mock": MockTool})

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_properties(self, mock_get_parser):
        """Test key properties."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        assert agent.chat_completions == agent.messages
        assert agent.trajectory == agent._trajectory
        assert isinstance(agent.chat_completions, list)
        assert isinstance(agent.trajectory, Trajectory)

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_reset(self, mock_get_parser):
        """Test the reset method."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        initial_message_count = len(agent.messages)

        # Add some state to reset
        agent.messages.append({"role": "user", "content": "test"})
        agent._trajectory.steps.append(Step())

        agent.reset()

        assert isinstance(agent._trajectory, Trajectory)
        assert agent._trajectory.steps == []
        assert len(agent.messages) == initial_message_count  # Should have system message
        assert agent.messages[0]["role"] == "system"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_format_observation_as_messages(self, mock_get_parser):
        """Test _format_observation_as_messages with different input types."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Test with dict containing question
        obs = {"question": "What is the weather?"}
        messages = agent._format_observation_as_messages(obs)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the weather?"

        # Test with string
        obs = "Hello, how can I help?"
        messages = agent._format_observation_as_messages(obs)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how can I help?"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_env_basic(self, mock_get_parser):
        """Test basic update_from_env functionality."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        initial_message_count = len(agent.messages)

        observation = {"question": "What's the weather today?"}
        agent.update_from_env(observation, 0.0, False, {})

        # Check that message was added
        assert len(agent.messages) == initial_message_count + 1
        assert agent.messages[-1]["role"] == "user"
        assert agent.messages[-1]["content"] == "What's the weather today?"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_model_basic(self, mock_get_parser):
        """Test basic update_from_model functionality."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"

        # Mock tool call parsing
        mock_tool_call = Mock()
        mock_tool_call.to_dict.return_value = {"name": "search", "arguments": {"query": "weather"}}
        mock_parser_instance.parse.return_value = [mock_tool_call]

        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # First provide observation
        agent.update_from_env({"question": "test"}, 0.0, False, {})

        response = "I'll search for the weather information."
        action = agent.update_from_model(response)

        # Check that step was created
        assert len(agent._trajectory.steps) == 1
        current_step = agent._trajectory.steps[0]
        assert current_step.model_response == response
        assert isinstance(current_step.action, list)
        assert len(current_step.action) == 1
        assert current_step.action[0]["type"] == "function"

        # Check that message was added
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == response

        # Check return value
        assert isinstance(action.action, list)

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_model_with_parsing_error(self, mock_get_parser):
        """Test update_from_model when tool parsing fails."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"

        # Mock parsing error
        mock_parser_instance.parse.side_effect = Exception("Parsing failed")

        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # First provide observation
        agent.update_from_env({"question": "test"}, 0.0, False, {})

        response = "This is a response without tool calls."
        _ = agent.update_from_model(response)

        # Should create a finish tool call
        current_step = agent._trajectory.steps[0]
        assert isinstance(current_step.action, list)
        assert len(current_step.action) == 1
        assert current_step.action[0]["function"]["name"] == "finish"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_basic_interaction_flow(self, mock_get_parser):
        """Test a basic complete interaction flow."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"

        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.to_dict.return_value = {"name": "search", "arguments": {"query": "weather"}}
        mock_parser_instance.parse.return_value = [mock_tool_call]

        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        initial_message_count = len(agent.messages)

        # Step 1: Initial environment update
        observation1 = {"question": "What's the weather?"}
        agent.update_from_env(observation1, 0.0, False, {})

        assert len(agent.messages) == initial_message_count + 1

        # Step 2: Model response with tool call
        response1 = "I'll search for weather information."
        action = agent.update_from_model(response1)

        assert len(agent._trajectory.steps) == 1
        assert len(agent.messages) == initial_message_count + 2
        assert isinstance(action.action, list)

        # Step 3: Environment feedback with tool results
        tool_outputs = {"tool_outputs": {"call_1": "Weather is sunny, 25°C"}}
        agent.update_from_env(tool_outputs, 0.0, False, {})

        # Should have tool message
        assert any(msg["role"] == "tool" for msg in agent.messages)

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_trajectory_to_dict(self, mock_get_parser):
        """Test that trajectory can be converted to dict."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_instance.parse.return_value = []
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add basic interaction
        agent.update_from_env({"question": "test"}, 0.0, False, {})
        agent.update_from_model("test response")

        trajectory_dict = agent.trajectory.to_dict()
        assert isinstance(trajectory_dict, dict)
        assert "steps" in trajectory_dict
        assert isinstance(trajectory_dict["steps"], list)
