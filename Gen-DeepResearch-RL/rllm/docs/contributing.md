# Contributing to rLLM

We welcome contributions to rLLM! This guide will help you get started with the development process.

## Getting Started

## Development Workflow

### Creating a New Feature

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, ensuring you follow the code style guidelines
3. Add tests for your feature
4. Run the tests locally to ensure they pass
5. Commit your changes with a descriptive message
6. Push your branch to GitHub
7. Create a pull request from your branch to the main repository

### Running Tests

To run the tests:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_specific_file.py
```

### Code Style

We use pre-commit to perform code formatting and linting with Ruff:
We use pre-commit to help improve code quality. To initialize pre-commit, run:

```
pip install pre-commit
pre-commit install
```

To resolve CI errors locally, you can manually run pre-commit by:
```
pre-commit run
```

<!-- ## Adding New Components

### Adding a New Agent

1. Create a new file in `rllm/agents/` (e.g., `my_agent.py`)
2. Implement your agent by inheriting from the base `Agent` class
3. Add any necessary system prompts to `rllm/agents/system_prompts.py`
4. Register your agent in `rllm/agents/__init__.py`
5. Add tests in `tests/agents/`

### Adding a New Environment

1. Create a new file in `rllm/environments/` (e.g., `my_env.py`)
2. Implement your environment by inheriting from the base `BaseEnv` class
3. Register your environment in `rllm/environments/__init__.py`
4. Add tests in `tests/environments/`

### Adding a New Reward Function

1. Create a new file in `rllm/rewards/` (e.g., `my_reward.py`)
2. Implement your reward function
3. Register your reward function in `rllm/rewards/__init__.py`
4. Add tests in `tests/rewards/` -->

## Submitting a Pull Request

When you're ready to submit your changes:

1. Make sure your branch is up-to-date with the main branch
2. Ensure all tests pass and pre-commit check passes
3. Create a pull request with a descriptive title and detailed description
4. Link any relevant issues in your pull request description

## Code Review

After submitting a pull request:

1. Core maintainers will review your code
2. Address any feedback or comments
3. Once approved, your changes will be merged into the main branch

## Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions and ideas
- **Pull Requests**: For submitting code changes

## License

By contributing to rLLM, you agree that your contributions will be licensed under the project's MIT license. 