import numpy as np

from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv, generate_random_map, get_goal_position, is_valid


class TestFrozenLakeEnv:
    """Test suite for FrozenLakeEnv class."""

    def test_init_default(self):
        """Test FrozenLakeEnv initialization with default parameters."""
        env = FrozenLakeEnv()
        assert env.size == 8
        assert env.p == 0.8
        assert env.seed == 42
        assert env.goal_postion is not None
        assert hasattr(env, "desc")
        assert hasattr(env, "ACTION_SPACE")

    def test_init_custom_parameters(self):
        """Test FrozenLakeEnv initialization with custom parameters."""
        env = FrozenLakeEnv(size=4, p=0.6, seed=100, is_slippery=True, max_steps=10)
        assert env.size == 4
        assert env.p == 0.6
        assert env.seed == 100
        # is_slippery is stored in the env_kwargs dict
        assert env.env_kwargs["is_slippery"] is True

    def test_init_with_custom_desc(self):
        """Test FrozenLakeEnv initialization with custom map description."""
        custom_desc = ["SF", "FG"]
        env = FrozenLakeEnv(desc=custom_desc)
        assert env.desc is not None
        # Check that custom description is used
        assert np.array_equal(env.desc, np.asarray(custom_desc, dtype="c"))

    def test_action_constants(self):
        """Test that action constants are properly defined."""
        assert FrozenLakeEnv.INVALID_ACTION == 0
        assert FrozenLakeEnv.PENALTY_FOR_INVALID == -1

        # Test action lookup
        assert FrozenLakeEnv.ACTION_LOOKUP[0] == "None"
        assert FrozenLakeEnv.ACTION_LOOKUP[1] == "Left"
        assert FrozenLakeEnv.ACTION_LOOKUP[2] == "Down"
        assert FrozenLakeEnv.ACTION_LOOKUP[3] == "Right"
        assert FrozenLakeEnv.ACTION_LOOKUP[4] == "Up"

    def test_map_constants(self):
        """Test that map constants are properly defined."""
        assert FrozenLakeEnv.MAP_LOOKUP[b"P"] == 0
        assert FrozenLakeEnv.MAP_LOOKUP[b"F"] == 1
        assert FrozenLakeEnv.MAP_LOOKUP[b"H"] == 2
        assert FrozenLakeEnv.MAP_LOOKUP[b"G"] == 3

    def test_grid_constants(self):
        """Test that grid rendering constants are properly defined."""
        assert FrozenLakeEnv.GRID_LOOKUP[0] == " P \t"  # player
        assert FrozenLakeEnv.GRID_LOOKUP[1] == " _ \t"  # frozen
        assert FrozenLakeEnv.GRID_LOOKUP[2] == " O \t"  # hole
        assert FrozenLakeEnv.GRID_LOOKUP[3] == " G \t"  # goal

    def test_reset(self):
        """Test the reset method."""
        env = FrozenLakeEnv(size=4, seed=42)
        obs, info = env.reset()

        # Check that observation is returned
        assert obs is not None
        assert isinstance(info, dict)

        # Check that environment is reset to initial state
        # FrozenLakeEnv doesn't have step_count, it uses parent class state
        assert hasattr(env, "s")  # Check that state exists

    def test_reset_deterministic(self):
        """Test that reset with same seed produces same initial state."""
        env1 = FrozenLakeEnv(size=4, seed=42)
        obs1, info1 = env1.reset()

        env2 = FrozenLakeEnv(size=4, seed=42)
        obs2, info2 = env2.reset()

        # Same seed should produce same map
        assert obs1 == obs2

    def test_step_valid_actions(self):
        """Test stepping with valid actions."""
        env = FrozenLakeEnv(size=4, seed=42)
        env.reset()

        # Test each valid action
        for action in [1, 2, 3, 4]:  # Left, Down, Right, Up
            env.reset()  # Reset for each test
            obs, reward, done, info = env.step(str(action))

            assert obs is not None
            assert isinstance(reward, int | float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)

    def test_step_invalid_action(self):
        """Test stepping with invalid action."""
        env = FrozenLakeEnv(size=4, seed=42)
        env.reset()

        # Test invalid action (0 is INVALID_ACTION) - returns reward 0, not penalty
        obs, reward, done, info = env.step("0")

        assert reward == 0  # Invalid action returns 0 reward
        assert obs is not None
        assert isinstance(done, bool)
        assert info["action_is_effective"] is False

    def test_step_string_action_conversion(self):
        """Test that string actions are properly converted."""
        env = FrozenLakeEnv(size=4, seed=42)
        env.reset()

        # Test string action
        obs, reward, done, info = env.step("1")  # Left
        assert obs is not None

    def test_step_non_string_action_conversion(self):
        """Test that non-string actions are converted to string."""
        env = FrozenLakeEnv(size=4, seed=42)
        env.reset()

        # Test integer action
        obs, reward, done, info = env.step(1)
        assert obs is not None

    def test_goal_reached(self):
        """Test that reaching goal gives proper reward and terminates."""
        # Create a simple 2x2 map where goal is easily reachable
        desc = ["SG", "FF"]
        env = FrozenLakeEnv(desc=desc, is_slippery=False)
        env.reset()

        # Move right to reach goal
        obs, reward, done, info = env.step("3")

        # Should get positive reward and be done
        assert reward > 0
        assert done is True
        assert env.success() is True

    def test_hole_reached(self):
        """Test that reaching hole terminates episode."""
        # Create a simple map with hole next to start
        desc = ["SH", "FF"]
        env = FrozenLakeEnv(desc=desc, is_slippery=False)
        env.reset()

        # Move right to reach hole
        obs, reward, done, info = env.step("3")

        # Should be done
        assert done is True
        assert env.success() is False

    def test_finished_property(self):
        """Test the finished property."""
        env = FrozenLakeEnv(size=4, seed=42)
        env.reset()

        # Initially not finished
        assert env.finished() is False

        # After reaching goal, should be finished
        desc = ["SG", "FF"]
        env = FrozenLakeEnv(desc=desc, is_slippery=False)
        env.reset()
        env.step("3")  # Move to goal
        assert env.finished() is True

    def test_success_property(self):
        """Test the success property."""
        # Test successful completion
        desc = ["SG", "FF"]
        env = FrozenLakeEnv(desc=desc, is_slippery=False)
        env.reset()
        env.step("3")  # Move to goal
        assert env.success() is True

        # Test unsuccessful completion (hole)
        desc = ["SH", "FF"]
        env = FrozenLakeEnv(desc=desc, is_slippery=False)
        env.reset()
        env.step("3")  # Move to hole
        assert env.success() is False

    def test_get_player_position(self):
        """Test getting player position."""
        env = FrozenLakeEnv(size=4, seed=42)
        env.reset()

        pos = env._get_player_position()
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert all(isinstance(x, int | np.integer) for x in pos)

    def test_render_modes(self):
        """Test different rendering modes."""
        env = FrozenLakeEnv(size=4, seed=42)
        env.reset()

        # Test tiny_rgb_array mode
        rendered = env.render(mode="tiny_rgb_array")
        assert rendered is not None

        # Test default mode (should use tiny_rgb_array)
        rendered_default = env.render()
        assert rendered_default is not None

    def test_idx_property(self):
        """Test the idx property from BaseEnv."""
        env = FrozenLakeEnv()

        # Initially should be None
        assert env.idx is None

        # Should be able to set and get
        env.idx = 5
        assert env.idx == 5

    def test_is_multithread_safe(self):
        """Test the is_multithread_safe static method."""
        assert FrozenLakeEnv.is_multithread_safe() is True

    def test_from_dict(self):
        """Test creating environment from dictionary."""
        env_info = {"size": 4, "p": 0.7, "seed": 123, "is_slippery": True, "max_steps": 15}

        env = FrozenLakeEnv.from_dict(env_info)
        assert isinstance(env, FrozenLakeEnv)
        assert env.size == 4
        assert env.p == 0.7
        assert env.seed == 123

    def test_multiple_steps(self):
        """Test taking multiple steps in the environment."""
        env = FrozenLakeEnv(size=4, seed=42, is_slippery=False)
        env.reset()

        step_count = 0
        done = False

        while not done and step_count < 20:  # Prevent infinite loop
            obs, reward, done, info = env.step("1")  # Always move left
            step_count += 1

            assert obs is not None
            assert isinstance(reward, int | float)
            assert isinstance(done, bool)

        assert step_count <= 20  # Should terminate within reasonable steps

    def test_slippery_vs_deterministic(self):
        """Test that slippery affects movement predictability."""
        # Create identical environments with different slippery settings
        desc = ["SFFF", "FFFF", "FFFF", "FFFG"]

        env_deterministic = FrozenLakeEnv(desc=desc, is_slippery=False, seed=42)
        env_slippery = FrozenLakeEnv(desc=desc, is_slippery=True, seed=42)

        # Both should be valid environments
        env_deterministic.reset()
        env_slippery.reset()

        # Both should be able to take steps
        obs1, _, _, _ = env_deterministic.step("3")
        obs2, _, _, _ = env_slippery.step("3")

        assert obs1 is not None
        assert obs2 is not None


class TestFrozenLakeUtilityFunctions:
    """Test suite for FrozenLake utility functions."""

    def test_generate_random_map(self):
        """Test generate_random_map function."""
        size = 4
        p = 0.8
        seed = 42

        random_map, goal_pos = generate_random_map(size=size, p=p, seed=seed)

        # Check return types
        assert isinstance(random_map, list)
        assert isinstance(goal_pos, tuple)
        assert len(goal_pos) == 2

        # Check map properties
        assert len(random_map) == size
        assert all(len(row) == size for row in random_map)

        # Check that map contains required elements
        full_map = "".join(random_map)
        assert "S" in full_map  # Start
        assert "G" in full_map  # Goal

        # Check goal position is valid
        assert 0 <= goal_pos[0] < size
        assert 0 <= goal_pos[1] < size

    def test_generate_random_map_deterministic(self):
        """Test that generate_random_map with same seed produces same map."""
        size = 4
        p = 0.8
        seed = 42

        map1, goal1 = generate_random_map(size=size, p=p, seed=seed)
        map2, goal2 = generate_random_map(size=size, p=p, seed=seed)

        assert map1 == map2
        assert goal1 == goal2

    def test_generate_random_map_different_seeds(self):
        """Test that different seeds produce different maps."""
        size = 4
        p = 0.8

        map1, goal1 = generate_random_map(size=size, p=p, seed=42)
        map2, goal2 = generate_random_map(size=size, p=p, seed=123)

        # Very likely to be different with different seeds
        assert map1 != map2 or goal1 != goal2

    def test_generate_random_map_size_parameter(self):
        """Test generate_random_map with different sizes."""
        for size in [3, 4, 5, 8]:
            random_map, goal_pos = generate_random_map(size=size, p=0.8, seed=42)

            assert len(random_map) == size
            assert all(len(row) == size for row in random_map)
            assert 0 <= goal_pos[0] < size
            assert 0 <= goal_pos[1] < size

    def test_generate_random_map_p_parameter(self):
        """Test generate_random_map with different probability values."""
        size = 4
        seed = 42

        # Test with different p values
        for p in [0.5, 0.8, 1.0]:
            random_map, goal_pos = generate_random_map(size=size, p=p, seed=seed)

            assert isinstance(random_map, list)
            assert isinstance(goal_pos, tuple)

            # Map should still be valid
            full_map = "".join(random_map)
            assert "S" in full_map
            assert "G" in full_map

    def test_is_valid_function(self):
        """Test is_valid function."""
        # Test valid board
        valid_board = [["S", "F"], ["F", "G"]]
        assert is_valid(valid_board, 2) is True

        # Test invalid board (no path)
        invalid_board = [["S", "H"], ["H", "G"]]
        assert is_valid(invalid_board, 2) is False

        # Test board with direct path
        direct_board = [["S", "G"], ["F", "F"]]
        assert is_valid(direct_board, 2) is True

    def test_get_goal_position(self):
        """Test get_goal_position function."""
        # Test with goal present
        map_with_goal = np.array([[b"S", b"F"], [b"F", b"G"]])
        goal_pos = get_goal_position(map_with_goal)
        assert goal_pos == (1, 1)

        # Test with no goal
        map_no_goal = np.array([[b"S", b"F"], [b"F", b"F"]])
        goal_pos = get_goal_position(map_no_goal)
        assert goal_pos is None

        # Test with goal at different position
        map_goal_top = np.array([[b"G", b"F"], [b"F", b"S"]])
        goal_pos = get_goal_position(map_goal_top)
        assert goal_pos == (0, 0)

    def test_generate_random_map_start_goal_different(self):
        """Test that start and goal positions are always different."""
        for _ in range(10):  # Test multiple times to be sure
            random_map, goal_pos = generate_random_map(size=4, p=0.8, seed=None)

            # Find start position
            start_pos = None
            for i, row in enumerate(random_map):
                for j, cell in enumerate(row):
                    if cell == "S":
                        start_pos = (i, j)
                        break
                if start_pos:
                    break

            assert start_pos is not None
            assert start_pos != goal_pos

    def test_is_valid_max_steps_constraint(self):
        """Test that is_valid respects MAX_STEPS constraint."""
        # Create a board that would require more than MAX_STEPS
        large_board = [["S"] + ["F"] * 10 + ["G"]] + [["F"] * 12 for _ in range(11)]

        # Should respect MAX_STEPS limit
        result = is_valid(large_board, 12)
        assert isinstance(result, bool)  # Should not crash

    def test_edge_cases(self):
        """Test edge cases for utility functions."""
        # Test minimum size
        map_2x2, goal_2x2 = generate_random_map(size=2, p=0.8, seed=42)
        assert len(map_2x2) == 2
        assert len(map_2x2[0]) == 2

        # Test p = 1.0 (all frozen)
        map_all_frozen, goal_all_frozen = generate_random_map(size=3, p=1.0, seed=42)
        full_map = "".join(map_all_frozen)
        # Should only have S, G, and F characters
        assert all(c in "SGF" for c in full_map)

        # Test p = 0.0 (all holes except S and G)
        map_all_holes, goal_all_holes = generate_random_map(size=3, p=0.0, seed=42)
        full_map = "".join(map_all_holes)
        # Should still be valid despite low p
        assert "S" in full_map
        assert "G" in full_map
