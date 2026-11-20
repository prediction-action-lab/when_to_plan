import numpy as np
import gymnasium as gym
import torch as th
import heapq
from PIL import Image
from typing import Tuple, List, Optional, Set
import copy
import os
import pickle


class BoxPushEnv(gym.Env):
    """
    Sokoban-style environment where the agent must push 1-3 boxes to goal locations.
    Uses fully observable symbolic observations with relative positions.
    """

    def __init__(self, cfg, seed=42):
        self.max_time = 250
        self.cfg = cfg

        # Configuration
        self.grid_size_range = cfg.env.get("grid_size_range", (8, 8))
        self.num_boxes_range = cfg.env.get("num_boxes_range", (3, 3))
        self.include_walls = cfg.env.get("include_walls", False)
        self.scaling_factor = cfg.env.scaling_factor
        self.num_cached_tasks = cfg.experiment.get("num_cached_tasks", 1)

        # Max number of boxes (for fixed observation size)
        self.max_boxes = self.num_boxes_range[1]

        obs_size = 2 + (self.max_boxes * 3) + (self.max_boxes * 2)
        if self.include_walls:
            obs_size += 4

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self.rng = np.random.default_rng(seed=seed)

        # Load or generate puzzles — keep full list for OOD ablation access
        self._all_cached_puzzles = self._load_or_generate_puzzles()
        self.cached_puzzles = self._all_cached_puzzles[: self.num_cached_tasks]

        # Initialize with first puzzle
        self._load_puzzle(0)

    def _get_puzzles_filepath(self) -> str:
        """Get the path to the saved puzzles file."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Create a filename based on configuration parameters
        filename = f"puzzles/puzzles_grid{self.grid_size_range[0]}-{self.grid_size_range[1]}_boxes{self.num_boxes_range[0]}-{self.num_boxes_range[1]}_walls{self.include_walls}.pkl"
        return os.path.join(base_dir, filename)

    def set_cache_task_range(self, start_idx: int, end_idx: int):
        """Set the range of cached tasks to use."""
        if (
            start_idx < 0
            or end_idx > len(self._all_cached_puzzles)
            or start_idx >= end_idx
        ):
            raise ValueError(
                f"Invalid cache task range: start_idx={start_idx}, end_idx={end_idx}, total_puzzles={len(self._all_cached_puzzles)}"
            )
        self.cached_puzzles = self._all_cached_puzzles[start_idx:end_idx]
        self.num_cached_tasks = end_idx - start_idx
        print(
            f"Set cached tasks to range: {start_idx} to {end_idx} (total {self.num_cached_tasks})"
        )

        # Load the first puzzle in the new range
        self._load_puzzle(0)

    def _load_or_generate_puzzles(self) -> List:
        """Load puzzles from file or generate them if needed."""
        filepath = self._get_puzzles_filepath()

        # Try to load existing puzzles
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    saved_puzzles = pickle.load(f)

                # Check if we have enough puzzles
                if len(saved_puzzles) >= self.num_cached_tasks:
                    print(f"Loaded {len(saved_puzzles)} puzzles from {filepath}")
                    return saved_puzzles
                else:
                    print(
                        f"Found {len(saved_puzzles)} puzzles but need {self.num_cached_tasks}. Regenerating..."
                    )
            except Exception as e:
                print(f"Error loading puzzles from {filepath}: {e}. Regenerating...")

        # Generate new puzzles
        print(f"Generating {self.num_cached_tasks} solvable puzzles...")
        puzzles = self._generate_solvable_puzzles(self.num_cached_tasks)

        # Save to file
        try:
            with open(filepath, "wb") as f:
                pickle.dump(puzzles, f)
            print(f"Saved {len(puzzles)} puzzles to {filepath}")
        except Exception as e:
            print(f"Warning: Could not save puzzles to {filepath}: {e}")

        return puzzles

    def _generate_solvable_puzzles(self, num_puzzles: int) -> List:
        """Generate a list of solvable puzzles."""
        puzzles = []
        attempts = 0
        max_attempts = num_puzzles * 100

        while len(puzzles) < num_puzzles and attempts < max_attempts:
            attempts += 1
            puzzle = self.generate_puzzle()
            maze, agent_pos, box_locs, goal_locs = puzzle

            # Validate puzzle is solvable
            result = sokoban_a_star(
                maze, agent_pos, box_locs, goal_locs, max_expansions=10000
            )
            if result is not None:  # Found complete solution
                puzzles.append(puzzle)
                if len(puzzles) % 10 == 0:
                    print(f"  Generated {len(puzzles)}/{num_puzzles} puzzles...")

        if len(puzzles) < num_puzzles:
            print(f"Warning: Only generated {len(puzzles)} solvable puzzles")
            if len(puzzles) == 0:
                raise RuntimeError("Failed to generate any solvable puzzles")

        return puzzles

    def _load_puzzle(self, puzzle_idx):
        """Load a cached puzzle by index."""
        maze, agent_pos, box_locs, goal_locs = self.cached_puzzles[puzzle_idx]
        # Deep copy to avoid modifying the cached version
        self.task = np.copy(maze)
        self.agent_location = np.copy(agent_pos)
        self.box_locations = copy.deepcopy(box_locs)
        self.goal_locations = copy.deepcopy(goal_locs)

    def generate_puzzle(self):
        """
        Generate a Sokoban puzzle with boxes and goals.

        Returns:
            maze: 2D numpy array (1=free, 0=wall)
            agent_pos: (row, col) starting position
            box_positions: set of (row, col) box positions
            goal_positions: set of (row, col) goal positions
        """
        # Create a room with walls around the edges
        size = self.rng.integers(self.grid_size_range[0], self.grid_size_range[1] + 1)
        maze = np.ones((size, size), dtype=np.int8)

        # Add walls around the border
        maze[0, :] = 0
        maze[-1, :] = 0
        maze[:, 0] = 0
        maze[:, -1] = 0

        # Optionally add some internal walls for complexity
        if self.include_walls:
            num_internal_walls = self.rng.integers(0, 3)
            for _ in range(num_internal_walls):
                if self.rng.random() < 0.5:  # Horizontal wall
                    row = self.rng.integers(2, size - 2)
                    col_start = self.rng.integers(1, size - 3)
                    col_end = col_start + self.rng.integers(1, 3)
                    maze[row, col_start : min(col_end, size - 1)] = 0
                else:  # Vertical wall
                    col = self.rng.integers(2, size - 2)
                    row_start = self.rng.integers(1, size - 3)
                    row_end = row_start + self.rng.integers(1, 3)
                    maze[row_start : min(row_end, size - 1), col] = 0

        # Get all free positions
        free_positions = [
            (r, c)
            for r in range(1, size - 1)
            for c in range(1, size - 1)
            if maze[r, c] == 1
        ]

        # Place agent
        agent_idx = self.rng.integers(0, len(free_positions))
        agent_pos = np.array(free_positions[agent_idx])
        free_positions.pop(agent_idx)

        # Determine number of boxes
        num_boxes = self.rng.integers(
            self.num_boxes_range[0], self.num_boxes_range[1] + 1
        )

        # Place boxes
        box_positions = set()
        for _ in range(num_boxes):
            if not free_positions:
                break
            box_idx = self.rng.integers(0, len(free_positions))
            box_positions.add(free_positions[box_idx])
            free_positions.pop(box_idx)

        # Place goals
        goal_positions = set()
        for _ in range(num_boxes):
            if not free_positions:
                break
            goal_idx = self.rng.integers(0, len(free_positions))
            goal_positions.add(free_positions[goal_idx])
            free_positions.pop(goal_idx)

        return maze, agent_pos, box_positions, goal_positions

    def is_failed_state(self) -> bool:
        """
        Check if a box is in an unrecoverable position (corner or against wall).
        A box is stuck if it's not on a goal and:
        - In a corner (two adjacent walls)
        - Against a wall and blocked by another box or wall on perpendicular side
        """
        H, W = self.task.shape

        for box_pos in self.box_locations:
            # Skip if box is on a goal
            if box_pos in self.goal_locations:
                continue

            r, c = box_pos

            # Check each direction
            up_blocked = (
                r <= 0 or self.task[r - 1, c] == 0 or (r - 1, c) in self.box_locations
            )
            down_blocked = (
                r >= H - 1
                or self.task[r + 1, c] == 0
                or (r + 1, c) in self.box_locations
            )
            left_blocked = (
                c <= 0 or self.task[r, c - 1] == 0 or (r, c - 1) in self.box_locations
            )
            right_blocked = (
                c >= W - 1
                or self.task[r, c + 1] == 0
                or (r, c + 1) in self.box_locations
            )

            # Check for corner or edge cases
            if (
                (up_blocked and left_blocked)
                or (up_blocked and right_blocked)
                or (down_blocked and left_blocked)
                or (down_blocked and right_blocked)
            ):
                return True

            # Check for box stuck against wall with perpendicular blockage
            if up_blocked or down_blocked:
                if left_blocked or right_blocked:
                    return True

        return False

    def render(self, mode="human", scaling_factor=None):
        """Render for visualization using MiniGrid-style rendering."""
        img = self._render_to_image(scaled=True, scaling_factor=scaling_factor)

        if mode == "rgb_array":
            return img
        elif mode == "human":
            img = Image.fromarray(img)
            return img

    def _render_to_image(self, scaled=False, scaling_factor=None):
        """
        Render environment state to a H x W x 3 uint8 numpy image.
        For visualization only - not used for observations.
        """
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor

        H, W = self.task.shape
        image = np.zeros((H, W, 3), dtype=np.uint8)

        # Walls black, free white
        image[self.task == 0] = [0, 0, 0]
        image[self.task == 1] = [255, 255, 255]

        # Goals (red)
        for gr, gc in self.goal_locations:
            image[gr, gc] = [255, 0, 0]

        # Boxes (blue)
        for br, bc in self.box_locations:
            image[br, bc] = [0, 0, 255]

        # Boxes on goals (purple)
        for pos in self.box_locations.intersection(self.goal_locations):
            br, bc = pos
            image[br, bc] = [128, 0, 128]

        # Agent (green)
        ar, ac = int(self.agent_location[0]), int(self.agent_location[1])
        image[ar, ac] = [0, 255, 0]

        if scaled and self.scaling_factor > 1:
            image = image.repeat(self.scaling_factor, axis=0).repeat(
                self.scaling_factor, axis=1
            )

        return image

    def get_distance_to_goal(self):
        # Return l2 distance of boxes to nearest goals as a simple heuristic metric
        total_dist = 0.0
        for box_pos in self.box_locations:
            min_dist = float("inf")
            for goal_pos in self.goal_locations:
                dist = abs(box_pos[0] - goal_pos[0]) + abs(box_pos[1] - goal_pos[1])
                min_dist = min(min_dist, dist)
            total_dist += min_dist

        return total_dist / self.cfg.env.distance_to_goal_scaling_factor

    def get_observation(self):
        """
        Get fully observable symbolic observation with relative positions.

        Returns:
            Feature vector with:
            - Agent position normalized (2): row/H, col/W
            - For each box slot (max_boxes):
              - Relative position to agent (2): (box_row - agent_row) / H, (box_col - agent_col) / W
              - On goal flag (1): 1.0 if box on goal, 0.0 otherwise
            - For each goal slot (max_boxes):
              - Relative position to agent (2): (goal_row - agent_row) / H, (goal_col - agent_col) / W
              (If fewer goals exist, remaining slots are padded with zeros)
            - Optional: Wall distances (4): normalized distance to wall in each direction
        """
        H, W = self.task.shape
        agent_r, agent_c = self.agent_location

        obs_parts = []

        # 1. Normalized agent position
        obs_parts.append(agent_r / H)
        obs_parts.append(agent_c / W)

        # 2. Box information (relative position + on_goal flag)
        boxes_list = sorted(list(self.box_locations))  # Sort for consistency
        for i in range(self.max_boxes):
            if i < len(boxes_list):
                box_r, box_c = boxes_list[i]
                # Relative position (normalized)
                obs_parts.append((box_r - agent_r) / H)
                obs_parts.append((box_c - agent_c) / W)
                # On goal flag
                obs_parts.append(1.0 if boxes_list[i] in self.goal_locations else 0.0)
            else:
                # Padding for missing boxes
                obs_parts.extend([0.0, 0.0, 0.0])

        # 3. Goal information (relative position)
        goals_list = sorted(list(self.goal_locations))  # Sort for consistency
        for i in range(self.max_boxes):
            if i < len(goals_list):
                goal_r, goal_c = goals_list[i]
                # Relative position (normalized)
                obs_parts.append((goal_r - agent_r) / H)
                obs_parts.append((goal_c - agent_c) / W)
            else:
                # Padding for missing goals
                obs_parts.extend([0.0, 0.0])

        # 4. Optional: Wall distances in each direction
        if self.include_walls:
            directions = [
                (-1, 0),  # Up
                (1, 0),  # Down
                (0, -1),  # Left
                (0, 1),  # Right
            ]
            for dr, dc in directions:
                dist = 0
                r, c = agent_r, agent_c
                while True:
                    r += dr
                    c += dc
                    dist += 1
                    if not (0 <= r < H and 0 <= c < W) or self.task[r, c] == 0:
                        break
                # Normalize distance by grid size
                obs_parts.append(dist / max(H, W))

        return np.array(obs_parts, dtype=np.float32)

    def get_tensor_state(self):
        """Return observation as torch tensor (1D vector)."""
        state = self.get_observation()
        return th.from_numpy(state).to(dtype=th.float32)

    def reset(self, puzzle_idx=None, **kwargs):
        self.time = 0

        # Sample a random puzzle from the cache
        if puzzle_idx is None:
            puzzle_idx = self.rng.integers(0, self.num_cached_tasks)
        # print(puzzle_idx)
        self._load_puzzle(puzzle_idx)

        self.map_size = max(self.task.shape)
        state = self.get_observation()

        info = {"task_idx": puzzle_idx}
        self.task_idx = puzzle_idx
        return state, info

    def step(self, action):
        self.time += 1
        info = {"task_idx": self.task_idx}

        # Epsilon action noise: with probability action_noise, replace with a random
        # movement action (0-3), leaving the no-op (4) unchanged.
        action_noise = getattr(self.cfg, "action_noise", 0.0)
        if action != 4 and action_noise > 0.0 and self.rng.random() < action_noise:
            action = int(self.rng.integers(0, 4))

        if action == 4:  # No-op action
            return_state = self.get_observation()
            reward = -self._compute_distance_heuristic()
            term = False
            trunc = self.time > 200
            return return_state, reward, term, trunc, info

        action_map = {
            0: np.array([1, 0]),  # Down
            1: np.array([0, 1]),  # Right
            2: np.array([-1, 0]),  # Up
            3: np.array([0, -1]),  # Left
        }

        direction = action_map[action]
        new_agent_pos = self.agent_location + direction

        H, W = self.task.shape

        # Check if new position is valid
        if not (0 <= new_agent_pos[0] < H and 0 <= new_agent_pos[1] < W):
            # Out of bounds, don't move
            pass
        elif self.task[new_agent_pos[0], new_agent_pos[1]] == 0:
            # Hit a wall, don't move
            pass
        elif tuple(new_agent_pos) in self.box_locations:
            # Trying to push a box
            box_new_pos = new_agent_pos + direction

            # Check if box can be pushed
            if (
                0 <= box_new_pos[0] < H
                and 0 <= box_new_pos[1] < W
                and self.task[box_new_pos[0], box_new_pos[1]] == 1
                and tuple(box_new_pos) not in self.box_locations
            ):
                # Valid push
                self.box_locations.remove(tuple(new_agent_pos))
                self.box_locations.add(tuple(box_new_pos))
                self.agent_location = new_agent_pos
        else:
            # Free space, move
            self.agent_location = new_agent_pos

        # Get new state
        return_state = self.get_observation()

        # Reward: negative distance from boxes to nearest goals
        reward = -self._compute_distance_heuristic()

        # Check termination
        term = self.box_locations == self.goal_locations
        # if term:
        #     reward += 100.0

        trunc = self.time > self.max_time

        return return_state, reward, term, trunc, info

    def get_model_free_spaces(self):
        """Return observation and action space shapes for model-free actor."""
        obs_space = self.observation_space
        action_space = self.action_space

        return obs_space, action_space

    def model_free_step(self, action):
        return self.step(action)

    def _compute_distance_heuristic(self) -> float:
        """Compute sum of minimum distances from each box to nearest goal."""
        if not self.box_locations or not self.goal_locations:
            return 0.0

        total_dist = 0.0
        for box_pos in self.box_locations:
            min_dist = float("inf")
            for goal_pos in self.goal_locations:
                dist = abs(box_pos[0] - goal_pos[0]) + abs(box_pos[1] - goal_pos[1])
                min_dist = min(min_dist, dist)
            total_dist += min_dist

        return total_dist

    def get_plan(self, max_expansions=None, plan_length=None, pretrain=False, **kwargs):
        """Get optimal plan using A* search for Sokoban.

        Args:
            max_expansions: Maximum number of search expansions
            plan_length: "short", "medium", or "long" - controls search budget and depth
        """
        # Map plan length to max search expansions and plan depth
        if plan_length == "short":
            max_expansions = 5000
            plan_depth = self.cfg.env.short_plan_horizon
        elif plan_length == "medium":
            max_expansions = 5000
            plan_depth = self.cfg.env.medium_plan_horizon
        elif plan_length == "long":
            max_expansions = 5000
            plan_depth = self.cfg.env.long_plan_horizon
        else:
            max_expansions = 400
            plan_depth = None  # No depth limit

        if pretrain:
            max_expansions = 5000
            plan_depth = None

        return sokoban_a_star(
            self.task,
            self.agent_location,
            self.box_locations,
            self.goal_locations,
            max_expansions=max_expansions,
            plan_depth=plan_depth,
        )


def sokoban_a_star(
    maze: np.ndarray,
    agent_start: np.ndarray,
    boxes_start: Set[Tuple[int, int]],
    goals: Set[Tuple[int, int]],
    max_expansions: Optional[int] = None,
    plan_depth: Optional[int] = None,
) -> Optional[List[int]]:
    """
    Optimized A* search for Sokoban.

    Args:
        maze: 2D grid representing the environment
        agent_start: Starting position of the agent
        boxes_start: Set of starting box positions
        goals: Set of goal positions
        max_expansions: Maximum number of search expansions
        plan_depth: If specified, only search up to this depth and return best plan found

    Returns:
        List of actions representing the plan
    """
    H, W = maze.shape

    # Precompute goal distances for faster heuristic
    goals_list = list(goals)
    goal_boxes = frozenset(goals)

    # Cache for heuristic values
    heuristic_cache = {}

    def heuristic(box_set):
        """Cached sum of Manhattan distances from boxes to nearest goals."""
        if box_set in heuristic_cache:
            return heuristic_cache[box_set]

        if box_set == goal_boxes:
            heuristic_cache[box_set] = 0
            return 0

        total = 0
        for box in box_set:
            min_dist = min(abs(box[0] - g[0]) + abs(box[1] - g[1]) for g in goals_list)
            total += min_dist

        heuristic_cache[box_set] = total
        return total

    # Precompute deadlock zones (walls and corners that are not goals)
    deadlock_positions = set()
    for r in range(H):
        for c in range(W):
            if (r, c) in goals or maze[r, c] == 0:
                continue

            up_wall = r <= 0 or maze[r - 1, c] == 0
            down_wall = r >= H - 1 or maze[r + 1, c] == 0
            left_wall = c <= 0 or maze[r, c - 1] == 0
            right_wall = c >= W - 1 or maze[r, c + 1] == 0

            # Corner deadlock
            if (
                (up_wall and left_wall)
                or (up_wall and right_wall)
                or (down_wall and left_wall)
                or (down_wall and right_wall)
            ):
                deadlock_positions.add((r, c))

    def is_deadlock_fast(box_set):
        """Fast deadlock check using precomputed positions."""
        # Check precomputed corner deadlocks
        if box_set & deadlock_positions:
            return True

        # Check box-on-box deadlocks
        for box in box_set:
            if box in goals:
                continue
            r, c = box

            up_blocked = r <= 0 or maze[r - 1, c] == 0 or (r - 1, c) in box_set
            down_blocked = r >= H - 1 or maze[r + 1, c] == 0 or (r + 1, c) in box_set
            left_blocked = c <= 0 or maze[r, c - 1] == 0 or (r, c - 1) in box_set
            right_blocked = c >= W - 1 or maze[r, c + 1] == 0 or (r, c + 1) in box_set

            # Check if stuck against wall with perpendicular blockage
            if (up_blocked or down_blocked) and (left_blocked or right_blocked):
                return True
        return False

    # State: (agent_pos, frozenset(box_positions))
    start_state = (tuple(agent_start), frozenset(boxes_start))
    start_h = heuristic(start_state[1])

    # A* setup
    open_heap = [(start_h, 0, start_state)]
    came_from = {start_state: None}
    g_score = {start_state: 0}
    closed = set()

    expansions = 0
    best_h = start_h
    best_state = start_state
    best_at_target_depth = None  # Track best plan at target depth
    best_h_at_target_depth = float("inf")

    # Precompute directions as tuples (much faster than numpy arrays)
    directions = [
        ((1, 0), 0),  # Down
        ((0, 1), 1),  # Right
        ((-1, 0), 2),  # Up
        ((0, -1), 3),  # Left
    ]

    while open_heap:
        current_f, current_g, current_state = heapq.heappop(open_heap)

        if current_state in closed:
            continue

        expansions += 1
        agent_pos, box_set = current_state
        curr_h = current_f - current_g  # Reuse computed heuristic from f-score

        # If plan_depth is specified, track best plan at that depth
        if plan_depth is not None and current_g == plan_depth:
            if curr_h < best_h_at_target_depth:
                best_h_at_target_depth = curr_h
                best_at_target_depth = current_state

                # If we reached goal at target depth, return early
                if box_set == goal_boxes:
                    actions = _reconstruct_actions(came_from, current_state)
                    return actions

            # Don't expand beyond target depth
            continue

        if curr_h < best_h:
            best_h = curr_h
            best_state = current_state

        # Check expansion limit
        if max_expansions is not None and expansions >= max_expansions:
            actions = _reconstruct_actions(came_from, best_state)
            # If no actions found (best_state is start_state), try to take at least one valid step
            if not actions:
                actions = _get_greedy_action(maze, agent_start, boxes_start, goals_list)
            return actions

        # Check goal
        if box_set == goal_boxes:
            actions = _reconstruct_actions(came_from, current_state)
            return actions

        closed.add(current_state)

        # Expand neighbors
        agent_r, agent_c = agent_pos
        for (dr, dc), action in directions:
            new_agent_r = agent_r + dr
            new_agent_c = agent_c + dc
            new_agent_pos = (new_agent_r, new_agent_c)

            # Check bounds and walls
            if not (0 <= new_agent_r < H and 0 <= new_agent_c < W):
                continue
            if maze[new_agent_r, new_agent_c] == 0:
                continue

            new_box_set = box_set

            # Check if pushing a box
            if new_agent_pos in box_set:
                box_new_r = new_agent_r + dr
                box_new_c = new_agent_c + dc
                box_new_pos = (box_new_r, box_new_c)

                # Check if box can be pushed
                if not (0 <= box_new_r < H and 0 <= box_new_c < W):
                    continue
                if maze[box_new_r, box_new_c] == 0:
                    continue
                if box_new_pos in box_set:
                    continue

                # Create new box set with moved box
                new_box_set = (box_set - {new_agent_pos}) | {box_new_pos}

                # Check for deadlock
                if is_deadlock_fast(new_box_set):
                    continue

            new_state = (new_agent_pos, new_box_set)

            if new_state in closed:
                continue

            tentative_g = current_g + 1

            if new_state not in g_score or tentative_g < g_score[new_state]:
                g_score[new_state] = tentative_g
                came_from[new_state] = (current_state, action)
                h_score = heuristic(new_box_set)
                f_score = tentative_g + h_score
                heapq.heappush(open_heap, (f_score, tentative_g, new_state))

    # No solution found - return best state encountered
    # If plan_depth was specified and we found a plan at that depth, use it
    if plan_depth is not None and best_at_target_depth is not None:
        actions = _reconstruct_actions(came_from, best_at_target_depth)
        if actions:  # If we have a valid plan at target depth
            return actions

    # Otherwise, use the overall best state
    actions = _reconstruct_actions(came_from, best_state)

    # If no actions found (best_state is start_state), try to take at least one valid step
    if not actions:
        actions = _get_greedy_action(maze, agent_start, boxes_start, goals_list)

    # Final safety: ensure we always return at least one action
    if not actions:
        actions = [0]  # Default to moving down

    return actions


def _get_greedy_action(maze, agent_pos, box_set, goals_list) -> List[int]:
    """Get a single greedy action that moves towards the goals."""
    H, W = maze.shape
    agent_pos = tuple(agent_pos) if isinstance(agent_pos, np.ndarray) else agent_pos

    # Try all 4 directions
    directions = [
        ((1, 0), 0),  # Down
        ((0, 1), 1),  # Right
        ((-1, 0), 2),  # Up
        ((0, -1), 3),  # Left
    ]

    best_action = None
    best_heuristic = float("inf")
    valid_actions = []

    for (dr, dc), action in directions:
        new_agent_r = agent_pos[0] + dr
        new_agent_c = agent_pos[1] + dc
        new_agent_pos = (new_agent_r, new_agent_c)

        # Check if move is valid
        if not (0 <= new_agent_r < H and 0 <= new_agent_c < W):
            continue
        if maze[new_agent_r, new_agent_c] == 0:
            continue

        new_box_set = box_set

        # Check if pushing a box
        if new_agent_pos in box_set:
            box_new_r = new_agent_r + dr
            box_new_c = new_agent_c + dc
            box_new_pos = (box_new_r, box_new_c)

            # Check if box can be pushed
            if not (0 <= box_new_r < H and 0 <= box_new_c < W):
                continue
            if maze[box_new_r, box_new_c] == 0:
                continue
            if box_new_pos in box_set:
                continue

            new_box_set = (box_set - {new_agent_pos}) | {box_new_pos}

        # This is a valid action
        valid_actions.append(action)

        # Compute heuristic for this move
        h = sum(
            min(abs(box[0] - g[0]) + abs(box[1] - g[1]) for g in goals_list)
            for box in new_box_set
        )

        if h < best_heuristic:
            best_heuristic = h
            best_action = action

    # If we found valid actions, return the best one; otherwise return first valid action
    if best_action is not None:
        return [best_action]
    elif valid_actions:
        return [valid_actions[0]]
    else:
        # No valid moves at all - return action 0 as last resort
        # This shouldn't happen in a valid puzzle, but prevents returning empty list
        return [0]


def _reconstruct_actions(came_from, state) -> List[int]:
    """Reconstruct action sequence from came_from dict."""
    actions = []
    current = state

    while current in came_from and came_from[current] is not None:
        prev_state, action = came_from[current]
        actions.append(action)
        current = prev_state

    return actions[::-1]
