import numpy as np
import gymnasium as gym
from PIL import Image
import torch as th
import os
import heapq
from when2plan.envs.maze.utils.generate_tasks import generate_vin_style_gridworld


class MazeEnv(gym.Env):
    def __init__(self, cfg, train=True, seed=42):
        self.cfg = cfg
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load or generate maze tasks
        train_tasks_file = base_dir + "/mazes/" + cfg.env.task_type + "/tasks.npy"
        eval_tasks_file = base_dir + "/mazes/" + cfg.env.task_type + "/tasks_eval.npy"

        train_tasks = load_or_generate_tasks(train_tasks_file, num_tasks=200, seed=seed)
        eval_tasks = load_or_generate_tasks(
            eval_tasks_file, num_tasks=50, seed=seed + 10000
        )

        self.tasks = train_tasks if train else eval_tasks
        self.task = self.tasks[0]

        # Determine how many reset positions we need
        self.max_reset_positions = cfg.experiment.get("max_reset_positions", None)
        required_positions = (
            self.max_reset_positions if self.max_reset_positions else 100
        )

        # Load or generate reset positions
        reset_pos_file = (
            base_dir
            + "/mazes/"
            + cfg.env.task_type
            + ("/reset_positions.npy" if train else "/reset_positions_eval.npy")
        )

        self.all_reset_positions = load_or_generate_reset_positions(
            reset_pos_file, self.tasks, required_positions, seed
        )

        # Keep original full arrays so set_cache_task_range can index into them absolutely
        self._all_tasks = self.tasks
        self._all_reset_positions = self.all_reset_positions

        num_cached_tasks = cfg.experiment.get("num_cached_tasks", None)
        if num_cached_tasks is not None and num_cached_tasks > 0:
            self.tasks = self.tasks[:num_cached_tasks]
            if self.all_reset_positions is not None:
                self.all_reset_positions = self.all_reset_positions[:num_cached_tasks]

        self.current_reset_position_idx = (
            0  # Cycles through positions deterministically
        )

        self.scaling_factor = cfg.env.scaling_factor
        H, W = self.task.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(H, W, 3), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self.rng = np.random.default_rng(seed=seed)
        
    def model_free_step(self, action):
        return self.step(action)
    
    def get_model_free_spaces(self):
        obs_space = self.observation_space
        action_space = self.action_space
        return obs_space, action_space
    
    def get_distance_to_goal(self):
        agent = self.state[:2]
        goal = self.state[2:]
        
        return np.linalg.norm(agent - goal) / self.cfg.env.distance_to_goal_scaling_factor

    def get_plan(self, max_expansions=None, plan_length=None, pretrain=False, **kwargs):
        """Get plan using A* search for Maze.
        
        Args:
            max_expansions: Maximum number of search expansions
            plan_length: "short", "medium", or "long" - controls search budget and depth
            pretrain: If True, use larger budget for pretraining
        """
        # Map plan length to max search expansions and plan depth
        if plan_length == "short":
            max_expansions = 2000
            plan_depth = self.cfg.env.short_plan_horizon
        elif plan_length == "medium":
            max_expansions = 2000
            plan_depth = self.cfg.env.medium_plan_horizon
        elif plan_length == "long":
            max_expansions = 2000
            plan_depth = self.cfg.env.long_plan_horizon
        else:
            plan_depth = None  # No depth limit
            
        if pretrain:
            max_expansions = 3000
            plan_depth = None
            
        return a_star_search(
            self.task,
            self.agent_location,
            self.goal_location,
            max_expansions=max_expansions,
            plan_depth=plan_depth,
        )

    def set_cache_task_range(self, min_idx, max_idx):
        self.tasks = self._all_tasks[min_idx:max_idx]
        if self._all_reset_positions is not None:
            self.all_reset_positions = self._all_reset_positions[min_idx:max_idx]

    def set_num_reset_positions(self, num_reset_positions):
        self.max_reset_positions = num_reset_positions

    def render(self, mode="human"):
        img = self.get_observation(scaled=True, as_float=False)
        if mode == "rgb_array":
            return img
        elif mode == "human":
            img = Image.fromarray(img)
            return img

    def get_observation(self, scaled=True, as_float=True):
        """
        Return an H x W x C numpy image observation suitable for Gym:
        - as_float=True returns float32 in [0,1]
        - as_float=False returns uint8 in [0,255]
        """
        H, W = self.task.shape
        image = np.zeros((H, W, 3), dtype=np.uint8)

        # Walls black, free white
        image[self.task == 0] = [0, 0, 0]
        image[self.task == 1] = [255, 255, 255]

        # Goal and agent
        gr, gc = int(self.goal_location[0]), int(self.goal_location[1])
        ar, ac = int(self.agent_location[0]), int(self.agent_location[1])
        image[gr, gc] = [255, 0, 0]
        image[ar, ac] = [0, 255, 0]

        if scaled and self.scaling_factor > 1:
            image = image.repeat(self.scaling_factor, axis=0).repeat(
                self.scaling_factor, axis=1
            )

        if as_float:
            return image.astype(np.float32) / 255.0
        return image

    def get_tensor_state(self, scaled=False):
        """
        Return a torch.FloatTensor shaped (C, H, W) in range [0,1], suitable for CNN input.
        """
        image = self.get_observation(scaled=scaled, as_float=False)
        tensor = th.from_numpy(image).permute(2, 0, 1).to(dtype=th.float32) / 255.0
        return tensor

    def reset(self, **kwargs):
        self.time = 0

        self.task_idx = self.rng.integers(low=0, high=len(self.tasks))
        self.task = self.tasks[self.task_idx]
        # Use pre-generated reset positions
        if self.all_reset_positions is not None:
            available_positions = self.all_reset_positions[self.task_idx]

            # Limit to max_reset_positions if specified
            if self.max_reset_positions is not None and self.max_reset_positions > 0:
                available_positions = available_positions[: self.max_reset_positions]

            # Cycle through positions deterministically
            pos_idx = self.rng.integers(0, len(available_positions))
            self.current_reset_position_idx += 1

            agent_loc, goal_loc = available_positions[pos_idx]
            self.agent_location = np.array(agent_loc, dtype=np.int64)
            self.goal_location = np.array(goal_loc, dtype=np.int64)

        self.state = np.concatenate((self.agent_location, self.goal_location))
        info = {"task_idx": self.task_idx}
        obs = self.get_observation(scaled=True, as_float=True)
        return obs, info

    def step(self, action):
        self.time += 1

        action_map = {
            0: [1, 0],  # Down
            1: [0, 1],  # Right
            2: [-1, 0],  # Up
            3: [0, -1],  # Left
            4: [0, 0],  # Don't move (thinking)
        }

        possible_new_location = self.agent_location + action_map[action]
        # Space is free so allow to move, otherwise just do nothing
        if self.task[possible_new_location[0], possible_new_location[1]] == 1:
            self.agent_location = self.agent_location + action_map[action]

        self.state = np.concatenate((self.agent_location, self.goal_location))
        reward = -np.linalg.norm(self.agent_location - self.goal_location)

        term = all(self.agent_location == self.goal_location)
        # if term:
        #     reward += 100.0
        trunc = self.time > 100

        obs = self.get_observation(scaled=True, as_float=True)
        info = {"task_idx": self.task_idx}
        return obs, reward, term, trunc, info


def load_or_generate_tasks(
    filepath, num_tasks=200, seed=42, height=20, width=20, obstacle_prob=0.2
):
    """Load maze tasks from file or generate if not found."""
    if os.path.exists(filepath):
        tasks = np.load(filepath)
        print(f"Loaded {len(tasks)} tasks from {filepath}")
        return tasks

    print(f"Tasks file not found at {filepath}. Generating {num_tasks} mazes...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    tasks = []
    rng = np.random.default_rng(seed)  # Create RNG once

    for i in range(num_tasks):
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_tasks} mazes")
        maze = generate_vin_style_gridworld(
            rng, height, width, obstacle_prob
        )  # Pass rng object
        tasks.append(maze)

    tasks = np.array(tasks, dtype=np.int8)
    np.save(filepath, tasks)
    print(f"Saved {len(tasks)} tasks to {filepath}")
    return tasks


def generate_start_goal_pairs(maze, num_pairs=100, seed=42):
    """Generate solvable start-goal pairs for a single maze."""
    rng = np.random.default_rng(seed=seed)
    valid_pairs = []
    rows, cols = np.where(maze == 1)

    attempts = 0
    max_attempts = num_pairs * 20

    while len(valid_pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = rng.choice(np.arange(len(rows)), size=2, replace=False)
        start = np.array([rows[idx1], cols[idx1]])
        goal = np.array([rows[idx2], cols[idx2]])

        result = a_star_search(maze, start, goal)
        if result is not None:
            valid_pairs.append([start.tolist(), goal.tolist()])

    return valid_pairs


def load_or_generate_reset_positions(filepath, tasks, num_positions=100, seed=42):
    """Load or generate reset positions for all tasks."""
    needs_regeneration = False

    if os.path.exists(filepath):
        try:
            positions = np.load(filepath, allow_pickle=True)
            if len(positions) != len(tasks):
                needs_regeneration = True
                print(
                    f"Position file has {len(positions)} tasks but need {len(tasks)}. Regenerating..."
                )
            else:
                min_pos = min(len(pos) for pos in positions)
                if min_pos < num_positions:
                    needs_regeneration = True
                    print(
                        f"Existing file has only {min_pos} pairs, need {num_positions}. Regenerating..."
                    )
                else:
                    print(
                        f"Loaded reset positions with {min_pos} pairs per task from {filepath}"
                    )
                    return positions
        except Exception as e:
            needs_regeneration = True
            print(f"Error loading positions: {e}. Regenerating...")
    else:
        needs_regeneration = True

    if needs_regeneration:
        print(f"Generating {num_positions} start-goal pairs for {len(tasks)} tasks...")
        all_positions = []

        for i, task in enumerate(tasks):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Generated pairs for {i + 1}/{len(tasks)} tasks")
            pairs = generate_start_goal_pairs(task, num_positions, seed + i)
            all_positions.append(pairs)

        all_positions = np.array(all_positions, dtype=object)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, all_positions)
        print(f"Saved reset positions to {filepath}")
        return all_positions


def a_star_search(maze, start, goal, max_expansions=None, plan_depth=None):
    """
    A* search on a maze. Returns the sequence of actions (0=Down, 1=Right, 2=Up, 3=Left)
    to move from start toward goal.
    
    Args:
        maze: 2D grid representing the maze
        start: Starting position
        goal: Goal position
        max_expansions: Maximum number of search expansions
        plan_depth: If specified, only search up to this depth and return best plan found
    """
    H, W = maze.shape
    start_r, start_c = int(start[0]), int(start[1])
    goal_r, goal_c = int(goal[0]), int(goal[1])
    start_idx = start_r * W + start_c
    goal_idx = goal_r * W + goal_c

    if maze[goal_r, goal_c] == 0 or maze[start_r, start_c] == 0:
        return None

    N = H * W
    inf = 10**9
    g = np.full(N, inf, dtype=np.int32)
    g[start_idx] = 0
    came_from = np.full(N, -1, dtype=np.int32)
    closed = np.zeros(N, dtype=np.bool_)

    def h(idx):
        r = idx // W
        c = idx % W
        return abs(r - goal_r) + abs(c - goal_c)

    open_heap = []
    heapq.heappush(open_heap, (h(start_idx), start_idx))

    expansions = 0
    best_h = h(start_idx)
    best_node = start_idx
    best_node_at_depth = None
    best_h_at_depth = float('inf')

    def reconstruct_path(node_idx):
        path = []
        cur = node_idx
        while cur != -1:
            r = cur // W
            c = cur % W
            path.append((r, c))
            cur = came_from[cur]
        path = path[::-1]

        if len(path) < 2:
            return []

        actions = []
        for i in range(len(path) - 1):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            if r1 == r0 + 1 and c1 == c0:
                actions.append(0)
            elif r1 == r0 - 1 and c1 == c0:
                actions.append(2)
            elif r1 == r0 and c1 == c0 + 1:
                actions.append(1)
            elif r1 == r0 and c1 == c0 - 1:
                actions.append(3)
        return actions

    while open_heap:
        f, current = heapq.heappop(open_heap)
        if closed[current]:
            continue

        expansions += 1
        current_g = g[current]
        curr_h = h(current)
        
        # If plan_depth is specified, track best plan at that depth
        if plan_depth is not None and current_g == plan_depth:
            if curr_h < best_h_at_depth:
                best_h_at_depth = curr_h
                best_node_at_depth = current
                
                # If we reached goal at target depth, return early
                if current == goal_idx:
                    actions = reconstruct_path(current)
                    return actions
            
            # Don't expand beyond target depth
            continue
        
        if curr_h < best_h:
            best_h = curr_h
            best_node = current

        if (max_expansions is not None) and (expansions >= max_expansions):
            if current != goal_idx:
                actions = reconstruct_path(best_node)
                return actions

        if current == goal_idx:
            actions = reconstruct_path(current)
            return actions

        closed[current] = True
        curr_r = current // W
        curr_c = current % W

        for nr, nc, action in [
            (curr_r + 1, curr_c, 0),
            (curr_r - 1, curr_c, 2),
            (curr_r, curr_c + 1, 1),
            (curr_r, curr_c - 1, 3),
        ]:
            if nr < 0 or nr >= H or nc < 0 or nc >= W or maze[nr, nc] == 0:
                continue
            neighbor_idx = nr * W + nc
            if closed[neighbor_idx]:
                continue

            tentative_g = g[current] + 1
            if tentative_g < g[neighbor_idx]:
                came_from[neighbor_idx] = current
                g[neighbor_idx] = tentative_g
                f_score = tentative_g + h(neighbor_idx)
                heapq.heappush(open_heap, (f_score, neighbor_idx))

    # No solution found - return best state encountered
    # If plan_depth was specified and we found a plan at that depth, use it
    if plan_depth is not None and best_node_at_depth is not None:
        actions = reconstruct_path(best_node_at_depth)
        if actions:  # If we have a valid plan at target depth
            return actions
    
    # Otherwise, return None if no path found
    return None
