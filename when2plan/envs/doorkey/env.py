import numpy as np
import gymnasium as gym
from typing import NamedTuple
from PIL import Image
import torch as th
from minigrid.envs.doorkey import DoorKeyEnv as SuperDoorKeyEnv
import heapq


class DoorKeyEnv(SuperDoorKeyEnv):
    def __init__(self, cfg, seed=None):
        super().__init__(size=8)

        self.cfg = cfg
        self.rng = np.random.default_rng(seed=seed)
        self.scaling_factor = self.cfg.env.scaling_factor

        # New: Limit the number of unique environment configurations (seeds)
        self.num_cached_tasks = self.cfg.experiment.get("num_cached_tasks", None)
        if self.num_cached_tasks is not None and self.num_cached_tasks > 0:
            # Keep a large pool so OOD ablation (tasks N to 2N, or beyond) is accessible
            self._all_fixed_seeds = np.arange(400)
            self.fixed_seeds = self._all_fixed_seeds[: self.num_cached_tasks]
        else:
            self._all_fixed_seeds = None
            self.fixed_seeds = None

        self.action_space = gym.spaces.Discrete(
            self.action_space.n + 1
        )  # +1 for "do nothing" action

        # Create symbolic obs space
        # observation = [agent_x, agent_y, agent_dir, has_key,
        #                key_x, key_y, door_x, door_y, door_is_open,
        #                goal_x, goal_y]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32,
        )

        self.max_time = 250

    def set_cache_task_range(self, start, end):
        """Set the range of cached tasks (by index) to sample from."""
        if self.num_cached_tasks is None or self._all_fixed_seeds is None:
            raise ValueError(
                "num_cached_tasks is not set; cannot set cache task range."
            )
        if start < 0 or end > len(self._all_fixed_seeds) or start >= end:
            raise ValueError(
                f"Invalid cache task range: start={start}, end={end}, total_seeds={len(self._all_fixed_seeds)}"
            )
        self.fixed_seeds = self._all_fixed_seeds[start:end]
        self.num_cached_tasks = end - start

    def model_free_step(self, action):
        return self.step(action)

    def get_plan(self, max_expansions=None, plan_length=None, pretrain=False, **kwargs):
        """Get plan using A* search for DoorKey.

        Args:
            max_expansions: Maximum number of search expansions
            plan_length: "short", "medium", or "long" - controls search budget and depth
            pretrain: If True, use larger budget for pretraining
        """
        # Map plan length to max search expansions and plan depth
        if plan_length == "short":
            max_expansions = 3000
            plan_depth = self.cfg.env.short_plan_horizon
        elif plan_length == "medium":
            max_expansions = 3000
            plan_depth = self.cfg.env.medium_plan_horizon
        elif plan_length == "long":
            max_expansions = 3000
            plan_depth = self.cfg.env.long_plan_horizon
        else:
            max_expansions = 1000
            plan_depth = None  # No depth limit

        if pretrain:
            max_expansions = 3000
            plan_depth = None

        return plan_doorkey_minigrid(
            self, max_expansions=max_expansions, plan_depth=plan_depth
        )

    def render(self, mode="human"):
        # Use MiniGrid's native rendering
        img = self.get_frame(highlight=False, tile_size=32)

        if mode == "rgb_array":
            return np.array(img)
        elif mode == "human":
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            return img

    def get_model_free_spaces(self):
        obs_space = self.observation_space
        action_space = self.action_space
        return obs_space, action_space

    def get_distance_to_goal(self):
        """Compute L2 distance from agent to goal"""
        agent_x, agent_y = self.agent_pos
        goal_x, goal_y = -1, -1
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "goal":
                    goal_x, goal_y = x, y
                    break
            if goal_x != -1:
                break

        if goal_x == -1:
            return float("inf")

        dist = np.linalg.norm(np.array([agent_x, agent_y]) - np.array([goal_x, goal_y]))
        return dist / self.cfg.env.distance_to_goal_scaling_factor

    def get_observation(self):
        """Return symbolic observation as 11-element feature vector."""
        W, H = self.grid.width, self.grid.height
        agent_x, agent_y = self.agent_pos
        agent_dir = self.agent_dir / 3.0  # Normalize to [0,1]

        # Check if agent is carrying the key
        has_key = (
            1.0 if self.carrying is not None and self.carrying.type == "key" else 0.0
        )

        # Find key position (if not carried)
        key_x, key_y = -1, -1
        if has_key == 0.0:  # Only search if not carrying
            for x in range(W):
                for y in range(H):
                    obj = self.grid.get(x, y)
                    if obj is not None and obj.type == "key":
                        key_x, key_y = x, y
                        break
                if key_x != -1:
                    break

        # Normalize key position (use 0.0 if picked up)
        key_x_norm = key_x / W if key_x != -1 else 0.0
        key_y_norm = key_y / H if key_y != -1 else 0.0

        # Find door position and state
        door_x, door_y, door_is_open = -1, -1, 0.0
        for x in range(W):
            for y in range(H):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "door":
                    door_x, door_y = x, y
                    door_is_open = 1.0 if obj.is_open else 0.0
                    break
            if door_x != -1:
                break

        door_x_norm = door_x / W if door_x != -1 else 0.0
        door_y_norm = door_y / H if door_y != -1 else 0.0

        # Find goal position
        goal_x, goal_y = -1, -1
        for x in range(W):
            for y in range(H):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type == "goal":
                    goal_x, goal_y = x, y
                    break
            if goal_x != -1:
                break

        goal_x_norm = goal_x / W if goal_x != -1 else 0.0
        goal_y_norm = goal_y / H if goal_y != -1 else 0.0

        arr = np.array(
            [
                agent_x / W,
                agent_y / H,
                agent_dir,
                has_key,
                key_x_norm,
                key_y_norm,
                door_x_norm,
                door_y_norm,
                door_is_open,
                goal_x_norm,
                goal_y_norm,
            ],
            dtype=np.float32,
        )

        return arr

    def get_tensor_state(self):
        """Return observation as torch tensor (1D vector)."""
        state = self.get_observation()
        return th.from_numpy(state).to(dtype=th.float32)

    def reset(self, **kwargs):
        # If we have a fixed set of seeds, pick one to constrain the environment
        if self.fixed_seeds is not None:
            seed = int(self.rng.choice(self.fixed_seeds))
            kwargs["seed"] = seed

        obs, info = super().reset(**kwargs)

        self.time = 0

        self.state = self.get_observation()
        self.task_idx = seed
        info = {"task_idx": self.task_idx}
        return self.state, info

    def step(self, action):
        self.time += 1
        info = {"task_idx": self.task_idx}

        meta_trunc = self.time > self.max_time

        # Epsilon action noise: with probability action_noise, replace with a random
        # real action (excludes the no-op which is action_space.n - 1).
        action_noise = getattr(self.cfg, "action_noise", 0.0)
        n_real_actions = self.action_space.n - 1
        if (
            action != self.action_space.n - 1
            and action_noise > 0.0
            and self.rng.random() < action_noise
        ):
            action = int(self.rng.integers(0, n_real_actions))

        if action == self.action_space.n - 1:
            return self.state, -1, False, meta_trunc, info

        # MiniGrid step returns a dict obs, we want our symbolic state
        _, reward, term, trunc, info = super().step(action)
        # if term:
        #     reward += 100.0

        trunc = meta_trunc or trunc
        self.state = self.get_observation()

        return self.state, reward, term, trunc, info


def plan_doorkey_minigrid(env, max_expansions=1000, plan_depth=None):
    """
    Plan a sequence of low-level env.actions to solve the Doorkey Minigrid.
    Returns a list of integer actions (using env.actions.*).
    Planning is done on a grid-state (x,y,has_key) with A* and then converted
    into turn/forward/pickup/toggle actions that should work when executed
    in the environment.

    Args:
        env: The DoorKey environment
        max_expansions: Maximum number of search expansions
        plan_depth: If specified, only search up to this depth and return best plan found
    """

    # helpers to inspect cells
    def cell(x, y):
        return env.grid.get(x, y)

    def is_wall(obj):
        return obj is not None and getattr(obj, "type", None) == "wall"

    def is_door(obj):
        return obj is not None and getattr(obj, "type", None) == "door"

    def door_is_open(obj):
        return getattr(obj, "is_open", False)

    def is_key(obj):
        return obj is not None and getattr(obj, "type", None) == "key"

    def is_goal(obj):
        return obj is not None and getattr(obj, "type", None) == "goal"

    W, H = env.grid.width, env.grid.height
    sx, sy = env.agent_pos
    start_dir = env.agent_dir

    # find goal, key, and door positions
    goal_pos = None
    key_pos = None
    door_pos = None
    for x in range(W):
        for y in range(H):
            obj = cell(x, y)
            if is_goal(obj) and goal_pos is None:
                goal_pos = (x, y)
            if is_key(obj) and key_pos is None:
                key_pos = (x, y)
            if is_door(obj) and door_pos is None:
                door_pos = (x, y)
    if goal_pos is None:
        return []

    def manhattan(ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)

    # detect whether agent is already carrying the key (MiniGrid stores carried obj in env.carrying)
    start_has_key = False
    carrying = getattr(env, "carrying", None)
    if carrying is not None and getattr(carrying, "type", None) == "key":
        start_has_key = True
        key_pos = None  # already carrying — key not on grid

    def staged_heuristic(x, y, has_key):
        """Staged heuristic: key → door → goal, guiding search through required waypoints."""
        gx, gy = goal_pos
        if not has_key and key_pos is not None:
            kx, ky = key_pos
            if door_pos is not None:
                dx, dy = door_pos
                return (
                    manhattan(x, y, kx, ky)
                    + manhattan(kx, ky, dx, dy)
                    + manhattan(dx, dy, gx, gy)
                )
            else:
                return manhattan(x, y, kx, ky) + manhattan(kx, ky, gx, gy)
        elif has_key and door_pos is not None:
            dx, dy = door_pos
            # Take min so we don't overestimate if the agent is already past the door
            return min(
                manhattan(x, y, dx, dy) + manhattan(dx, dy, gx, gy),
                manhattan(x, y, gx, gy),
            )
        else:
            return manhattan(x, y, gx, gy)

    start_state = (sx, sy, start_has_key)
    goal_xy = goal_pos
    start_h = staged_heuristic(sx, sy, start_has_key)

    open_heap = []
    heapq.heappush(open_heap, (start_h, 0, start_state))
    came_from = {start_state: None}
    gscore = {start_state: 0}
    closed = set()

    expansions = 0
    best_state_at_depth = None
    best_h_at_depth = float("inf")

    DIRS = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    ]  # assumed env direction order (right, down, left, up)

    while open_heap and expansions < max_expansions:
        f_score, cost, state = heapq.heappop(open_heap)
        x, y, has_key = state

        # Skip stale heap entries (state already expanded with lower cost)
        if state in closed:
            continue
        closed.add(state)
        expansions += 1

        # If plan_depth is specified, track best plan at that depth
        if plan_depth is not None and cost >= plan_depth:
            curr_h = staged_heuristic(x, y, has_key)
            if curr_h < best_h_at_depth:
                best_h_at_depth = curr_h
                best_state_at_depth = state

                # If we reached goal at target depth, return early
                if (x, y) == goal_xy:
                    state_path = []
                    cur = state
                    while cur is not None:
                        state_path.append(cur)
                        cur = came_from[cur]
                    state_path.reverse()
                    break

            # Don't expand beyond target depth
            continue

        if (x, y) == goal_xy:
            # reconstruct path (states: x,y,has_key)
            state_path = []
            cur = state
            while cur is not None:
                state_path.append(cur)
                cur = came_from[cur]
            state_path.reverse()
            break

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            obj = cell(nx, ny)
            if is_wall(obj):
                continue

            # can't move into closed locked door without key
            if is_door(obj) and (not door_is_open(obj)) and (not has_key):
                continue

            n_has_key = has_key or is_key(obj)
            nstate = (nx, ny, n_has_key)
            if nstate in closed:
                continue
            tentative_g = cost + 1
            if tentative_g < gscore.get(nstate, 1e9):
                gscore[nstate] = tentative_g
                came_from[nstate] = state
                f = tentative_g + staged_heuristic(nx, ny, n_has_key)
                heapq.heappush(open_heap, (f, tentative_g, nstate))
    else:
        # no path found or max_expansions reached
        # If plan_depth was specified and we found a state at that depth, use it
        if plan_depth is not None and best_state_at_depth is not None:
            state_path = []
            cur = best_state_at_depth
            while cur is not None:
                state_path.append(cur)
                cur = came_from[cur]
            state_path.reverse()
        else:
            return []

    # state_path is a list of (x,y,has_key) from start to goal
    # convert state_path to actions (turns + forward + pickup + toggle)
    actions = []
    # start state
    cur_x, cur_y, cur_has_key = state_path[0]
    cur_dir = start_dir

    ACT = env.actions

    # if starting on a key cell and not carrying it, emit pickup and return so caller replans after pickup
    if is_key(cell(cur_x, cur_y)) and (not cur_has_key):
        actions.append(ACT.pickup)
        return actions

    for i in range(1, len(state_path)):
        nx, ny, next_has_key = state_path[i]
        mv = (nx - cur_x, ny - cur_y)
        # find desired direction index
        try:
            desired_dir = DIRS.index(mv)
        except ValueError:
            # unexpected move (shouldn't happen)
            cur_x, cur_y = nx, ny
            cur_has_key = next_has_key
            continue

        next_obj = cell(nx, ny)

        # If next cell is a closed door, we need to toggle (open) from current cell before moving
        if is_door(next_obj) and (not door_is_open(next_obj)):
            # ensure we have the key before toggling; if the key is on current cell & not picked, pick it
            if not cur_has_key and is_key(cell(cur_x, cur_y)):
                actions.append(ACT.pickup)
                cur_has_key = True

            # turn to face the door
            diff = (desired_dir - cur_dir) % 4
            if diff == 1:
                actions.append(ACT.right)
            elif diff == 2:
                actions.append(ACT.right)
                actions.append(ACT.right)
            elif diff == 3:
                actions.append(ACT.left)
            cur_dir = desired_dir

            # only toggle if we (now) have the key or the door is already open
            if cur_has_key or door_is_open(next_obj):
                actions.append(ACT.toggle)
            else:
                # unexpected: A* allowed this move but we don't have the key -> abort remaining plan
                return actions

            # If the object behind the door is a key, pick it up now (still on current cell).
            # Return immediately after pickup so caller can replan with env.carrying updated.
            if is_key(next_obj) and (not cur_has_key):
                actions.append(ACT.pickup)
                return actions

            # move forward into the door cell
            actions.append(ACT.forward)
            cur_x, cur_y = nx, ny

            continue

        # otherwise just turn to desired_dir then move forward
        diff = (desired_dir - cur_dir) % 4
        if diff == 1:
            actions.append(ACT.right)
        elif diff == 2:
            actions.append(ACT.right)
            actions.append(ACT.right)
        elif diff == 3:
            actions.append(ACT.left)
        cur_dir = desired_dir

        # If the next cell contains a key, pick it up now (before moving) and return so caller replans.
        if is_key(next_obj) and (not cur_has_key):
            actions.append(ACT.pickup)
            return actions

        actions.append(ACT.forward)
        cur_x, cur_y = nx, ny

        # if we stepped on a key, pick it up
        if is_key(next_obj) and (not cur_has_key):
            actions.append(ACT.pickup)
            cur_has_key = True

    return actions
