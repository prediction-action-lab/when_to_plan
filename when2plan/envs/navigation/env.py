import numpy as np
import gymnasium as gym
from PIL import Image
import torch as th
import os
import heapq
from shapely.geometry import Point, Polygon, box as Box
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import box as ShapelyBox
from PIL import ImageDraw, ImageFont
from when2plan.envs.navigation.utils import (
    _check_collision_with_obstacles,
    generate_tasks,
    draw_circle,
    draw_rectangle,
)

"""
Continuous maze navigation environment

Robot navigates through a single room with obstacles.
Must reach goal position from starting position.

Obstacles are physical - robot cannot pass through them.
"""

OBSTACLE_TYPES = {
    0: "chair",
    1: "horizontal_table",
    2: "vertical_table",
    3: "sofa",
    4: "person",
    5: "small_clutter",
    6: "large_clutter",
}

# (width, height) for rectangles, radius for circles
OBSTACLE_SHAPES = {
    "chair": ("rect", 1.0, 1.0),
    "horizontal_table": ("rect", 2.0, 1.0),
    "vertical_table": ("rect", 1.0, 2.0),
    "sofa": ("rect", 2.0, 2.0),
    "person": ("circle", 0.5),
    "small_clutter": ("circle", 0.3),
    "large_clutter": ("circle", 1.0),
}

OBSTACLE_PROBS = {
    "overall": 0.5,
    "chair": 0.2,
    "horizontal_table": 0.15,
    "vertical_table": 0.15,
    "sofa": 0.1,
    "person": 0.2,
    "small_clutter": 0.1,
    "large_clutter": 0.1,
}


class NavigationEnv(gym.Env):
    def __init__(self, cfg, seed=42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=seed)

        self.obstacle_types = OBSTACLE_TYPES
        self.obstacle_shapes = OBSTACLE_SHAPES
        self.obstacle_probs = OBSTACLE_PROBS

        self.room_size = cfg.env.get("room_size", 20.0)
        self.num_cached_tasks = cfg.experiment.get("num_cached_tasks", 100)
        self.robot_radius = cfg.env.get("robot_radius", 0.5)
        self.max_obstacles = cfg.env.get("max_obstacles", 10)
        self.min_obstacles = cfg.env.get("min_obstacles", 5)
        self.max_steps = 250
        self.step_size = cfg.env.get("step_size", 0.5)  # Max movement per step
        self.min_start_goal_dist = cfg.env.get("min_start_goal_dist", 10.0)

        # Load or generate tasks
        base_dir = os.path.dirname(os.path.abspath(__file__)) + "/tasks/"
        os.makedirs(base_dir, exist_ok=True)

        tasks_file = os.path.join(base_dir, "navigation_tasks.npz")
        enough_tasks_exist = False
        if os.path.exists(tasks_file):
            data = np.load(tasks_file, allow_pickle=True)
            self.start_positions = data["start_positions"]
            self.goal_positions = data["goal_positions"]
            self.obstacles = data["obstacles"]
            self.obstacle_types_data = data["obstacle_types"]

            if len(self.start_positions) < self.num_cached_tasks:
                print(
                    f"Warning: only {len(self.start_positions)} cached tasks, need {self.num_cached_tasks}"
                )
                enough_tasks_exist = False
            else:
                enough_tasks_exist = True

        if not enough_tasks_exist:
            print(f"Generating {self.num_cached_tasks} navigation tasks...")
            (
                self.start_positions,
                self.goal_positions,
                self.obstacles,
                self.obstacle_types_data,
            ) = generate_tasks(
                rng=self.rng,
                num_tasks=self.num_cached_tasks,
                max_obstacles=self.max_obstacles,
                min_obstacles=self.min_obstacles,
                obstacle_types=self.obstacle_types,
                obstacle_probs=self.obstacle_probs,
                obstacle_shapes=self.obstacle_shapes,
                room_size=self.room_size,
                robot_radius=self.robot_radius,
                min_start_goal_dist=self.min_start_goal_dist,
            )
            np.savez(
                tasks_file,
                start_positions=self.start_positions,
                goal_positions=self.goal_positions,
                obstacles=self.obstacles,
                obstacle_types=self.obstacle_types_data,
            )

        # Keep original full arrays so set_cache_task_range can index absolutely
        self._all_start_positions = self.start_positions
        self._all_goal_positions = self.goal_positions
        self._all_obstacles = self.obstacles
        self._all_obstacle_types_data = self.obstacle_types_data

        # Current task state
        self.current_task_idx = 0
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.time = 0

        # Rendering cache
        # _cached_background: static layer (walls + obstacles + goal) — rebuilt on reset()
        # _cached_obs: last full observation — reused by get_tensor_state() to avoid double-render
        self._cached_background = None
        self._cached_obs = None

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )

        # Continuous control: [dx, dy] normalized to [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _check_collision(self, new_pos):
        """Check if robot collides with walls or obstacles at new_pos."""
        # Check room boundaries
        if (
            new_pos[0] < self.robot_radius
            or new_pos[0] > self.room_size - self.robot_radius
            or new_pos[1] < self.robot_radius
            or new_pos[1] > self.room_size - self.robot_radius
        ):
            return True

        return _check_collision_with_obstacles(
            new_pos,
            self.obstacles[self.current_task_idx],
            self.obstacle_shapes,
            self.obstacle_types_data[self.current_task_idx],
            self.robot_radius,
        )

    def _check_goal_reached(self):
        """Check if robot reached goal position."""
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        return dist < self.robot_radius * 2

    def _build_background_cache(self):
        """Pre-render static scene elements (walls, obstacles, goal) into a cached array.

        This is called once per episode in reset().  get_observation() then copies
        this array and draws only the agent on top, saving every wall/obstacle
        draw call on every step.
        """
        img_size = 64
        pixels_per_unit = img_size / self.room_size

        img_array = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # light gray

        # Walls
        wall_thickness = 0.3
        wall_color = [60, 60, 60]
        for cx, cy, w, h in [
            (
                self.room_size / 2,
                self.room_size - wall_thickness / 2,
                self.room_size,
                wall_thickness,
            ),
            (self.room_size / 2, wall_thickness / 2, self.room_size, wall_thickness),
            (wall_thickness / 2, self.room_size / 2, wall_thickness, self.room_size),
            (
                self.room_size - wall_thickness / 2,
                self.room_size / 2,
                wall_thickness,
                self.room_size,
            ),
        ]:
            draw_rectangle(
                img_array,
                cx,
                cy,
                w,
                h,
                wall_color,
                pixels_per_unit=pixels_per_unit,
                room_size=self.room_size,
                img_size=img_size,
                fill=True,
            )

        # Color map for obstacles
        color_map = {
            "chair": [139, 69, 19],
            "horizontal_table": [210, 105, 30],
            "vertical_table": [210, 105, 30],
            "sofa": [205, 133, 63],
            "person": [255, 107, 107],
            "small_clutter": [78, 205, 196],
            "large_clutter": [149, 225, 211],
        }

        # Obstacles
        room_obstacles = self.obstacles[self.current_task_idx]
        room_obstacle_types = self.obstacle_types_data[self.current_task_idx]
        for i in range(self.max_obstacles):
            if room_obstacles[i, 2] > 0.5:  # valid
                obs_type = room_obstacle_types[i]
                pos_x, pos_y = room_obstacles[i, 0], room_obstacles[i, 1]
                shape_info = self.obstacle_shapes[obs_type]
                color = color_map.get(obs_type, [128, 128, 128])
                if shape_info[0] == "circle":
                    draw_circle(
                        img_array,
                        pos_x,
                        pos_y,
                        shape_info[1],
                        color,
                        pixels_per_unit=pixels_per_unit,
                        room_size=self.room_size,
                        img_size=img_size,
                        fill=True,
                    )
                else:
                    draw_rectangle(
                        img_array,
                        pos_x,
                        pos_y,
                        shape_info[1],
                        shape_info[2],
                        color,
                        pixels_per_unit=pixels_per_unit,
                        room_size=self.room_size,
                        img_size=img_size,
                        fill=True,
                    )

        # Goal (static within an episode)
        draw_circle(
            img_array,
            self.goal_pos[0],
            self.goal_pos[1],
            self.robot_radius * 1.5,
            [50, 200, 50],
            pixels_per_unit=pixels_per_unit,
            room_size=self.room_size,
            img_size=img_size,
            fill=True,
        )

        self._cached_background = img_array

    def get_observation(self):
        """Return current observation as numpy array (RGB image).

        Copies the pre-built static background (walls + obstacles + goal) and
        draws only the agent on top.  Falls back to a full redraw if the
        background cache has not been built yet (e.g. before first reset).
        """
        img_size = 64
        pixels_per_unit = img_size / self.room_size

        if self._cached_background is not None:
            # Fast path: copy static scene, paint agent only
            img_array = self._cached_background.copy()
        else:
            # Slow path (first call before reset, or cache invalidated)
            img_array = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
            wall_thickness = 0.3
            wall_color = [60, 60, 60]
            for cx, cy, w, h in [
                (
                    self.room_size / 2,
                    self.room_size - wall_thickness / 2,
                    self.room_size,
                    wall_thickness,
                ),
                (
                    self.room_size / 2,
                    wall_thickness / 2,
                    self.room_size,
                    wall_thickness,
                ),
                (
                    wall_thickness / 2,
                    self.room_size / 2,
                    wall_thickness,
                    self.room_size,
                ),
                (
                    self.room_size - wall_thickness / 2,
                    self.room_size / 2,
                    wall_thickness,
                    self.room_size,
                ),
            ]:
                draw_rectangle(
                    img_array,
                    cx,
                    cy,
                    w,
                    h,
                    wall_color,
                    pixels_per_unit=pixels_per_unit,
                    room_size=self.room_size,
                    img_size=img_size,
                    fill=True,
                )
            color_map = {
                "chair": [139, 69, 19],
                "horizontal_table": [210, 105, 30],
                "vertical_table": [210, 105, 30],
                "sofa": [205, 133, 63],
                "person": [255, 107, 107],
                "small_clutter": [78, 205, 196],
                "large_clutter": [149, 225, 211],
            }
            room_obstacles = self.obstacles[self.current_task_idx]
            room_obstacle_types = self.obstacle_types_data[self.current_task_idx]
            for i in range(self.max_obstacles):
                if room_obstacles[i, 2] > 0.5:
                    obs_type = room_obstacle_types[i]
                    pos_x, pos_y = room_obstacles[i, 0], room_obstacles[i, 1]
                    shape_info = self.obstacle_shapes[obs_type]
                    color = color_map.get(obs_type, [128, 128, 128])
                    if shape_info[0] == "circle":
                        draw_circle(
                            img_array,
                            pos_x,
                            pos_y,
                            shape_info[1],
                            color,
                            pixels_per_unit=pixels_per_unit,
                            room_size=self.room_size,
                            img_size=img_size,
                            fill=True,
                        )
                    else:
                        draw_rectangle(
                            img_array,
                            pos_x,
                            pos_y,
                            shape_info[1],
                            shape_info[2],
                            color,
                            pixels_per_unit=pixels_per_unit,
                            room_size=self.room_size,
                            img_size=img_size,
                            fill=True,
                        )
            draw_circle(
                img_array,
                self.goal_pos[0],
                self.goal_pos[1],
                self.robot_radius * 1.5,
                [50, 200, 50],
                pixels_per_unit=pixels_per_unit,
                room_size=self.room_size,
                img_size=img_size,
                fill=True,
            )

        # Draw agent (only moving element)
        draw_circle(
            img_array,
            self.agent_pos[0],
            self.agent_pos[1],
            self.robot_radius,
            [100, 150, 255],
            pixels_per_unit=pixels_per_unit,
            room_size=self.room_size,
            img_size=img_size,
            fill=True,
        )

        self._cached_obs = img_array
        return img_array

    def get_tensor_state(self):
        """Return observation as torch tensor for NN input (CHW format, float32).

        Reuses the observation cached by the most recent step() / reset() call
        so that meta_env's inference pass does not trigger a second full render.
        """
        obs = (
            self._cached_obs if self._cached_obs is not None else self.get_observation()
        )
        # Convert to CHW format (channels first) and normalize to [0, 1]
        obs_chw = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
        obs_normalized = obs_chw.astype(np.float32) / 255.0
        return th.from_numpy(obs_normalized)

    def reset(self, **kwargs):
        self.time = 0
        self.current_task_idx = self.rng.integers(0, self.num_cached_tasks)

        # Set start and goal positions from cached task
        self.agent_pos = self.start_positions[self.current_task_idx].copy()
        self.goal_pos = self.goal_positions[self.current_task_idx].copy()

        # Build background cache for this episode (walls + obstacles + goal are static)
        self._build_background_cache()

        info = {"task_idx": self.current_task_idx}
        obs = self.get_observation()  # paints agent on top of fresh background
        return obs, info

    def model_free_step(self, action):
        return self.step(action)

    def get_distance_to_goal(self):
        """Return Euclidean distance from agent to goal."""
        return (
            float(np.linalg.norm(self.agent_pos - self.goal_pos))
            / self.cfg.env.distance_to_goal_scaling_factor
        )

    def get_model_free_spaces(self):
        """Return observation and action space shapes for model-free actor."""
        obs_space = self.observation_space
        action_space = self.action_space

        return obs_space, action_space

    def step(self, action):
        self.time += 1
        info = {"task_idx": self.current_task_idx}

        # Clip and scale action
        action = np.asarray(action).flatten()
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.step_size

        # Attempt move with collision detection
        new_pos = self.agent_pos + delta

        if not self._check_collision(new_pos):
            self.agent_pos = new_pos
        # else: stay at current position (collision)

        # Check if reached goal
        if self._check_goal_reached():
            reward = 100.0
            term = True
            trunc = False
            info["success"] = True
        else:
            # Distance-based reward to encourage progress toward goal
            dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
            reward = -0.01 * dist_to_goal
            term = False
            trunc = self.time >= self.max_steps

        obs = self.get_observation()
        return obs, reward, term, trunc, info

    def get_plan(self, max_expansions=None, plan_length=None, pretrain=False, **kwargs):
        """Get plan using A* search for Navigation.

        Args:
            max_expansions: Maximum number of search expansions
            plan_length: "short", "medium", or "long" - controls search budget and depth
            pretrain: If True, use larger budget for pretraining
        """
        # Map plan length to max search expansions and plan depth
        if plan_length == "short":
            max_expansions = 1000
            plan_depth = self.cfg.env.short_plan_horizon
        elif plan_length == "medium":
            max_expansions = 1000
            plan_depth = self.cfg.env.medium_plan_horizon
        elif plan_length == "long":
            max_expansions = 2000
            plan_depth = self.cfg.env.long_plan_horizon
        else:
            max_expansions = 5000
            plan_depth = None  # No depth limit

        if pretrain:
            max_expansions = 10000
            plan_depth = None

        return a_star_continuous(
            start_pos=self.agent_pos.copy(),
            goal_pos=self.goal_pos.copy(),
            obstacles=self.obstacles[self.current_task_idx],
            obstacle_types=self.obstacle_types_data[self.current_task_idx],
            obstacle_shapes=self.obstacle_shapes,
            robot_radius=self.robot_radius,
            room_size=self.room_size,
            step_size=self.step_size,
            max_expansions=max_expansions,
            plan_depth=plan_depth,
        )

    def set_cache_task_range(self, min_idx, max_idx):
        self.start_positions = self._all_start_positions[min_idx:max_idx]
        self.goal_positions = self._all_goal_positions[min_idx:max_idx]
        self.obstacles = self._all_obstacles[min_idx:max_idx]
        self.obstacle_types_data = self._all_obstacle_types_data[min_idx:max_idx]
        self.num_cached_tasks = max_idx - min_idx

    def render(self, mode="human", render_mode=None):
        """Render current room state directly to numpy/PIL.

        Accepts both ``mode`` (legacy gymnasium) and ``render_mode`` (passed by
        MetaEnv) so the signature is compatible with all callers.  The env
        always returns an RGB numpy array regardless of the mode value.
        """
        # Image size in pixels
        img_size = 600
        pixels_per_unit = img_size / self.room_size

        # Create RGB image
        img_array = (
            np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        )  # Light gray background

        # Draw walls
        wall_thickness = 0.3
        wall_color = [60, 60, 60]  # Dark gray

        # Top wall
        draw_rectangle(
            img_array,
            self.room_size / 2,
            self.room_size - wall_thickness / 2,
            self.room_size,
            wall_thickness,
            wall_color,
            fill=True,
        )

        # Bottom wall
        draw_rectangle(
            img_array,
            self.room_size / 2,
            wall_thickness / 2,
            self.room_size,
            wall_thickness,
            wall_color,
            fill=True,
        )

        # Left wall
        draw_rectangle(
            img_array,
            wall_thickness / 2,
            self.room_size / 2,
            wall_thickness,
            self.room_size,
            wall_color,
            fill=True,
        )

        # Right wall
        draw_rectangle(
            img_array,
            self.room_size - wall_thickness / 2,
            self.room_size / 2,
            wall_thickness,
            self.room_size,
            wall_color,
            fill=True,
        )

        # Color map for obstacles
        color_map = {
            "chair": [139, 69, 19],  # Brown
            "horizontal_table": [210, 105, 30],  # Chocolate
            "vertical_table": [210, 105, 30],
            "sofa": [205, 133, 63],  # Peru
            "person": [255, 107, 107],  # Red
            "small_clutter": [78, 205, 196],  # Teal
            "large_clutter": [149, 225, 211],  # Light teal
        }

        # Draw obstacles
        room_obstacles = self.obstacles[self.current_task_idx]
        room_obstacle_types = self.obstacle_types_data[self.current_task_idx]

        for i in range(self.max_obstacles):
            if room_obstacles[i, 2] > 0.5:  # valid
                obs_type = room_obstacle_types[i]
                pos_x, pos_y = room_obstacles[i, 0], room_obstacles[i, 1]
                shape_info = self.obstacle_shapes[obs_type]
                color = color_map.get(obs_type, [128, 128, 128])

                if shape_info[0] == "circle":
                    radius = shape_info[1]
                    draw_circle(
                        img_array,
                        pos_x,
                        pos_y,
                        radius,
                        color,
                        pixels_per_unit=img_size / self.room_size,
                        room_size=self.room_size,
                        img_size=img_size,
                        fill=True,
                    )
                    draw_circle(
                        img_array,
                        pos_x,
                        pos_y,
                        radius,
                        [0, 0, 0],
                        pixels_per_unit=img_size / self.room_size,
                        room_size=self.room_size,
                        img_size=img_size,
                        fill=False,
                    )
                else:  # rectangle
                    width, height = shape_info[1], shape_info[2]
                    draw_rectangle(
                        img_array,
                        pos_x,
                        pos_y,
                        width,
                        height,
                        color,
                        pixels_per_unit=img_size / self.room_size,
                        room_size=self.room_size,
                        img_size=img_size,
                        fill=True,
                    )
                    draw_rectangle(
                        img_array,
                        pos_x,
                        pos_y,
                        width,
                        height,
                        [0, 0, 0],
                        pixels_per_unit=img_size / self.room_size,
                        room_size=self.room_size,
                        img_size=img_size,
                        fill=False,
                    )

        # Draw goal (green circle with star)
        draw_circle(
            img_array,
            self.goal_pos[0],
            self.goal_pos[1],
            self.robot_radius * 1.5,
            [50, 200, 50],
            pixels_per_unit=img_size / self.room_size,
            room_size=self.room_size,
            img_size=img_size,
            fill=True,
        )
        draw_circle(
            img_array,
            self.goal_pos[0],
            self.goal_pos[1],
            self.robot_radius * 1.5,
            [0, 100, 0],
            pixels_per_unit=img_size / self.room_size,
            room_size=self.room_size,
            img_size=img_size,
            fill=False,
        )

        # Draw robot (blue circle with darker outline)
        draw_circle(
            img_array,
            self.agent_pos[0],
            self.agent_pos[1],
            self.robot_radius,
            [100, 150, 255],
            pixels_per_unit=img_size / self.room_size,
            room_size=self.room_size,
            img_size=img_size,
            fill=True,
        )
        draw_circle(
            img_array,
            self.agent_pos[0],
            self.agent_pos[1],
            self.robot_radius,
            [0, 0, 139],
            pixels_per_unit=img_size / self.room_size,
            room_size=self.room_size,
            img_size=img_size,
            fill=False,
        )

        # Draw direction indicator (small white circle in direction of goal)
        dx = self.goal_pos[0] - self.agent_pos[0]
        dy = self.goal_pos[1] - self.agent_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            dx, dy = dx / dist, dy / dist
            indicator_x = self.agent_pos[0] + dx * self.robot_radius * 0.5
            indicator_y = self.agent_pos[1] + dy * self.robot_radius * 0.5
            draw_circle(
                img_array,
                indicator_x,
                indicator_y,
                self.robot_radius * 0.3,
                [255, 255, 255],
                pixels_per_unit=img_size / self.room_size,
                room_size=self.room_size,
                img_size=img_size,
                fill=True,
            )

        # Convert to PIL Image
        img = Image.fromarray(img_array, mode="RGB")

        draw = ImageDraw.Draw(img)

        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Draw info text
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        info_text = f"Time: {self.time}  Dist to Goal: {dist_to_goal:.1f}"
        draw.text((10, 10), info_text, fill=(0, 0, 0), font=font)

        if mode == "human":
            return img
        elif mode == "rgb_array":
            return np.array(img)
        else:
            return img


def a_star_continuous(
    start_pos,
    goal_pos,
    obstacles,
    obstacle_types,
    obstacle_shapes,
    robot_radius,
    room_size,
    step_size,
    max_expansions=5000,
    plan_depth=None,
):
    """
    A* search in continuous 2D space with obstacle avoidance.
    Optimized with simple geometric collision checks (no Shapely in hot path).

    Args:
        start_pos: Starting position [x, y]
        goal_pos: Goal position [x, y]
        obstacles: Array of obstacle positions and validity flags
        obstacle_types: Types of obstacles
        obstacle_shapes: Shape information for obstacles
        robot_radius: Radius of the robot
        room_size: Size of the room
        step_size: Step size for motion
        max_expansions: Maximum number of search expansions
        plan_depth: If specified, only search up to this depth and return best plan found

    Returns:
        List of [dx, dy] actions or None if no path found.
    """

    # Pre-process obstacles into simple geometric primitives for fast collision checks
    circle_obstacles = []  # [(x, y, radius)]
    rect_obstacles = []  # [(x_min, y_min, x_max, y_max)]

    for i in range(len(obstacles)):
        if obstacles[i, 2] > 0.5:  # valid
            obs_type = obstacle_types[i]
            obs_x, obs_y = obstacles[i, 0], obstacles[i, 1]
            shape_info = obstacle_shapes[obs_type]

            if shape_info[0] == "circle":
                radius = shape_info[1]
                # Store as circle: (center_x, center_y, radius + robot_radius)
                circle_obstacles.append((obs_x, obs_y, radius + robot_radius))
            else:  # rectangle
                width, height = shape_info[1], shape_info[2]
                # Expand rectangle by robot_radius for Minkowski sum
                rect_obstacles.append(
                    (
                        obs_x - width / 2 - robot_radius,
                        obs_y - height / 2 - robot_radius,
                        obs_x + width / 2 + robot_radius,
                        obs_y + height / 2 + robot_radius,
                    )
                )

    def heuristic(pos):
        """Euclidean distance to goal."""
        return np.linalg.norm(pos - goal_pos)

    def check_collision(pos):
        """Fast collision check using simple geometric math."""
        px, py = pos[0], pos[1]

        # Fast boundary check
        if (
            px < robot_radius
            or px > room_size - robot_radius
            or py < robot_radius
            or py > room_size - robot_radius
        ):
            return True

        # Check circle obstacles (point-to-circle distance)
        for cx, cy, r in circle_obstacles:
            dx, dy = px - cx, py - cy
            if dx * dx + dy * dy < r * r:  # Avoid sqrt
                return True

        # Check rectangle obstacles (AABB collision)
        for x_min, y_min, x_max, y_max in rect_obstacles:
            if x_min <= px <= x_max and y_min <= py <= y_max:
                return True

        return False

    # Pre-compute normalized directions and their magnitudes
    directions = np.array(
        [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],  # Cardinal
            [0.7071, 0.7071],
            [0.7071, -0.7071],
            [-0.7071, 0.7071],
            [-0.7071, -0.7071],  # Diagonal (1/sqrt(2))
        ],
        dtype=np.float32,
    )
    # Pre-scale directions by step_size
    scaled_directions = directions * step_size
    # Pre-compute step costs (cardinal=1.0, diagonal=1.414)
    step_costs = (
        np.array([1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414], dtype=np.float32)
        * step_size
    )

    def get_neighbors(pos):
        """Generate neighboring positions in 8 directions."""
        neighbors = []
        px, py = pos[0], pos[1]

        for i, delta in enumerate(scaled_directions):
            new_x = px + delta[0]
            new_y = py + delta[1]

            # Early boundary check before full collision
            if (
                new_x < robot_radius
                or new_x > room_size - robot_radius
                or new_y < robot_radius
                or new_y > room_size - robot_radius
            ):
                continue

            # Check circle obstacles
            collision = False
            for cx, cy, r in circle_obstacles:
                dx, dy = new_x - cx, new_y - cy
                if dx * dx + dy * dy < r * r:
                    collision = True
                    break

            if collision:
                continue

            # Check rectangle obstacles
            for x_min, y_min, x_max, y_max in rect_obstacles:
                if x_min <= new_x <= x_max and y_min <= new_y <= y_max:
                    collision = True
                    break

            if not collision:
                new_pos = np.array([new_x, new_y], dtype=np.float32)
                neighbors.append((new_pos, delta, step_costs[i]))

        return neighbors

    def pos_to_key(pos, resolution=0.25):  # Slightly coarser for fewer nodes
        """Discretize position for dictionary key."""
        return (round(pos[0] / resolution), round(pos[1] / resolution))

    # A* initialization
    open_set = []
    closed_set = set()  # Track expanded nodes
    came_from = {}
    g_score = {}
    action_from = {}

    start_key = pos_to_key(start_pos)
    g_score[start_key] = 0
    f_score = heuristic(start_pos)

    counter = 0
    heapq.heappush(open_set, (f_score, counter, 0, start_pos.copy(), start_key))
    counter += 1

    expansions = 0
    goal_threshold = robot_radius * 2

    # Track best state at target depth
    best_key_at_depth = None
    best_h_at_depth = float("inf")

    while open_set and expansions < max_expansions:
        _, _, current_g, current_pos, current_key = heapq.heappop(open_set)

        # Skip if already expanded
        if current_key in closed_set:
            continue

        closed_set.add(current_key)
        expansions += 1

        current_h = np.linalg.norm(current_pos - goal_pos)

        # If plan_depth is specified, track best plan at that depth
        if plan_depth is not None and current_g >= plan_depth * step_size:
            if current_h < best_h_at_depth:
                best_h_at_depth = current_h
                best_key_at_depth = current_key

                # If we reached goal at target depth, return early
                if current_h < goal_threshold:
                    path_actions = []
                    node = current_key
                    while node in came_from:
                        path_actions.append(action_from[node])
                        node = came_from[node]
                    path_actions.reverse()
                    normalized_actions = [action / step_size for action in path_actions]
                    return normalized_actions

            # Don't expand beyond target depth
            continue

        # Check if reached goal
        if current_h < goal_threshold:
            # Reconstruct path
            path_actions = []
            node = current_key

            while node in came_from:
                path_actions.append(action_from[node])
                node = came_from[node]

            path_actions.reverse()

            # Normalize actions to [-1, 1] range
            normalized_actions = [action / step_size for action in path_actions]
            return normalized_actions

        # Explore neighbors
        for neighbor_pos, action, action_cost in get_neighbors(current_pos):
            neighbor_key = pos_to_key(neighbor_pos)

            # Skip if already fully expanded
            if neighbor_key in closed_set:
                continue

            tentative_g = g_score[current_key] + action_cost

            # Skip if we already have a better path to this neighbor
            if neighbor_key in g_score and tentative_g >= g_score[neighbor_key]:
                continue

            came_from[neighbor_key] = current_key
            action_from[neighbor_key] = action
            g_score[neighbor_key] = tentative_g
            f_score_neighbor = tentative_g + heuristic(neighbor_pos)
            heapq.heappush(
                open_set,
                (
                    f_score_neighbor,
                    counter,
                    tentative_g,
                    neighbor_pos.copy(),
                    neighbor_key,
                ),
            )
            counter += 1

    # No path found
    # If plan_depth was specified and we found a state at that depth, use it
    if plan_depth is not None and best_key_at_depth is not None:
        path_actions = []
        node = best_key_at_depth
        while node in came_from:
            path_actions.append(action_from[node])
            node = came_from[node]

        if path_actions:  # If we have a valid plan at target depth
            path_actions.reverse()
            normalized_actions = [action / step_size for action in path_actions]
            return normalized_actions

    return None
