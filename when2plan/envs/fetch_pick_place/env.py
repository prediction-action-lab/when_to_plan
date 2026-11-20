import numpy as np
import gymnasium as gym
import torch
import os
import mujoco
import heapq
from copy import deepcopy
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch.fetch_env import MujocoFetchEnv
import platform


"""
Description:
Custom fetch obstacle avoidance task

The table has 10 rectangular obstacles placed on it. The robot arm starts on one
side of the table and must reach a goal position on the table while avoiding
collisions with the obstacles.

Task: Navigate gripper from start position to goal while avoiding rectangular obstacles.
Configurations are pre-generated for repeatability (1000 cached setups).
"""

# Path to custom XML with obstacles (absolute path)
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_XML_PATH = os.path.join(_ASSETS_DIR, "pick_and_place_obstacles.xml")

# Path to store/load configurations
CONFIGS_PATH = os.path.join(_ASSETS_DIR, "obstacle_configs.npy")


class FetchPickPlaceEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, cfg, reward_type="sparse", seed=42, render_mode=None, **kwargs):
        # Set MuJoCo rendering backend for headless environments (clusters/Docker)
        # osmesa = CPU-based rendering (works without GPU)
        # egl = GPU-based rendering (faster but requires GPU + drivers)
        # Users can override by setting MUJOCO_GL env var before running
        if "MUJOCO_GL" not in os.environ and platform.system() != "Darwin":
            os.environ["MUJOCO_GL"] = "osmesa"

        self.cfg = cfg
        self.rng = np.random.default_rng(seed=seed)

        # Action repetition (frame skip) - reduces search space and helps learning
        # effective_action_repeat = max(1, 25 // cfg.env.get("n_substeps", 1))

        effective_action_repeat = max(1, 25 // cfg.env.get("n_substeps", 1))
        self.action_repeat = effective_action_repeat

        # In-distribution vs out-of-distribution task split
        self.in_distribution_task_amount = cfg.env.get("num_pretrained_tasks", 10)

        # Whether to use different camera angles for OOD tasks
        # If True, OOD tasks use cameras rotated 180 degrees
        # If False, OOD tasks use the same camera angles as in-distribution tasks
        self.use_ood_camera_angle = cfg.env.get("use_ood_camera_angle", True)

        # Camera configurations for multi-view observations
        # In-distribution: cameras on one side of the table
        self.camera_configs_in_dist = {
            "camera1": {  # Front-side view
                "distance": 1.2,
                "azimuth": 135,
                "elevation": -25,
                "lookat": [1.3, 0.75, 0.42],
            },
            "camera2": {  # Top-down view
                "distance": 1.5,
                "azimuth": 90,
                "elevation": -60,
                "lookat": [1.3, 0.75, 0.42],
            },
        }
        # Out-of-distribution: cameras on opposite side of the table (azimuth rotated 180°)
        self.camera_configs_out_dist = {
            "camera1": {  # Front-side view from opposite side
                "distance": 1.2,
                "azimuth": 315,  # 135 + 180
                "elevation": -25,
                "lookat": [1.3, 0.75, 0.42],
            },
            "camera2": {  # Top-down view from opposite side
                "distance": 1.5,
                "azimuth": 270,  # 90 + 180
                "elevation": -60,
                "lookat": [1.3, 0.75, 0.42],
            },
        }
        self.image_size = cfg.env.get("image_size", 224)  # Size for model input

        # Camera view configuration: if True, use only top-down view (camera2)
        # if False, use both side view (camera1) and top-down view (camera2)
        self.use_top_down_only = False

        # Obstacle configuration
        self.num_obstacles = (
            10  # Number of rectangular obstacles (limited by XML slots)
        )
        self.table_center = np.array([1.3, 0.75, 0.42])  # x, y, z
        self.table_dims = np.array(
            [0.3, 0.9]
        )  # Table is 0.5m x 1.05m (1.5x longer in y)

        # Obstacle size range (width, length, height)
        self.obstacle_size_min = np.array([0.02, 0.02, 0.02])
        self.obstacle_size_max = np.array([0.06, 0.06, 0.1])

        # Task tracking (must be set before loading configs)
        self.num_cached_tasks = cfg.experiment.get("num_cached_tasks", 1000)
        self.task_idx = 0
        self.current_start_pos = None
        self.current_goal_pos = None
        self.time = 0
        self.max_time = 250

        # Pre-generated configurations
        self.configurations = None  # Will load from file or generate
        self.current_config_idx = 0

        # Task range for sampling (can be limited via set_cache_task_range)
        self.cached_task_start = 0
        self.cached_task_end = self.num_cached_tasks

        # Load or generate configurations
        self._load_or_generate_configs()

        # Initial joint positions - robot at standard position
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }

        # Get n_substeps from config (default to 5 if not specified)
        n_substeps = cfg.env.get("n_substeps", 1)

        # Calculate render_fps based on n_substeps
        # dt = n_substeps * model_timestep (model_timestep = 0.002 for Fetch)
        # render_fps should be 1.0 / dt
        model_timestep = 0.002
        dt = n_substeps * model_timestep
        render_fps = int(np.round(1.0 / dt))

        # Update metadata to match n_substeps before calling parent __init__
        if not hasattr(self, "metadata"):
            self.metadata = {}
        self.metadata["render_fps"] = render_fps

        # Initialize parent MujocoFetchEnv
        # Change to assets directory so relative paths in XML work correctly
        original_cwd = os.getcwd()
        try:
            os.chdir(_ASSETS_DIR)
            MujocoFetchEnv.__init__(
                self,
                model_path=MODEL_XML_PATH,
                has_object=False,
                block_gripper=False,
                n_substeps=n_substeps,
                gripper_extra_height=0.0,  # Start at table height for reach task
                target_in_the_air=False,  # Keep targets on table (in grid cells)
                target_offset=0.0,
                obj_range=0.15,
                target_range=0.15,
                distance_threshold=0.05,
                initial_qpos=initial_qpos,
                reward_type=reward_type,
                render_mode=render_mode,
                width=128,
                height=128,
                default_camera_config={
                    "distance": 1.5,
                    "azimuth": 120,
                    "elevation": -30,
                    "lookat": [1.3, 0.75, 0.42],
                },
                **kwargs,
            )
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)

        EzPickle.__init__(
            self,
            cfg=cfg,
            reward_type=reward_type,
            seed=seed,
            render_mode=render_mode,
            **kwargs,
        )

        # Build per-joint normalisation parameters now that self.model exists.
        self._init_joint_normalization()

    # ── Joint normalisation helpers ───────────────────────────────────────────

    def _init_joint_normalization(self):
        """Initialise per-joint normalisation from MuJoCo joint limits.

        Each of the 7 arm joints is mapped to ``[-1, 1]`` by
        ``_normalize_joints`` / ``_denormalize_joints``.  Unlimited
        (continuous-rotation) joints fall back to ``[-π, π]``.

        This is called once at the end of ``__init__`` and produces:
            _joint_mid   (7,) – midpoint of each joint's range
            _joint_scale (7,) – half-range; divide / multiply to normalise
        """
        _ARM_JOINT_NAMES = [
            "robot0:shoulder_pan_joint",
            "robot0:shoulder_lift_joint",
            "robot0:upperarm_roll_joint",  # typically unlimited → [-π, π]
            "robot0:elbow_flex_joint",
            "robot0:forearm_roll_joint",  # typically unlimited → [-π, π]
            "robot0:wrist_flex_joint",
            "robot0:wrist_roll_joint",  # typically unlimited → [-π, π]
        ]
        _FALLBACK = np.pi  # half-range for unlimited joints

        lo = np.empty(7)
        hi = np.empty(7)
        for i, jname in enumerate(_ARM_JOINT_NAMES):
            try:
                jid = self.model.joint(jname).id
                if self.model.jnt_limited[jid]:
                    lo[i] = self.model.jnt_range[jid, 0]
                    hi[i] = self.model.jnt_range[jid, 1]
                else:
                    lo[i], hi[i] = -_FALLBACK, _FALLBACK
            except Exception:
                lo[i], hi[i] = -_FALLBACK, _FALLBACK

        self._joint_lo = lo
        self._joint_hi = hi
        self._joint_mid = (lo + hi) * 0.5
        self._joint_scale = (hi - lo) * 0.5  # half-range

    def _normalize_joints(self, qpos7: np.ndarray) -> np.ndarray:
        """Normalise 7-DOF raw joint angles to ``[-1, 1]`` per joint.

        Inverse of ``_denormalize_joints``.
        """
        return (
            (np.asarray(qpos7, dtype=np.float64) - self._joint_mid) / self._joint_scale
        ).astype(np.float32)

    def _denormalize_joints(self, norm7: np.ndarray) -> np.ndarray:
        """Convert normalised ``[-1, 1]`` joint values back to raw qpos angles.

        Inverse of ``_normalize_joints``.
        """
        return np.asarray(norm7, dtype=np.float64) * self._joint_scale + self._joint_mid

    def _load_or_generate_configs(self):
        """Load pre-generated configurations or generate them if not exists."""
        if os.path.exists(CONFIGS_PATH):
            print(f"Loading configurations from {CONFIGS_PATH}")
            self.configurations = np.load(CONFIGS_PATH, allow_pickle=True).item()
        else:
            print(f"Generating {self.num_cached_tasks} obstacle configurations...")
            self._generate_and_save_configs()

    def get_model_free_spaces(self):
        num_cameras = 1 if self.use_top_down_only else len(self.camera_configs_in_dist)
        obs_shape = (self.image_size, self.image_size, 3 * num_cameras)
        obs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )
        # Get output shape (joint angles — normalised to [-1, 1] per joint)
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )  # 7-DOF arm joints, normalised; see _normalize_joints / _denormalize_joints

        return obs_space, action_space

    def _geometric_path_exists(
        self,
        start_pos,
        goal_pos,
        obstacles,
        safe_margin: float = 0.03,
        step_size: float = 0.05,
        max_expansions: int = 2000,
    ) -> bool:
        """Return True if a collision-free path from *start_pos* to *goal_pos*
        exists, using a lightweight geometric A* search (no MuJoCo).

        The collision model mirrors ``_check_collision_geometric`` but takes
        *obstacles* as an explicit argument so it can be called during config
        generation before ``self.current_obstacles`` is populated.

        All heap operations are over integer grid keys + a monotone counter to
        avoid ever handing numpy arrays to the heap comparator.
        """
        start_pos_f = np.asarray(start_pos, dtype=np.float64)
        goal_pos_f = np.asarray(goal_pos, dtype=np.float64)

        # Short-circuit: already within reach — no planning needed.
        if float(np.linalg.norm(start_pos_f - goal_pos_f)) < 0.05:
            return True

        tc = self.table_center
        td = self.table_dims
        ws_x_min = tc[0] - td[0] / 2 - 0.15
        ws_x_max = tc[0] + td[0] / 2 + 0.15
        ws_y_min = tc[1] - td[1] / 2 - 0.15
        ws_y_max = tc[1] + td[1] / 2 + 0.15
        # Use the bare table surface (tc[2]) as the floor, not tc[2]+safe_margin.
        # The gripper legitimately operates at tc[2]+0.02 m; applying safe_margin
        # here would reject both start and goal and make every low-z config fail.
        ws_z_min = tc[2]
        ws_z_max = tc[2] + 0.55

        inv = 1.0 / step_size

        def to_key(pos):
            return (
                int(round(pos[0] * inv)),
                int(round(pos[1] * inv)),
                int(round(pos[2] * inv)),
            )

        goal_k = to_key(goal_pos_f)

        def _collides_key(k, is_goal=False):
            x, y, z = k[0] * step_size, k[1] * step_size, k[2] * step_size
            if not (
                ws_x_min <= x <= ws_x_max
                and ws_y_min <= y <= ws_y_max
                and ws_z_min <= z <= ws_z_max
            ):
                return True
            # For the goal cell itself use a tighter margin (0.0) so that a
            # goal placed legitimately close to an obstacle face doesn't get
            # incorrectly rejected by grid snap-induced over-expansion.
            effective_margin = 0.0 if is_goal else safe_margin
            if effective_margin == 0.0:
                return False
            sm2 = effective_margin * effective_margin
            for obs_pos, obs_size in obstacles:
                dx = max(0.0, abs(x - obs_pos[0]) - obs_size[0])
                dy = max(0.0, abs(y - obs_pos[1]) - obs_size[1])
                dz = max(0.0, z - (obs_pos[2] + obs_size[2]))
                dzb = max(0.0, (obs_pos[2] - obs_size[2]) - z)
                dz_t = max(dz, dzb)
                if dx * dx + dy * dy + dz_t * dz_t < sm2:
                    return True
            return False

        # 26-connected integer deltas
        int_deltas = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                for ddz in (-1, 0, 1):
                    if ddx == ddy == ddz == 0:
                        continue
                    cost = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
                    int_deltas.append(((ddx, ddy, ddz), cost * step_size))

        start_k = to_key(start_pos_f)

        def h(k):
            p = np.array([k[0] * step_size, k[1] * step_size, k[2] * step_size])
            return float(np.linalg.norm(p - goal_pos_f))

        best_g: dict = {start_k: 0.0}
        # heap entries: (f, g, tie, key_tuple) — no numpy arrays in the heap
        counter = 0
        heap = [(h(start_k), 0.0, counter, start_k)]

        expansions = 0
        while heap and expansions < max_expansions:
            f, g, _, cur_k = heapq.heappop(heap)

            if best_g.get(cur_k, float("inf")) < g - 1e-9:
                continue  # stale entry

            expansions += 1

            if cur_k == goal_k:
                return True
            # Within one grid step of the goal is close enough.
            if h(cur_k) < step_size * 1.5:
                return True

            for (ddx, ddy, ddz), cost in int_deltas:
                nk = (cur_k[0] + ddx, cur_k[1] + ddy, cur_k[2] + ddz)
                ng = g + cost
                if ng >= best_g.get(nk, float("inf")):
                    continue
                if _collides_key(nk, is_goal=(nk == goal_k)):
                    continue
                best_g[nk] = ng
                counter += 1
                heapq.heappush(heap, (ng + h(nk), ng, counter, nk))

        return False

    def _generate_and_save_configs(self):
        """Generate obstacle configurations and save to file.

        Configs where no collision-free goal can be found, or where the
        geometric planner cannot solve the task, are silently discarded and
        regenerated so that every saved config is guaranteed to be solvable.
        """
        configs = {}
        discarded = 0
        idx = 0
        while idx < self.num_cached_tasks:
            config = self._generate_single_config()
            if config is None:
                discarded += 1
                continue  # retry with fresh random obstacles
            configs[idx] = config
            idx += 1
            if idx % 100 == 0:
                print(
                    f"Generated {idx}/{self.num_cached_tasks} configurations "
                    f"({discarded} discarded)"
                )

        if discarded:
            print(f"Total discarded (unsolvable/bad goal): {discarded}")

        # Save to file
        np.save(CONFIGS_PATH, configs)
        print(f"Saved configurations to {CONFIGS_PATH}")
        self.configurations = configs

    def _generate_single_config(self):
        """Generate a single obstacle configuration.

        Returns:
            dict with:
                - obstacles: list of (position, size) tuples for each obstacle
                - start_pos: starting position for gripper (on left side of table)
                - goal_pos: goal position (somewhere on table, avoiding obstacles)
        """
        # Table bounds
        x_min = self.table_center[0] - self.table_dims[0] / 2
        x_max = self.table_center[0] + self.table_dims[0] / 2
        y_min = self.table_center[1] - self.table_dims[1] / 2
        y_max = self.table_center[1] + self.table_dims[1] / 2
        z_table = self.table_center[2]

        obstacles = []

        # Generate random obstacles
        for _ in range(self.num_obstacles):
            # Random size
            size = self.rng.uniform(self.obstacle_size_min, self.obstacle_size_max)

            # Random position on table (with margins to keep obstacles fully on table)
            x = self.rng.uniform(x_min + size[0], x_max - size[0])
            y = self.rng.uniform(y_min + size[1], y_max - size[1])
            z = z_table + size[2] / 2  # Bottom of obstacle at table height

            pos = np.array([x, y, z])
            obstacles.append((pos, size))

        # Start position: left edge of table (where gripper can reach)
        start_x = x_min + 0.02  # 2cm from left edge (very close to edge)
        start_y = self.rng.uniform(y_min + 0.05, y_max - 0.05)
        start_z = z_table + 0.02  # Slightly above table
        start_pos = np.array([start_x, start_y, start_z])

        # Inline 3D sphere-vs-AABB helper — mirrors _check_collision_geometric
        # so generated goals are collision-free from the planner's perspective.
        def _pos_in_collision_3d(pos, obs_list, margin):
            px, py, pz = pos
            for obs_pos, obs_size in obs_list:
                dx = max(0.0, abs(px - obs_pos[0]) - obs_size[0])
                dy = max(0.0, abs(py - obs_pos[1]) - obs_size[1])
                dz_above = max(0.0, pz - (obs_pos[2] + obs_size[2]))
                dz_below = max(0.0, (obs_pos[2] - obs_size[2]) - pz)
                dz_total = max(dz_above, dz_below)
                if dx * dx + dy * dy + dz_total * dz_total < margin * margin:
                    return True
            return False

        # Goal position: random position on table, not in collision with any
        # obstacle (3-D sphere-vs-AABB check that matches the planner).
        #
        # The old code used a 0.2 m margin (far too large — with 10 obstacles
        # it can cover the entire 0.3 × 0.9 m table) and had no fallback when
        # the loop exhausted, silently returning a colliding goal.
        #
        # Multi-tier search: progressively relax the minimum distance and the
        # safe margin until a valid goal is found.  The tight margin tiers
        # (0.03 → 0.01) match the planner's default safe_margin.
        goal_pos = None
        for goal_safe_margin in (0.03, 0.01):
            for min_dist in (0.4, 0.2, 0.0):
                for _ in range(300):
                    goal_x = self.rng.uniform(x_min + 0.1, x_max - 0.05)
                    goal_y = self.rng.uniform(y_min + 0.05, y_max - 0.05)
                    goal_z = z_table + 0.02
                    candidate = np.array([goal_x, goal_y, goal_z])

                    if np.linalg.norm(candidate[:2] - start_pos[:2]) < min_dist:
                        continue
                    if not _pos_in_collision_3d(candidate, obstacles, goal_safe_margin):
                        goal_pos = candidate
                        break
                if goal_pos is not None:
                    break
            if goal_pos is not None:
                break

        # Last-resort: brute-force a collision-free point on a dense grid
        # (margin = 0.01 m — just enough to not be inside the geom face).
        if goal_pos is None:
            for gx in np.linspace(x_min + 0.1, x_max - 0.05, 40):
                for gy in np.linspace(y_min + 0.05, y_max - 0.05, 40):
                    candidate = np.array([gx, gy, z_table + 0.02])
                    if not _pos_in_collision_3d(candidate, obstacles, 0.01):
                        goal_pos = candidate
                        break
                if goal_pos is not None:
                    break

        # No collision-free goal found — discard this config and let the caller
        # regenerate rather than silently saving a bad config.
        if goal_pos is None:
            return None

        assert not _pos_in_collision_3d(
            goal_pos, obstacles, 0.01
        ), f"goal_pos {goal_pos} is in collision — this should never happen."

        # Final gate: verify the geometric planner can actually reach the goal.
        # Uses the same A* + AABB logic as get_plan() but without MuJoCo so it
        # is fast enough to run during config generation.  Configs where the
        # start and goal are separated by an impassable wall of obstacles are
        # discarded here and regenerated by the caller.
        if not self._geometric_path_exists(start_pos, goal_pos, obstacles):
            return None

        return {"obstacles": obstacles, "start_pos": start_pos, "goal_pos": goal_pos}

    def load_config(self, config_idx):
        """Load a specific configuration by index."""
        if self.configurations is None:
            raise RuntimeError("Configurations not loaded!")

        if config_idx not in self.configurations:
            config_idx = config_idx % len(self.configurations)

        return self.configurations[config_idx]

    def get_distance_to_goal(self):
        # Return l2 distance of gripper to goal as a simple heuristic metric
        obs = self._get_obs()
        gripper_pos = obs["observation"][:3]
        goal_pos = obs["desired_goal"]
        distance_to_goal = np.linalg.norm(gripper_pos - goal_pos)
        return distance_to_goal / self.cfg.env.distance_to_goal_scaling_factor

    def get_observation(self):
        """Get multi-camera RGB observation for model-free actor."""
        # Update goal visualization before rendering
        obs_dict = self._get_obs()
        site_id = self.model.site("target0").id
        self.model.site_pos[site_id] = obs_dict["desired_goal"]
        mujoco.mj_forward(self.model, self.data)

        # Choose camera configs based on whether task is in-distribution or out-of-distribution
        # If use_ood_camera_angle is False, always use in-distribution camera angles
        if self.task_idx < self.in_distribution_task_amount:
            camera_configs = self.camera_configs_in_dist
        else:
            # Use different camera angle for OOD only if configured to do so
            camera_configs = (
                self.camera_configs_out_dist
                if self.use_ood_camera_angle
                else self.camera_configs_in_dist
            )

        # Filter to only top-down camera if configured
        if self.use_top_down_only:
            camera_configs = {"camera2": camera_configs["camera2"]}

        # Render from camera(s)
        images = []

        for camera_name, camera_config in camera_configs.items():
            # Render from this specific camera configuration
            # We need to manually set the camera and render
            img = self._render_from_camera(camera_config)

            # Resize to model input size
            if img is not None:
                from PIL import Image

                pil_img = Image.fromarray(img)
                # pil_img = pil_img.resize(
                #     (self.image_size, self.image_size), Image.Resampling.BILINEAR
                # )
                img_resized = np.array(pil_img)
                images.append(img_resized)

        # Stack images along channel dimension: (H, W, C*num_cameras)
        # For top-down only: (H, W, 3) | For both cameras: (H, W, 6)
        multi_camera_obs = np.concatenate(images, axis=2)

        return multi_camera_obs

    def _render_from_camera(self, camera_config):
        """Render from a specific camera configuration."""
        # Force rgb_array mode for image observations
        render_mode = self.render_mode if self.render_mode is not None else "rgb_array"

        # Ensure renderer is initialized
        if self.mujoco_renderer._viewers.get(render_mode) is None:
            self.mujoco_renderer._get_viewer(render_mode)

        # Get the camera for the viewer
        viewer = self.mujoco_renderer._viewers.get(render_mode)
        if viewer is not None and hasattr(viewer, "cam"):
            cam = viewer.cam

            # Set camera parameters
            cam.distance = camera_config["distance"]
            cam.azimuth = camera_config["azimuth"]
            cam.elevation = camera_config["elevation"]
            cam.lookat[:] = camera_config["lookat"]

        # Render image with the configured camera
        return self.mujoco_renderer.render(render_mode)

    def get_tensor_state(self):
        """Return observation as torch tensor for model-free actor input."""
        obs = self.get_observation()
        # Convert HWC to CHW for PyTorch (C, H, W)
        obs = np.transpose(obs, (2, 0, 1))
        return torch.from_numpy(obs).to(dtype=torch.float32) / 255.0

    def get_observation_dict(self):
        """Get the current observation dict (for compatibility with parent class)."""
        return self._get_obs()

    def render(self, render_mode=None):
        self.render_mode = render_mode

        """Render the environment."""
        # Update goal visualization before rendering
        obs_dict = self._get_obs()
        site_id = self.model.site("target0").id
        self.model.site_pos[site_id] = obs_dict["desired_goal"]
        mujoco.mj_forward(self.model, self.data)

        return super().render()

    def _sample_goal(self):
        """Override parent method to use goal from loaded configuration."""
        if self.current_goal_pos is not None:
            return self.current_goal_pos.copy()
        else:
            # Fallback to random goal on table
            x_min = self.table_center[0] - self.table_dims[0] / 2
            x_max = self.table_center[0] + self.table_dims[0] / 2
            y_min = self.table_center[1] - self.table_dims[1] / 2
            y_max = self.table_center[1] + self.table_dims[1] / 2
            z = self.table_center[2] + 0.02

            x = self.rng.uniform(x_min + 0.05, x_max - 0.05)
            y = self.rng.uniform(y_min + 0.05, y_max - 0.05)
            return np.array([x, y, z])

    def set_cache_task_range(self, start_idx, end_idx):
        """Set the range of cached tasks to sample from."""
        self.cached_task_start = start_idx
        self.cached_task_end = end_idx

    def reset(self, seed=None, options=None):
        """Override reset to load configuration and spawn obstacles."""
        # if seed is not None:
        #     self.rng = np.random.default_rng(seed)

        # Load configuration for this task
        if self.num_cached_tasks is not None:
            self.task_idx = self.rng.integers(
                self.cached_task_start, self.cached_task_end
            )
        else:
            self.task_idx = 0

        # self.task_idx = 9

        config = self.load_config(self.task_idx)
        self.current_start_pos = config["start_pos"]
        self.current_goal_pos = config["goal_pos"]
        self.current_obstacles = config["obstacles"]

        self.time = 0

        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)

        # Spawn obstacles dynamically by positioning the obstacle bodies
        # MuJoCo requires obstacles to be defined in XML, so we'll position them
        for i, (obs_pos, obs_size) in enumerate(self.current_obstacles):
            if i < 10:  # We have 10 obstacle slots in XML
                # Find the obstacle body and geom
                body_name = f"obstacle{i}"
                geom_name = f"obstacle{i}"

                try:
                    # Set obstacle position
                    body_id = self.model.body(body_name).id
                    self.model.body_pos[body_id] = obs_pos

                    # Set obstacle size
                    geom_id = self.model.geom(geom_name).id
                    self.model.geom_size[geom_id] = obs_size
                except:
                    # If obstacle doesn't exist in XML yet, skip
                    pass

        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = np.array(
                # Format [X, Y, Z]
                # +X brings it outward (away from base)
                # +Y brings it to the left of the base (base frame)
                # Note 0 Y seems to be as far left as the gripper can go
                [1.05, 0.25, 0.44]
            )
            self.data.mocap_quat[0] = np.array([0.707, 0.0, 0.707, 0.0])

        # Let IK settle on the new mocap target
        for _ in range(30):
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        # Get updated observation
        obs = self._get_obs()

        # Update target site visualization to show goal position
        site_id = self.model.site("target0").id
        self.model.site_pos[site_id] = obs["desired_goal"]

        # Forward kinematics to update site positions for rendering
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        info["task_idx"] = self.task_idx

        return obs, info

    def step(self, action):
        self.time += 1

        # Gaussian action noise with std = action_noise, applied in normalised [-1, 1]
        # space before execution. A value of 0.1 is roughly comparable in effect to
        # epsilon = 0.1 in the discrete environments.
        action_noise = getattr(self.cfg, "action_noise", 0.0)
        if action_noise > 0.0:
            action = np.asarray(action, dtype=np.float64).copy()
            action = action + self.rng.normal(0.0, action_noise, size=action.shape)

        # Keep grippers closed
        self.data.qpos[13] = 0.0  # r_gripper_finger_joint
        self.data.qpos[14] = 0.0  # l_gripper_finger_joint

        # Repeat action for action_repeat steps
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = super().step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # Check if goal is reached (gripper at goal position)
        gripper_pos = obs["observation"][:3]
        goal_pos = obs["desired_goal"]
        distance_to_goal = np.linalg.norm(gripper_pos - goal_pos)

        if distance_to_goal < 0.05:  # Within 5cm of goal
            terminated = True
            info["is_success"] = True

        # Add time-based truncation
        truncated = truncated or (self.time >= self.max_time)
        info["task_idx"] = self.task_idx

        return obs, total_reward, terminated, truncated, info

    def model_free_step(self, action):
        return self.step_positions(action)

    def step_positions(self, target_joint_positions):
        """Set joint positions from a normalised ``[-1, 1]`` target vector.

        The input uses the same normalised representation produced by
        ``get_plan(pretrain=True)``: each element is in ``[-1, 1]`` and is
        de-normalised back to raw joint angles (via ``_denormalize_joints``)
        before being written to ``qpos``.

        Args:
            target_joint_positions: Normalised joint targets for arm (7 DOFs),
                each element expected in ``[-1, 1]``.
        """
        self.time += 1

        target_joint_positions = np.asarray(
            target_joint_positions, dtype=np.float64
        ).copy()

        # Flatten if batched (shape (1, 7) -> (7,))
        if target_joint_positions.ndim > 1:
            target_joint_positions = target_joint_positions.flatten()

        # De-normalise from [-1, 1] to raw joint angles, then write to qpos.
        # Arm has 7 DOFs: shoulder_pan, shoulder_lift, upperarm_roll,
        #   elbow_flex, forearm_roll, wrist_flex, wrist_roll (qpos indices 6-12).
        if len(target_joint_positions) == 7:
            raw_joints = self._denormalize_joints(target_joint_positions)
            self.data.qpos[6:13] = raw_joints

        # Keep grippers closed
        self.data.qpos[13] = 0.0  # r_gripper_finger_joint
        self.data.qpos[14] = 0.0  # l_gripper_finger_joint

        # Update physics state
        mujoco.mj_forward(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        gripper_pos = obs["observation"][:3]
        goal_pos = obs["desired_goal"]
        distance_to_goal = np.linalg.norm(gripper_pos - goal_pos)

        if self.reward_type == "sparse":
            reward = -1.0 if distance_to_goal > self.distance_threshold else 0.0
        else:
            reward = -distance_to_goal

        # Check termination
        terminated = distance_to_goal < 0.05
        truncated = self.time >= self.max_time

        info = {"task_idx": self.task_idx, "is_success": terminated}

        return obs, reward, terminated, truncated, info

    def _check_collision_geometric(self, pos, safe_margin: float = 0.03) -> bool:
        """Return True if *pos* collides with any obstacle, the table surface,
        or is outside the reachable workspace.

        Obstacles are axis-aligned boxes (AABB) stored as (center, half-size)
        pairs; the gripper is treated as a sphere of radius ``safe_margin``.

        Args:
            pos: (3,) gripper xyz in world frame.
            safe_margin: Extra clearance (metres) added to every obstacle face
                and to the table floor.

        Returns:
            True  → position is *unsafe* (do not visit)
            False → position is *safe*
        """
        x, y, z = pos

        # ── workspace bounds (generous volume above the table) ──────────
        ws_x_min = self.table_center[0] - self.table_dims[0] / 2 - 0.15
        ws_x_max = self.table_center[0] + self.table_dims[0] / 2 + 0.15
        ws_y_min = self.table_center[1] - self.table_dims[1] / 2 - 0.15
        ws_y_max = self.table_center[1] + self.table_dims[1] / 2 + 0.15
        ws_z_min = self.table_center[2] + safe_margin  # floor = table surface
        ws_z_max = self.table_center[2] + 0.55  # ceiling ~55 cm above table

        if not (
            ws_x_min <= x <= ws_x_max
            and ws_y_min <= y <= ws_y_max
            and ws_z_min <= z <= ws_z_max
        ):
            return True  # out of workspace

        # ── obstacle AABB checks ─────────────────────────────────────────
        # Each obstacle: center obs_pos, half-extents obs_size (MuJoCo box).
        # A sphere of radius safe_margin at *pos* is in collision if the
        # signed distance to the expanded box along every axis is <= 0.
        for obs_pos, obs_size in self.current_obstacles:
            # Sphere vs AABB: closest point on box to sphere centre, then check distance.
            # Equivalent to checking each axis independently for gripper-as-point + margin.
            dx = max(0.0, abs(x - obs_pos[0]) - obs_size[0])
            dy = max(0.0, abs(y - obs_pos[1]) - obs_size[1])
            # Only care about heights up to the top face of the obstacle
            dz = max(0.0, z - (obs_pos[2] + obs_size[2]))  # 0 if below top face
            # dz below bottom: obstacle is floating above table, skip lower volume
            dz_below = max(0.0, (obs_pos[2] - obs_size[2]) - z)
            dz_total = max(dz, dz_below)

            sq_dist = dx * dx + dy * dy + dz_total * dz_total
            if sq_dist < safe_margin * safe_margin:
                return True  # collision

        return False  # safe

    def get_plan(
        self,
        safe_margin: float = 0.02,
        step_size: float = None,
        max_expansions: int = 8000,
        pretrain: bool = False,
        plan_length: str = None,
        return_plan_plus_joint: bool = False,
    ):
        """Plan in gripper Cartesian space using A* with analytic collision
        detection, then verify / materialise the plan with a MuJoCo rollout.

        Unlike ``get_plan``, the search never calls ``mj_step``; obstacles and
        table geometry are represented as AABBs and checked analytically.  This
        is orders-of-magnitude faster than the physics-based planner.

        After the search succeeds (or exhausts its budget), the waypoints are
        replayed through a mocap-IK MuJoCo rollout so that the returned
        trajectory contains *true* physics-consistent gripper positions and
        joint angles rather than the idealized grid positions.

        Args:
            safe_margin: Clearance (m) added around every obstacle face and the
                table surface.  Larger values produce safer but potentially
                longer paths.
            step_size: Grid resolution (m).  Defaults to 0.05 m.  Smaller
                values give smoother paths but increase node count.
            max_expansions: Hard cap on A* node expansions.
            pretrain: If True the rollout also returns joint angles (7 DOFs)
                at each waypoint, matching the format expected by the
                pretraining pipeline.
            plan_length: ``"short"`` / ``"medium"`` / ``"long"`` — maps to the
                same depth limits as ``get_plan`` but scaled for A*.
            return_plan_plus_joint: If True (and pretrain is False), return a
                tuple ``(exec_actions, pretrain_actions)`` where exec_actions
                are the usual 4-D delta actions and pretrain_actions are the
                aligned normalised 7-DOF joint angles.  The two lists have the
                same length, so index i of exec_actions and pretrain_actions
                both correspond to the same waypoint transition.  No extra A*
                search or rollout is performed.

        Returns:
            If pretrain is False and return_plan_plus_joint is False:
                List of np.ndarray (4,) — 4-D delta actions [dx, dy, dz, gripper]
                compatible with env.step(), one per plan waypoint transition.
            If pretrain is True:
                List of np.ndarray (7,) — 7-DOF arm joint angles compatible
                with env.step_positions(), one per plan waypoint.
            If return_plan_plus_joint is True:
                Tuple (exec_actions, pretrain_actions) — both lists aligned
                by transition index.
            None if no waypoints could be found at all.
        """
        if step_size is None:
            step_size = 0.05  # 5 cm grid — finer than physics planner, but analytic

        # ── depth (horizon) limits by plan length ───────────────────────
        if plan_length == "short":
            plan_depth = self.cfg.env.short_plan_horizon
        elif plan_length == "medium":
            plan_depth = self.cfg.env.medium_plan_horizon
        elif plan_length == "long":
            plan_depth = self.cfg.env.long_plan_horizon
        else:
            plan_depth = 200  # generous default; analytic search is cheap

        max_g_cost = plan_depth * step_size  # fixed depth budget

        # ── start / goal ─────────────────────────────────────────────────
        grip_site_id = self.model.site("robot0:grip").id
        start_pos = self.data.site_xpos[grip_site_id].copy()
        obs_dict = self._get_obs()
        goal_pos = obs_dict["desired_goal"].copy()

        # ── reduce safe_margin if start is inside the exclusion zone ─────
        # After env.step() the gripper can end up clipping an obstacle face.
        # Rather than returning None immediately, quietly reduce the margin
        # so A* can still expand.  We never go below 0.01 m.
        if self._check_collision_geometric(start_pos, safe_margin):
            for try_margin in (
                safe_margin * 0.75,
                safe_margin * 0.5,
                safe_margin * 0.25,
                0.01,
            ):
                if not self._check_collision_geometric(start_pos, try_margin):
                    safe_margin = try_margin
                    break

        # ── action primitives: 26-connected grid ─────────────────────────
        # Axis-aligned (6) + face-diagonal (12) + body-diagonal (8).
        # Each action has exact Euclidean cost for A* admissibility.
        deltas = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    vec = np.array([dx, dy, dz], dtype=np.float64) * step_size
                    deltas.append((vec, float(np.linalg.norm(vec))))

        # ── state discretisation ─────────────────────────────────────────
        inv_step = 1.0 / step_size

        def state_key(pos):
            return (
                int(round(pos[0] * inv_step)),
                int(round(pos[1] * inv_step)),
                int(round(pos[2] * inv_step)),
            )

        # ── node pool (avoids duplicating large arrays in each heap entry) ─
        node_pos = [start_pos.copy()]
        node_parent = [-1]
        node_g = [0.0]

        start_key = state_key(start_pos)
        start_h = float(np.linalg.norm(start_pos - goal_pos))

        # best_g[key] = lowest g-cost at which this cell was reached
        best_g = {start_key: 0.0}

        # Priority queue entries: (f, g, node_idx)
        open_set = [(start_h, 0.0, 0)]

        expansions = 0
        goal_node = -1  # set when goal is reached
        best_node = 0  # closest node seen (fallback)
        best_dist = start_h

        while open_set and expansions < max_expansions:
            f, g, nidx = heapq.heappop(open_set)

            # ── stale-entry pruning ──────────────────────────────────────
            key = state_key(node_pos[nidx])
            if best_g.get(key, float("inf")) < g - 1e-9:
                continue

            expansions += 1

            cur_pos = node_pos[nidx]
            cur_dist = float(np.linalg.norm(cur_pos - goal_pos))

            if cur_dist < best_dist:
                best_dist = cur_dist
                best_node = nidx

            # ── goal check ───────────────────────────────────────────────
            if cur_dist < 0.05:
                goal_node = nidx
                break

            # ── depth / budget limit ─────────────────────────────────────
            if g >= max_g_cost:
                continue

            # ── expand neighbours ────────────────────────────────────────
            for delta, cost in deltas:
                new_pos = cur_pos + delta
                new_g = g + cost
                new_key = state_key(new_pos)

                if new_g >= best_g.get(new_key, float("inf")):
                    continue  # already visited with lower or equal cost

                if self._check_collision_geometric(new_pos, safe_margin):
                    continue  # geometric collision

                best_g[new_key] = new_g
                new_h = float(np.linalg.norm(new_pos - goal_pos))
                new_f = new_g + new_h

                new_nidx = len(node_pos)
                node_pos.append(new_pos.copy())
                node_parent.append(nidx)
                node_g.append(new_g)

                heapq.heappush(open_set, (new_f, new_g, new_nidx))

        # ── reconstruct waypoint path ─────────────────────────────────────
        terminal = goal_node if goal_node != -1 else best_node
        waypoints = []
        idx = terminal
        while idx != -1:
            waypoints.append(node_pos[idx].copy())
            idx = node_parent[idx]
        waypoints.reverse()

        if len(waypoints) < 2:
            waypoints = [start_pos.copy(), goal_pos.copy()]

        # ── MuJoCo rollout to materialise the plan ────────────────────────
        # Always collect both xyz and joints from the rollout; format is
        # converted below to match the original get_plan return contract.
        rollout = self._rollout_geometric_plan(waypoints, pretrain=True)

        if pretrain:
            # Return List[np.ndarray(7,)] — normalised joint angles in [-1, 1].
            # Each raw qpos[6:13] vector is mapped to [-1, 1] per joint using
            # the MuJoCo joint limits (see _normalize_joints).  step_positions()
            # de-normalises before applying, so training targets and inference
            # application share an identical, bounded representation.
            #
            # rollout[0] is the joints at the *current* position — a no-op
            # target that adds no useful information.  We skip it here so that
            # plan[0] is always the first real waypoint target, matching the
            # convention used by all other envs (e.g. NavigationEnv) where
            # plan[0] is the first action to execute from the current state.
            # collect_pretrain_data_with_planner iterates over plan (not
            # plan[1:]) and relies on this contract.
            return [self._normalize_joints(joints) for (_xyz, joints) in rollout[1:]]
        else:
            # Return List[np.ndarray(4,)] — 4D delta actions for env.step().
            # _set_action scales action[:3] by 0.05 m, so dividing the xyz
            # displacement by 0.05 recovers the normalised action component.
            ACTION_SCALE = 0.05
            exec_actions = []
            for i in range(1, len(rollout)):
                delta = rollout[i][0] - rollout[i - 1][0]
                act = np.zeros(4, dtype=np.float32)
                act[:3] = np.clip(delta / ACTION_SCALE, -1.0, 1.0)
                act[3] = -1.0  # keep gripper closed
                exec_actions.append(act)

            if not exec_actions:
                return (None, None) if return_plan_plus_joint else None

            if return_plan_plus_joint:
                # Joint angles aligned to exec_actions: rollout[1:] matches
                # exec_actions[0..n-1] (rollout[0] is the no-op current state).
                pretrain_actions = [
                    self._normalize_joints(joints) for (_xyz, joints) in rollout[1:]
                ]
                return exec_actions, pretrain_actions

            return exec_actions

    def _rollout_geometric_plan(self, waypoints, pretrain: bool = False):
        """Execute a sequence of Cartesian waypoints via mocap-IK rollout.

        The gripper is driven to each waypoint by directly writing to
        ``data.mocap_pos`` and running a short burst of ``mj_step`` calls so
        that the internal IK controller settles.  This is the same mechanism
        used in :meth:`reset`.

        Crucially, the environment's physics state is ***saved before*** the
        rollout and ***restored afterwards***, so the method is side-effect
        free from the caller's perspective.

        Args:
            waypoints: Ordered list of (3,) target gripper positions.
            pretrain: If True, also record the 7-DOF arm joint angles at each
                waypoint (indices 6–12 in ``qpos``).

        Returns:
            If pretrain is False:
                List[np.ndarray (3,)]  — true gripper xyz per waypoint.
            If pretrain is True:
                List[Tuple[np.ndarray (3,), np.ndarray (7,)]]  — (xyz, joints).
        """
        # ── save environment state ────────────────────────────────────────
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_time = self.time

        grip_site_id = self.model.site("robot0:grip").id

        # Number of simulation steps to let IK settle per waypoint.
        # More substeps → better convergence; ~20 substep-groups is enough.
        settle_steps = max(20, self.action_repeat * 4)

        results = []

        try:
            for wp in waypoints:
                # Move mocap target to waypoint (same technique as reset)
                if self.model.nmocap > 0:
                    self.data.mocap_pos[0] = wp.copy()
                    self.data.mocap_quat[0] = np.array([0.707, 0.0, 0.707, 0.0])

                # Let IK controller converge
                for _ in range(settle_steps):
                    mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

                # Read true gripper position from forward kinematics
                mujoco.mj_forward(self.model, self.data)
                true_xyz = self.data.site_xpos[grip_site_id].copy()

                if pretrain:
                    joints = self.data.qpos[6:13].copy()
                    results.append((true_xyz, joints))
                else:
                    results.append(true_xyz)

        finally:
            # ── restore environment state (side-effect free) ─────────────
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)
            self.time = saved_time

        return results
