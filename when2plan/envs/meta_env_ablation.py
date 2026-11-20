import numpy as np
import gymnasium as gym
from when2plan.models.model_free_actor import (
    ModelFreeActorTensor,
    ModelFreeActorImage,
    ModelFreeActorTensorContinuous,
    ModelFreeActorImageContinuous,
)
import torch
import torch.optim as optim
from gymnasium.spaces import Dict, Box
from collections import deque
import time

"""
ablation options:
- raw_state: provide the raw state from the sub-environment as input to the meta-controller, instead of the history-based observation
- no_history: keep the same observation space but ablate the history by only providing the current
- no_model_free_uncertainty: ablate the model free uncertainty from the observation (set to 0)
- no_distance_to_goal: ablate the distance to goal from the observation (set to
- no_option_duration: ablate the option duration from the observation (set to 0)
- no_last_action: ablate the last action from the observation (set to 0)

"""


class MetaEnvAblation(gym.Env):
    def __init__(
        self, cfg, sub_env, model_free_actors=None, model_free_actor_optimizers=None
    ):
        self.ablation = cfg.ablation
        self.sub_env = sub_env
        self.cfg = cfg
        self.logging = cfg.experiment.get("logging", False)
        self.step_type = cfg.get(
            "meta_step_type", "execute"
        )  # "bandit", "execute", "seperate"
        self.num_pretrained_tasks = cfg.experiment.get("num_pretrained_tasks", 0)
        self.continuous_sub_env_action_space = (
            cfg.env.model_free_observation_space == "continuous"
        )

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        self.device = device

        self.running_time = 0

        if model_free_actors is None:
            print("Model free actors not provided, initializing new ones")

            model_free_actors = []
            model_free_actor_optimizers = []

            # Get the model free actor class
            if self.continuous_sub_env_action_space:
                action_dim = self.sub_env.action_space.shape[0]
                if cfg.env.model_free_space == "image":
                    model_free_actor_class = ModelFreeActorImageContinuous
                else:
                    model_free_actor_class = ModelFreeActorTensorContinuous
            else:
                action_dim = self.sub_env.action_space.n
                if cfg.env.model_free_space == "image":
                    model_free_actor_class = ModelFreeActorImage
                else:
                    model_free_actor_class = ModelFreeActorTensor

            for _ in range(cfg.num_actors):
                # Condition on type of action space
                model_free_actor = model_free_actor_class(
                    self.sub_env.observation_space, action_dim
                )
                model_free_actor.to(device)
                model_free_actors.append(model_free_actor)
                model_free_actor_optimizers.append(
                    optim.Adam(model_free_actor.parameters(), lr=1e-3)
                )
        else:
            for actor in model_free_actors:
                actor.to(device)

        # Save args and tasks
        self.model_free_actors = model_free_actors
        self.model_free_actor_optimizers = model_free_actor_optimizers

        self.energy_T = cfg.energy_T
        self.rewards = cfg.experiment.rewards

        self.max_planning_budget = cfg.env.max_planning_budget

        self.uncertainty_average = 0
        self.num_uncertainty_samples = 0

        self.action_space = gym.spaces.Discrete(4)

        # Clips/normalizes for observation space

        if self.cfg.env.model_free_action_space == "continuous":
            max_uncertainty = 1.0
        else:
            max_uncertainty = np.log(self.sub_env.action_space.n)
        self.model_free_uncertainty_bounds = [0.0, max_uncertainty]

        self.history_length = 4
        # Each history entry: (distance_to_goal, model_free_uncertainty, option_duration, last_action)
        # option_duration is normalized by max_steps; last_action is normalized by (num_actions - 1)
        _feature_ablations = {
            "no_model_free_uncertainty",
            "no_distance_to_goal",
            "no_option_duration",
            "no_last_action",
        }
        # When a feature is ablated it is excluded entirely, shrinking the entry by 1
        self.history_entry_size = 3 if self.ablation in _feature_ablations else 4
        observation_size = self.history_entry_size * self.history_length

        if self.ablation == "raw_state":
            # The raw state fed to the meta-controller should match what the
            # model-free actors see: get_tensor_state() returns (C, H, W) for
            # image envs and a 1-D vector for state-based envs.
            sub_obs_shape = self.sub_env.observation_space.shape
            if len(sub_obs_shape) == 3:  # HWC image -> CHW tensor
                h, w, c = sub_obs_shape
                raw_state_shape = (c, h, w)
            else:
                raw_state_shape = sub_obs_shape
            self.observation_space = Box(
                low=0, high=1, shape=raw_state_shape, dtype=np.float32
            )
        elif self.ablation == "no_history":
            self.observation_space = Box(
                low=0, high=np.inf, shape=(self.history_entry_size,), dtype=np.float32
            )
        else:
            self.observation_space = Box(
                low=0, high=np.inf, shape=(observation_size,), dtype=np.float32
            )

        self.history = deque(maxlen=self.history_length)
        self.last_option_duration = 0
        self.last_action = 0

    def sub_env_render(self, mode="human", scaling_factor=None):
        return self.sub_env.render(mode=mode, scaling_factor=scaling_factor)

    def get_model_free_nets_and_opts(self):
        return self.model_free_actors, self.model_free_actor_optimizers

    def construct_obs(self):
        if self.ablation == "raw_state":
            tensor = self.sub_env.get_tensor_state()
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy()
            return np.array(tensor, dtype=np.float32)

        model_free_uncertainty = np.clip(
            self.model_free_uncertainty,
            self.model_free_uncertainty_bounds[0],
            self.model_free_uncertainty_bounds[1],
        )
        model_free_uncertainty = (
            model_free_uncertainty / self.model_free_uncertainty_bounds[1]
        )

        distance_to_goal = self.sub_env.get_distance_to_goal()

        # Normalize option duration by max episode steps so it stays in [0, 1]
        max_steps = self.cfg.env.get("max_steps", 100)
        option_duration_norm = self.last_option_duration / max_steps

        # Normalize last action to [0, 1]
        num_meta_actions = self.action_space.n  # 4
        last_action_norm = self.last_action / (num_meta_actions - 1)

        # Build entry, excluding the ablated feature entirely
        features = [
            distance_to_goal,
            model_free_uncertainty,
            option_duration_norm,
            last_action_norm,
        ]
        feature_names = [
            "distance_to_goal",
            "model_free_uncertainty",
            "option_duration",
            "last_action",
        ]
        ablation_map = {
            "no_distance_to_goal": "distance_to_goal",
            "no_model_free_uncertainty": "model_free_uncertainty",
            "no_option_duration": "option_duration",
            "no_last_action": "last_action",
        }
        excluded = ablation_map.get(self.ablation)
        features = [v for name, v in zip(feature_names, features) if name != excluded]

        current_entry = np.array(features, dtype=np.float32)

        if self.ablation == "no_history":
            return current_entry

        self.history.append(current_entry)
        obs = np.concatenate(list(self.history))
        return obs

    def render(self, render_mode="rgb_array"):
        return self.sub_env.render(render_mode=render_mode)

    def reset(self, **kwargs):
        self.time = 0
        self.last_option_duration = 0
        self.last_action = 0

        initial_entry = np.zeros(self.history_entry_size, dtype=np.float32)
        for _ in range(self.history_length):
            self.history.append(initial_entry)

        _, sub_info = self.sub_env.reset()
        self.model_free_action, self.model_free_uncertainty = (
            self.model_free_actor_inference()
        )

        obs = self.construct_obs()
        info = {}
        self.sub_env_task_idx = sub_info["task_idx"]

        return obs, info

    def model_free_actor_inference(self):
        """
        Performs one round of inference with the model-free actor ensemble.
        Returns the chosen action and ensemble disagreement as uncertainty metric.

        uncertainty metric: Higher values = more uncertainty/disagreement (OOD-like)
                        Lower values = more agreement (ID-like)
        """
        input_tensor = self.sub_env.get_tensor_state()

        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.to(self.device)

        if self.cfg.env.model_free_action_space == "continuous":
            # Continuous action space: actors output normalised predictions.
            # Denormalise using per-joint stats stored in actor buffers.
            all_means = []
            for actor in self.model_free_actors:
                with torch.inference_mode():
                    output = actor(input_tensor)
                    # Handle tuple output (mean, log_std)
                    mean = output[0] if isinstance(output, tuple) else output
                    # # Apply per-joint denormalisation if stats are available
                    # if hasattr(actor, 'denormalize'):
                    #     mean = actor.denormalize(mean)
                    # elif hasattr(actor, 'action_mean') and hasattr(actor, 'action_std'):
                    #     mean = mean * actor.action_std.to(mean.device) + actor.action_mean.to(mean.device)
                    all_means.append(mean)

            # Stack means: (num_actors, action_dim)
            all_means = torch.stack(all_means, dim=0)

            # Get ensemble prediction (average means)
            mean_action = all_means.mean(dim=0)  # (action_dim,)
            chosen_action = mean_action.cpu().numpy()

            # Use variance across ensemble predictions as uncertainty
            # Higher variance = more disagreement = likely OOD
            action_variance = all_means.var(dim=0).sum().item()
            uncertainty = action_variance

        else:
            # Discrete action space: actors output logits
            all_logits = []
            for actor in self.model_free_actors:
                with torch.inference_mode():
                    logits = actor(input_tensor)
                    all_logits.append(logits)

            # Stack logits: (num_actors, num_actions)
            all_logits = torch.stack(all_logits, dim=0)

            # Convert to probabilities
            all_probs = torch.softmax(all_logits, dim=-1)  # (num_actors, num_actions)

            # Get ensemble prediction (average probabilities)
            mean_probs = all_probs.mean(dim=0)  # (num_actions,)
            chosen_action = int(torch.argmax(mean_probs).item())

            # Higher entropy = more uncertain = likely OOD
            epsilon = 1e-10  # numerical stability
            predictive_entropy = -torch.sum(
                mean_probs * torch.log(mean_probs + epsilon)
            ).item()
            uncertainty = predictive_entropy

        return chosen_action, uncertainty

    def step(self, action):
        return self.step_commit(action)

    """
    Action space of:
    0) Model free
    1) Short plan
    2) Medium plan
    3) Long plan
    
    and note we follow the plan fully once committed
    """

    def step_commit(self, action):
        self.time += 1

        logging = {}
        expert_data = {}
        info = {}

        meta_action = None

        # gamma for within-option discounting so accumulated reward is SMDP-correct
        gamma = self.cfg.meta_discount

        if action == 0:  # Model free
            meta_action = "model_free"
            rew = self.rewards["act"]
            option_duration = 1

            _, _, term, trunc, info = self.sub_env.model_free_step(
                self.model_free_action
            )
        else:  # Plan
            plan_scaling_factor = self.cfg.get("plan_cost_scaling_factor", 1.0)

            if action == 1:
                plan = self.sub_env.get_plan(
                    max_expansions=self.max_planning_budget, plan_length="short"
                )
                short_plan_depth = self.cfg.env.short_plan_horizon
                # Overhead cost for invoking the planner (separate from execution cost)
                rew = -1 * short_plan_depth * plan_scaling_factor
                meta_action = "short_plan"

            elif action == 2:
                plan = self.sub_env.get_plan(
                    max_expansions=self.max_planning_budget, plan_length="medium"
                )
                medium_plan_depth = self.cfg.env.medium_plan_horizon
                rew = -1 * medium_plan_depth * plan_scaling_factor
                meta_action = "medium_plan"

            elif action == 3:
                plan = self.sub_env.get_plan(
                    max_expansions=self.max_planning_budget, plan_length="long"
                )
                long_plan_depth = self.cfg.env.long_plan_horizon
                rew = -1 * long_plan_depth * plan_scaling_factor
                meta_action = "long_plan"

            # Follow the plan with geometric discounting (SMDP-correct accumulation)
            # r_total = overhead + sum_k gamma^k * r_k
            # This ensures rewards from later steps in the option are properly down-weighted
            # so that PPO's single gamma application at the meta level is less distorting.
            option_duration = 0
            discount = 1.0
            for plan_action in plan:
                _, sub_reward, term, trunc, info = self.sub_env.step(plan_action)
                rew += discount * self.rewards["act"]
                discount *= gamma
                option_duration += 1

                if term or trunc:
                    break

        # Track option duration and last action for next observation
        self.last_option_duration = option_duration
        self.last_action = action

        # Needs to be after reward calculation and before reassignment of model free action and uncertainty
        if self.logging:
            logging["model_free_actor_uncertainty"] = self.model_free_uncertainty
            logging["reward"] = rew
            logging["time"] = self.time
            logging["option_duration"] = option_duration

            in_distribution = self.sub_env_task_idx < self.num_pretrained_tasks

            if meta_action == "model_free":
                logging["model_free_actor_uncertainty_on_model_free"] = (
                    self.model_free_uncertainty
                )
                logging["meta_action"] = [1, 0, 0, 0]
                if in_distribution:
                    logging["in_distribution_model_free"] = 1
                    logging["out_distribution_model_free"] = 0
                else:
                    logging["in_distribution_model_free"] = 0
                    logging["out_distribution_model_free"] = 1
            elif meta_action == "short_plan":
                logging["model_free_actor_uncertainty_on_short_plan"] = (
                    self.model_free_uncertainty
                )
                logging["meta_action"] = [0, 1, 0, 0]
                if in_distribution:
                    logging["in_distribution_short_plan"] = 1
                    logging["out_distribution_short_plan"] = 0
                else:
                    logging["in_distribution_short_plan"] = 0
                    logging["out_distribution_short_plan"] = 1
            elif meta_action == "medium_plan":
                logging["model_free_actor_uncertainty_on_medium_plan"] = (
                    self.model_free_uncertainty
                )
                logging["meta_action"] = [0, 0, 1, 0]
                if in_distribution:
                    logging["in_distribution_medium_plan"] = 1
                    logging["out_distribution_medium_plan"] = 0
                else:
                    logging["in_distribution_medium_plan"] = 0
                    logging["out_distribution_medium_plan"] = 1
            elif meta_action == "long_plan":
                logging["model_free_actor_uncertainty_on_long_plan"] = (
                    self.model_free_uncertainty
                )
                logging["meta_action"] = [0, 0, 0, 1]
                if in_distribution:
                    logging["in_distribution_long_plan"] = 1
                    logging["out_distribution_long_plan"] = 0
                else:
                    logging["in_distribution_long_plan"] = 0
                    logging["out_distribution_long_plan"] = 1

        # Get uncertainty and action from model free actor
        self.model_free_action, self.model_free_uncertainty = (
            self.model_free_actor_inference()
        )

        obs_dict = self.construct_obs()

        info.update(logging)
        info.update(expert_data)

        return obs_dict, rew, term, trunc, info
