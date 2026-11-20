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


class MetaEnv(gym.Env):
    def __init__(
        self, cfg, sub_env, model_free_actors=None, model_free_actor_optimizers=None
    ):
        self.sub_env = sub_env
        self.cfg = cfg
        self.logging = cfg.experiment.get("logging", False)
        self.joint_training = cfg.experiment.get("joint_training", False)
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

        if self.joint_training:
            # Create optimizers
            model_free_actor_optimizers = []
            for actor in model_free_actors:
                model_free_actor_optimizers.append(
                    optim.Adam(actor.parameters(), lr=1e-4)
                )
            # Create fifo for storing data (BC so only need obs, action)
            self.joint_training_buffer = {
                "observations": deque(
                    maxlen=self.cfg.experiment.get("replay_buffer_size", 10000)
                ),
                "actions": deque(
                    maxlen=self.cfg.experiment.get("replay_buffer_size", 10000)
                ),
                "total_added": 0,
            }

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
        # Each history entry: (distance_to_goal, model_free_uncertainty, last_action)
        # last_action is normalized by (num_actions - 1)
        self.history_entry_size = 3
        observation_size = self.history_entry_size * self.history_length

        self.observation_space = Box(
            low=0, high=np.inf, shape=(observation_size,), dtype=np.float32
        )

        self.history = deque(maxlen=self.history_length)
        self.last_action = 0

    def sub_env_render(self, mode="human", scaling_factor=None):
        return self.sub_env.render(mode=mode, scaling_factor=scaling_factor)

    def get_model_free_nets_and_opts(self):
        return self.model_free_actors, self.model_free_actor_optimizers

    def construct_obs(self):
        model_free_uncertainty = np.clip(
            self.model_free_uncertainty,
            self.model_free_uncertainty_bounds[0],
            self.model_free_uncertainty_bounds[1],
        )
        model_free_uncertainty = (
            model_free_uncertainty / self.model_free_uncertainty_bounds[1]
        )

        distance_to_goal = self.sub_env.get_distance_to_goal()

        # Normalize last action to [0, 1]
        num_meta_actions = self.action_space.n  # 4
        last_action_norm = self.last_action / (num_meta_actions - 1)

        current_entry = np.array(
            [
                distance_to_goal,
                model_free_uncertainty,
                last_action_norm,
            ],
            dtype=np.float32,
        )
        self.history.append(current_entry)

        obs = np.concatenate(list(self.history))
        return obs

    def render(self, render_mode="rgb_array"):
        return self.sub_env.render(render_mode=render_mode)

    def reset(self, **kwargs):
        self.time = 0
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

            _, _, term, trunc, info = self.sub_env.model_free_step(
                self.model_free_action
            )
        else:  # Plan
            plan_scaling_factor = self.cfg.get("plan_cost_scaling_factor", 1.0)
            # Only for fetch since has unusual action space difference
            fetch_pretrain_actions = None

            if action == 1:
                if self.joint_training and self.cfg.env.type == "fetch_pick_place":
                    plan, fetch_pretrain_actions = self.sub_env.get_plan(
                        max_expansions=self.max_planning_budget,
                        plan_length="short",
                        return_plan_plus_joint=True,
                    )
                else:
                    plan = self.sub_env.get_plan(
                        max_expansions=self.max_planning_budget, plan_length="short"
                    )
                short_plan_depth = self.cfg.env.short_plan_horizon
                rew = -1 * short_plan_depth * plan_scaling_factor
                meta_action = "short_plan"

            elif action == 2:
                if self.joint_training and self.cfg.env.type == "fetch_pick_place":
                    plan, fetch_pretrain_actions = self.sub_env.get_plan(
                        max_expansions=self.max_planning_budget,
                        plan_length="medium",
                        return_plan_plus_joint=True,
                    )
                else:
                    plan = self.sub_env.get_plan(
                        max_expansions=self.max_planning_budget, plan_length="medium"
                    )
                medium_plan_depth = self.cfg.env.medium_plan_horizon
                rew = -1 * medium_plan_depth * plan_scaling_factor
                meta_action = "medium_plan"

            elif action == 3:
                if self.joint_training and self.cfg.env.type == "fetch_pick_place":
                    plan, fetch_pretrain_actions = self.sub_env.get_plan(
                        max_expansions=self.max_planning_budget,
                        plan_length="long",
                        return_plan_plus_joint=True,
                    )
                else:
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
            discount = 1.0
            for step_idx, plan_action in enumerate(plan):
                # Collect BC data for joint training: capture state before the action is taken.
                # For continuous envs (fetch) store the normalised 7-DOF joint angles so
                # that the BC targets match the model-free actor's output space, not the
                # 4-D delta actions used for environment stepping.
                if self.joint_training:
                    obs_tensor = self.sub_env.get_tensor_state()
                    if isinstance(obs_tensor, torch.Tensor):
                        obs_tensor = obs_tensor.detach().cpu()
                    if fetch_pretrain_actions is not None and step_idx < len(
                        fetch_pretrain_actions
                    ):
                        bc_action = fetch_pretrain_actions[step_idx]
                    else:
                        bc_action = plan_action
                    self.joint_training_buffer["observations"].append(obs_tensor)
                    self.joint_training_buffer["actions"].append(bc_action)
                    self.joint_training_buffer["total_added"] += 1

                _, sub_reward, term, trunc, info = self.sub_env.step(plan_action)
                rew += discount * self.rewards["act"]
                discount *= gamma

                if term or trunc:
                    break

        self.last_action = action

        # Needs to be after reward calculation and before reassignment of model free action and uncertainty
        if self.logging:
            logging["model_free_actor_uncertainty"] = self.model_free_uncertainty
            logging["reward"] = rew
            logging["time"] = self.time

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
