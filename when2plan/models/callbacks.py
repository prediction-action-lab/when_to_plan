from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import collections
from when2plan.utils import behavior_clone, behavior_clone_continuous
import torch
import pandas as pd
import copy


class TrainLoggingCallback(BaseCallback):
    def __init__(self, cfg=None, output_dir=None, env=None):
        super().__init__()

        self.plot_freq = cfg.log_freq
        self.num_eval_resets = cfg.num_eval_resets
        self.output_dir = output_dir
        self.env_name = cfg.env.type

        # Store overall data across all rollouts
        self.data = {
            "meta_action": [],
            "reward": [],
            "time": [],
            "env_name": [],
            "model_free_actor_uncertainty": [],
            # Standard mode uncertaintys (execute, seperate, bandit)
            "model_free_actor_uncertainty_on_plan": [],
            "model_free_actor_uncertainty_on_follow_plan": [],
            "model_free_actor_uncertainty_on_model_free": [],
            "model_free_actor_uncertainty_on_clear_plan": [],
            # Commit mode uncertaintys
            "model_free_actor_uncertainty_on_short_plan": [],
            "model_free_actor_uncertainty_on_medium_plan": [],
            "model_free_actor_uncertainty_on_long_plan": [],
            # Standard mode distribution tracking
            "in_distribution_model_free_percentage": [],
            "out_of_distribution_model_free_percentage": [],
            # Bandit mode distribution tracking (separate for plan and model_free)
            "in_distribution_plan": [],
            "out_distribution_plan": [],
            # Seperate mode distribution tracking
            "in_distribution_follow_plan": [],
            "out_distribution_follow_plan": [],
            # Execute mode distribution tracking (includes clear_plan)
            "in_distribution_clear_plan": [],
            "out_distribution_clear_plan": [],
            # Commit mode distribution tracking (all 4 actions)
            "in_distribution_model_free": [],
            "in_distribution_short_plan": [],
            "in_distribution_medium_plan": [],
            "in_distribution_long_plan": [],
            "out_distribution_model_free": [],
            "out_distribution_short_plan": [],
            "out_distribution_medium_plan": [],
            "out_distribution_long_plan": [],
            "steps": [],
            "rollout_number": [],
        }

        # Track data per rollout (accumulated across all envs and episodes)
        self.rollout_data = self._empty_rollout_data()
        self.rollout_number = 0

        # NEW: Track per-episode metrics within a rollout
        self.episode_rewards = []
        self.episode_steps = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0

    def _empty_rollout_data(self):
        return {
            "meta_action": [0, 0, 0, 0],
            "reward": 0,
            "time": 0,
            "model_free_actor_uncertainty": 0.0,
            # Standard mode uncertaintys
            "model_free_actor_uncertainty_on_plan": 0.0,
            "model_free_actor_uncertainty_on_follow_plan": 0.0,
            "model_free_actor_uncertainty_on_model_free": 0.0,
            "model_free_actor_uncertainty_on_clear_plan": 0.0,
            # Commit mode uncertaintys
            "model_free_actor_uncertainty_on_short_plan": 0.0,
            "model_free_actor_uncertainty_on_medium_plan": 0.0,
            "model_free_actor_uncertainty_on_long_plan": 0.0,
            "steps": 0,
            # Standard mode distribution tracking
            "in_distribution_model_free_count": 0,
            "in_distribution_total_count": 0,
            "out_of_distribution_model_free_count": 0,
            "out_of_distribution_total_count": 0,
            # Bandit mode distribution tracking
            "in_distribution_plan": 0,
            "out_distribution_plan": 0,
            # Seperate mode distribution tracking
            "in_distribution_follow_plan": 0,
            "out_distribution_follow_plan": 0,
            # Execute mode distribution tracking (includes clear_plan)
            "in_distribution_clear_plan": 0,
            "out_distribution_clear_plan": 0,
            # Commit mode distribution tracking (counts for all 4 actions)
            "in_distribution_model_free": 0,
            "in_distribution_short_plan": 0,
            "in_distribution_medium_plan": 0,
            "in_distribution_long_plan": 0,
            "out_distribution_model_free": 0,
            "out_distribution_short_plan": 0,
            "out_distribution_medium_plan": 0,
            "out_distribution_long_plan": 0,
            # Track counts for each action type
            "count_plan": 0,
            "count_follow_plan": 0,
            "count_model_free": 0,
            "count_clear_plan": 0,
            # Commit mode action counts
            "count_short_plan": 0,
            "count_medium_plan": 0,
            "count_long_plan": 0,
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False])

        for i, info in enumerate(infos):
            if info:
                self.rollout_data["steps"] += 1
                self.current_episode_steps += 1

                # Update action counts
                if "meta_action" in info:
                    action = info["meta_action"]
                    if action == [1, 0, 0, 0]:
                        self.rollout_data["meta_action"][0] += 1
                        self.rollout_data["count_plan"] += 1
                    elif action == [0, 1, 0, 0]:
                        self.rollout_data["meta_action"][1] += 1
                        self.rollout_data["count_follow_plan"] += 1
                    elif action == [0, 0, 1, 0]:
                        self.rollout_data["meta_action"][2] += 1
                        self.rollout_data["count_model_free"] += 1
                    elif action == [0, 0, 0, 1]:
                        self.rollout_data["meta_action"][3] += 1
                        self.rollout_data["count_clear_plan"] += 1

                # Update reward
                if "reward" in info:
                    self.rollout_data["reward"] += info["reward"]
                    self.current_episode_reward += info["reward"]

                # Update actor uncertaintys
                if "model_free_actor_uncertainty" in info:
                    self.rollout_data["model_free_actor_uncertainty"] += info[
                        "model_free_actor_uncertainty"
                    ]
                # Standard mode uncertaintys
                if "model_free_actor_uncertainty_on_plan" in info:
                    self.rollout_data["model_free_actor_uncertainty_on_plan"] += info[
                        "model_free_actor_uncertainty_on_plan"
                    ]
                if "model_free_actor_uncertainty_on_model_free" in info:
                    self.rollout_data[
                        "model_free_actor_uncertainty_on_model_free"
                    ] += info["model_free_actor_uncertainty_on_model_free"]
                if "model_free_actor_uncertainty_on_follow_plan" in info:
                    self.rollout_data[
                        "model_free_actor_uncertainty_on_follow_plan"
                    ] += info["model_free_actor_uncertainty_on_follow_plan"]
                if "model_free_actor_uncertainty_on_clear_plan" in info:
                    self.rollout_data[
                        "model_free_actor_uncertainty_on_clear_plan"
                    ] += info["model_free_actor_uncertainty_on_clear_plan"]
                # Commit mode uncertaintys
                if "model_free_actor_uncertainty_on_short_plan" in info:
                    self.rollout_data[
                        "model_free_actor_uncertainty_on_short_plan"
                    ] += info["model_free_actor_uncertainty_on_short_plan"]
                    self.rollout_data["count_short_plan"] += 1
                if "model_free_actor_uncertainty_on_medium_plan" in info:
                    self.rollout_data[
                        "model_free_actor_uncertainty_on_medium_plan"
                    ] += info["model_free_actor_uncertainty_on_medium_plan"]
                    self.rollout_data["count_medium_plan"] += 1
                if "model_free_actor_uncertainty_on_long_plan" in info:
                    self.rollout_data[
                        "model_free_actor_uncertainty_on_long_plan"
                    ] += info["model_free_actor_uncertainty_on_long_plan"]
                    self.rollout_data["count_long_plan"] += 1

                # Track in-distribution vs out-of-distribution model-free usage (standard modes)
                if "in_distribution_model_free" in info:
                    self.rollout_data["in_distribution_total_count"] += 1
                    if info["in_distribution_model_free"] == 1:
                        self.rollout_data["in_distribution_model_free_count"] += 1

                if "out_of_distribution_model_free" in info:
                    self.rollout_data["out_of_distribution_total_count"] += 1
                    if info["out_of_distribution_model_free"] == 1:
                        self.rollout_data["out_of_distribution_model_free_count"] += 1

                # Track bandit mode distribution (plan and model_free separately)
                if "in_distribution_plan" in info:
                    self.rollout_data["in_distribution_plan"] += info[
                        "in_distribution_plan"
                    ]
                    self.rollout_data["out_distribution_plan"] += info.get(
                        "out_distribution_plan", 0
                    )

                # Track seperate mode distribution (plan, follow_plan, model_free)
                if "in_distribution_follow_plan" in info:
                    self.rollout_data["in_distribution_follow_plan"] += info[
                        "in_distribution_follow_plan"
                    ]
                    self.rollout_data["out_distribution_follow_plan"] += info.get(
                        "out_distribution_follow_plan", 0
                    )

                # Track execute mode distribution (includes clear_plan)
                if "in_distribution_clear_plan" in info:
                    self.rollout_data["in_distribution_clear_plan"] += info[
                        "in_distribution_clear_plan"
                    ]
                    self.rollout_data["out_distribution_clear_plan"] += info.get(
                        "out_distribution_clear_plan", 0
                    )

                # Track commit mode distribution (all 4 actions)
                # Each action logs its own distribution fields separately
                if (
                    "in_distribution_model_free" in info
                    and "out_distribution_model_free" in info
                ):
                    # Commit mode model_free action
                    self.rollout_data["in_distribution_model_free"] += info[
                        "in_distribution_model_free"
                    ]
                    self.rollout_data["out_distribution_model_free"] += info[
                        "out_distribution_model_free"
                    ]
                if "in_distribution_short_plan" in info:
                    self.rollout_data["in_distribution_short_plan"] += info[
                        "in_distribution_short_plan"
                    ]
                    self.rollout_data["out_distribution_short_plan"] += info.get(
                        "out_distribution_short_plan", 0
                    )
                if "in_distribution_medium_plan" in info:
                    self.rollout_data["in_distribution_medium_plan"] += info[
                        "in_distribution_medium_plan"
                    ]
                    self.rollout_data["out_distribution_medium_plan"] += info.get(
                        "out_distribution_medium_plan", 0
                    )
                if "in_distribution_long_plan" in info:
                    self.rollout_data["in_distribution_long_plan"] += info[
                        "in_distribution_long_plan"
                    ]
                    self.rollout_data["out_distribution_long_plan"] += info.get(
                        "out_distribution_long_plan", 0
                    )

            if dones[i]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_steps.append(self.current_episode_steps)
                self.current_episode_reward = 0
                self.current_episode_steps = 0

        return True

    def _on_rollout_end(self):
        steps = self.rollout_data["steps"]

        if steps > 0:
            # NEW: Average time and reward per episode
            if len(self.episode_steps) > 0:
                self.rollout_data["time"] = np.mean(self.episode_steps)
                self.rollout_data["reward"] = np.mean(self.episode_rewards)

            # Average overall uncertainty by total steps
            self.rollout_data["model_free_actor_uncertainty"] /= steps

            # Average conditional uncertaintys by their respective counts
            if self.rollout_data["count_plan"] > 0:
                self.rollout_data[
                    "model_free_actor_uncertainty_on_plan"
                ] /= self.rollout_data["count_plan"]
            if self.rollout_data["count_follow_plan"] > 0:
                self.rollout_data[
                    "model_free_actor_uncertainty_on_follow_plan"
                ] /= self.rollout_data["count_follow_plan"]
            if self.rollout_data["count_model_free"] > 0:
                self.rollout_data[
                    "model_free_actor_uncertainty_on_model_free"
                ] /= self.rollout_data["count_model_free"]
            if self.rollout_data["count_clear_plan"] > 0:
                self.rollout_data[
                    "model_free_actor_uncertainty_on_clear_plan"
                ] /= self.rollout_data["count_clear_plan"]
            # Commit mode uncertaintys
            if self.rollout_data["count_short_plan"] > 0:
                self.rollout_data[
                    "model_free_actor_uncertainty_on_short_plan"
                ] /= self.rollout_data["count_short_plan"]
            if self.rollout_data["count_medium_plan"] > 0:
                self.rollout_data[
                    "model_free_actor_uncertainty_on_medium_plan"
                ] /= self.rollout_data["count_medium_plan"]
            if self.rollout_data["count_long_plan"] > 0:
                self.rollout_data[
                    "model_free_actor_uncertainty_on_long_plan"
                ] /= self.rollout_data["count_long_plan"]

            # Calculate percentages for in-distribution and out-of-distribution
            if self.rollout_data["in_distribution_total_count"] > 0:
                self.rollout_data["in_distribution_model_free_percentage"] = (
                    self.rollout_data["in_distribution_model_free_count"]
                    / self.rollout_data["in_distribution_total_count"]
                    * 100
                )
            else:
                self.rollout_data["in_distribution_model_free_percentage"] = 0

            if self.rollout_data["out_of_distribution_total_count"] > 0:
                self.rollout_data["out_of_distribution_model_free_percentage"] = (
                    self.rollout_data["out_of_distribution_model_free_count"]
                    / self.rollout_data["out_of_distribution_total_count"]
                    * 100
                )
            else:
                self.rollout_data["out_of_distribution_model_free_percentage"] = 0

        # Don't save the count fields to CSV
        self.rollout_data["rollout_number"] = self.rollout_number
        self.rollout_data["env_name"] = self.env_name

        # Append to overall data (excluding count fields)
        for key in self.data:
            self.data[key].append(self.rollout_data[key])

        self.rollout_number += 1

        # Save at specified frequency
        if self.rollout_number % self.plot_freq == 0:
            self.save_data()

        # Reset for next rollout
        self.rollout_data = self._empty_rollout_data()
        # NEW: Reset episode tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0

        return True

    def save_data(self):
        if self.rollout_number == 0 or len(self.data["reward"]) == 0:
            return
        data_df = pd.DataFrame(self.data)
        data_df.to_csv(
            os.path.join(self.output_dir, "training_log.csv"),
            index=False,
        )
        # print(
        #     "Saved training log to ", os.path.join(self.output_dir, "training_log.csv")
        # )


class BehaviorCloningCallback(BaseCallback):
    def __init__(
        self,
        model_free_networks,
        model_free_optimizers,
        buffer_size,
        batch_size,
        bc_epochs,
        bc_buffer_samples,
        joint_training_buffer=None,
        is_continuous=False,
        bc_update_freq=1,
        bc_min_new_samples=500,
        verbose=0,
    ):
        """Behavior cloning callback.

        Args:
            model_free_networks: Ensemble list of model-free networks.
            model_free_optimizers: Corresponding list of optimizers.
            buffer_size: maxlen for the fallback info-dict replay buffer.
            batch_size: Mini-batch size for BC updates.
            bc_epochs: Training epochs per rollout-end update.
            bc_buffer_samples: Max samples drawn from buffer per update.
            joint_training_buffer: If provided, a dict with 'observations' and
                'actions' deques sourced directly from MetaEnv.  When set, the
                info-dict replay buffer is not used.
            is_continuous: If True, use MSE-based behavior_clone_continuous
                (for continuous action spaces such as fetch_pick_place).
            bc_update_freq: Run a BC update every this many rollout-ends.
            bc_min_new_samples: Minimum number of new samples that must have been
                added since the last BC update before training is triggered.
        """
        super().__init__(verbose)
        # Fallback replay buffer (used when joint_training_buffer is None)
        self.replay_buffer = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.bc_epochs = bc_epochs
        self.bc_buffer_samples = bc_buffer_samples

        self.model_free_networks = model_free_networks
        self.optimizers = model_free_optimizers

        # Direct reference to MetaEnv's buffer (mutable deque — always up to date)
        self.joint_training_buffer = joint_training_buffer
        self.is_continuous = is_continuous

        self.num_rollouts = 0
        self.bc_update_freq = bc_update_freq
        self.bc_min_new_samples = bc_min_new_samples
        # Monotonic counters for change detection (robust to full deques)
        self._last_trained_total_added = 0  # for joint_training_buffer
        self._replay_total_added = 0  # for fallback replay_buffer
        self._last_trained_replay_count = 0

    def _on_step(self) -> bool:
        # Only collect from info dicts when not using the joint_training_buffer
        if self.joint_training_buffer is not None:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            if info and "expert_obs" in info and "expert_action" in info:
                self.replay_buffer.append((info["expert_obs"], info["expert_action"]))
                self._replay_total_added += 1

        return True

    def _on_rollout_end(self) -> bool:
        self.num_rollouts += 1
        if self.num_rollouts % self.bc_update_freq != 0:
            return True

        # Select the buffer to train from
        if self.joint_training_buffer is not None:
            buffer = list(
                zip(
                    self.joint_training_buffer["observations"],
                    self.joint_training_buffer["actions"],
                )
            )
        else:
            buffer = self.replay_buffer

        if len(buffer) == 0:
            return True

        # Use monotonic counters so a full deque doesn't fool the check.
        if self.joint_training_buffer is not None:
            current_total = self.joint_training_buffer["total_added"]
            if current_total - self._last_trained_total_added < self.bc_min_new_samples:
                return True
        else:
            if (
                self._replay_total_added - self._last_trained_replay_count
                < self.bc_min_new_samples
            ):
                return True

        bc_fn = behavior_clone_continuous if self.is_continuous else behavior_clone

        # Pass the full ensemble in one call so bootstrapping works correctly
        # (behavior_clone only bootstraps when len(networks) > 1).
        bc_fn(
            self.model_free_networks,
            self.optimizers,
            buffer,
            batch_size=self.batch_size,
            epochs=self.bc_epochs,
            max_samples=self.bc_buffer_samples,
        )

        if self.joint_training_buffer is not None:
            self._last_trained_total_added = current_total
        else:
            self._last_trained_replay_count = self._replay_total_added

        # behavior_clone moves networks to CUDA for training; move back to CPU
        # so that MetaEnv inference (which always runs on CPU) doesn't break.
        for network in self.model_free_networks:
            network.to("cpu")

        return True
