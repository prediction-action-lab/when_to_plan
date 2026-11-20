import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import heapq
import copy
from typing import Any, Dict
from omegaconf import OmegaConf

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from typing import List, Optional
import os
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap


class BCDataSet(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, action = self.data[idx]
        return image, action


def behavior_clone(
    networks,
    optimizers,
    dataset,
    batch_size,
    epochs,
    max_samples,
    num_workers=0,
    base_seed=42,
    use_bootstrap=True,
):
    """
    Trains an ensemble of model-free networks to mimic actions in a dataset.
    Each network gets a different random seed and optionally a bootstrapped dataset.

    Args:
        networks: A list of PyTorch models to train (or single model for backward compat).
        optimizers: A list of optimizers (or single optimizer).
        dataset: A list of (observation, action) pairs.
        batch_size: The batch size for training.
        epochs: The number of training epochs.
        max_samples: Maximum number of samples to use from dataset.
        num_workers: Number of DataLoader workers.
        base_seed: Base random seed (actor i gets base_seed + i).
        use_bootstrap: If True, each actor trains on a bootstrap sample (bagging).

    Returns:
        Dictionary mapping actor_idx -> training_metrics array, or single array if 1 actor.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Handle backward compatibility: single network/optimizer
    if not isinstance(networks, list):
        networks = [networks]
        optimizers = [optimizers]
        single_network_mode = True
    else:
        single_network_mode = False

    # Ensure models are on device and in train mode
    for network in networks:
        network.to(device)
        network.train()

    # Convert dataset to list
    dataset_list = list(dataset)

    # Skip training if dataset is empty
    if len(dataset_list) == 0:
        return {} if not single_network_mode else np.array([])

    # Sample down if too large (applied before bootstrap)
    if len(dataset_list) > max_samples:
        rng_subsample = np.random.default_rng(base_seed)
        indices = rng_subsample.choice(
            len(dataset_list), size=max_samples, replace=False
        )
        dataset_list = [dataset_list[i] for i in indices]

    loss_fn = nn.CrossEntropyLoss()
    all_metrics = {}

    # Train each actor independently
    for actor_idx, (network, optimizer) in enumerate(zip(networks, optimizers)):
        # Set unique seed for this actor
        actor_seed = base_seed + actor_idx
        torch.manual_seed(actor_seed)
        np.random.seed(actor_seed)
        random.seed(actor_seed)

        # Create bootstrap sample if requested
        if use_bootstrap and len(networks) > 1:
            rng = np.random.default_rng(actor_seed)
            bootstrap_indices = rng.integers(
                0, len(dataset_list), size=len(dataset_list)
            )
            actor_dataset = [dataset_list[i] for i in bootstrap_indices]
        else:
            actor_dataset = dataset_list

        # Create dataset and dataloader for this actor
        bc_dataset = BCDataSet(actor_dataset)
        dataloader = DataLoader(
            bc_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=False,
        )

        training_metrics = []

        # Training loop for this actor
        epoch_pbar = tqdm(range(epochs), desc=f"Training Actor {actor_idx}", leave=True)
        for epoch in epoch_pbar:
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for batch_idx, (observations, actions) in enumerate(dataloader):
                # Handle both tensor and numpy inputs
                if not isinstance(observations, torch.Tensor):
                    observations = torch.from_numpy(observations)
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions, dtype=torch.long)

                observations, actions = observations.to(device), actions.to(device)
                observations = observations.to(torch.float32)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                logits = network(observations)

                # Calculate the loss
                loss = loss_fn(logits, actions)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate accuracy
                predicted_actions = torch.max(logits, 1)[1]
                correct_predictions += (predicted_actions == actions).sum().item()
                total_samples += actions.size(0)

            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_samples
            training_metrics.append([avg_loss, accuracy, epoch])

            # Update progress bar with metrics
            epoch_pbar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}"}
            )

        epoch_pbar.close()
        all_metrics[actor_idx] = np.array(training_metrics)

    # Backward compatibility: return single array if only 1 network
    if single_network_mode:
        return all_metrics[0]

    return all_metrics


def behavior_clone_continuous(
    networks,
    optimizers,
    dataset,
    batch_size,
    epochs,
    max_samples,
    num_workers=0,
    base_seed=42,
    use_bootstrap=True,
):
    """
    Trains an ensemble of model-free networks for continuous action spaces.
    Uses MSE loss instead of cross-entropy since actions are continuous.

    Args:
        networks: A list of PyTorch models to train (or single model for backward compat).
        optimizers: A list of optimizers (or single optimizer).
        dataset: A list of (observation, action) pairs where action is continuous.
        batch_size: The batch size for training.
        epochs: The number of training epochs.
        max_samples: Maximum number of samples to use from dataset.
        num_workers: Number of DataLoader workers.
        base_seed: Base random seed (actor i gets base_seed + i).
        use_bootstrap: If True, each actor trains on a bootstrap sample (bagging).

    Returns:
        Dictionary mapping actor_idx -> training_metrics array, or single array if 1 actor.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Handle backward compatibility: single network/optimizer
    if not isinstance(networks, list):
        networks = [networks]
        optimizers = [optimizers]
        single_network_mode = True
    else:
        single_network_mode = False

    # Ensure models are on device and in train mode
    for network in networks:
        network.to(device)
        network.train()

    # Convert dataset to list
    dataset_list = list(dataset)

    # Skip training if dataset is empty
    if len(dataset_list) == 0:
        return {} if not single_network_mode else np.array([])

    # Sample down if too large (applied before bootstrap)
    if len(dataset_list) > max_samples:
        rng_subsample = np.random.default_rng(base_seed)
        indices = rng_subsample.choice(
            len(dataset_list), size=max_samples, replace=False
        )
        dataset_list = [dataset_list[i] for i in indices]

    loss_fn = nn.MSELoss()
    all_metrics = {}

    # Train each actor independently
    for actor_idx, (network, optimizer) in enumerate(zip(networks, optimizers)):
        # Set unique seed for this actor
        actor_seed = base_seed + actor_idx
        torch.manual_seed(actor_seed)
        np.random.seed(actor_seed)
        random.seed(actor_seed)

        # Create bootstrap sample if requested
        if use_bootstrap and len(networks) > 1:
            rng = np.random.default_rng(actor_seed)
            bootstrap_indices = rng.integers(
                0, len(dataset_list), size=len(dataset_list)
            )
            actor_dataset = [dataset_list[i] for i in bootstrap_indices]
        else:
            actor_dataset = dataset_list

        # Create dataset and dataloader for this actor
        bc_dataset = BCDataSet(actor_dataset)
        dataloader = DataLoader(
            bc_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=False,
        )

        training_metrics = []

        # Cosine annealing: starts at the optimizer's initial LR and decays to
        # near-zero over all epochs.  This is critical for memorisation — a
        # constant LR keeps oscillating around per-task minima, while annealing
        # lets gradients settle tightly at the end of training.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # Training loop for this actor
        epoch_pbar = tqdm(range(epochs), desc=f"Training Actor {actor_idx}", leave=True)
        for epoch in epoch_pbar:
            total_loss = 0
            total_samples = 0

            for batch_idx, (observations, actions) in enumerate(dataloader):
                # Handle both tensor and numpy inputs
                if not isinstance(observations, torch.Tensor):
                    observations = torch.from_numpy(observations)
                if not isinstance(actions, torch.Tensor):
                    actions = torch.from_numpy(actions)

                observations = observations.to(torch.float32)
                actions = actions.to(torch.float32)
                observations, actions = observations.to(device), actions.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass - continuous models now output action directly
                predicted_actions = network(observations)

                # Calculate MSE loss on predicted actions
                loss = loss_fn(predicted_actions, actions)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_samples += actions.size(0)

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            # For continuous actions, we track loss instead of accuracy
            training_metrics.append(
                [avg_loss, 0.0, epoch]
            )  # 0.0 placeholder for accuracy

            # Update progress bar with metrics
            epoch_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        epoch_pbar.close()
        all_metrics[actor_idx] = np.array(training_metrics)

    # Backward compatibility: return single array if only 1 network
    if single_network_mode:
        return all_metrics[0]

    return all_metrics


def collect_pretrain_data_with_planner(environment, n_steps):
    # pretrain_env = copy.deepcopy(environment)
    pretrain_env = environment
    pretrain_data = []

    steps_collected = 0
    pbar = tqdm(total=n_steps, desc="Collecting pretrain data")

    while steps_collected < n_steps:
        _, info = pretrain_env.reset()
        done = False
        while not done and steps_collected < n_steps:
            plan = pretrain_env.get_plan(pretrain=True, plan_length="medium")
            if plan is None:
                break

            # All envs return a plan where every element is a real action /
            # target for the current state:
            #   - NavigationEnv: list of [dx, dy] delta actions
            #   - FetchPickPlaceEnv: list of 7-DOF normalised joint targets
            # Iterate over the full plan (not plan[1:]) — each env's
            # get_plan(pretrain=True) already strips any current-state no-op.
            for action in plan:
                img = pretrain_env.get_tensor_state()
                pretrain_data.append((img, action))
                steps_collected += 1
                pbar.update(1)
                _, _, term, trunc, _ = pretrain_env.model_free_step(action)
                done = term or trunc

                if term or trunc or steps_collected >= n_steps:
                    break

    pbar.close()
    return pretrain_data


def collect_render_data_with_planner(environment, n_steps):
    # Use environment directly (no deepcopy) to preserve task range settings
    pretrain_env = environment
    render_data = []

    steps_collected = 0
    while steps_collected < n_steps:
        _, info = pretrain_env.reset()
        done = False
        while not done and steps_collected < n_steps:
            # path = pretrain_env.get_plan(max_expansions=1000)
            path = pretrain_env.get_plan(pretrain=True, plan_length="medium")
            print(len(path))
            if path is None:
                break  # no plan found, reset environment
            render_img = pretrain_env.render(mode="rgb_array")
            render_data.append(render_img)
            action = path[0]  # take first action in planned path
            # pretrain_data.append((img, action))
            steps_collected += 1
            _, _, term, trunc, _ = pretrain_env.step(action)
            done = term or trunc

    return render_data


"""
Note environments here are actually meta environments
"""


def get_model_free_metrics(
    in_distribution_environment,
    out_of_distribution_environment,
    n_steps,
    model_free_action,
):
    in_dist_episode_rewards = []
    in_dist_step_uncertaintys = []
    out_of_dist_episode_rewards = []
    out_of_dist_step_uncertaintys = []

    # Collect metrics for in-distribution environment
    _, _ = in_distribution_environment.reset()
    episode_reward = 0
    while len(in_dist_step_uncertaintys) < n_steps:
        _, reward, term, trunc, info = in_distribution_environment.step(
            model_free_action
        )
        episode_reward += reward
        in_dist_step_uncertaintys.append(info.get("model_free_actor_uncertainty", 0))
        if term or trunc:
            in_dist_episode_rewards.append(episode_reward)
            episode_reward = 0
            _, _ = in_distribution_environment.reset()

    # Collect metrics for out-of-distribution environment
    _, _ = out_of_distribution_environment.reset()
    episode_reward = 0
    while len(out_of_dist_step_uncertaintys) < n_steps:
        _, reward, term, trunc, info = out_of_distribution_environment.step(
            model_free_action
        )
        episode_reward += reward
        out_of_dist_step_uncertaintys.append(
            info.get("model_free_actor_uncertainty", 0)
        )
        if term or trunc:
            out_of_dist_episode_rewards.append(episode_reward)
            episode_reward = 0
            _, _ = out_of_distribution_environment.reset()

    return {
        "in_dist_step_uncertaintys": in_dist_step_uncertaintys,
        "out_dist_step_uncertaintys": out_of_dist_step_uncertaintys,
        "in_dist_episode_rewards": in_dist_episode_rewards,
        "out_dist_episode_rewards": out_of_dist_episode_rewards,
    }


"""
    Plot the mean of both rewards and uncertaintys
    Use 95% bootstrap CIs
    Return the plot, optimally save it as a figure
"""


def plot_model_free_metrics(metrics, save_path=None):
    """
    Plots the model-free metrics based on the output of get_model_free_metrics.

    Args:
        metrics: A dictionary containing the following keys:
            - in_dist_step_uncertaintys
            - out_dist_step_uncertaintys
            - in_dist_episode_rewards
            - out_dist_episode_rewards
    """
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Define color palette
    colors = sns.color_palette("muted")

    # Step-level uncertaintys
    in_step_conf_mean = np.mean(metrics["in_dist_step_uncertaintys"])
    out_step_conf_mean = np.mean(metrics["out_dist_step_uncertaintys"])

    in_step_conf_ci = bootstrap(
        (np.array(metrics["in_dist_step_uncertaintys"]),),
        np.mean,
        uncertainty_level=0.99,
    ).uncertainty_interval
    out_step_conf_ci = bootstrap(
        (np.array(metrics["out_dist_step_uncertaintys"]),),
        np.mean,
        uncertainty_level=0.99,
    ).uncertainty_interval

    axs[0].bar(
        ["In-Distribution", "Out-of-Distribution"],
        [in_step_conf_mean, out_step_conf_mean],
        yerr=[
            (
                in_step_conf_mean - in_step_conf_ci.low,
                out_step_conf_mean - out_step_conf_ci.low,
            ),
            (
                in_step_conf_ci.high - in_step_conf_mean,
                out_step_conf_ci.high - out_step_conf_mean,
            ),
        ],
        capsize=5,
        color=colors[:2],
    )
    axs[0].set_title("Step-Level uncertaintys", fontsize=16)
    axs[0].set_ylabel("Mean uncertainty", fontsize=14)
    axs[0].tick_params(axis="both", labelsize=12)
    axs[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Episode-level rewards
    in_episode_reward_mean = np.mean(metrics["in_dist_episode_rewards"])
    out_episode_reward_mean = np.mean(metrics["out_dist_episode_rewards"])

    in_episode_reward_ci = bootstrap(
        (np.array(metrics["in_dist_episode_rewards"]),), np.mean, uncertainty_level=0.99
    ).uncertainty_interval
    out_episode_reward_ci = bootstrap(
        (np.array(metrics["out_dist_episode_rewards"]),),
        np.mean,
        uncertainty_level=0.99,
    ).uncertainty_interval

    axs[1].bar(
        ["In-Distribution", "Out-of-Distribution"],
        [in_episode_reward_mean, out_episode_reward_mean],
        yerr=[
            (
                in_episode_reward_mean - in_episode_reward_ci.low,
                out_episode_reward_mean - out_episode_reward_ci.low,
            ),
            (
                in_episode_reward_ci.high - in_episode_reward_mean,
                out_episode_reward_ci.high - out_episode_reward_mean,
            ),
        ],
        capsize=5,
        color=colors[2:4],
    )
    axs[1].set_title("Episode-Level Rewards", fontsize=16)
    axs[1].set_ylabel("Mean Reward", fontsize=14)
    axs[1].tick_params(axis="both", labelsize=12)
    axs[1].grid(axis="y", linestyle="--", alpha=0.7)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.tight_layout()
    plt.show()
