import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

from when2plan.envs.get_sub_env import get_sub_env_class
import torch.optim as optim
from when2plan.models.model_free_actor import (
    ModelFreeActorTensor,
    ModelFreeActorImage,
    ModelFreeActorTensorContinuous,
    ModelFreeActorImageContinuous,
)
from when2plan.utils import (
    collect_pretrain_data_with_planner,
    behavior_clone,
    behavior_clone_continuous,
)
from gymnasium.spaces import Dict
import torch
import argparse
import os
from omegaconf import OmegaConf
import numpy as np
from typing import List, Optional
import copy
import matplotlib.pyplot as plt


def load_hydra_config(
    config_dir: str, config_name: str = "config", overrides: Optional[List[str]] = None
):
    """
    Notebook-friendly Hydra loader.
    - config_dir: relative path (preferred) or absolute path to config dir
    - config_name: top-level config filename (without .yaml)
    - overrides: list of override strings, e.g. ["env=medium_maze","experiment=maze_default"]
    Returns an omegaconf.DictConfig (resolved).
    """
    overrides = overrides or []
    try:
        # hydra 1.x
        from hydra import initialize, compose, initialize_config_dir
    except Exception:
        # older hydra
        from hydra.experimental import initialize, compose, initialize_config_dir

    # When a user passed an absolute path use initialize_config_dir (accepts abs paths),
    # otherwise use initialize with a relative config_path.
    if os.path.isabs(config_dir):
        # Added version_base="1.1" here
        with initialize_config_dir(
            config_dir=config_dir, job_name="notebook", version_base="1.1"
        ):
            cfg = compose(config_name=config_name, overrides=overrides)
    else:
        # Added version_base="1.1" here
        with initialize(
            config_path=config_dir, job_name="notebook", version_base="1.1"
        ):
            cfg = compose(config_name=config_name, overrides=overrides)

    # return a resolved DictConfig
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def get_env(env_name, num_cached_tasks=None):
    env_class = get_sub_env_class(env_name)
    configs_path = "../../experiments/configs"
    env_type = "env=" + env_name
    experiment_type = "experiment=" + env_name + "_default"
    cfg = load_hydra_config(
        config_dir=configs_path,
        overrides=[env_type, experiment_type],
    )
    if num_cached_tasks is not None:
        cfg.env.num_cached_tasks = num_cached_tasks
        print("Overriding num_cached_tasks to:", num_cached_tasks)

    env = env_class(cfg, seed=42)
    return env


def pretrain(
    args, env, num_actors, num_steps, batch_size, epochs, plot, base_seed=42, save=True
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    actors = []
    optimizers = []

    # Detect if action space is discrete or continuous
    obs_space, action_space = env.get_model_free_spaces()

    if args.model_free_action_space == "continuous":
        is_continuous = True
    else:
        is_continuous = False

    print(args)

    print(
        f"Detected action space type: {'Continuous' if is_continuous else 'Discrete'}"
    )

    for i in range(num_actors):
        # Set seed before creating network for unique initialization
        torch.manual_seed(base_seed + i)
        np.random.seed(base_seed + i)

        # Create appropriate actor based on input type and action space
        if is_continuous:
            action_dim = action_space.shape[0]
            if args.model_free_observation_space == "tensor":
                actor = ModelFreeActorTensorContinuous(obs_space, action_dim)
            elif args.model_free_observation_space == "image":
                actor = ModelFreeActorImageContinuous(obs_space, action_dim)
            else:
                raise ValueError(
                    f"Unknown model_free_observation_space type for continuous actions: {args.model_free_observation_space}"
                )
        else:
            action_dim = action_space.n
            if args.model_free_observation_space == "tensor":
                actor = ModelFreeActorTensor(obs_space, action_dim)
            elif args.model_free_observation_space == "image":
                actor = ModelFreeActorImage(obs_space, action_dim)
            else:
                raise ValueError(
                    f"Unknown model_free_observation_space type for discrete actions: {args.model_free_observation_space}"
                )

        actor.to(device)
        actors.append(actor)
        optimizers.append(optim.Adam(actor.parameters(), lr=1e-3))

    pretrain_data = collect_pretrain_data_with_planner(env, num_steps)

    print("Collected pretrain data of length:", len(pretrain_data))

    print(f"Training {num_actors} actors with ensemble diversity...")

    # Use appropriate behavior cloning function based on action space type
    is_continuous = args.model_free_action_space == "continuous"

    if is_continuous:
        metrics = behavior_clone_continuous(
            actors,
            optimizers,
            pretrain_data,
            batch_size=batch_size,
            epochs=epochs,
            max_samples=1000000,
            base_seed=base_seed,
            use_bootstrap=True,
        )
    else:
        metrics = behavior_clone(
            actors,
            optimizers,
            pretrain_data,
            batch_size=batch_size,
            epochs=epochs,
            max_samples=1000000,
            base_seed=base_seed,
            use_bootstrap=True,
        )

    if save:
        save_dir = os.path.join(os.path.dirname(__file__), "../actors")
        os.makedirs(save_dir, exist_ok=True)

        # Save each actor separately
        for i, actor in enumerate(actors):
            save_path = os.path.join(
                save_dir, f"pretrained_{args.sub_env}_model_free_actor_{i}.pth"
            )
            torch.save(actor.state_dict(), save_path)
            print(f"Saved actor {i} to: {save_path}")

    # Show training plots with all actors
    if plot and metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss for each actor
        for actor_idx, actor_metrics in metrics.items():
            ax1.plot(
                actor_metrics[:, 2],
                actor_metrics[:, 0],
                label=f"Actor {actor_idx}",
                alpha=0.8,
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy for each actor
        for actor_idx, actor_metrics in metrics.items():
            ax2.plot(
                actor_metrics[:, 2],
                actor_metrics[:, 1],
                label=f"Actor {actor_idx}",
                alpha=0.8,
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return actors


def pretrain_model_free(args):
    if args.sub_env is None:
        raise ValueError("Please specify a sub-environment using --sub_env")

    print("Training pretrain model-free on sub-environment:", args.sub_env)
    env = get_env(args.sub_env, num_cached_tasks=args.num_cached_tasks)
    pretrained_actors = pretrain(
        args,
        env,
        num_actors=args.num_actors,
        num_steps=args.steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        plot=args.plot,
    )
    return pretrained_actors, env


def pretrain_model_free_sub_env(args, env):
    print("Training pretrain model-free on sub-environment:")
    pretrained_actors = pretrain(
        args,
        env,
        num_actors=args.num_actors,
        num_steps=args.steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        plot=args.plot,
        save=args.save_actor,
    )
    return pretrained_actors, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub_env", default=None, type=str, help="Sub-environment to train"
    )
    parser.add_argument(
        "--num_cached_tasks",
        default=None,
        type=int,
        help="Number of cached tasks for the sub-environment",
    )
    parser.add_argument(
        "--steps",
        default=10000,
        type=int,
        help="Number of pretrain data collection steps",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Whether to plot training metrics"
    )
    parser.add_argument(
        "--num_actors", default=4, type=int, help="Number of ensemble actors to train"
    )
    parser.add_argument(
        "--model_free_input",
        default="tensor",
        type=str,
        choices=[
            "tensor",
            "image",
            "discrete_tensor",
            "continuous_tensor",
            "discrete_image",
            "continuous_image",
        ],
        help="Type of model-free actor input. Use 'tensor' or 'image' for auto-detection based on action space, or specify explicitly with discrete_/continuous_ prefix.",
    )
    parser.add_argument(
        "--save_actor",
        action="store_true",
        default=True,
        help="Whether to save trained actors",
    )
    args = parser.parse_args()
    pretrain_model_free(args)
