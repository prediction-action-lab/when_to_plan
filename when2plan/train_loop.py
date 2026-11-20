from when2plan.envs.meta_env import MetaEnv
from when2plan.envs.get_sub_env import get_sub_env_class
from when2plan.models.callbacks import TrainLoggingCallback, BehaviorCloningCallback
from when2plan.envs.experiments.heuristic_policies import HeuristicPolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO, DQN
import torch
import torch.optim as optim
from when2plan.models.model_free_actor import (
    ModelFreeActorImageContinuous,
    ModelFreeActorTensor,
    ModelFreeActorImage,
    ModelFreeActorTensorContinuous,
)
import os
import csv


def create_sub_env(cfg):
    env_class = get_sub_env_class(cfg.env.type)
    sub_env = env_class(cfg, seed=cfg.seed)
    return sub_env


def train_loop(cfg, output_dir):
    dummy_env = create_sub_env(cfg)
    obs_space, action_space = dummy_env.get_model_free_spaces()

    device = "cpu"
    if cfg.env.model_free_observation_space == "image":
        if cfg.env.model_free_action_space == "continuous":
            model_free_actor_class = ModelFreeActorImageContinuous
        else:
            model_free_actor_class = ModelFreeActorImage
    else:
        if cfg.env.model_free_action_space == "continuous":
            model_free_actor_class = ModelFreeActorTensorContinuous
        else:
            model_free_actor_class = ModelFreeActorTensor

    model_free_networks = []
    model_free_optimizers = []
    num_envs = cfg.get("num_envs", 1)

    for _ in range(cfg.num_actors):
        # Check if action space is discrete or continuous and create appropriate actor
        print(cfg.env.model_free_action_space)
        if cfg.env.model_free_action_space == "continuous":
            action_dim = action_space.shape[0]
        else:
            action_dim = action_space.n

        network = model_free_actor_class(obs_space, action_dim)
        network.to(device)
        if num_envs > 1:
            network.share_memory()
        model_free_networks.append(network)
        model_free_optimizers.append(optim.Adam(network.parameters(), lr=1e-3))

    # 3. Create Vectorized Environment
    def make_env():
        sub_env = create_sub_env(cfg)
        return MetaEnv(cfg, sub_env, model_free_actors=model_free_networks)

    vec_env_cls = DummyVecEnv if num_envs > 1 else None
    meta_env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=vec_env_cls)

    callback_list = []
    if cfg.experiment.experiment == "default":
        bc_callback = BehaviorCloningCallback(
            model_free_networks=model_free_networks,
            model_free_optimizers=model_free_optimizers,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.bc_batch_size,
            bc_epochs=cfg.bc_epochs,
            bc_buffer_samples=cfg.bc_buffer_samples,
        )
        callback_list.append(bc_callback)
    elif cfg.experiment.experiment == "joint_training":
        # Create lr=1e-4 optimizers for joint training directly here.
        # MetaEnv also creates its own copy internally but they go unused;
        # these are the ones the callback will train with.
        joint_optimizers = [
            optim.Adam(network.parameters(), lr=1e-4) for network in model_free_networks
        ]
        # Unwrap DummyVecEnv -> Monitor -> MetaEnv to reach the live buffer.
        base_meta_env = meta_env.envs[0].env
        bc_callback = BehaviorCloningCallback(
            model_free_networks=model_free_networks,
            model_free_optimizers=joint_optimizers,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.bc_batch_size,
            bc_epochs=cfg.bc_epochs,
            bc_buffer_samples=cfg.bc_buffer_samples,
            joint_training_buffer=base_meta_env.joint_training_buffer,
            is_continuous=(cfg.env.model_free_action_space == "continuous"),
        )
        callback_list.append(bc_callback)
    elif cfg.experiment.experiment == "pretrain":
        # Load in pretrained model-free networks
        for i, network in enumerate(model_free_networks):
            pretrained_path = os.path.join(
                os.path.dirname(__file__),
                f"./actors/pretrained_{cfg.env.type}_model_free_actor_{i}.pth",
            )
            print(f"Loading pretrained model-free network from {pretrained_path}")
            network.load_state_dict(torch.load(pretrained_path, map_location=device))

    if cfg.experiment.logging:
        logging_callback = TrainLoggingCallback(cfg, output_dir, meta_env)
        callback_list.append(logging_callback)

    callbacklist = CallbackList(callback_list)
    device = "cpu"
    model = PPO(
        "MlpPolicy",
        meta_env,
        learning_rate=cfg.meta_learning_rate,
        gamma=cfg.meta_discount,
        ent_coef=cfg.meta_ent_coef,
        verbose=1,
        device=device,
    )
    model.learn(total_timesteps=cfg.training_timesteps, callback=callbacklist)

    # Evaluation after training
    eval_rewards = []
    eval_action_counts = []
    heuristic_results = {}

    if cfg.evaluate:
        print(f"\nRunning evaluation for {cfg.evaluation_episodes} episodes...")

        # Create a single evaluation environment (non-vectorized)
        eval_sub_env = create_sub_env(cfg)
        eval_env = MetaEnv(cfg, eval_sub_env, model_free_actors=model_free_networks)
        # eval_env = meta_env

        for episode in range(cfg.evaluation_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            action_counts = {}

            while not done:
                action, _states = model.predict(obs, deterministic=True)

                # Track action frequency (action is always 0-3)
                action_key = int(action)
                action_counts[action_key] = action_counts.get(action_key, 0) + 1

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            eval_rewards.append(episode_reward)
            eval_action_counts.append(action_counts)
            if (episode + 1) % 10 == 0:
                print(f"Completed {episode + 1}/{cfg.evaluation_episodes} episodes")

    # Evaluate heuristic policies
    if cfg.get("evaluate_heuristics", False):
        print(
            f"\nRunning heuristic policy evaluation for {cfg.evaluation_episodes} episodes..."
        )

        heuristic_types = [
            "always_model_free",
            "always_short_plan",
            "always_medium_plan",
            "always_long_plan",
            "random_policy",
        ]

        for policy_type in heuristic_types:
            print(f"Evaluating {policy_type}...")
            heuristic_policy = HeuristicPolicy(policy_type)
            policy_rewards = []
            policy_action_counts = []

            # Create a single evaluation environment for this policy
            eval_sub_env = create_sub_env(cfg)
            eval_env = MetaEnv(cfg, eval_sub_env, model_free_actors=model_free_networks)

            for episode in range(cfg.evaluation_episodes):
                obs, _ = eval_env.reset()
                done = False
                episode_reward = 0
                action_counts = {}

                while not done:
                    action = heuristic_policy.get_action()

                    # Track action frequency
                    action_counts[action] = action_counts.get(action, 0) + 1

                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                policy_rewards.append(episode_reward)
                policy_action_counts.append(action_counts)

            heuristic_results[policy_type] = {
                "rewards": policy_rewards,
                "action_counts": policy_action_counts,
            }
            print(f"Completed {policy_type} evaluation")

    # Save all evaluation results to CSV
    if cfg.evaluate or cfg.get("evaluate_heuristics", False):
        eval_csv_path = os.path.join(output_dir, "evaluation_results.csv")
        with open(eval_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Create header
            header = ["episode"]
            if cfg.evaluate:
                header.extend(
                    ["reward", "action_0", "action_1", "action_2", "action_3"]
                )
            if cfg.get("evaluate_heuristics", False):
                for policy_type in [
                    "always_model_free",
                    "always_short_plan",
                    "always_medium_plan",
                    "always_long_plan",
                    "random_policy",
                ]:
                    header.append(f"{policy_type}_reward")
            writer.writerow(header)

            # Write data rows
            num_episodes = cfg.evaluation_episodes
            for i in range(num_episodes):
                row = [i]

                # Add learned policy results
                if cfg.evaluate:
                    reward = eval_rewards[i] if i < len(eval_rewards) else ""
                    counts = (
                        eval_action_counts[i] if i < len(eval_action_counts) else {}
                    )
                    row.extend(
                        [
                            reward,
                            counts.get(0, 0) if counts else "",
                            counts.get(1, 0) if counts else "",
                            counts.get(2, 0) if counts else "",
                            counts.get(3, 0) if counts else "",
                        ]
                    )

                # Add heuristic policy results
                if cfg.get("evaluate_heuristics", False):
                    for policy_type in [
                        "always_model_free",
                        "always_short_plan",
                        "always_medium_plan",
                        "always_long_plan",
                        "random_policy",
                    ]:
                        if policy_type in heuristic_results:
                            heuristic_reward = heuristic_results[policy_type][
                                "rewards"
                            ][i]
                            row.append(heuristic_reward)
                        else:
                            row.append("")

                writer.writerow(row)

    if cfg.get("post_run_ablation", False):
        print("\nRunning post-training ablation...")
        # Use trained policy with OOD tasks (N+1 to 2N) and fresh post-ablation actors.

        # --- Build post-ablation actors ---
        post_ablation_networks = []
        for i in range(cfg.num_actors):
            if cfg.env.model_free_action_space == "continuous":
                action_dim = action_space.shape[0]
            else:
                action_dim = action_space.n
            network = model_free_actor_class(obs_space, action_dim)
            network.to("cpu")
            pretrained_path = os.path.join(
                os.path.dirname(__file__),
                f"./actors/post_ablation/pretrained_{cfg.env.type}_model_free_actor_post_ablation_{i}.pth",
            )
            print(f"Loading post-ablation actor from {pretrained_path}")
            network.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
            network.eval()
            post_ablation_networks.append(network)

        # --- Build OOD sub-env with task range num_cached_tasks -> num_cached_tasks*2 ---
        # e.g. num_cached_tasks=20 → tasks 20-39 (ID for ablation: 20-29, total: 20-39)
        num_cached_tasks = cfg.experiment.num_cached_tasks
        ablation_sub_env = create_sub_env(cfg)
        ablation_sub_env.set_cache_task_range(
            num_cached_tasks, num_cached_tasks * 2
        )

        ablation_meta_env = MetaEnv(
            cfg, ablation_sub_env, model_free_actors=post_ablation_networks
        )

        # --- Run evaluation ---
        ablation_rewards = []
        ablation_action_counts = []
        print(
            f"Running post-ablation eval for {cfg.evaluation_episodes} episodes "
            f"(tasks {num_cached_tasks} – {num_cached_tasks * 2 - 1})..."
        )

        for episode in range(cfg.evaluation_episodes):
            obs, _ = ablation_meta_env.reset()
            done = False
            episode_reward = 0
            action_counts = {}

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                action_key = int(action)
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
                obs, reward, terminated, truncated, info = ablation_meta_env.step(
                    action
                )
                done = terminated or truncated
                episode_reward += reward

            ablation_rewards.append(episode_reward)
            ablation_action_counts.append(action_counts)
            if (episode + 1) % 10 == 0:
                print(f"  Ablation: {episode + 1}/{cfg.evaluation_episodes} episodes")

        # --- Save results ---
        ablation_csv_path = os.path.join(output_dir, "post_ablation_results.csv")
        with open(ablation_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["episode", "reward", "action_0", "action_1", "action_2", "action_3"]
            )
            for i, (r, counts) in enumerate(
                zip(ablation_rewards, ablation_action_counts)
            ):
                writer.writerow(
                    [
                        i,
                        r,
                        counts.get(0, 0),
                        counts.get(1, 0),
                        counts.get(2, 0),
                        counts.get(3, 0),
                    ]
                )
        mean_ablation_reward = (
            sum(ablation_rewards) / len(ablation_rewards)
            if ablation_rewards
            else float("nan")
        )
        print(f"Post-ablation mean reward: {mean_ablation_reward:.3f}")
        print(f"Saved post-ablation results to {ablation_csv_path}")
