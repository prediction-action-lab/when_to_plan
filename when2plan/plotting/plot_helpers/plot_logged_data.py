import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from collections import defaultdict
from tqdm import tqdm


def create_unique_labels(overrides_list):
    # Find keys that differ across overrides
    all_keys = set().union(*(o.keys() for o in overrides_list))
    differing_keys = set()
    for key in all_keys:
        values = set(o.get(key, None) for o in overrides_list)
        if len(values) > 1:
            # Skip env-related keys and base experiment config
            # but keep hyperparameters like 'experiment.rewards.plan'
            if key not in ["env", "experiment"]:
                differing_keys.add(key)
    # Build labels using only differing keys
    labels = []
    for overrides in overrides_list:
        label_parts = [
            f"{key}={overrides[key]}" for key in differing_keys if key in overrides
        ]
        label = ", ".join(label_parts) if label_parts else "default"
        labels.append(label)
    return labels


def load_folder(path):
    # Load in training_log.csv and overrides.yaml from a folder
    if not os.path.isdir(path):
        print(f"No directory {path}")
        return None

    log_file = os.path.join(path, "training_log.csv")
    if not os.path.isfile(log_file):
        print(f"No log file found at {log_file}")
        return None
    df = pd.read_csv(log_file)
    overrides = {}
    overrides_file = os.path.join(path, ".hydra", "overrides.yaml")
    if os.path.isfile(overrides_file):
        with open(overrides_file, "r") as fh:
            overrides = yaml.safe_load(fh)

        if isinstance(overrides, list):  # If it's a list of strings, convert to dict
            overrides_dict = {}
            for item in overrides:
                if "=" in item:
                    k, v = item.split("=", 1)
                    overrides_dict[k] = v
            overrides = overrides_dict

    # Add after loading overrides:
    config = {}
    config_file = os.path.join(path, ".hydra", "config.yaml")
    if os.path.isfile(config_file):
        with open(config_file, "r") as fh:
            config = yaml.safe_load(fh)

    return df, overrides, config  # Return 3 items instead of 2


def compute_cached_task_ratio(overrides, config):
    """
    Compute the ratio of num_cached_tasks to num_pretrained_tasks.

    Returns the ratio as a float, or None if cannot be computed.
    """
    # Get num_cached_tasks from overrides or config
    num_cached = None
    if "experiment.num_cached_tasks" in overrides:
        try:
            num_cached = float(overrides["experiment.num_cached_tasks"])
        except (ValueError, TypeError):
            pass
    elif (
        config and "experiment" in config and "num_cached_tasks" in config["experiment"]
    ):
        try:
            num_cached = float(config["experiment"]["num_cached_tasks"])
        except (ValueError, TypeError):
            pass

    # Get num_pretrained_tasks from config
    num_pretrained = None
    if (
        config
        and "experiment" in config
        and "num_pretrained_tasks" in config["experiment"]
    ):
        try:
            num_pretrained = float(config["experiment"]["num_pretrained_tasks"])
        except (ValueError, TypeError):
            pass

    # Compute ratio
    if num_cached is not None and num_pretrained is not None and num_pretrained > 0:
        return num_cached / num_pretrained

    return None


def group_runs_by_config(data):
    # data: list of (df, overrides, config)  # Update comment
    groups = defaultdict(list)
    for df, overrides, config in data:  # Unpack 3 values
        # Remove 'seed', 'env', and the base 'experiment' config name
        # but keep hyperparameters like 'experiment.rewards.plan'
        # ALSO remove absolute num_cached_tasks since we'll use ratio instead
        overrides_wo_seed = {
            k: v
            for k, v in overrides.items()
            if k not in ["seed", "env", "experiment", "experiment.num_cached_tasks"]
        }

        # Compute and add the ratio instead
        ratio = compute_cached_task_ratio(overrides, config)
        if ratio is not None:
            overrides_wo_seed["experiment.num_cached_tasks_ratio"] = f"{ratio:.1f}"

        key = tuple(sorted(overrides_wo_seed.items()))
        groups[key].append(df)
    return groups


def print_final_data_info(data):
    """
    Print summary info about the final data structure:
    data: list of [label, list_of_dfs, overrides_dict]
    """
    print(f"Number of unique labels/configs: {len(data)}")
    for i, (label, dfs, overrides) in enumerate(data):
        print(f"  Config {i+1}:")
        print(f"    Label: {label}")
        print(f"    Number of runs: {len(dfs)}")
        print(f"    Overrides: {overrides}")
    # Optionally, count unique overrides
    unique_overrides = {frozenset(overrides.items()) for _, _, overrides in data}
    print(f"Number of unique overrides: {len(unique_overrides)}")


def detect_step_type(df, overrides=None, config=None):
    """
    Detect the meta step type from config overrides or by analyzing meta_action patterns.
    Returns: 'bandit', 'execute', 'seperate', or 'commit'

    First checks overrides for 'meta_step_type' field, then falls back to pattern detection.
    """
    # Check full config first (most reliable)
    if config is not None:
        if "meta_step_type" in config:
            return config["meta_step_type"]

    # Then check overrides...
    if overrides is not None:
        if "meta_step_type" in overrides:
            return overrides["meta_step_type"]
        # Also check with experiment prefix
        if "experiment.meta_step_type" in overrides:
            return overrides["experiment.meta_step_type"]

    # Fall back to pattern detection from logged data
    # Check for commit-specific columns first (most reliable for commit mode)
    commit_columns = [
        "in_distribution_short_plan",
        "in_distribution_medium_plan",
        "in_distribution_long_plan",
    ]
    if any(col in df.columns for col in commit_columns):
        return "commit"

    # Check for bandit-specific columns
    bandit_columns = ["in_distribution_plan", "out_distribution_plan"]
    if any(col in df.columns for col in bandit_columns):
        return "bandit"

    # Parse the string representation of lists
    def parse_action(s):
        s = s.strip("[]")
        values = np.array([int(x) for x in s.split(",")], dtype=float)
        return values

    # Sample a subset of meta_actions to detect pattern
    sample_size = min(1000, len(df))
    sample_actions = (
        df["meta_action"].sample(n=sample_size, random_state=42).apply(parse_action)
    )

    # Stack into array for analysis
    action_matrix = np.stack(sample_actions.values)

    # Check which positions have non-zero values
    has_plan = (action_matrix[:, 0] > 0).any()  # Position 0: plan
    has_follow_plan = (
        action_matrix[:, 1] > 0
    ).any()  # Position 1: follow_plan or model_free (bandit)
    has_model_free = (action_matrix[:, 2] > 0).any()  # Position 2: model_free
    has_clear_plan = (action_matrix[:, 3] > 0).any()  # Position 3: clear_plan

    # Execute: uses all 4 positions (plan, follow_plan, model_free, clear_plan)
    if has_clear_plan:
        return "execute"

    # Seperate: uses first 3 positions (plan, follow_plan, model_free)
    if has_plan and has_follow_plan and has_model_free:
        return "seperate"

    # Default to bandit if unclear
    return "bandit"


def parse_meta_actions_vectorized(df, step_type="execute"):
    """
    Vectorized parsing of meta_action column into percentage columns.
    Much faster than row-by-row iteration.

    Args:
        df: DataFrame with meta_action column
        step_type: One of 'bandit', 'execute', 'seperate', or 'commit'
    """

    # Parse the string representation of lists
    def parse_action(s):
        s = s.strip("[]")
        values = np.array([int(x) for x in s.split(",")], dtype=float)
        total = values.sum()
        if total > 0:
            return values / total * 100.0
        # Return appropriate size based on step_type
        if step_type == "commit":
            return np.zeros(4)
        else:
            return np.zeros(4)  # Keep 4 for backward compatibility

    # Apply vectorized operation and convert directly to array
    parsed = np.array(df["meta_action"].apply(parse_action).tolist())

    if step_type == "commit":
        # Commit mode: model_free, short_plan, medium_plan, long_plan
        df["meta_action_model_free_percent"] = parsed[:, 0]
        df["meta_action_short_plan_percent"] = parsed[:, 1]
        df["meta_action_medium_plan_percent"] = parsed[:, 2]
        df["meta_action_long_plan_percent"] = parsed[:, 3]
    else:
        # Other modes (execute, bandit, seperate)
        df["meta_action_plan_percent"] = parsed[:, 0]
        df["meta_action_follow_plan_percent"] = parsed[:, 1]
        df["meta_action_model_free_percent"] = parsed[:, 2]
        df["meta_action_clear_plan_percent"] = parsed[:, 3]

    return df


def compute_distribution_percentages(df, step_type="execute"):
    """
    Compute percentage of each action within in-distribution and out-of-distribution tasks.

    Args:
        df: DataFrame with distribution count columns
        step_type: One of 'bandit', 'execute', 'seperate', or 'commit'
    """
    if step_type == "commit":
        actions = ["model_free", "short_plan", "medium_plan", "long_plan"]
    elif step_type == "bandit":
        actions = ["plan", "model_free"]
    elif step_type == "seperate":
        actions = ["plan", "follow_plan", "model_free"]
    elif step_type == "execute":
        actions = ["plan", "follow_plan", "model_free", "clear_plan"]
    else:
        return df

    # Check if distribution columns exist
    in_cols = [f"in_distribution_{action}" for action in actions]
    out_cols = [f"out_distribution_{action}" for action in actions]

    if not all(col in df.columns for col in in_cols + out_cols):
        return df

    # Compute totals for in-distribution and out-of-distribution
    df["total_in_distribution"] = sum(df[col] for col in in_cols)
    df["total_out_distribution"] = sum(df[col] for col in out_cols)

    # Compute percentages for each action
    for action in actions:
        in_col = f"in_distribution_{action}"
        out_col = f"out_distribution_{action}"

        # In-distribution percentage
        df[f"in_distribution_{action}_percent"] = np.where(
            df["total_in_distribution"] > 0,
            df[in_col] / df["total_in_distribution"] * 100,
            0,
        )

        # Out-of-distribution percentage
        df[f"out_distribution_{action}_percent"] = np.where(
            df["total_out_distribution"] > 0,
            df[out_col] / df["total_out_distribution"] * 100,
            0,
        )

    return df


def compute_ci(values, uncertainty=0.95, n_bootstrap=1000, random_state=None):
    """Compute bootstrapped uncertainty interval for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        val = values[0] if len(values) == 1 else 0.0
        return val, val, val

    rng = np.random.default_rng(random_state)
    # Vectorized bootstrap
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)

    alpha = (1 - uncertainty) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    mean = values.mean()
    return mean, lower, upper


def sort_data_by_differing_keys(data):
    """
    Sorts the data list by all differing keys, in alphabetical order.
    Values are sorted numerically if possible, else lexicographically.
    """
    # Find all differing keys
    overrides_list = [overrides for _, _, overrides in data]
    all_keys = set().union(*(o.keys() for o in overrides_list))
    differing_keys = []
    for key in sorted(all_keys):
        values = set(o.get(key, None) for o in overrides_list)
        if len(values) > 1:
            # Skip env-related keys and base experiment config
            if key not in ["env", "experiment"]:
                differing_keys.append(key)

    def sort_key(item):
        _, _, overrides = item
        result = []
        for k in differing_keys:
            v = overrides.get(k, "")
            # Try numeric sort
            try:
                v = float(v)
            except (ValueError, TypeError):
                pass
            result.append(v)
        return tuple(result)

    return sorted(data, key=sort_key)


def plot_metrics_by_config(
    data, metrics, smooth_alpha=0.3, metric_label_overrides=None
):
    """
    data: list of [label, list_of_dfs, overrides_dict]
    metrics: list of column names to plot
    smooth_alpha: smoothing factor for exponential moving average (0=no smoothing, 1=raw data)
    Creates a grid where rows=configs, columns=metrics
    Each environment is plotted as a separate line within each config
    All plots in the same column (same metric) share the same y-axis scale
    """
    n_configs = len(data)
    n_metrics = len(metrics)

    # Create figure with more space for labels
    fig, axes = plt.subplots(
        n_configs,
        n_metrics,
        figsize=(4 * n_metrics, 4 * n_configs),
        constrained_layout=True,
    )

    # Handle single row/column cases
    if n_configs == 1 and n_metrics == 1:
        axes = np.array([[axes]])
    elif n_configs == 1:
        axes = axes.reshape(1, -1)
    elif n_metrics == 1:
        axes = axes.reshape(-1, 1)

    def smooth_data(values, alpha=0.3):
        """Apply exponential moving average smoothing."""
        if len(values) == 0:
            return values
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    # First pass: collect all data to determine y-axis limits for each metric
    metric_limits = {}
    for metric_idx, metric in enumerate(metrics):
        all_values = []
        for config_idx, (label, dfs, overrides) in enumerate(data):
            combined_df = pd.concat(dfs, ignore_index=True)
            if metric in combined_df.columns:
                # Get all mean values and CI bounds for this metric
                if "env_name" in combined_df.columns:
                    env_names = sorted(combined_df["env_name"].unique())
                else:
                    env_names = ["default"]

                for env_name in env_names:
                    if "env_name" in combined_df.columns:
                        env_df = combined_df[combined_df["env_name"] == env_name]
                    else:
                        env_df = combined_df

                    grouped = env_df.groupby("rollout_number")
                    for rollout_num, group in grouped:
                        values = group[metric].values
                        mean, lower, upper = compute_ci(values)
                        all_values.extend([lower, upper, mean])

        if all_values:
            # Add 5% padding to the range for better visualization
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            range_val = max_val - min_val
            padding = range_val * 0.05 if range_val > 0 else 0.05

            # Special handling for percent metrics
            if "percent" in metric:
                metric_limits[metric] = (0, 100)
            if "reward" in metric:
                metric_limits[metric] = (-200, 100)
            else:
                metric_limits[metric] = (min_val - padding, max_val + padding)

    # Second pass: plot the data with shared y-axis limits
    for config_idx, (label, dfs, overrides) in enumerate(
        tqdm(data, desc="Plotting configs", position=0)
    ):
        combined_df = pd.concat(dfs, ignore_index=True)

        # Get unique environments in this config
        if "env_name" in combined_df.columns:
            env_names = sorted(combined_df["env_name"].unique())
        else:
            env_names = ["default"]

        # Use a color palette for environments
        env_colors = sns.color_palette("husl", len(env_names))
        env_color_map = {env: color for env, color in zip(env_names, env_colors)}

        for metric_idx, metric in enumerate(
            tqdm(metrics, desc=f"  {label}", position=1, leave=False)
        ):
            ax = axes[config_idx, metric_idx]

            # Plot each environment as a separate line
            for env_name in env_names:
                if "env_name" in combined_df.columns:
                    env_df = combined_df[combined_df["env_name"] == env_name]
                else:
                    env_df = combined_df

                # Group by rollout_number and compute CIs for each metric
                grouped = env_df.groupby("rollout_number")

                # Compute mean and CIs for each rollout_number
                rollout_numbers = []
                means = []
                lower_bounds = []
                upper_bounds = []

                for rollout_num, group in grouped:
                    rollout_numbers.append(rollout_num)
                    values = group[metric].values
                    mean, lower, upper = compute_ci(values)
                    means.append(mean)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)

                rollout_numbers = np.array(rollout_numbers)
                means = np.array(means)
                lower_bounds = np.array(lower_bounds)
                upper_bounds = np.array(upper_bounds)

                # Apply smoothing
                means_smoothed = smooth_data(means, smooth_alpha)
                lower_smoothed = smooth_data(lower_bounds, smooth_alpha)
                upper_smoothed = smooth_data(upper_bounds, smooth_alpha)

                # Plot with uncertainty interval shading (no markers)
                color = env_color_map[env_name]
                ax.plot(
                    rollout_numbers,
                    means_smoothed,
                    linewidth=2.5,
                    label=env_name,
                    color=color,
                    alpha=0.9,
                )
                ax.fill_between(
                    rollout_numbers,
                    lower_smoothed,
                    upper_smoothed,
                    alpha=0.15,
                    color=color,
                )

            # Styling
            if metric_label_overrides and metric in metric_label_overrides:
                metric_label = metric_label_overrides[metric]
            else:
                metric_label = metric.replace("_", " ").title()
            ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Rollout Number", fontsize=11, fontweight="bold")

            # Apply shared y-axis limits for this metric
            if metric in metric_limits:
                ax.set_ylim(metric_limits[metric])

            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add title only to top row
            if config_idx == 0:
                if metric_label_overrides and metric in metric_label_overrides:
                    metric_title = metric_label_overrides[metric]
                else:
                    metric_title = metric.replace("_", " ").title()
                # Use smaller font and add wrapping for long titles
                ax.set_title(
                    metric_title, fontsize=9, fontweight="bold", pad=12, wrap=True
                )

            # Add config label on the left side (outside the plot area)
            if metric_idx == 0:
                # Create a text box with the label
                label_text = label
                ax.annotate(
                    label_text,
                    xy=(0, 0.5),
                    xytext=(-0.25, 0.5),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    fontsize=9,
                    ha="right",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3
                    ),
                )

            # Add legend to every plot for clarity (especially important with multiple envs)
            ax.legend(loc="best", fontsize=8, framealpha=0.9, title="Environment")

    return fig, axes


def plot_commit_distribution(data, smooth_alpha=0.3):
    """
    Create stacked area plots for commit mode showing action distribution over time.
    Creates separate plots for each environment's in-distribution and out-of-distribution tasks.
    Each line represents one of the 4 actions (model_free, short_plan, medium_plan, long_plan).

    data: list of [label, list_of_dfs, overrides_dict]
    """
    n_configs = len(data)

    # First, determine the number of unique environments across all configs
    all_env_names = set()
    for label, dfs, overrides in data:
        combined_df = pd.concat(dfs, ignore_index=True)
        if "env_name" in combined_df.columns:
            all_env_names.update(combined_df["env_name"].unique())
        else:
            all_env_names.add("default")

    env_names_sorted = sorted(all_env_names)
    n_envs = len(env_names_sorted)
    n_cols = n_envs * 2  # 2 columns per environment (in-dist and out-of-dist)

    # Create figure with columns for each environment's in/out distribution
    # Increase width to avoid constrained_layout collapsing
    fig, axes = plt.subplots(
        n_configs,
        n_cols,
        figsize=(max(6 * n_envs, 12), 4 * n_configs),
        constrained_layout=True,
    )

    # Handle single config case
    if n_configs == 1:
        axes = axes.reshape(1, -1)

    # Handle single environment case
    if n_cols == 2:
        if n_configs == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(n_configs, -1)

    # Action names and colors
    action_names = ["Model Free", "Short Plan", "Medium Plan", "Long Plan"]
    action_colors = sns.color_palette("husl", 4)

    def smooth_data(values, alpha=0.3):
        """Apply exponential moving average smoothing."""
        if len(values) == 0:
            return values
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def compute_action_percentages(env_df, distribution_type):
        """
        Compute percentage of each action over time for a given distribution type.
        distribution_type: 'in' or 'out'
        """
        # Columns to check for distribution
        action_dist_cols = {
            "model_free": f"{distribution_type}_distribution_model_free",
            "short_plan": f"{distribution_type}_distribution_short_plan",
            "medium_plan": f"{distribution_type}_distribution_medium_plan",
            "long_plan": f"{distribution_type}_distribution_long_plan",
        }

        # Check if columns exist
        if not all(col in env_df.columns for col in action_dist_cols.values()):
            return None

        # Group by rollout_number
        grouped = env_df.groupby("rollout_number")

        rollout_numbers = []
        action_percentages = {action: [] for action in action_names}

        for rollout_num, group in grouped:
            rollout_numbers.append(rollout_num)

            # Count how many times each action was taken in this distribution
            total_count = 0
            action_counts = {}

            for action_key, col_name in action_dist_cols.items():
                count = group[col_name].sum()
                action_counts[action_key] = count
                total_count += count

            # Convert to percentages
            if total_count > 0:
                action_percentages["Model Free"].append(
                    action_counts["model_free"] / total_count * 100
                )
                action_percentages["Short Plan"].append(
                    action_counts["short_plan"] / total_count * 100
                )
                action_percentages["Medium Plan"].append(
                    action_counts["medium_plan"] / total_count * 100
                )
                action_percentages["Long Plan"].append(
                    action_counts["long_plan"] / total_count * 100
                )
            else:
                for action in action_names:
                    action_percentages[action].append(0)

        return np.array(rollout_numbers), action_percentages

    # Plot for each config
    for config_idx, (label, dfs, overrides) in enumerate(
        tqdm(data, desc="Plotting commit distributions", position=0)
    ):
        combined_df = pd.concat(dfs, ignore_index=True)

        # Get unique environments in this config
        if "env_name" in combined_df.columns:
            config_env_names = sorted(combined_df["env_name"].unique())
        else:
            config_env_names = ["default"]

        # Plot each environment in its own pair of columns
        for env_idx, env_name in enumerate(env_names_sorted):
            # Check if this environment exists in this config
            if env_name not in config_env_names:
                # Fill with "No data" for both columns
                for dist_idx in range(2):
                    col_idx = env_idx * 2 + dist_idx
                    ax = axes[config_idx, col_idx]
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {env_name}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                continue

            # Get data for this specific environment
            env_df = (
                combined_df[combined_df["env_name"] == env_name]
                if "env_name" in combined_df.columns
                else combined_df
            )

            # Plot in-distribution and out-of-distribution for this environment
            for dist_idx, (dist_type, dist_label) in enumerate(
                [("in", "In-Distribution"), ("out", "Out-of-Distribution")]
            ):
                col_idx = env_idx * 2 + dist_idx
                ax = axes[config_idx, col_idx]

                result = compute_action_percentages(env_df, dist_type)

                if result is None:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                rollout_numbers, action_percentages = result

                # Apply smoothing
                for action in action_names:
                    action_percentages[action] = smooth_data(
                        action_percentages[action], smooth_alpha
                    )

                # Create stacked area plot
                ax.stackplot(
                    rollout_numbers,
                    *[action_percentages[action] for action in action_names],
                    labels=action_names,
                    colors=action_colors,
                    alpha=0.7,
                )

                # Styling
                ax.set_ylabel("Action Percentage", fontsize=11, fontweight="bold")
                ax.set_xlabel("Rollout Number", fontsize=11, fontweight="bold")
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Add title only to top row
                if config_idx == 0:
                    title = f"{env_name} - {dist_label}"
                    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

                # Add config label on the leftmost column only
                if col_idx == 0:
                    ax.annotate(
                        label,
                        xy=(0, 0.5),
                        xytext=(-0.25, 0.5),
                        xycoords="axes fraction",
                        textcoords="axes fraction",
                        fontsize=9,
                        ha="right",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3
                        ),
                    )

                # Add legend to first plot only to avoid clutter
                if config_idx == 0 and col_idx == 0:
                    ax.legend(loc="best", fontsize=8, framealpha=0.9, title="Action")

    return fig, axes


def plot_log_exp_data(experiment_dir):
    if not os.path.isdir(experiment_dir):
        print(f"No directory {experiment_dir}")
        return

    # Look for data in the results subfolder
    results_dir = os.path.join(experiment_dir, "results")
    if not os.path.isdir(results_dir):
        print(f"No results directory found at {results_dir}")
        return

    # Check if single run (has training_log.csv) or multiple runs (subfolders)
    single_run = os.path.isfile(os.path.join(results_dir, "training_log.csv"))

    if single_run:
        print("Plotting from a single run folder")
    else:
        print("Plotting from multiple run folders")

    if single_run:  # Look in a single folder
        data = []
        df, overrides, config = load_folder(results_dir)
        if df is None:
            return
        label = "Single Run"
        data.append([label, [df], overrides])
    else:  # Look at all folders in results_dir
        raw_data = []
        for entry in os.scandir(results_dir):
            if entry.is_dir():
                folder = load_folder(entry.path)
                if folder is not None:
                    df, overrides, config = folder
                    raw_data.append((df, overrides, config))
        if len(raw_data) == 0:
            print("No log data found in subdirectories.")
            return

        grouped_data = group_runs_by_config(raw_data)
        config_dicts = [dict(config_key) for config_key in grouped_data.keys()]
        labels = create_unique_labels(config_dicts)

        data = []
        for label, (config_key, dfs) in zip(labels, grouped_data.items()):
            data.append([label, dfs, dict(config_key)])

    if len(data) == 0:
        print("No log data found.")
        return

    data = sort_data_by_differing_keys(data)

    # Print some information
    print_final_data_info(data)

    # Detect step type from first dataframe and overrides
    first_df = data[0][1][0]
    first_overrides = data[0][2]
    first_config = None
    if single_run:
        # Config was already loaded
        result = load_folder(results_dir)
        if result is not None:
            _, _, first_config = result
    else:
        # Load config from first folder
        for entry in os.scandir(results_dir):
            if entry.is_dir():
                result = load_folder(entry.path)
                if result is not None:
                    _, _, first_config = result
                    break
    step_type = detect_step_type(first_df, first_overrides, first_config)
    print(f"\nDetected step type: {step_type}")

    # Preprocess the data for specific metrics (vectorized)
    for i in range(len(data)):
        label, dfs, config = data[i]

        # Process each dataframe once with vectorized operations
        processed_dfs = []
        for df in dfs:
            df = parse_meta_actions_vectorized(
                df.copy(), step_type=step_type
            )  # Use .copy() to avoid warnings
            df = compute_distribution_percentages(df, step_type=step_type)
            processed_dfs.append(df)

        data[i] = (label, processed_dfs, config)

    # Define metrics based on step type
    if step_type == "bandit":
        # Bandit: plan (action 0) vs model_free (action 1)
        # Now tracks in/out distribution separately for plan and model_free
        metrics_to_plot = [
            "meta_action_plan_percent",
            "meta_action_follow_plan_percent",
            "reward",
            "time",
            "model_free_actor_uncertainty",
            "model_free_actor_uncertainty_on_plan",
            "model_free_actor_uncertainty_on_model_free",
            # Bandit distribution tracking (raw counts)
            "in_distribution_plan",
            "out_distribution_plan",
            "in_distribution_model_free",
            "out_distribution_model_free",
            # Bandit distribution percentages
            "in_distribution_plan_percent",
            "out_distribution_plan_percent",
            "in_distribution_model_free_percent",
            "out_distribution_model_free_percent",
        ]
    elif step_type == "execute":
        # Execute: plan, execute (follow or model_free), clear
        metrics_to_plot = [
            "meta_action_plan_percent",
            "meta_action_follow_plan_percent",
            "meta_action_model_free_percent",
            "meta_action_clear_plan_percent",
            "reward",
            "time",
            "model_free_actor_uncertainty",
            "model_free_actor_uncertainty_on_plan",
            "model_free_actor_uncertainty_on_follow_plan",
            "model_free_actor_uncertainty_on_model_free",
            "model_free_actor_uncertainty_on_clear_plan",
            # Execute distribution tracking (raw counts)
            "in_distribution_plan",
            "out_distribution_plan",
            "in_distribution_follow_plan",
            "out_distribution_follow_plan",
            "in_distribution_model_free",
            "out_distribution_model_free",
            "in_distribution_clear_plan",
            "out_distribution_clear_plan",
            # Execute distribution percentages
            "in_distribution_plan_percent",
            "out_distribution_plan_percent",
            "in_distribution_follow_plan_percent",
            "out_distribution_follow_plan_percent",
            "in_distribution_model_free_percent",
            "out_distribution_model_free_percent",
            "in_distribution_clear_plan_percent",
            "out_distribution_clear_plan_percent",
        ]
    elif step_type == "commit":
        # Commit: model_free, short_plan, medium_plan, long_plan
        metrics_to_plot = [
            "reward",
            "time",
            "model_free_actor_uncertainty",
            "model_free_actor_uncertainty_on_model_free",
            "model_free_actor_uncertainty_on_short_plan",
            "model_free_actor_uncertainty_on_medium_plan",
            "model_free_actor_uncertainty_on_long_plan",
            # Commit distribution percentages
            "in_distribution_model_free_percent",
            "out_distribution_model_free_percent",
            "in_distribution_short_plan_percent",
            "out_distribution_short_plan_percent",
            "in_distribution_medium_plan_percent",
            "out_distribution_medium_plan_percent",
            "in_distribution_long_plan_percent",
            "out_distribution_long_plan_percent",
        ]
    else:  # seperate
        # Seperate: plan, follow_plan, model_free (no clear)
        metrics_to_plot = [
            "meta_action_plan_percent",
            "meta_action_follow_plan_percent",
            "meta_action_model_free_percent",
            "reward",
            "time",
            "model_free_actor_uncertainty",
            "model_free_actor_uncertainty_on_plan",
            "model_free_actor_uncertainty_on_follow_plan",
            "model_free_actor_uncertainty_on_model_free",
            # Seperate distribution tracking (raw counts)
            "in_distribution_plan",
            "out_distribution_plan",
            "in_distribution_follow_plan",
            "out_distribution_follow_plan",
            "in_distribution_model_free",
            "out_distribution_model_free",
            # Seperate distribution percentages
            "in_distribution_plan_percent",
            "out_distribution_plan_percent",
            "in_distribution_follow_plan_percent",
            "out_distribution_follow_plan_percent",
            "in_distribution_model_free_percent",
            "out_distribution_model_free_percent",
        ]

    # Create label overrides for bandit mode
    metric_label_overrides = {}
    if step_type == "bandit":
        metric_label_overrides = {
            "meta_action_plan_percent": "Use Planning Percent",
            "meta_action_follow_plan_percent": "Use Model Free Percent",
        }

    # Create plots subfolder
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate a separate plot for each config
    saved_files = []
    for config_idx, (label, dfs, config) in enumerate(
        tqdm(data, desc="Generating plots per config")
    ):
        # Create a clean filename from the label
        # Replace special characters and limit length
        clean_label = (
            label.replace("=", "_")
            .replace(",", "")
            .replace(" ", "")
            .replace("/", "_")
            .replace(".", "_")
        )
        # Limit filename length
        if len(clean_label) > 100:
            clean_label = clean_label[:100]

        # Create single-config data list
        single_config_data = [(label, dfs, config)]

        # For commit mode, create both distribution and metrics plots
        if step_type == "commit":
            # Distribution plot
            fig_dist, ax_dist = plot_commit_distribution(
                single_config_data, smooth_alpha=0.3
            )
            dist_filename = os.path.join(plots_dir, f"{clean_label}_distribution.png")
            fig_dist.savefig(dist_filename, dpi=150, bbox_inches="tight")
            plt.close(fig_dist)
            saved_files.append(dist_filename)

            # Metrics plot
            fig_metrics, ax_metrics = plot_metrics_by_config(
                single_config_data,
                metrics_to_plot,
                metric_label_overrides=metric_label_overrides,
            )
            metrics_filename = os.path.join(plots_dir, f"{clean_label}_metrics.png")
            fig_metrics.savefig(metrics_filename, dpi=100, bbox_inches="tight")
            plt.close(fig_metrics)
            saved_files.append(metrics_filename)
        else:
            # Standard plot
            fig, ax = plot_metrics_by_config(
                single_config_data,
                metrics_to_plot,
                metric_label_overrides=metric_label_overrides,
            )
            filename = os.path.join(plots_dir, f"{clean_label}.png")
            fig.savefig(filename, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(filename)

    print(f"\nSaved {len(saved_files)} plot(s) to {plots_dir}/")
    for f in saved_files:
        print(f"  - {os.path.basename(f)}")

    return saved_files
