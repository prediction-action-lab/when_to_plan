"""
Plot evaluation results comparing learned meta-policy against heuristic baselines.

For each environment, plots 5 bars showing average reward with 95% CIs:
- Meta agent (learned policy)
- Always model-free
- Always short plan
- Always medium plan
- Always long plan
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from collections import defaultdict
import scipy.stats as stats

# ===== PLOT CONFIGURATION =====
USE_HATCHING_PATTERNS = (
    False  # Set to True to add hatching patterns to bars (useful for grayscale)
)
LATEX_TABLE_TRANSPOSED = (
    False  # Set to True: rows=policies, cols=environments (narrower table)
    # Set to False: rows=environments, cols=policies (wider table)
)
# ==============================

# Set publication-quality plot style
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "text.usetex": False,  # Set to True if LaTeX is available
    }
)


def load_evaluation_folder(path):
    """Load evaluation_results.csv and config from a folder."""
    if not os.path.isdir(path):
        print(f"No directory {path}")
        return None

    eval_file = os.path.join(path, "evaluation_results.csv")
    if not os.path.isfile(eval_file):
        print(f"No evaluation file found at {eval_file}")
        return None

    df = pd.read_csv(eval_file)

    # Load config to determine environment
    config = {}
    config_file = os.path.join(path, ".hydra", "config.yaml")
    if os.path.isfile(config_file):
        with open(config_file, "r") as fh:
            config = yaml.safe_load(fh)

    # Load overrides
    overrides = {}
    overrides_file = os.path.join(path, ".hydra", "overrides.yaml")
    if os.path.isfile(overrides_file):
        with open(overrides_file, "r") as fh:
            overrides = yaml.safe_load(fh)

        if isinstance(overrides, list):
            overrides_dict = {}
            for item in overrides:
                if "=" in item:
                    k, v = item.split("=", 1)
                    overrides_dict[k] = v
            overrides = overrides_dict

    return df, config, overrides


def get_env_name(config, overrides):
    """Extract environment name from config or overrides."""
    # Check overrides first
    if "env" in overrides:
        return overrides["env"]

    # Then check config
    if "env" in config and "type" in config["env"]:
        return config["env"]["type"]

    return "unknown"


def compute_stats(rewards):
    """Compute mean and 95% CI using percentile bootstrap."""
    if len(rewards) == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(rewards)
    if len(rewards) == 1:
        return mean, 0, 0

    # 95% CI using percentile bootstrap
    n_bootstrap = 10000
    bootstrap_means = []

    rewards_array = np.array(rewards)
    n_samples = len(rewards_array)

    # Generate bootstrap samples
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(rewards_array, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Compute 95% CI using 2.5th and 97.5th percentiles
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    # Return mean and error bars (distance from mean to CI bounds)
    lower_error = mean - ci_lower
    upper_error = ci_upper - mean

    return mean, lower_error, upper_error


def group_runs_by_env(experiment_path):
    """
    Load all evaluation results and group by environment.

    Returns: dict mapping env_name -> list of dataframes
    """
    groups = defaultdict(list)

    # First check if experiment_path itself is a run directory
    if os.path.isfile(os.path.join(experiment_path, "evaluation_results.csv")):
        subdirs = [experiment_path]
    else:
        # Check if we have a 'results' subdirectory (cluster case)
        results_dir = os.path.join(experiment_path, "results")
        if os.path.isdir(results_dir):
            # Look for run directories inside results/
            subdirs = [
                os.path.join(results_dir, d)
                for d in os.listdir(results_dir)
                if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith(".")
            ]
        else:
            # Check if experiment_path contains subdirectories (multiple runs)
            subdirs = [
                os.path.join(experiment_path, d)
                for d in os.listdir(experiment_path)
                if os.path.isdir(os.path.join(experiment_path, d))
                and not d.startswith(".")  # Ignore hidden folders like .hydra
                and d not in ["plots", "__pycache__"]  # Ignore common non-run folders
            ]

        # If no subdirectories found, treat experiment_path itself as a single run
        if not subdirs:
            subdirs = [experiment_path]

    print(f"Found {len(subdirs)} run directories")

    for subdir in subdirs:
        result = load_evaluation_folder(subdir)
        if result is None:
            continue

        df, config, overrides = result
        env_name = get_env_name(config, overrides)
        groups[env_name].append(df)
        print(f"  Loaded {subdir} -> env: {env_name}")

    return groups


def plot_evaluation_comparison(experiment_path, output_path=None):
    """
    Create bar plot comparing learned policy vs heuristic baselines.

    Args:
        experiment_path: Path to folder containing run subdirectories
        output_path: Where to save the plot (default: experiment_path/plots/)
    """
    # Group runs by environment
    env_groups = group_runs_by_env(experiment_path)

    if not env_groups:
        print("No evaluation data found!")
        return

    # Compute statistics for each environment and policy type
    results = {}
    for env_name, dfs in env_groups.items():
        # For each policy type, collect all episode rewards across all runs
        meta_rewards = []
        mf_rewards = []
        short_rewards = []
        medium_rewards = []
        long_rewards = []
        random_rewards = []

        for df in dfs:
            # Meta agent rewards
            if "reward" in df.columns:
                meta_rewards.extend(df["reward"].values)

            # Heuristic rewards
            if "always_model_free_reward" in df.columns:
                mf_rewards.extend(df["always_model_free_reward"].values)
            if "always_short_plan_reward" in df.columns:
                short_rewards.extend(df["always_short_plan_reward"].values)
            if "always_medium_plan_reward" in df.columns:
                medium_rewards.extend(df["always_medium_plan_reward"].values)
            if "always_long_plan_reward" in df.columns:
                long_rewards.extend(df["always_long_plan_reward"].values)
            if "random_policy_reward" in df.columns:
                random_rewards.extend(df["random_policy_reward"].values)

        # Compute stats
        results[env_name] = {
            "Meta Agent": compute_stats(meta_rewards),
            "Always Reactive": compute_stats(mf_rewards),
            "Always Short": compute_stats(short_rewards),
            "Always Medium": compute_stats(medium_rewards),
            "Always Long": compute_stats(long_rewards),
            "Random Policy": compute_stats(random_rewards),
        }

    # Create plot with publication-quality settings
    # Scale figure size based on number of environments
    # Each environment group needs about 1.2 inches of width
    base_width = 3.0  # Minimum width for axis labels and legend
    width_per_env = 1.4  # Width allocated per environment
    fig_width = base_width + len(env_groups) * width_per_env

    # Ensure minimum width for readability (especially for single environment)
    fig_width = max(5, fig_width)

    # Height scales slightly with number of environments to maintain good proportions
    fig_height = min(5.5, max(3.5, 3.5 + len(env_groups) * 0.1))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    policy_types = [
        "Meta Agent",
        "Always Reactive",
        "Always Short",
        "Always Medium",
        "Always Long",
        "Random Policy",
    ]

    # Use pastel color palette for a softer, professional look
    colors = [
        "#ED9EE6",
        "#FDB462",  # Pastel orange
        "#80B1D3",  # Pastel blue
        "#B3DE69",  # Pastel green
        "#FB8072",  # Pastel red/coral
        "#BEBEBE",  # Pastel gray
    ]

    # Use hatching patterns for additional distinction (useful in grayscale)
    hatches = ["", "///", "\\\\\\", "|||", "...", "xxx"]

    num_envs = len(env_groups)
    num_policies = len(policy_types)

    # Width of each bar and spacing - thinner bars for better readability with many envs
    bar_width = 0.10
    group_spacing = 0.3

    # Sort environments for consistent ordering
    env_names = sorted(env_groups.keys())

    # Capitalize environment names for publication
    env_names_display = [name.replace("_", " ").title() for name in env_names]

    # Collect all data points to determine y-axis limits
    all_values = []
    for env_name in env_names:
        for policy_type in policy_types:
            mean, lower_ci, upper_ci = results[env_name][policy_type]
            all_values.append(mean - lower_ci)  # Lower bound with error
            all_values.append(mean + upper_ci)  # Upper bound with error

    # Plot bars
    for policy_idx, policy_type in enumerate(policy_types):
        positions = []
        means = []
        lower_errors = []
        upper_errors = []

        for env_idx, env_name in enumerate(env_names):
            # Position: env group center + offset for this policy
            group_center = env_idx * (num_policies * bar_width + group_spacing)
            pos = group_center + policy_idx * bar_width
            positions.append(pos)

            mean, lower_ci, upper_ci = results[env_name][policy_type]
            means.append(mean)
            lower_errors.append(lower_ci)
            upper_errors.append(upper_ci)

        # Plot bars with error bars
        bars = ax.bar(
            positions,
            means,
            bar_width,
            label=policy_type,
            color=colors[policy_idx],
            edgecolor="black",
            linewidth=0.8,
            yerr=[lower_errors, upper_errors],
            capsize=4,
            error_kw={
                "elinewidth": 1.5,
                "capthick": 1.5,
                "ecolor": "black",
                "alpha": 0.8,
            },
            hatch=hatches[policy_idx] if USE_HATCHING_PATTERNS else None,
            alpha=0.9 if policy_idx == 0 else 0.85,  # Slightly emphasize Meta Agent
            zorder=3,
        )

    # Set x-axis labels (environment names)
    group_centers = [
        i * (num_policies * bar_width + group_spacing)
        + (num_policies - 1) * bar_width / 2
        for i in range(num_envs)
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(env_names_display, rotation=0, ha="center")

    # Set y-axis limits with some padding
    y_min, y_max = min(all_values), max(all_values)
    y_min = max(y_min, -250)  # Set a reasonable floor for y-axis
    y_max = min(y_max, 100)  # Set a reasonable ceiling for y-axis
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Labels and styling
    ax.set_ylabel("Average Episode Return", fontsize=12, fontweight="medium")
    ax.set_xlabel("Environment", fontsize=12, fontweight="medium")

    # Grid styling
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5, zorder=1)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Add legend below plot in horizontal layout
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),  # Move legend further down from x-axis
        frameon=True,
        fancybox=False,
        shadow=False,
        ncol=6,  # Horizontal layout with all 6 policies in one row
        framealpha=0.95,
        edgecolor="black",
        facecolor="white",
    )
    legend.get_frame().set_linewidth(0.8)

    # Adjust layout to make room for legend
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.18)  # Make more room for legend below

    # Save plot in multiple formats
    if output_path is None:
        output_path = os.path.join(experiment_path, "plots")

    os.makedirs(output_path, exist_ok=True)

    # Save high-resolution PNG for presentations
    save_path_png = os.path.join(output_path, "evaluation_comparison.png")
    plt.savefig(save_path_png, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Saved high-res PNG to {save_path_png}")

    # Save PDF for papers (vector format)
    save_path_pdf = os.path.join(output_path, "evaluation_comparison.pdf")
    plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"✓ Saved vector PDF to {save_path_pdf}")

    # Save EPS for some journals
    save_path_eps = os.path.join(output_path, "evaluation_comparison.eps")
    plt.savefig(save_path_eps, format="eps", bbox_inches="tight", facecolor="white")
    print(f"✓ Saved EPS to {save_path_eps}")

    plt.close()

    # Print summary statistics (formatted for LaTeX table)
    print("\n=== Summary Statistics ===")
    print("\nPlain text format:")
    for env_name in env_names:
        print(f"\n{env_name}:")
        for policy_type in policy_types:
            mean, lower_ci, upper_ci = results[env_name][policy_type]
            print(
                f"  {policy_type:15s}: {mean:7.2f} ± [{lower_ci:.2f}, {upper_ci:.2f}]"
            )

    # Also print LaTeX table format for easy copy-paste
    # Requires: \usepackage{booktabs}
    print("\n\nLaTeX table format:")
    print("% Requires: \\usepackage{booktabs}")

    # Short display names for policies (order must match policy_types)
    policy_short_names = {
        "Meta Agent": "Meta Agent",
        "Always Reactive": "Reactive",
        "Always Short": "Short",
        "Always Medium": "Medium",
        "Always Long": "Long",
        "Random Policy": "Random",
    }

    # Pre-compute best-policy CI bounds per environment (for overlap test)
    best_bounds = {}
    for env_name in env_names:
        best_pt = max(policy_types, key=lambda pt: results[env_name][pt][0])
        bm, bl, bu = results[env_name][best_pt]
        best_bounds[env_name] = (bm - bl, bm + bu)  # (ci_lo, ci_hi)

    def mean_cell(env_name, policy_type):
        """Return the mean cell (bold if CI overlaps best)."""
        mean, lower_ci, upper_ci = results[env_name][policy_type]
        ci_lo = mean - lower_ci
        ci_hi = mean + upper_ci
        best_ci_lo, best_ci_hi = best_bounds[env_name]
        overlaps_best = ci_hi >= best_ci_lo and ci_lo <= best_ci_hi
        if overlaps_best:
            return f"$\\mathbf{{{mean:.1f}}}$"
        return f"${mean:.1f}$"

    def ci_cell(env_name, policy_type):
        """Return the scriptsize CI margin cell."""
        _, lower_ci, upper_ci = results[env_name][policy_type]
        return f"\\scriptsize$[{lower_ci:.1f},\\,{upper_ci:.1f}]$"

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\normalsize")
    print("\\setlength{\\tabcolsep}{8pt}")

    if LATEX_TABLE_TRANSPOSED:
        # Rows = policies, each with a mean row + CI row; columns = environments
        col_format = "l" + "c" * len(env_names)
        print("\\begin{tabular}{" + col_format + "}")
        print("\\toprule")
        header_cells = ["\\textbf{Setup}"] + [
            f"\\textbf{{{name.replace('_', ' ').title()}}}" for name in env_names
        ]
        print(" & ".join(header_cells) + " \\\\")
        print("\\midrule")
        for i, policy_type in enumerate(policy_types):
            short = policy_short_names.get(policy_type, policy_type)
            mean_row = [short] + [
                mean_cell(env_name, policy_type) for env_name in env_names
            ]
            ci_row = [""] + [ci_cell(env_name, policy_type) for env_name in env_names]
            print(" & ".join(mean_row) + " \\\\")
            print(" & ".join(ci_row) + " \\\\")
            if i < len(policy_types) - 1:
                print("\\addlinespace")
    else:
        # Rows = environments, each with a mean row + CI row; columns = policies
        col_format = "l" + "c" * len(policy_types)
        print("\\begin{tabular}{" + col_format + "}")
        print("\\toprule")
        header_cells = ["\\textbf{Env.}"] + [
            f"\\textbf{{{policy_short_names.get(pt, pt)}}}" for pt in policy_types
        ]
        print(" & ".join(header_cells) + " \\\\")
        print("\\midrule")
        for i, env_name in enumerate(env_names):
            env_display = env_name.replace("_", " ").title()
            mean_row = [env_display] + [mean_cell(env_name, pt) for pt in policy_types]
            ci_row = [""] + [ci_cell(env_name, pt) for pt in policy_types]
            print(" & ".join(mean_row) + " \\\\")
            print(" & ".join(ci_row) + " \\\\")
            if i < len(env_names) - 1:
                print("\\addlinespace")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Average episode return with 95\\% bootstrap uncertainty intervals")
    print("         (margins shown below each mean). Bold entries overlap in CI with")
    print("         the best performer per environment.}")
    print("\\label{tab:evaluation_results}")
    print("\\end{table}")

    return save_path_png


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Directory containing experiment run folders with evaluation_results.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to save plots (default: experiment_path/plots/)",
    )

    args = parser.parse_args()

    plot_evaluation_comparison(args.experiment_path, args.output_path)
