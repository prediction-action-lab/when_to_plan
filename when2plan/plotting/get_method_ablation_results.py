"""
Print method-ablation results (mean reward + 95% CI) grouped by ablation type and
environment, with a LaTeX table for easy copy-paste into papers.

The ablation type is read from the top-level ``ablation`` key in each run's hydra
config.  Runs where the key is missing / None / "none" / "null" are labelled
"Main method".

Usage – edit EXPERIMENT_FOLDERS below and run:
    .venv/bin/python when2plan/plotting/get_method_ablation_results.py

Or pass folders via CLI:
    .venv/bin/python when2plan/plotting/get_method_ablation_results.py \\
        --experiment_paths outputs/chtc_results_nav_ablation \\
                           outputs/chtc_results_box_push_ablation \\
                           outputs/chtc_results_doorkey_ablation \\
                           outputs/chtc_results_maze_ablation
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# *** Edit this list to set the folders used when running without CLI args ***
# ---------------------------------------------------------------------------
EXPERIMENT_FOLDERS = [
    "outputs/chtc_results_nav_ablation",
    "outputs/chtc_results_box_push_ablation",
    "outputs/chtc_results_doorkey_ablation",
    "outputs/chtc_results_maze_ablation",
    "outputs/chtc_results_fetch_ablation",
]
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 10_000
CI_LEVEL = 95  # percent

# Display-name mappings (extend as needed)
ABLATION_DISPLAY = {
    "Main method": "Main method",
    "no_distance_to_goal": "Distance",
    "no_history": "History",
    "no_model_free_uncertainty": "Uncertainty",
    "no_mf_uncertainty": "Uncertainty",
    "no_option_duration": "Duration",
    "step_separate": "Step separate",
    "step_seperate": "Step separate",
}

ENV_DISPLAY = {
    "fetch_pick_place": "Fetch P\\&P",
    "doorkey": "DoorKey",
    "box_push": "Box Push",
    "navigation": "Navigation",
    "maze": "Maze",
}

# ---------------------------------------------------------------------------
# Ablation label remapping.
# Any ablation key listed here will be loaded and stored under the given name.
# Use this to fold a specific ablation into "Main method" without hardcoding.
# ---------------------------------------------------------------------------
REMAP_ABLATIONS = {
    "no_option_duration": "Main method",
}

# Ablation labels to exclude from loading entirely (still supported if needed).
EXCLUDE_ABLATIONS: set = set()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_evaluation_folder(path):
    """Load evaluation_results.csv + hydra config/overrides from a run folder."""
    if not os.path.isdir(path):
        return None

    eval_file = os.path.join(path, "evaluation_results.csv")
    if not os.path.isfile(eval_file):
        return None

    df = pd.read_csv(eval_file)

    config = {}
    config_file = os.path.join(path, ".hydra", "config.yaml")
    if os.path.isfile(config_file):
        with open(config_file, "r") as fh:
            config = yaml.safe_load(fh) or {}

    overrides = {}
    overrides_file = os.path.join(path, ".hydra", "overrides.yaml")
    if os.path.isfile(overrides_file):
        with open(overrides_file, "r") as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw, list):
            for item in raw:
                if "=" in item:
                    k, v = item.split("=", 1)
                    overrides[k] = v
        elif isinstance(raw, dict):
            overrides = raw

    return df, config, overrides


def discover_subdirs(experiment_path):
    """Return the list of run subdirectories inside an experiment folder."""
    if os.path.isfile(os.path.join(experiment_path, "evaluation_results.csv")):
        return [experiment_path]

    results_dir = os.path.join(experiment_path, "results")
    if os.path.isdir(results_dir):
        return [
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith(".")
        ]

    candidates = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
        and not d.startswith(".")
        and d not in ["plots", "__pycache__"]
    ]
    return candidates or [experiment_path]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_env_name(config, overrides):
    """Extract environment name from hydra config / overrides."""
    if "env" in overrides:
        return overrides["env"]
    if "env" in config and isinstance(config["env"], dict) and "type" in config["env"]:
        return config["env"]["type"]
    return "unknown"


def get_ablation(config, overrides):
    """Extract the ablation label from config / overrides.  None → 'Main method'."""
    raw = overrides.get("ablation") or config.get("ablation")
    if raw is None or str(raw).lower() in ("none", "null", ""):
        return "Main method"
    return str(raw)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=42):
    """Percentile bootstrap of the mean. Returns (mean, lower_err, upper_err)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0, 0.0

    rng = np.random.RandomState(seed)
    boot_means = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(
        axis=1
    )

    alpha = (100 - ci) / 2
    ci_lo = float(np.percentile(boot_means, alpha))
    ci_hi = float(np.percentile(boot_means, 100 - alpha))
    return mean, mean - ci_lo, ci_hi - mean


def bootstrap_average_ci(
    env_rewards_dict, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=42
):
    """
    Bootstrap CI for the average of per-environment means (fully vectorised).

    For each bootstrap iteration, resample within every environment independently
    and average the resulting per-env means.
    """
    arrays = [
        np.asarray(r, dtype=float) for r in env_rewards_dict.values() if len(r) > 0
    ]
    if not arrays:
        return np.nan, np.nan, np.nan

    mean = float(np.mean([arr.mean() for arr in arrays]))

    rng = np.random.RandomState(seed)
    # shape: (n_bootstrap, num_envs) — each column = resampled means for one env
    boot_env_means = np.column_stack(
        [
            rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)
            for arr in arrays
        ]
    )
    boot_avgs = boot_env_means.mean(axis=1)  # shape: (n_bootstrap,)

    alpha = (100 - ci) / 2
    ci_lo = float(np.percentile(boot_avgs, alpha))
    ci_hi = float(np.percentile(boot_avgs, 100 - alpha))
    return mean, mean - ci_lo, ci_hi - mean


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_all_runs(experiment_folders):
    """
    Load every run from every folder and group rewards by (ablation, env_name).

    Returns
    -------
    dict[(ablation_label, env_name) -> list[float]]   episode rewards
    """
    groups = defaultdict(list)
    total_dirs = 0

    for folder in experiment_folders:
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            print(f"  [WARNING] Folder not found, skipping: {folder}")
            continue

        subdirs = discover_subdirs(folder)
        total_dirs += len(subdirs)
        print(f"  {folder}  →  {len(subdirs)} run(s)")

        for subdir in subdirs:
            result = load_evaluation_folder(subdir)
            if result is None:
                continue
            df, config, overrides = result
            env = get_env_name(config, overrides)
            ablation = get_ablation(config, overrides)

            if ablation in EXCLUDE_ABLATIONS:
                continue
            ablation = REMAP_ABLATIONS.get(ablation, ablation)
            if "reward" in df.columns:
                groups[(ablation, env)].extend(df["reward"].tolist())

    print(f"\nTotal run directories loaded: {total_dirs}")
    return groups


def compute_stats(groups):
    """Bootstrap each (ablation, env) group. Returns same-shaped dict of (mean, lo, hi, n)."""
    return {
        key: (*bootstrap_ci(rewards), len(rewards)) for key, rewards in groups.items()
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def ablation_display(label):
    return ABLATION_DISPLAY.get(label, label.replace("_", " ").title())


def env_display(name):
    return ENV_DISPLAY.get(name, name.replace("_", " ").title())


def print_results(stats, groups):
    """Plain-text summary, then LaTeX table."""
    ablation_labels = sorted(stats.keys(), key=lambda k: (k[0] != "Main method", k[0]))
    ablation_values = sorted(
        set(k[0] for k in ablation_labels),
        key=lambda a: (a != "Main method", a),
    )
    env_names = sorted(set(k[1] for k in ablation_labels))

    sep = "-" * 80

    # -----------------------------------------------------------------------
    # Plain-text
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  METHOD ABLATION RESULTS  (mean ± 95% CI)")
    print("=" * 80)

    for ablation in ablation_values:
        print(f"\n  {ablation_display(ablation)}")
        print(sep)
        for env in env_names:
            key = (ablation, env)
            if key not in stats:
                continue
            mean, lo, hi, n = stats[key]
            ci_lo = mean - lo
            ci_hi = mean + hi
            print(
                f"  {env_display(env):<28s}  {mean:>8.2f}"
                f"  [{ci_lo:>7.2f},  {ci_hi:>7.2f}]  (n={n})"
            )
        print(sep)

    # -----------------------------------------------------------------------
    # LaTeX table  – rows = environments, columns = ablation methods
    # -----------------------------------------------------------------------
    col_format = "l" + "c" * len(ablation_values)

    print("\n\nLaTeX table format:")
    print("% Requires: \\usepackage{booktabs}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\normalsize")
    print("\\setlength{\\tabcolsep}{8pt}")
    print("\\begin{tabular}{" + col_format + "}")
    print("\\toprule")

    header_cells = ["\\textbf{Env.}"] + [
        f"\\textbf{{{ablation_display(a)}}}" for a in ablation_values
    ]
    print(" & ".join(header_cells) + " \\\\")
    print("\\midrule")

    # Pre-compute best-ablation bounds per environment (for bolding)
    best_bounds = {}
    for env in env_names:
        candidates = [
            (ablation, stats[(ablation, env)])
            for ablation in ablation_values
            if (ablation, env) in stats
        ]
        if not candidates:
            continue
        best_abl, (bm, bl, bh, _) = max(
            candidates, key=lambda t: t[1][0] if not np.isnan(t[1][0]) else -np.inf
        )
        best_bounds[env] = (bm - bl, bm + bh)  # (ci_lo, ci_hi)

    for i, env in enumerate(env_names):
        env_disp = env_display(env)

        def _mean_cell(ablation, env=env):
            key = (ablation, env)
            if key not in stats:
                return "$-$"
            mean, lo, hi, _ = stats[key]
            if np.isnan(mean):
                return "$-$"
            ci_lo_val = mean - lo
            ci_hi_val = mean + hi
            best_lo, best_hi = best_bounds.get(env, (np.inf, -np.inf))
            overlaps = ci_hi_val >= best_lo and ci_lo_val <= best_hi
            if overlaps:
                return f"$\\mathbf{{{mean:.1f}}}$"
            return f"${mean:.1f}$"

        def _ci_cell(ablation, env=env):
            key = (ablation, env)
            if key not in stats:
                return ""
            _, lo, hi, _ = stats[key]
            if np.isnan(lo):
                return ""
            return f"\\scriptsize$[{lo:.1f},\\,{hi:.1f}]$"

        mean_row = [env_disp] + [_mean_cell(a) for a in ablation_values]
        ci_row = [""] + [_ci_cell(a) for a in ablation_values]

        print(" & ".join(mean_row) + " \\\\")
        print(" & ".join(ci_row) + " \\\\")
        if i < len(env_names) - 1:
            print("\\addlinespace")

    # Average row across all environments (with bootstrapped CI where raw data exists)
    print("\\midrule")

    # Pre-compute per-ablation average stats
    avg_stats = {}  # ablation -> (mean, lo_err, hi_err) or None
    for ablation in ablation_values:
        env_rewards = {
            env: groups[(ablation, env)]
            for env in env_names
            if (ablation, env) in groups and len(groups[(ablation, env)]) > 0
        }
        if env_rewards:
            avg_stats[ablation] = bootstrap_average_ci(env_rewards)
        else:
            # Hardcoded entries: derive mean from stats, no CI
            means = [
                stats[(ablation, env)][0]
                for env in env_names
                if (ablation, env) in stats and not np.isnan(stats[(ablation, env)][0])
            ]
            avg_stats[ablation] = (float(np.mean(means)), None, None) if means else None

    # Determine best average for bolding
    all_avg_means = [
        avg_stats[a][0]
        for a in ablation_values
        if avg_stats.get(a) and not np.isnan(avg_stats[a][0])
    ]
    best_avg = max(all_avg_means) if all_avg_means else np.inf

    def _avg_mean_cell(ablation):
        entry = avg_stats.get(ablation)
        if entry is None or np.isnan(entry[0]):
            return "$-$"
        avg = entry[0]
        return f"$\\mathbf{{{avg:.1f}}}$" if avg >= best_avg else f"${avg:.1f}$"

    def _avg_ci_cell(ablation):
        entry = avg_stats.get(ablation)
        if entry is None or entry[1] is None or np.isnan(entry[1]):
            return ""
        _, lo, hi = entry
        return f"\\scriptsize$[{lo:.1f},\\,{hi:.1f}]$"

    avg_row = ["\\textbf{Average}"] + [_avg_mean_cell(a) for a in ablation_values]
    avg_ci_row = [""] + [_avg_ci_cell(a) for a in ablation_values]
    print(" & ".join(avg_row) + " \\\\")
    print(" & ".join(avg_ci_row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Mean episode return per method ablation with 95\\% bootstrap")
    print("         uncertainty intervals (margins below each mean). Bold entries")
    print("         overlap in CI with the best performer per environment.}")
    print("\\label{tab:method_ablation}")
    print("\\end{table}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def get_method_ablation_results(experiment_folders):
    print(f"\nLoading runs from {len(experiment_folders)} folder(s):")
    groups = load_all_runs(experiment_folders)

    if not groups:
        print("No evaluation data found!")
        return

    stats = compute_stats(groups)

    print_results(stats, groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print method-ablation results (mean reward + 95% CI) grouped by "
            "ablation type and environment, plus a LaTeX table."
        )
    )
    parser.add_argument(
        "--experiment_paths",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "One or more experiment folders to load. "
            "If omitted, uses the EXPERIMENT_FOLDERS list at the top of this file."
        ),
    )
    args = parser.parse_args()

    folders = args.experiment_paths if args.experiment_paths else EXPERIMENT_FOLDERS
    get_method_ablation_results(folders)
