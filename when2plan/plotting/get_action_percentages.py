"""  
Print action-type percentages (mf, short, medium, long) for the Main method,
grouped by environment, with 95% bootstrap uncertainty intervals.

Loads from multiple experiment folders the same way get_method_ablation_results
does: runs whose ``ablation`` key maps to "Main method" via REMAP_ABLATIONS are
included; everything else is ignored.

Usage – edit EXPERIMENT_FOLDERS below and run:
    .venv/bin/python when2plan/plotting/get_action_percentages.py

Or pass folders via CLI:
    .venv/bin/python when2plan/plotting/get_action_percentages.py \\
        --experiment_paths outputs/chtc_results_nav_ablation \\
                           outputs/chtc_results_box_push_ablation ...
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# Action index -> human-readable label
ACTION_LABELS = {
    0: "mf",
    1: "short",
    2: "medium",
    3: "long",
}

N_BOOTSTRAP = 10_000
CI_LEVEL = 95  # percent

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

# Ablation label remapping: runs matching a key are stored under the mapped name.
REMAP_ABLATIONS = {
    "no_option_duration": "Main method",
}

# Only keep runs that resolve to one of these ablation labels after remapping.
KEEP_ABLATIONS = {"Main method"}

ENV_DISPLAY = {
    "fetch_pick_place": "Fetch P&P",
    "doorkey": "DoorKey",
    "box_push": "Box Push",
    "navigation": "Navigation",
    "maze": "Maze",
}


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors plot_evaluation_results.py)
# ---------------------------------------------------------------------------


def load_evaluation_folder(path):
    """Load evaluation_results.csv + hydra config from a run folder."""
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


def get_env_name(config, overrides):
    """Extract environment name from hydra config / overrides."""
    if "env" in overrides:
        return overrides["env"]
    if "env" in config and isinstance(config["env"], dict) and "type" in config["env"]:
        return config["env"]["type"]
    return "unknown"


def get_ablation(config, overrides):
    """Extract the ablation label, applying REMAP_ABLATIONS. None → 'Main method'."""
    raw = overrides.get("ablation") or config.get("ablation")
    if raw is None or str(raw).lower() in ("none", "null", ""):
        return "Main method"
    label = str(raw)
    return REMAP_ABLATIONS.get(label, label)


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


def group_runs_by_env(experiment_folders):
    """
    Load runs from all experiment folders, keeping only those that map to
    KEEP_ABLATIONS, and group their DataFrames by environment name.

    Returns: dict[env_name -> list[pd.DataFrame]]
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
            ablation = get_ablation(config, overrides)
            if ablation not in KEEP_ABLATIONS:
                continue
            env_name = get_env_name(config, overrides)
            groups[env_name].append(df)

    print(f"\nTotal run directories loaded: {total_dirs}")
    print(
        f"Kept {sum(len(v) for v in groups.values())} Main method run(s) across {len(groups)} environment(s)\n"
    )
    return groups


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=42):
    """
    Percentile bootstrap of the mean.

    Parameters
    ----------
    values : array-like of floats (one value per episode)
    n_bootstrap : int
    ci : float, uncertainty level in percent

    Returns
    -------
    mean, lower_error, upper_error
        where lower_error = mean - ci_low  (both >= 0)
              upper_error = ci_high - mean
    """
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
    ci_low = float(np.percentile(boot_means, alpha))
    ci_high = float(np.percentile(boot_means, 100 - alpha))

    return mean, mean - ci_low, ci_high - mean


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_action_percentages_by_env(env_groups):
    """
    For each environment, compute per-episode action percentages and summarise.

    Returns
    -------
    dict[env_name -> dict[action_label -> (mean_pct, lower_err, upper_err)]]
    """
    results = {}

    for env_name, dfs in env_groups.items():
        # Collect per-episode percentage for each action type
        per_episode = {label: [] for label in ACTION_LABELS.values()}

        for df in dfs:
            # Check which action columns are present
            action_cols = {
                i: f"action_{i}" for i in ACTION_LABELS if f"action_{i}" in df.columns
            }

            if not action_cols:
                continue

            # Total actions per episode (sum across all present action columns)
            totals = sum(df[col] for col in action_cols.values())

            for idx, label in ACTION_LABELS.items():
                col = f"action_{idx}"
                if col in df.columns:
                    # Episode-level percentage (0 when total is 0)
                    episode_pcts = np.where(
                        totals > 0, df[col].values / totals.values * 100, 0.0
                    )
                    per_episode[label].extend(episode_pcts.tolist())
                # If column missing, leave list empty → will show NaN

        results[env_name] = {
            label: bootstrap_ci(per_episode[label]) for label in ACTION_LABELS.values()
        }

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_results(results):
    """Pretty-print results table."""
    env_names = sorted(results.keys())

    sep = "-" * 72

    print()
    print("=" * 72)
    print("  ACTION PERCENTAGES  (mean ± 95% CI)")
    print("=" * 72)

    for env_name in env_names:
        print()
        print(f"  Environment: {env_name}")
        print(sep)
        print(
            f"  {'Action':<10}  {'Mean %':>8}  {'CI lower':>10}  {'CI upper':>10}  {'±':>6}"
        )
        print(sep)

        action_stats = results[env_name]
        for label in ACTION_LABELS.values():
            mean, lo_err, hi_err = action_stats[label]
            if np.isnan(mean):
                print(f"  {label:<10}  {'N/A':>8}")
            else:
                ci_lo = mean - lo_err
                ci_hi = mean + hi_err
                sym_err = (lo_err + hi_err) / 2  # approximate symmetric CI half-width
                print(
                    f"  {label:<10}  {mean:>7.2f}%"
                    f"  [{ci_lo:>7.2f}%,  {ci_hi:>7.2f}%]"
                    f"  ±{sym_err:>5.2f}%"
                )
        print(sep)

    print()

    # ------------------------------------------------------------------
    # LaTeX table  (rows = environments, columns = action types)
    # ------------------------------------------------------------------
    action_labels = list(ACTION_LABELS.values())  # [mf, short, medium, long]
    col_format = "l" + "c" * len(action_labels)

    print("\n\nLaTeX table format:")
    print("% Requires: \\usepackage{booktabs}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\normalsize")
    print("\\setlength{\\tabcolsep}{8pt}")
    print("\\begin{tabular}{" + col_format + "}")
    print("\\toprule")

    label_display = {"mf": "MF", "short": "Short", "medium": "Medium", "long": "Long"}
    header_cells = ["\\textbf{Env.}"] + [
        f"\\textbf{{{label_display.get(lbl, lbl.title())}}}" for lbl in action_labels
    ]
    print(" & ".join(header_cells) + " \\\\")
    print("\\midrule")

    for i, env_name in enumerate(env_names):
        action_stats = results[env_name]

        # Find the action with the highest mean to bold it
        best_label = max(
            action_labels,
            key=lambda lbl: (
                action_stats[lbl][0] if not np.isnan(action_stats[lbl][0]) else -np.inf
            ),
        )
        best_mean, best_lo, best_hi = action_stats[best_label]
        best_ci_lo = best_mean - best_lo
        best_ci_hi = best_mean + best_hi

        def _mean_cell(lbl):
            mean, lo_err, hi_err = action_stats[lbl]
            if np.isnan(mean):
                return "$-$"
            ci_lo = mean - lo_err
            ci_hi = mean + hi_err
            overlaps_best = ci_hi >= best_ci_lo and ci_lo <= best_ci_hi
            if overlaps_best:
                return f"$\\mathbf{{{mean:.1f}}}$"
            return f"${mean:.1f}$"

        def _ci_cell(lbl):
            _, lo_err, hi_err = action_stats[lbl]
            if np.isnan(lo_err):
                return ""
            return f"\\scriptsize$[{lo_err:.1f},\\,{hi_err:.1f}]$"

        env_display_name = ENV_DISPLAY.get(env_name, env_name.replace("_", " ").title())
        mean_row = [env_display_name] + [_mean_cell(lbl) for lbl in action_labels]
        ci_row = [""] + [_ci_cell(lbl) for lbl in action_labels]

        print(" & ".join(mean_row) + " \\\\")
        print(" & ".join(ci_row) + " \\\\")
        if i < len(env_names) - 1:
            print("\\addlinespace")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        "\\caption{Action-type percentages with 95\\% bootstrap uncertainty intervals"
    )
    print("         (margins shown below each mean). Bold entries overlap in CI with")
    print("         the most frequent action per environment.}")
    print("\\label{tab:action_percentages}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def get_action_percentages(experiment_folders):
    print(f"\nLoading Main method runs from {len(experiment_folders)} folder(s):")
    env_groups = group_runs_by_env(experiment_folders)

    if not env_groups:
        print("No Main method evaluation data found!")
        return

    results = compute_action_percentages_by_env(env_groups)
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print action-type percentages (mf, short, medium, long) for the Main "
            "method runs, grouped by environment, with 95% bootstrap CIs."
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
    get_action_percentages(folders)
