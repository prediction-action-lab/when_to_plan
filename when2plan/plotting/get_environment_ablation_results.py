"""
Compute and save action-percentage statistics for environment-level ablations.

Supports two ablation types (auto-detected per folder):
  - Cost ablation     : ``plan_cost_scaling_factor`` varies across runs
  - ID/OOD ablation   : ``experiment.num_cached_tasks`` varies across runs

For each folder the script:
1. Scans all runs and reads their Hydra configs.
2. Detects which parameter is varying.
3. Groups runs by parameter value.
4. Computes per-episode action percentages (mf/short/medium/long).
5. Bootstraps 95 % uncertainty intervals on the mean of each percentage.
6. Saves a JSON summary to  ``<folder>/environment_ablation_action_percentages.json``.
7. Prints a plain-text summary and a LaTeX table to stdout.

Usage – edit EXPERIMENT_FOLDERS below and run::

    .venv/bin/python when2plan/plotting/get_environment_ablation_results.py

Or supply folders via CLI::

    .venv/bin/python when2plan/plotting/get_environment_ablation_results.py \\
        --experiment_paths outputs/when2plan_maze_cost_ablation \\
                           outputs/when2plan_maze_id_ablation
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_LABELS = {0: "mf", 1: "short", 2: "medium", 3: "long"}

N_BOOTSTRAP = 10_000
CI_LEVEL = 95  # percent

# *** Edit this list to set the folders used when running without CLI args ***
EXPERIMENT_FOLDERS: list[str] = [
    # "outputs/when2plan_navigation_cost_ablation",
    # "outputs/when2plan_navigation_id_ablation",
    "outputs/when2plan_box_push_cost_ablation",
    "outputs/when2plan_box_push_id_ablation",
    "outputs/when2plan_doorkey_cost_ablation",
    "outputs/when2plan_doorkey_id_ablation",
    "outputs/when2plan_maze_cost_ablation",
    "outputs/when2plan_maze_id_ablation",
    # "outputs/when2plan_fetch_cost_ablation",
    # "outputs/when2plan_fetch_id_ablation",
]

# Default values used when a run does NOT override the parameter.
DEFAULT_PLAN_COST = 0.5
DEFAULT_NUM_CACHED = 20

# Parameter meta-data: (config key path, override key, display name, default)
PARAM_COST = (
    "plan_cost_scaling_factor",
    "plan_cost_scaling_factor",
    "plan_cost_scaling_factor",
    DEFAULT_PLAN_COST,
)
PARAM_ID = (
    "experiment.num_cached_tasks",
    "experiment.num_cached_tasks",
    "num_cached_tasks",
    DEFAULT_NUM_CACHED,
)

ENV_DISPLAY = {
    "fetch_pick_place": "Fetch P&P",
    "doorkey": "DoorKey",
    "box_push": "Box Push",
    "navigation": "Navigation",
    "maze": "Maze",
}

OUTPUT_FILENAME = "environment_ablation_action_percentages.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def discover_subdirs(experiment_path: str) -> list[str]:
    """Return the list of run sub-directories inside an experiment folder."""
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
        and d not in ("plots", "__pycache__")
    ]
    return candidates or [experiment_path]


def load_run(run_dir: str):
    """
    Load evaluation CSV + Hydra config from *run_dir*.

    Returns ``(df, config, overrides)`` or ``None`` when data are missing.
    """
    eval_file = os.path.join(run_dir, "evaluation_results.csv")
    if not os.path.isfile(eval_file):
        return None

    df = pd.read_csv(eval_file)

    config: dict = {}
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            config = yaml.safe_load(fh) or {}

    overrides: dict = {}
    ovr_path = os.path.join(run_dir, ".hydra", "overrides.yaml")
    if os.path.isfile(ovr_path):
        with open(ovr_path) as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw, list):
            for item in raw:
                if "=" in str(item):
                    k, v = str(item).split("=", 1)
                    overrides[k] = v
        elif isinstance(raw, dict):
            overrides = raw

    return df, config, overrides


def get_env_name(config: dict, overrides: dict) -> str:
    if "env" in overrides:
        return overrides["env"]
    if "env" in config and isinstance(config["env"], dict):
        return config["env"].get("type", "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


def _get_nested(config: dict, dotted_key: str):
    """Walk *config* following a dot-separated key path."""
    parts = dotted_key.split(".")
    obj = config
    for part in parts:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def get_param_value(config: dict, overrides: dict, param_tuple: tuple):
    """
    Extract the parameter value for a single run.

    Checks overrides first (as a dot-notation string key), then config,
    then falls back to *default*.
    """
    cfg_key, ovr_key, _, default = param_tuple

    # 1. Overrides take precedence
    if ovr_key in overrides:
        try:
            return float(overrides[ovr_key])
        except (ValueError, TypeError):
            pass

    # 2. Config (potentially nested)
    val = _get_nested(config, cfg_key)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass

    return float(default)


# ---------------------------------------------------------------------------
# Auto-detection of which parameter varies in a folder
# ---------------------------------------------------------------------------


def detect_varying_param(run_data: list[tuple]) -> tuple | None:
    """
    Given a list of (df, config, overrides) tuples, return the PARAM_* tuple
    whose values vary across runs.

    Returns ``None`` when neither parameter varies (single run or constant values).
    """
    for param in (PARAM_COST, PARAM_ID):
        values = set()
        for df, config, overrides in run_data:
            values.add(get_param_value(config, overrides, param))
        if len(values) > 1:
            return param

    # No variation found – fall back to cost param as default detection hint.
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap of the mean.

    Returns ``(mean, lower_error, upper_error)`` where
    ``lower_error = mean - ci_low`` and ``upper_error = ci_high - mean``
    (both ≥ 0).
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


def compute_action_percentages(dfs: list[pd.DataFrame]) -> dict[str, tuple]:
    """
    Compute per-episode action percentages for a group of runs and bootstrap CIs.

    Returns: dict{action_label -> (mean_pct, lower_error, upper_error)}
    """
    per_episode: dict[str, list[float]] = {lbl: [] for lbl in ACTION_LABELS.values()}

    for df in dfs:
        action_cols = {
            i: f"action_{i}" for i in ACTION_LABELS if f"action_{i}" in df.columns
        }
        if not action_cols:
            continue

        totals = sum(df[col] for col in action_cols.values())
        for idx, label in ACTION_LABELS.items():
            col = f"action_{idx}"
            if col in df.columns:
                pcts = np.where(totals > 0, df[col].values / totals.values * 100, 0.0)
                per_episode[label].extend(pcts.tolist())

    return {label: bootstrap_ci(per_episode[label]) for label in ACTION_LABELS.values()}


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------


def process_folder(folder: str) -> dict | None:
    """
    Process a single experiment folder.

    Returns a summary dict ready for JSON serialisation, or ``None`` on failure.
    """
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        print(f"  [WARNING] Folder not found, skipping: {folder}")
        return None

    subdirs = discover_subdirs(folder)
    print(f"  {folder}  →  {len(subdirs)} run(s)")

    run_data = []
    for subdir in subdirs:
        result = load_run(subdir)
        if result is not None:
            run_data.append(result)

    if not run_data:
        print(f"  [WARNING] No valid runs in {folder}")
        return None

    # Detect varying parameter
    param = detect_varying_param(run_data)
    if param is None:
        _, _, param_display, _ = PARAM_COST  # default label for messages
        print(
            f"  [WARNING] No parameter variation detected in {folder}; "
            "cannot form groups. Treating all runs as a single group."
        )
        # Treat entire folder as one group with an artificial key
        groups: dict[float, list[pd.DataFrame]] = {
            get_param_value(run_data[0][1], run_data[0][2], PARAM_COST): [
                t[0] for t in run_data
            ]
        }
        param = PARAM_COST
    else:
        _, _, param_display, _ = param
        print(f"    Detected varying parameter: {param_display}")

        # Group runs by parameter value
        groups = defaultdict(list)
        for df, config, overrides in run_data:
            val = get_param_value(config, overrides, param)
            groups[val].append(df)

    # Environment name (from first run with a config)
    env_name = "unknown"
    for _, config, overrides in run_data:
        name = get_env_name(config, overrides)
        if name != "unknown":
            env_name = name
            break

    _, _, param_key, _ = param

    # Compute stats per parameter value
    param_stats: dict[str, dict[str, list]] = {}
    for val in sorted(groups.keys()):
        dfs = groups[val]
        stats = compute_action_percentages(dfs)
        param_stats[str(val)] = {label: list(triple) for label, triple in stats.items()}
        print(
            f"    {param_key}={val}: "
            + "  ".join(f"{lbl}={stats[lbl][0]:.1f}%" for lbl in ACTION_LABELS.values())
        )

    summary = {
        "folder": folder,
        "env": env_name,
        "ablation_type": param_key,
        "param_values": sorted(groups.keys()),
        "stats": param_stats,  # param_value_str -> action_label -> [mean, lo_err, hi_err]
    }
    return summary


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_folder_results(summary: dict) -> None:
    """Print plain-text + LaTeX output for a single folder's summary."""
    env_name = summary["env"]
    param_key = summary["ablation_type"]
    param_values = summary["param_values"]
    stats = summary["stats"]
    action_labels = list(ACTION_LABELS.values())

    env_display = ENV_DISPLAY.get(env_name, env_name.replace("_", " ").title())
    sep = "-" * 80

    print()
    print("=" * 80)
    print(f"  {env_display}  |  ablation: {param_key}")
    print("=" * 80)
    print(f"  {'Param':>8}  " + "  ".join(f"{'  ' + lbl:>13}" for lbl in action_labels))
    print(sep)

    for val in param_values:
        val_stats = stats[str(val)]
        parts = [f"  {val:>8.3g}  "]
        for lbl in action_labels:
            mean, lo, hi = val_stats[lbl]
            if np.isnan(mean):
                parts.append(f"{'N/A':>13}")
            else:
                parts.append(f"{mean:>6.1f}±{(lo + hi) / 2:>4.1f}%  ")
        print("".join(parts))

    print(sep)

    # LaTeX table
    col_fmt = "r" + "c" * len(action_labels)
    label_map = {"mf": "MF", "short": "Short", "medium": "Medium", "long": "Long"}

    print(f"\n% LaTeX – {env_display} | {param_key}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\normalsize")
    print("\\setlength{\\tabcolsep}{6pt}")
    print("\\begin{tabular}{" + col_fmt + "}")
    print("\\toprule")
    header = [f"\\textbf{{{param_key}}}"] + [
        f"\\textbf{{{label_map.get(lbl, lbl.title())}}}" for lbl in action_labels
    ]
    print(" & ".join(header) + " \\\\")
    print("\\midrule")

    for val in param_values:
        val_stats = stats[str(val)]

        # Bold action that overlaps with best CI
        best_lbl = max(
            action_labels,
            key=lambda lbl: (
                val_stats[lbl][0] if not np.isnan(val_stats[lbl][0]) else -np.inf
            ),
        )
        best_mean, best_lo, best_hi = val_stats[best_lbl]
        best_ci_lo = best_mean - best_lo
        best_ci_hi = best_mean + best_hi

        def _mean_cell(lbl: str) -> str:
            mean, lo_err, hi_err = val_stats[lbl]
            if np.isnan(mean):
                return "$-$"
            ci_lo = mean - lo_err
            ci_hi = mean + hi_err
            overlaps = ci_hi >= best_ci_lo and ci_lo <= best_ci_hi
            return f"$\\mathbf{{{mean:.1f}}}$" if overlaps else f"${mean:.1f}$"

        def _ci_cell(lbl: str) -> str:
            _, lo_err, hi_err = val_stats[lbl]
            if np.isnan(lo_err):
                return ""
            return f"\\scriptsize$[{lo_err:.1f},\\,{hi_err:.1f}]$"

        mean_row = [f"{val:.3g}"] + [_mean_cell(lbl) for lbl in action_labels]
        ci_row = [""] + [_ci_cell(lbl) for lbl in action_labels]
        print(" & ".join(mean_row) + " \\\\")
        print(" & ".join(ci_row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        f"\\caption{{Action-type percentages vs. \\texttt{{{param_key}}} "
        f"for {env_display} with 95\\% bootstrap CIs.}}"
    )
    print(f"\\label{{tab:env_ablation_{env_name}_{param_key}}}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(experiment_folders: list[str]) -> None:
    summaries = []

    print(f"\nProcessing {len(experiment_folders)} folder(s):\n")
    for folder in experiment_folders:
        summary = process_folder(folder)
        if summary is None:
            continue

        # Save JSON to folder
        out_path = os.path.join(os.path.expanduser(folder), OUTPUT_FILENAME)
        with open(out_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"    → Saved: {out_path}")

        summaries.append(summary)
        print_folder_results(summary)

    print(f"\nDone. Processed {len(summaries)} folder(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute action-percentage statistics for environment-level ablations "
            "(cost or ID/OOD) and save results to each folder."
        )
    )
    parser.add_argument(
        "--experiment_paths",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "One or more experiment folders. "
            "If omitted, uses EXPERIMENT_FOLDERS at the top of this file."
        ),
    )
    args = parser.parse_args()
    folders = args.experiment_paths if args.experiment_paths else EXPERIMENT_FOLDERS
    run(folders)
