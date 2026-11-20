"""
Plot action-percentage results for environment-level ablations.

Loads from the same folder list as ``get_environment_ablation_results.py`` and
produces:

1. **Grid plot** (one per ablation type: cost / ID):
   Rows = environments, Columns = hyperparam values.
   Each cell shows a 4-bar chart (mf / short / medium / long) with 95 % CI
   error bars.

2. **Averaged plot** (one per ablation type):
   All environments pooled. X-axis groups = hyperparam values, bars within
   each group = action types. 95 % bootstrap CI from pooled per-episode data.

3. **LaTeX table** (per ablation type):
   Rows = hyperparam values, Columns = action types. Printed to stdout.
   Bold entries overlap in CI with the highest percentage for that row.

All plots saved to ``<first_folder>/plots/`` unless overridden.

Usage::

    .venv/bin/python when2plan/plotting/plot_env_ablation_results.py

Or with explicit paths::

    .venv/bin/python when2plan/plotting/plot_env_ablation_results.py \\
        --experiment_paths outputs/when2plan_maze_cost_ablation \\
                           outputs/when2plan_maze_id_ablation
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Plot style  (mirrors plot_evaluation_results.py)
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 15,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "text.usetex": False,
    }
)

# ---------------------------------------------------------------------------
# Plot geometry & typography  – tune everything here to adjust all plots
# ---------------------------------------------------------------------------
PLOT_CFG: dict = {
    # Shared
    "dpi": 600,  # PNG save resolution for both plot types
    "spine_lw": 0.8,  # axis spine line width
    "grid_bg_alpha": 0.25,  # background grid line transparency
    "grid_bg_lw": 0.5,  # background grid line width
    # Grid plot (rows = envs, cols = param values)
    "grid_bar_w": 0.55,  # bar width inside each grid cell
    "grid_cell_w": 1.6,  # inches allocated per column
    "grid_cell_h": 1.5,  # inches allocated per row
    "grid_fig_w_margin": 1.8,  # extra fig width for row labels
    "grid_fig_h_margin": 1.5,  # extra fig height for legend
    "grid_bar_edge_lw": 0.6,  # bar border line width
    "grid_capsize": 3,  # error bar cap width (pts)
    "grid_elinewidth": 1.0,  # error bar line width
    "grid_capthick": 1.0,  # error bar cap thickness
    "grid_error_alpha": 0.75,  # error bar transparency
    "grid_tick_fs": 7,  # tick label font size inside each cell
    "grid_env_label_fs": 9,  # row (env name) label font size
    "grid_col_title_fs": 9,  # column (param value) title font size
    "grid_suplabel_fs": 11,  # supxlabel / supylabel font size
    "grid_legend_fs": 9,  # legend font size
    "grid_legend_patch_lw": 0.6,  # legend colour-patch border thickness
    "grid_tight_pad": 0.8,  # tight_layout overall pad
    "grid_tight_h_pad": 0.6,  # tight_layout row padding
    "grid_tight_w_pad": 0.4,  # tight_layout column padding
    # Averaged plot (grouped bars, one group per param value)
    "avg_bar_w": 0.15,  # width of each individual bar
    "avg_group_gap": 0.25,  # gap between param-value groups
    "avg_fig_h": 4.0,  # figure height (inches)
    "avg_min_fig_w": 6.0,  # minimum figure width (inches) – increase to expand axes
    "avg_bar_edge_lw": 0.7,  # bar border line width
    "avg_capsize": 4,  # error bar cap width (pts)
    "avg_elinewidth": 1.2,  # error bar line width
    "avg_capthick": 1.2,  # error bar cap thickness
    "avg_error_alpha": 0.8,  # error bar transparency
    "avg_xtick_fs": 15,  # x-tick (param value) font size
    "avg_label_fs": 18,  # axis label font size
    "avg_legend_fs": 14,  # legend font size
    "avg_legend_frame_lw": 0.7,  # legend frame line width
    "avg_tight_pad": 0.4,  # tight_layout pad
    "avg_bottom_adjust": 0.22,  # subplots_adjust bottom (room for legend)
    # Combined averaged plot (all ablation types side-by-side)
    "combined_fig_h": 4.0,  # figure height (inches)
    "combined_bottom_adjust": 0.22,  # subplots_adjust bottom (room for shared legend)
}

# Action colors (consistent across all plots)
ACTION_COLORS = {
    "reactive": "#ED9EE6",  # pastel pink
    "short": "#80B1D3",  # pastel blue
    "medium": "#B3DE69",  # pastel green
    "long": "#FDB462",  # pastel orange
}
ACTION_LABELS = {0: "reactive", 1: "short", 2: "medium", 3: "long"}
ACTION_DISPLAY = {
    "reactive": "Reactive",
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
}

N_BOOTSTRAP = 10_000
CI_LEVEL = 95

# ---------------------------------------------------------------------------
# *** Edit this list to set the folders used when running without CLI args ***
# ---------------------------------------------------------------------------
EXPERIMENT_FOLDERS: list[str] = [
    "outputs/when2plan_navigation_cost_ablation",
    "outputs/when2plan_navigation_id_ablation",
    "outputs/when2plan_box_push_cost_ablation",
    "outputs/when2plan_box_push_id_ablation",
    "outputs/when2plan_doorkey_cost_ablation",
    "outputs/when2plan_doorkey_id_ablation",
    "outputs/when2plan_maze_cost_ablation",
    "outputs/when2plan_maze_id_ablation",
    "outputs/when2plan_fetch_cost_ablation",
    "outputs/when2plan_fetch_id_ablation",
    "outputs/action_noise_exp",
]

DEFAULT_PLAN_COST = 0.5
DEFAULT_NUM_CACHED = 20

# Folders that hold the standard (main method) runs – used to supply the
# default plan_cost_scaling_factor=0.5 point that is missing from the cost
# ablation sweep.
MAIN_METHOD_FOLDERS: list[str] = [
    "outputs/chtc_results_nav_ablation",
    "outputs/chtc_results_box_push_ablation",
    "outputs/chtc_results_doorkey_ablation",
    "outputs/chtc_results_maze_ablation",
    "outputs/chtc_results_fetch_ablation",
]
# Ablation key that identifies the main method runs inside those folders.
MAIN_METHOD_ABLATION = "no_option_duration"

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
PARAM_NOISE = (
    "action_noise",
    "action_noise",
    "action_noise",
    0.0,
)

ENV_DISPLAY = {
    "fetch_pick_place": "Fetch",
    "doorkey": "DoorKey",
    "box_push": "Box Push",
    "navigation": "Navigation",
    "maze": "Maze",
}

PARAM_DISPLAY = {
    "plan_cost_scaling_factor": "Plan Cost (α)",
    "num_cached_tasks": "Percent ID Tasks",
    "action_noise": "Action Noise (ε)",
}

# Order of panels in the combined averaged plot (missing keys are appended at end)
COMBINED_PARAM_ORDER = ["num_cached_tasks", "plan_cost_scaling_factor", "action_noise"]

# Number of fixed in-distribution tasks used to convert num_cached_tasks -> % ID
NUM_ID_TASKS = 10


def format_param_val(param_key: str, val: float, latex: bool = False) -> str:
    """Return a human-readable string for a parameter value."""
    if param_key == "num_cached_tasks":
        pct = int(round(NUM_ID_TASKS / val * 100))
        return f"{pct}\\%" if latex else f"{pct}%"
    if param_key == "action_noise":
        return f"{val:.1f}"
    return f"{int(val)}" if val == int(val) else f"{val:.2g}"


# ---------------------------------------------------------------------------
# I/O helpers  (same as get_environment_ablation_results.py)
# ---------------------------------------------------------------------------


def discover_subdirs(experiment_path: str) -> list[str]:
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


def _get_nested(config: dict, dotted_key: str):
    parts = dotted_key.split(".")
    obj = config
    for part in parts:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def get_param_value(config: dict, overrides: dict, param_tuple: tuple) -> float:
    cfg_key, ovr_key, _, default = param_tuple
    if ovr_key in overrides:
        try:
            return float(overrides[ovr_key])
        except (ValueError, TypeError):
            pass
    val = _get_nested(config, cfg_key)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return float(default)


def detect_varying_param(run_data: list[tuple]) -> tuple | None:
    for param in (PARAM_COST, PARAM_ID, PARAM_NOISE):
        values = {get_param_value(d[1], d[2], param) for d in run_data}
        if len(values) > 1:
            return param
    return None


# ---------------------------------------------------------------------------
# Data loading: returns nested dict
#   data[ablation_type][param_value][env_name] = list[pd.DataFrame]
# ---------------------------------------------------------------------------


def load_all_folders(experiment_folders: list[str]) -> dict:
    """
    Load all run data and organise as:
        data[ablation_key][param_value][env_name] = [df, ...]
    Also returns param metadata:
        meta[ablation_key] = param_tuple
    """
    data: dict[str, dict[float, dict[str, list[pd.DataFrame]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    meta: dict[str, tuple] = {}

    for folder in experiment_folders:
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            print(f"  [WARNING] Folder not found, skipping: {folder}")
            continue

        subdirs = discover_subdirs(folder)
        run_data = [r for s in subdirs if (r := load_run(s)) is not None]

        if not run_data:
            print(f"  [WARNING] No valid runs in {folder}")
            continue

        param = detect_varying_param(run_data)
        if param is None:
            print(f"  [WARNING] No variation detected in {folder}, skipping.")
            continue

        _, _, param_key, _ = param
        meta[param_key] = param

        for df, config, overrides in run_data:
            env = get_env_name(config, overrides)
            val = get_param_value(config, overrides, param)
            data[param_key][val][env].append(df)

        print(f"  {folder}  →  {len(run_data)} run(s), param={param_key}")

    return data, meta


def load_main_method_runs(
    main_folders: list[str],
) -> dict[str, list[pd.DataFrame]]:
    """
    Load runs from the standard ablation folders where the ablation key equals
    MAIN_METHOD_ABLATION ("no_option_duration").  Returns
    ``{env_name: [df, ...]}``.
    """
    result: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for folder in main_folders:
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            print(f"  [WARNING] Main method folder not found, skipping: {folder}")
            continue
        subdirs = discover_subdirs(folder)
        for subdir in subdirs:
            r = load_run(subdir)
            if r is None:
                continue
            df, config, overrides = r
            raw = overrides.get("ablation") or config.get("ablation")
            if str(raw) != MAIN_METHOD_ABLATION:
                continue
            env = get_env_name(config, overrides)
            result[env].append(df)
    total = sum(len(v) for v in result.values())
    print(
        f"  Loaded {total} main method run(s) across {len(result)} env(s) for default cost value"
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_error, upper_error)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0, 0.0
    rng = np.random.RandomState(seed)
    boots = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)
    alpha = (100 - ci) / 2
    return (
        mean,
        mean - float(np.percentile(boots, alpha)),
        float(np.percentile(boots, 100 - alpha)) - mean,
    )


def action_pct_from_dfs(dfs: list[pd.DataFrame]) -> dict[str, list[float]]:
    """Collect per-episode action percentage vectors from a list of DataFrames."""
    per_ep: dict[str, list[float]] = {lbl: [] for lbl in ACTION_LABELS.values()}
    for df in dfs:
        acols = {i: f"action_{i}" for i in ACTION_LABELS if f"action_{i}" in df.columns}
        if not acols:
            continue
        totals = sum(df[c] for c in acols.values())
        for idx, lbl in ACTION_LABELS.items():
            col = f"action_{idx}"
            if col in df.columns:
                pcts = np.where(totals > 0, df[col].values / totals.values * 100.0, 0.0)
                per_ep[lbl].extend(pcts.tolist())
    return per_ep


def stats_for_cell(dfs: list[pd.DataFrame]) -> dict[str, tuple]:
    """Return {action_label: (mean, lo_err, hi_err)} for a group of DataFrames."""
    per_ep = action_pct_from_dfs(dfs)
    return {lbl: bootstrap_ci(per_ep[lbl]) for lbl in ACTION_LABELS.values()}


def stats_averaged_over_envs(
    param_val_env_data: dict[float, dict[str, list[pd.DataFrame]]],
) -> dict[float, dict[str, tuple]]:
    """
    For each param_value, pool per-episode percentages across all environments
    and bootstrap the mean.  Returns {param_val: {action_lbl: (mean, lo, hi)}}.
    """
    result = {}
    for val, env_dict in param_val_env_data.items():
        all_dfs = [df for dfs in env_dict.values() for df in dfs]
        per_ep = action_pct_from_dfs(all_dfs)
        result[val] = {lbl: bootstrap_ci(per_ep[lbl]) for lbl in ACTION_LABELS.values()}
    return result


# ---------------------------------------------------------------------------
# Stats computation + cache helpers
# ---------------------------------------------------------------------------


def compute_all_stats(data: dict) -> dict:
    """
    Pre-compute all bootstrap statistics needed for every plot and table.

    Returns::

        {
          param_key: {
            "grid":     {param_val: {env_name: {lbl: (mean, lo, hi)}}},
            "averaged": {param_val: {lbl: (mean, lo, hi)}},
          }
        }
    """
    cache: dict = {}
    for param_key, data_by_param in data.items():
        print(f"  Computing bootstrap CIs for {param_key} …")
        grid: dict = {}
        for val, env_dict in data_by_param.items():
            grid[val] = {}
            for env_name, dfs in env_dict.items():
                cell = stats_for_cell(dfs)
                grid[val][env_name] = {lbl: tuple(t) for lbl, t in cell.items()}
        averaged: dict = {}
        for val, lbl_dict in stats_averaged_over_envs(data_by_param).items():
            averaged[val] = {lbl: tuple(t) for lbl, t in lbl_dict.items()}
        cache[param_key] = {"grid": grid, "averaged": averaged}
    return cache


def save_stats_cache(cache: dict, path: str) -> None:
    """Serialise *cache* (float keys → string keys) to a JSON file."""

    def _serial(obj):
        if isinstance(obj, (float, int)) and not isinstance(obj, bool):
            return obj
        return obj  # tuples → lists automatically via json.dumps

    serializable: dict = {}
    for param_key, pdata in cache.items():
        serializable[param_key] = {
            "grid": {
                str(val): {
                    env: {lbl: list(tup) for lbl, tup in lbl_dict.items()}
                    for env, lbl_dict in env_dict.items()
                }
                for val, env_dict in pdata["grid"].items()
            },
            "averaged": {
                str(val): {lbl: list(tup) for lbl, tup in lbl_dict.items()}
                for val, lbl_dict in pdata["averaged"].items()
            },
        }
    with open(path, "w") as fh:
        json.dump(serializable, fh, indent=2)
    print(f"  Saved stats cache → {path}")


def load_stats_cache(path: str) -> dict:
    """Load a JSON cache file and restore float keys + tuple values."""
    with open(path) as fh:
        raw = json.load(fh)
    cache: dict = {}
    for param_key, pdata in raw.items():
        cache[param_key] = {
            "grid": {
                float(val): {
                    env: {lbl: tuple(tup) for lbl, tup in lbl_dict.items()}
                    for env, lbl_dict in env_dict.items()
                }
                for val, env_dict in pdata["grid"].items()
            },
            "averaged": {
                float(val): {lbl: tuple(tup) for lbl, tup in lbl_dict.items()}
                for val, lbl_dict in pdata["averaged"].items()
            },
        }
    print(f"  Loaded stats cache from {path}")
    return cache


# ---------------------------------------------------------------------------
# Shared axis styling helper
# ---------------------------------------------------------------------------


def _sorted_param_vals(param_key: str, vals) -> list:
    """Sort parameter values so the displayed axis increases left to right.

    For ``num_cached_tasks`` the display value is ``NUM_ID_TASKS / val * 100``,
    which *decreases* as val increases, so we reverse-sort by val to get an
    increasing percentage axis (100 % on the right).
    """
    reverse = param_key == "num_cached_tasks"
    return sorted(vals, reverse=reverse)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(
        axis="y",
        alpha=PLOT_CFG["grid_bg_alpha"],
        linestyle="--",
        linewidth=PLOT_CFG["grid_bg_lw"],
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(PLOT_CFG["spine_lw"])
    ax.spines["bottom"].set_linewidth(PLOT_CFG["spine_lw"])


# ---------------------------------------------------------------------------
# Plot 1 – Grid plot: rows = envs, cols = param values
# ---------------------------------------------------------------------------


def plot_grid(
    grid_stats: dict[float, dict[str, dict[str, tuple]]],
    param_key: str,
    output_dir: str,
) -> None:
    """
    Grid of mini bar charts: rows = environments, columns = param values.
    Each cell shows action percentages (reactive/short/medium/long) with 95 % CIs.

    Args:
        grid_stats: pre-computed {param_val: {env_name: {action_lbl: (mean, lo, hi)}}}
    """
    param_vals = _sorted_param_vals(param_key, grid_stats.keys())
    # Collect all env names across all param values, keep consistent order
    all_envs: list[str] = []
    seen: set[str] = set()
    for val in param_vals:
        for env in sorted(grid_stats[val].keys()):
            if env not in seen:
                all_envs.append(env)
                seen.add(env)

    n_rows = len(all_envs)
    n_cols = len(param_vals)

    action_lbls = list(ACTION_LABELS.values())
    bar_x = np.arange(len(action_lbls))
    bar_w = PLOT_CFG["grid_bar_w"]

    cell_w = PLOT_CFG["grid_cell_w"]
    cell_h = PLOT_CFG["grid_cell_h"]
    fig_w = max(5, n_cols * cell_w + PLOT_CFG["grid_fig_w_margin"])
    fig_h = max(4, n_rows * cell_h + PLOT_CFG["grid_fig_h_margin"])

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        sharey="row",
        constrained_layout=False,
    )

    # Ensure axes is always 2-D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    param_display = PARAM_DISPLAY.get(param_key, param_key)

    for row_idx, env_name in enumerate(all_envs):
        env_disp = ENV_DISPLAY.get(env_name, env_name.replace("_", " ").title())

        for col_idx, val in enumerate(param_vals):
            ax = axes[row_idx, col_idx]
            cell_stats = grid_stats[val].get(
                env_name,
                {lbl: (np.nan, np.nan, np.nan) for lbl in action_lbls},
            )

            means = [cell_stats[lbl][0] for lbl in action_lbls]
            lo_err = [cell_stats[lbl][1] for lbl in action_lbls]
            hi_err = [cell_stats[lbl][2] for lbl in action_lbls]
            colors = [ACTION_COLORS[lbl] for lbl in action_lbls]

            bars = ax.bar(
                bar_x,
                means,
                bar_w,
                color=colors,
                edgecolor="black",
                linewidth=PLOT_CFG["grid_bar_edge_lw"],
                yerr=[lo_err, hi_err],
                capsize=PLOT_CFG["grid_capsize"],
                error_kw={
                    "elinewidth": PLOT_CFG["grid_elinewidth"],
                    "capthick": PLOT_CFG["grid_capthick"],
                    "ecolor": "black",
                    "alpha": PLOT_CFG["grid_error_alpha"],
                },
                zorder=3,
            )

            ax.set_ylim(0, 105)
            ax.set_xticks(bar_x)
            ax.set_xticklabels(
                [ACTION_DISPLAY[lbl] for lbl in action_lbls],
                fontsize=PLOT_CFG["grid_tick_fs"],
            )
            ax.tick_params(axis="y", labelsize=PLOT_CFG["grid_tick_fs"])

            _style_ax(ax)

            # Row label (env name) on leftmost column
            if col_idx == 0:
                ax.set_ylabel(
                    env_disp,
                    fontsize=PLOT_CFG["grid_env_label_fs"],
                    fontweight="medium",
                )

            # Column label (param value) on top row
            if row_idx == 0:
                ax.set_title(
                    format_param_val(param_key, val),
                    fontsize=PLOT_CFG["grid_col_title_fs"],
                )

            # y-axis label only on leftmost
            if col_idx != 0:
                ax.set_ylabel("")

    # Super-labels
    fig.supylabel(
        "Action %", fontsize=PLOT_CFG["grid_suplabel_fs"], fontweight="medium", x=0.01
    )
    fig.supxlabel(
        param_display,
        fontsize=PLOT_CFG["grid_suplabel_fs"],
        fontweight="medium",
        y=0.01,
    )

    # Legend (action types) at bottom
    legend_handles = [
        mpatches.Patch(
            facecolor=ACTION_COLORS[lbl],
            edgecolor="black",
            linewidth=PLOT_CFG["grid_legend_patch_lw"],
            label=ACTION_DISPLAY[lbl],
        )
        for lbl in action_lbls
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(action_lbls),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=PLOT_CFG["grid_legend_fs"],
    )

    # (Removed: fig.suptitle)

    plt.tight_layout(
        pad=PLOT_CFG["grid_tight_pad"],
        h_pad=PLOT_CFG["grid_tight_h_pad"],
        w_pad=PLOT_CFG["grid_tight_w_pad"],
    )

    safe_key = param_key.replace(".", "_")
    png_path = os.path.join(output_dir, f"env_ablation_grid_{safe_key}.png")
    pdf_path = os.path.join(output_dir, f"env_ablation_grid_{safe_key}.pdf")
    plt.savefig(png_path, dpi=PLOT_CFG["dpi"], bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  Saved grid plot  →  {png_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2 – Averaged across envs: grouped bars per param value
# ---------------------------------------------------------------------------


def plot_averaged(
    avg_stats: dict[float, dict[str, tuple]],
    param_key: str,
    output_dir: str,
) -> None:
    """
    Grouped bar chart: x-groups = param values, bars = action types.
    Percentages pooled (per-episode) across all environments.

    Args:
        avg_stats: pre-computed {param_val: {action_lbl: (mean, lo, hi)}}
    """
    param_vals = _sorted_param_vals(param_key, avg_stats.keys())
    action_lbls = list(ACTION_LABELS.values())
    n_params = len(param_vals)
    n_actions = len(action_lbls)

    bar_w = PLOT_CFG["avg_bar_w"]
    group_gap = PLOT_CFG["avg_group_gap"]
    group_width = n_actions * bar_w

    param_display = PARAM_DISPLAY.get(param_key, param_key)

    fig_w = max(PLOT_CFG["avg_min_fig_w"], n_params * (group_width + group_gap) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, PLOT_CFG["avg_fig_h"]), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.05)

    for a_idx, lbl in enumerate(action_lbls):
        positions = []
        means, lo_errs, hi_errs = [], [], []
        for p_idx, val in enumerate(param_vals):
            group_center = p_idx * (group_width + group_gap)
            pos = group_center + a_idx * bar_w
            positions.append(pos)
            m, lo, hi = avg_stats[val][lbl]
            means.append(m)
            lo_errs.append(lo)
            hi_errs.append(hi)

        ax.bar(
            positions,
            means,
            bar_w,
            label=ACTION_DISPLAY[lbl],
            color=ACTION_COLORS[lbl],
            edgecolor="black",
            linewidth=PLOT_CFG["avg_bar_edge_lw"],
            yerr=[lo_errs, hi_errs],
            capsize=PLOT_CFG["avg_capsize"],
            error_kw={
                "elinewidth": PLOT_CFG["avg_elinewidth"],
                "capthick": PLOT_CFG["avg_capthick"],
                "ecolor": "black",
                "alpha": PLOT_CFG["avg_error_alpha"],
            },
            zorder=3,
        )

    # X-tick labels (param values) centred under each group
    group_centers = [
        p_idx * (group_width + group_gap) + (n_actions - 1) * bar_w / 2
        for p_idx in range(n_params)
    ]
    val_labels = [format_param_val(param_key, val) for val in param_vals]

    ax.set_xticks(group_centers)
    ax.set_xticklabels(val_labels, fontsize=PLOT_CFG["avg_xtick_fs"])
    ax.set_xlabel(param_display, fontsize=PLOT_CFG["avg_label_fs"], fontweight="medium")
    ax.set_ylabel("Action %", fontsize=PLOT_CFG["avg_label_fs"], fontweight="medium")
    ax.set_ylim(0, 105)

    # Tighten x-axis to actual bar extent
    x_pad = bar_w * 0.8
    x_right = (n_params - 1) * (group_width + group_gap) + (n_actions - 1) * bar_w
    ax.set_xlim(-x_pad, x_right + bar_w + x_pad)
    # (Removed: ax.set_title)

    _style_ax(ax)

    legend = fig.legend(
        loc="outside lower center",
        ncol=n_actions,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=PLOT_CFG["avg_legend_fs"],
    )
    legend.get_frame().set_linewidth(PLOT_CFG["avg_legend_frame_lw"])

    safe_key = param_key.replace(".", "_")
    png_path = os.path.join(output_dir, f"env_ablation_avg_{safe_key}.png")
    pdf_path = os.path.join(output_dir, f"env_ablation_avg_{safe_key}.pdf")
    plt.savefig(png_path, dpi=PLOT_CFG["dpi"], bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  Saved averaged plot  →  {png_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3 – Combined averaged: all ablation types side-by-side, one legend
# ---------------------------------------------------------------------------


def plot_combined_averaged(all_stats: dict, output_dir: str) -> None:
    """
    Single figure with one panel per ablation type, arranged side-by-side in the
    order defined by COMBINED_PARAM_ORDER.  A single shared legend is placed at
    the bottom so the individual panels stay uncluttered.
    """
    # Build ordered panel list; any key not in COMBINED_PARAM_ORDER goes at end
    ordered_keys = [k for k in COMBINED_PARAM_ORDER if k in all_stats]
    ordered_keys += [k for k in all_stats if k not in ordered_keys]
    if not ordered_keys:
        return

    action_lbls = list(ACTION_LABELS.values())
    n_actions = len(action_lbls)
    bar_w = PLOT_CFG["avg_bar_w"]
    group_gap = PLOT_CFG["avg_group_gap"]
    group_width = n_actions * bar_w

    # Compute natural width for each panel based on how many param values it has
    panel_widths = [
        max(
            PLOT_CFG["avg_min_fig_w"] / max(1, len(ordered_keys)),
            len(all_stats[k]["averaged"]) * (group_width + group_gap) + 0.5,
        )
        for k in ordered_keys
    ]
    fig_w = sum(panel_widths) + 0.5 * (len(ordered_keys) - 1)
    fig_h = PLOT_CFG["combined_fig_h"]

    fig, axes = plt.subplots(
        1,
        len(ordered_keys),
        figsize=(fig_w, fig_h),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": panel_widths},
    )
    if len(ordered_keys) == 1:
        axes = [axes]
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.05)

    for panel_idx, (ax, param_key) in enumerate(zip(axes, ordered_keys)):
        avg_stats = all_stats[param_key]["averaged"]
        param_vals = _sorted_param_vals(param_key, avg_stats.keys())
        n_params = len(param_vals)

        for a_idx, lbl in enumerate(action_lbls):
            positions = [
                p_idx * (group_width + group_gap) + a_idx * bar_w
                for p_idx in range(n_params)
            ]
            means = [avg_stats[v][lbl][0] for v in param_vals]
            lo_errs = [avg_stats[v][lbl][1] for v in param_vals]
            hi_errs = [avg_stats[v][lbl][2] for v in param_vals]

            ax.bar(
                positions,
                means,
                bar_w,
                label=ACTION_DISPLAY[lbl],
                color=ACTION_COLORS[lbl],
                edgecolor="black",
                linewidth=PLOT_CFG["avg_bar_edge_lw"],
                yerr=[lo_errs, hi_errs],
                capsize=PLOT_CFG["avg_capsize"],
                error_kw={
                    "elinewidth": PLOT_CFG["avg_elinewidth"],
                    "capthick": PLOT_CFG["avg_capthick"],
                    "ecolor": "black",
                    "alpha": PLOT_CFG["avg_error_alpha"],
                },
                zorder=3,
            )

        group_centers = [
            p_idx * (group_width + group_gap) + (n_actions - 1) * bar_w / 2
            for p_idx in range(n_params)
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [format_param_val(param_key, v) for v in param_vals],
            fontsize=PLOT_CFG["avg_xtick_fs"],
        )
        ax.set_xlabel(
            PARAM_DISPLAY.get(param_key, param_key),
            fontsize=PLOT_CFG["avg_label_fs"],
            fontweight="medium",
        )
        ax.set_ylim(0, 105)

        # Tighten x-axis to actual bar extent
        x_pad = bar_w * 0.8
        x_right = (n_params - 1) * (group_width + group_gap) + (n_actions - 1) * bar_w
        ax.set_xlim(-x_pad, x_right + bar_w + x_pad)

        if panel_idx == 0:
            ax.set_ylabel(
                "Action %", fontsize=PLOT_CFG["avg_label_fs"], fontweight="medium"
            )

        _style_ax(ax)

    # Single legend shared across all panels
    handles, _ = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles=handles,
        labels=[ACTION_DISPLAY[lbl] for lbl in action_lbls],
        loc="outside lower center",
        ncol=n_actions,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=PLOT_CFG["avg_legend_fs"],
    )
    leg.get_frame().set_linewidth(PLOT_CFG["avg_legend_frame_lw"])

    png_path = os.path.join(output_dir, "env_ablation_combined_avg.png")
    pdf_path = os.path.join(output_dir, "env_ablation_combined_avg.pdf")
    plt.savefig(png_path, dpi=PLOT_CFG["dpi"], bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  Saved combined averaged plot  \u2192  {png_path}")
    plt.close()


# ---------------------------------------------------------------------------
# LaTeX table  (rows = param values, cols = action types)
# ---------------------------------------------------------------------------


def print_latex_table(
    avg_stats: dict[float, dict[str, tuple]],
    param_key: str,
) -> None:
    param_vals = _sorted_param_vals(param_key, avg_stats.keys())
    action_lbls = list(ACTION_LABELS.values())

    param_display = PARAM_DISPLAY.get(param_key, param_key)
    col_fmt = "r" + "c" * len(action_lbls)

    print(f"\n% ===== LaTeX table: {param_display} =====")
    print("% Requires: \\usepackage{booktabs}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\normalsize")
    print("\\setlength{\\tabcolsep}{8pt}")
    print("\\begin{tabular}{" + col_fmt + "}")
    print("\\toprule")

    header = [f"\\textbf{{{param_display}}}"] + [
        f"\\textbf{{{ACTION_DISPLAY[lbl]}}}" for lbl in action_lbls
    ]
    print(" & ".join(header) + " \\\\")
    print("\\midrule")

    for i, val in enumerate(param_vals):
        val_stats = avg_stats[val]

        # Best action (highest mean)
        best_lbl = max(
            action_lbls,
            key=lambda lbl: (
                val_stats[lbl][0] if not np.isnan(val_stats[lbl][0]) else -np.inf
            ),
        )
        bm, blo, bhi = val_stats[best_lbl]
        best_ci_lo, best_ci_hi = bm - blo, bm + bhi

        def _mean_cell(lbl: str) -> str:
            mean, lo, hi = val_stats[lbl]
            if np.isnan(mean):
                return "$-$"
            ci_lo, ci_hi = mean - lo, mean + hi
            bold = ci_hi >= best_ci_lo and ci_lo <= best_ci_hi
            return f"$\\mathbf{{{mean:.1f}}}$" if bold else f"${mean:.1f}$"

        def _ci_cell(lbl: str) -> str:
            _, lo, hi = val_stats[lbl]
            if np.isnan(lo):
                return ""
            return f"\\scriptsize$[{lo:.1f},\\,{hi:.1f}]$"

        val_str = format_param_val(param_key, val, latex=True)
        mean_row = [val_str] + [_mean_cell(lbl) for lbl in action_lbls]
        ci_row = [""] + [_ci_cell(lbl) for lbl in action_lbls]

        print(" & ".join(mean_row) + " \\\\")
        print(" & ".join(ci_row) + " \\\\")
        if i < len(param_vals) - 1:
            print("\\addlinespace")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        f"\\caption{{Action-type percentages (averaged across environments) vs. "
        f"\\texttt{{{param_key}}} with 95\\% bootstrap CIs.}}"
    )
    print(f"\\label{{tab:avg_action_pct_{param_key.replace('.', '_')}}}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(experiment_folders: list[str], cache_file: str | None = None) -> None:
    # ── fast path: load pre-computed stats from cache ────────────────────
    if cache_file and os.path.isfile(cache_file):
        print(f"\nLoading pre-computed stats from cache: {cache_file}")
        all_stats = load_stats_cache(cache_file)

        # Derive output dir from the cache file location
        output_dir = os.path.join(os.path.dirname(os.path.abspath(cache_file)), "plots")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to: {output_dir}\n")

        for param_key, pdata in all_stats.items():
            print(f"--- {param_key} ---")
            plot_grid(pdata["grid"], param_key, output_dir)
            plot_averaged(pdata["averaged"], param_key, output_dir)
            print_latex_table(pdata["averaged"], param_key)

        plot_combined_averaged(all_stats, output_dir)
        print("\nDone.")
        return

    # ── slow path: load raw data and compute stats ────────────────────────
    print(f"\nLoading data from {len(experiment_folders)} folder(s):")
    data, meta = load_all_folders(experiment_folders)

    if not data:
        print("No valid ablation data found!")
        return

    # Determine output directory: first valid folder's plots/ subdir
    first_valid = next(
        (
            os.path.expanduser(f)
            for f in experiment_folders
            if os.path.isdir(os.path.expanduser(f))
        ),
        ".",
    )
    output_dir = os.path.join(first_valid, "plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}\n")

    # For the cost ablation, the default value (0.5) was not part of the sweep;
    # load it from the main method folders.
    if "plan_cost_scaling_factor" in data:
        print("\nLoading main method runs for plan_cost_scaling_factor=0.5:")
        main_runs = load_main_method_runs(MAIN_METHOD_FOLDERS)
        for env, dfs in main_runs.items():
            data["plan_cost_scaling_factor"][DEFAULT_PLAN_COST][env].extend(dfs)

    print("\nRunning bootstrap CI computation (this may take a while) …")
    all_stats = compute_all_stats(data)

    # Save cache alongside the plots
    _cache_path = cache_file or os.path.join(output_dir, "stats_cache.json")
    save_stats_cache(all_stats, _cache_path)

    for param_key, pdata in all_stats.items():
        print(f"--- {param_key} ---")
        plot_grid(pdata["grid"], param_key, output_dir)
        plot_averaged(pdata["averaged"], param_key, output_dir)
        print_latex_table(pdata["averaged"], param_key)

    plot_combined_averaged(all_stats, output_dir)
    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot action-percentage ablation results (grid + averaged) "
            "and print a LaTeX table."
        )
    )
    parser.add_argument(
        "--experiment_paths",
        nargs="+",
        default=None,
        metavar="PATH",
        help="One or more experiment folders. Defaults to EXPERIMENT_FOLDERS.",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Path to a stats_cache.json file. "
            "If the file exists, skip data loading and plot directly from it. "
            "If it does not exist (or is not given), compute stats and save the "
            "cache to this path (or to <plots>/stats_cache.json)."
        ),
    )
    args = parser.parse_args()
    folders = args.experiment_paths if args.experiment_paths else EXPERIMENT_FOLDERS
    run(folders, cache_file=args.cache_file)
