"""
Plot per-rollout reward and model-free percentage from pre-computed stats files.

Reads one or more ``*_rollout_stats.csv`` files produced by
``compute_rollout_stats.py`` and generates a two-panel figure:

  Left  – Average reward vs. rollout number (one line per domain)
  Right – Model-free action % vs. rollout number (one line per domain)

Confidence bands show 95 % bootstrap CIs computed per rollout.

Usage (load all stats from a directory)::

    .venv/bin/python when2plan/plotting/plot_rollout_stats.py \\
        --stats_dir outputs/test/rollout_stats

Or pass individual files directly::

    .venv/bin/python when2plan/plotting/plot_rollout_stats.py \\
        --stats_files outputs/test/rollout_stats/maze_rollout_stats.csv \\
                      outputs/test/rollout_stats/doorkey_rollout_stats.csv

Plots are saved to ``<stats_dir>/plots/`` (or alongside the first stats file).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 23,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 23,
        "axes.titlesize": 23,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 23,
        "figure.titlesize": 23,
        "lines.linewidth": 2.1,
        "axes.linewidth": 1.15,
        "grid.linewidth": 0.6,
        "text.usetex": False,
    }
)

# One colour per domain – cycle through a colourblind-friendly palette
_PALETTE = [
    "#4878CF",  # blue
    "#6ACC65",  # green
    "#D65F5F",  # red
    "#B47CC7",  # purple
    "#C4AD66",  # tan
    "#77BEDB",  # sky blue
    "#F28E2B",  # orange
    "#59A14F",  # dark green
]

ENV_DISPLAY = {
    "fetch_pick_place": "Fetch",
    "doorkey": "DoorKey",
    "box_push": "Box Push",
    "navigation": "Navigation",
    "maze": "Maze",
}

# ---------------------------------------------------------------------------
# *** Set to False to omit the reward panel and show only model-free % ***
# ---------------------------------------------------------------------------
SHOW_REWARD_PANEL: bool = True

# ---------------------------------------------------------------------------
# *** Set to True to add a model-free actor confidence panel ***
# ---------------------------------------------------------------------------
SHOW_CONFIDENCE_PANEL: bool = True

# ---------------------------------------------------------------------------
# *** Cap the x-axis at this many environment timesteps (None = show all) ***
# ---------------------------------------------------------------------------
MAX_TIMESTEPS: int | None = 1_000_000

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _collect_stats_files(stats_dir: str) -> list[str]:
    """Return all *_rollout_stats.csv files inside stats_dir."""
    return sorted(
        os.path.join(stats_dir, f)
        for f in os.listdir(stats_dir)
        if f.endswith("_rollout_stats.csv")
    )


def _env_name_from_path(path: str) -> str:
    """Derive the environment name from the stats filename."""
    basename = os.path.basename(path)
    return basename.replace("_rollout_stats.csv", "")


def load_stats(paths: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load each CSV and return ``{env_name: DataFrame}``.

    Each DataFrame must have columns:
        rollout_number,
        reward_mean, reward_ci_lo, reward_ci_hi,
        model_free_pct_mean, model_free_pct_ci_lo, model_free_pct_ci_hi
    """
    result: dict[str, pd.DataFrame] = {}
    for path in paths:
        env_name = _env_name_from_path(path)
        try:
            df = pd.read_csv(path)
            df.sort_values("rollout_number", inplace=True)
            result[env_name] = df
            print(f"  Loaded {path}  ({len(df)} rollouts, env={env_name})")
        except Exception as exc:
            print(f"  [WARNING] Could not read {path}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Axes styling helper
# ---------------------------------------------------------------------------


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(axis="both", alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Core plot
# ---------------------------------------------------------------------------


def plot_rollout_stats(
    stats: dict[str, pd.DataFrame],
    output_dir: str,
    smooth_window: int = 1,
) -> None:
    """
    Two-panel figure: reward (left) and model-free % (right) vs rollout number.

    Args:
        stats: mapping {env_name -> stats DataFrame}
        output_dir: directory to write PNG and PDF
        smooth_window: rolling-average window size (1 = no smoothing)
    """
    if not stats:
        print("No stats to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    env_names = sorted(stats.keys())
    colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(env_names)}

    n_panels = int(SHOW_REWARD_PANEL) + 1 + int(SHOW_CONFIDENCE_PANEL)
    fig_w = max(6.3, n_panels * 5.75)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.6))
    if n_panels == 1:
        axes = [axes]

    panel_iter = iter(axes)
    ax_reward = next(panel_iter) if SHOW_REWARD_PANEL else None
    ax_mf = next(panel_iter)
    ax_conf = next(panel_iter) if SHOW_CONFIDENCE_PANEL else None

    for env_name in env_names:
        df = stats[env_name]
        # Filter to MAX_TIMESTEPS if set
        if MAX_TIMESTEPS is not None and "timestep" in df.columns:
            df = df[df["timestep"] <= MAX_TIMESTEPS]
        x = (
            df["timestep"].values
            if "timestep" in df.columns
            else df["rollout_number"].values
        ) / 1_000
        color = colors[env_name]
        display_label = ENV_DISPLAY.get(env_name, env_name.replace("_", " ").title())

        def _smooth(arr: np.ndarray) -> np.ndarray:
            if smooth_window <= 1:
                return arr
            return (
                pd.Series(arr)
                .rolling(smooth_window, min_periods=1, center=True)
                .mean()
                .values
            )

        # ── reward panel (optional) ──────────────────────────────────────
        if SHOW_REWARD_PANEL:
            r_mean = _smooth(df["reward_mean"].values)
            r_lo = _smooth(df["reward_ci_lo"].values)
            r_hi = _smooth(df["reward_ci_hi"].values)

            ax_reward.plot(x, r_mean, color=color, label=display_label, zorder=3)
            ax_reward.fill_between(x, r_lo, r_hi, color=color, alpha=0.18, zorder=2)

        # ── model-free % panel ───────────────────────────────────────────
        m_mean = _smooth(df["model_free_pct_mean"].values)
        m_lo = _smooth(df["model_free_pct_ci_lo"].values)
        m_hi = _smooth(df["model_free_pct_ci_hi"].values)

        ax_mf.plot(x, m_mean, color=color, label=display_label, zorder=3)
        ax_mf.fill_between(x, m_lo, m_hi, color=color, alpha=0.18, zorder=2)

        # ── confidence panel (optional) ──────────────────────────────────
        if SHOW_CONFIDENCE_PANEL and ax_conf is not None:
            if (
                "uncertainty_mean" in df.columns
                and df["uncertainty_mean"].notna().any()
            ):
                c_mean = _smooth(df["uncertainty_mean"].values)
                c_lo = _smooth(df["uncertainty_ci_lo"].values)
                c_hi = _smooth(df["uncertainty_ci_hi"].values)
                ax_conf.plot(x, c_mean, color=color, label=display_label, zorder=3)
                ax_conf.fill_between(x, c_lo, c_hi, color=color, alpha=0.18, zorder=2)

    # ── labels / styling ────────────────────────────────────────────────
    if SHOW_REWARD_PANEL:
        ax_reward.set_xlabel("Timestep (Thousands)")
        ax_reward.set_ylabel("Average Reward")
        # ax_reward.set_title("Average Reward")
        _style_ax(ax_reward)

    ax_mf.set_xlabel("Timestep (Thousands)")
    ax_mf.set_ylabel("Reactive Action %")
    # ax_mf.set_title("Reactive Action %")
    ax_mf.set_ylim(0, 101)
    _style_ax(ax_mf)

    if SHOW_CONFIDENCE_PANEL and ax_conf is not None:
        ax_conf.set_xlabel("Timestep (Thousands)")
        ax_conf.set_ylabel("Reactive Policy Uncertainty")
        # ax_conf.set_title("Reactive Actor Confidence")
        _style_ax(ax_conf)

    # Shared legend below both panels
    legend_handles = [
        mpatches.Patch(
            facecolor=colors[env_name],
            edgecolor="black",
            linewidth=0.6,
            label=ENV_DISPLAY.get(env_name, env_name.replace("_", " ").title()),
        )
        for env_name in env_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(env_names), 5),
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    plt.tight_layout(pad=0.8, w_pad=1.2)
    plt.subplots_adjust(bottom=0.25)

    png_path = os.path.join(output_dir, "rollout_stats.png")
    pdf_path = os.path.join(output_dir, "rollout_stats.pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  Saved → {png_path}")
    print(f"  Saved → {pdf_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    stats_files: list[str] | None,
    stats_dir: str | None,
    smooth_window: int,
) -> None:
    # Resolve file list
    if stats_files:
        paths = stats_files
        base_dir = os.path.dirname(os.path.abspath(paths[0]))
    elif stats_dir:
        stats_dir = os.path.expanduser(stats_dir)
        if not os.path.isdir(stats_dir):
            raise FileNotFoundError(f"Stats directory not found: {stats_dir}")
        paths = _collect_stats_files(stats_dir)
        base_dir = stats_dir
    else:
        raise ValueError("Provide --stats_dir or --stats_files.")

    if not paths:
        print("No *_rollout_stats.csv files found.")
        return

    print(f"\nLoading {len(paths)} stats file(s):")
    stats = load_stats(paths)

    output_dir = os.path.join(base_dir, "plots")
    print(f"\nSaving plots to: {output_dir}")
    plot_rollout_stats(stats, output_dir, smooth_window=smooth_window)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot average reward and model-free % vs rollout number "
            "from pre-computed rollout stats CSVs."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--stats_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory containing *_rollout_stats.csv files.",
    )
    source.add_argument(
        "--stats_files",
        nargs="+",
        type=str,
        default=None,
        metavar="FILE",
        help="One or more *_rollout_stats.csv files to plot together.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        metavar="W",
        help="Rolling-average smoothing window (default: 1 = no smoothing).",
    )
    args = parser.parse_args()
    run(args.stats_files, args.stats_dir, args.smooth_window)
