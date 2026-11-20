"""
Compute per-rollout mean and 95 % bootstrap CIs from training logs.

For each environment found in an experiment folder this script produces one
CSV file containing, for every rollout number:

    rollout_number, reward_mean, reward_ci_lo, reward_ci_hi,
    model_free_pct_mean, model_free_pct_ci_lo, model_free_pct_ci_hi

These files are consumed by ``plot_rollout_stats.py``.

Usage (single experiment folder)::

    .venv/bin/python when2plan/plotting/compute_rollout_stats.py \\
        --experiment_path outputs/test

    # Saves: outputs/test/rollout_stats/<env_name>_rollout_stats.csv

Or with an explicit output directory::

    .venv/bin/python when2plan/plotting/compute_rollout_stats.py \\
        --experiment_path outputs/test \\
        --output_dir outputs/my_stats
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _discover_run_dirs(experiment_path: str) -> list[str]:
    """Return a flat list of run directories that contain training_log.csv."""
    log_name = "training_log.csv"

    # Case 1: the path itself is a run dir
    if os.path.isfile(os.path.join(experiment_path, log_name)):
        return [experiment_path]

    # Case 2: cluster layout – results/ subdirectory
    results_dir = os.path.join(experiment_path, "results")
    if os.path.isdir(results_dir):
        candidates = [
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith(".")
        ]
        found = [d for d in candidates if os.path.isfile(os.path.join(d, log_name))]
        if found:
            return found

    # Case 3: flat subdirectories inside experiment_path
    candidates = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
        and not d.startswith(".")
        and d not in ("plots", "__pycache__", "rollout_stats")
    ]
    found = [d for d in candidates if os.path.isfile(os.path.join(d, log_name))]
    return found or [experiment_path]


def _load_run(run_dir: str) -> pd.DataFrame | None:
    """Load training_log.csv from *run_dir*. Returns None on failure."""
    log_path = os.path.join(run_dir, "training_log.csv")
    if not os.path.isfile(log_path):
        return None
    try:
        return pd.read_csv(log_path)
    except Exception as exc:
        print(f"  [WARNING] Could not read {log_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Model-free percentage
# ---------------------------------------------------------------------------


def _parse_meta_action(s: str) -> np.ndarray:
    """Parse a string like '[528, 503, 520, 497]' into a numpy array."""
    s = str(s).strip().strip("[]")
    return np.array([float(x) for x in s.split(",")])


def _compute_model_free_pct(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of per-row model-free percentages.

    Uses the ``meta_action`` column (action counts [reactive, short, medium, long]).
    Falls back to zero if the column is absent.
    """
    if "meta_action" not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)

    parsed = df["meta_action"].apply(_parse_meta_action)
    arr = np.stack(parsed.values)  # shape (N, 4)
    totals = arr.sum(axis=1)
    mf_pct = np.where(totals > 0, arr[:, 0] / totals * 100.0, 0.0)
    return pd.Series(mf_pct, index=df.index)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_lower_bound, ci_upper_bound) via percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(values))
    if len(values) == 1:
        return (mean, mean, mean)
    rng = np.random.RandomState(seed)
    boots = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(
        axis=1
    )
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, alpha * 100))
    hi = float(np.percentile(boots, (1.0 - alpha) * 100))
    return (mean, lo, hi)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_stats_for_env(
    dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Given a list of DataFrames (one per seed / run) for a single environment,
    compute per-rollout statistics.

    Each DataFrame must contain ``rollout_number`` and ``reward`` columns.

    Returns a DataFrame with columns:
        rollout_number, timestep,
        reward_mean, reward_ci_lo, reward_ci_hi,
        model_free_pct_mean, model_free_pct_ci_lo, model_free_pct_ci_hi,
        uncertainty_mean, uncertainty_ci_lo, uncertainty_ci_hi,
        n_seeds
    """
    # Annotate each df with model-free pct, then concatenate
    annotated = []
    for df in dfs:
        df = df.copy()
        df["_mf_pct"] = _compute_model_free_pct(df)
        annotated.append(df)

    combined = pd.concat(annotated, ignore_index=True)

    if "rollout_number" not in combined.columns:
        raise ValueError("training_log.csv is missing a 'rollout_number' column.")

    has_uncertainty = "model_free_actor_uncertainty" in combined.columns
    has_steps = "steps" in combined.columns

    rows = []
    for rollout_num, group in combined.groupby("rollout_number", sort=True):
        rewards = group["reward"].dropna().values
        mf_pcts = group["_mf_pct"].dropna().values

        # Cumulative timestep: each rollout contributes `steps` env steps.
        # steps column stores the per-rollout batch size (constant, e.g. 2048).
        if has_steps:
            steps_per_rollout = int(round(group["steps"].median()))
        else:
            steps_per_rollout = 2048  # sensible default
        timestep = (int(rollout_num) + 1) * steps_per_rollout

        r_mean, r_lo, r_hi = _bootstrap_ci(rewards)
        m_mean, m_lo, m_hi = _bootstrap_ci(mf_pcts)

        if has_uncertainty:
            conf_vals = group["model_free_actor_uncertainty"].dropna().values
            c_mean, c_lo, c_hi = _bootstrap_ci(conf_vals)
        else:
            c_mean, c_lo, c_hi = np.nan, np.nan, np.nan

        rows.append(
            {
                "rollout_number": int(rollout_num),
                "timestep": timestep,
                "reward_mean": r_mean,
                "reward_ci_lo": r_lo,
                "reward_ci_hi": r_hi,
                "model_free_pct_mean": m_mean,
                "model_free_pct_ci_lo": m_lo,
                "model_free_pct_ci_hi": m_hi,
                "uncertainty_mean": c_mean,
                "uncertainty_ci_lo": c_lo,
                "uncertainty_ci_hi": c_hi,
                "n_seeds": int(len(group["reward"].dropna())),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(experiment_path: str, output_dir: str | None = None) -> None:
    experiment_path = os.path.expanduser(experiment_path)
    if not os.path.isdir(experiment_path):
        raise FileNotFoundError(f"Experiment path not found: {experiment_path}")

    if output_dir is None:
        output_dir = os.path.join(experiment_path, "rollout_stats")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDiscovering runs in: {experiment_path}")
    run_dirs = _discover_run_dirs(experiment_path)
    print(f"  Found {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}")

    # Group DataFrames by env_name found inside training_log
    env_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for run_dir in run_dirs:
        df = _load_run(run_dir)
        if df is None:
            print(f"  [SKIP] No training_log.csv in {run_dir}")
            continue

        if "env_name" in df.columns:
            for env_name, sub_df in df.groupby("env_name"):
                env_dfs[str(env_name)].append(sub_df.reset_index(drop=True))
        else:
            # Fall back to config / overrides
            env_name = "unknown"
            cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
            ovr_path = os.path.join(run_dir, ".hydra", "overrides.yaml")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as fh:
                    cfg = yaml.safe_load(fh) or {}
                if isinstance(cfg.get("env"), dict):
                    env_name = cfg["env"].get("type", env_name)
            if os.path.isfile(ovr_path):
                with open(ovr_path) as fh:
                    raw = yaml.safe_load(fh)
                if isinstance(raw, list):
                    for item in raw:
                        if str(item).startswith("env="):
                            env_name = str(item).split("=", 1)[1]
            env_dfs[env_name].append(df)

    if not env_dfs:
        print("No training data found – nothing to compute.")
        return

    print(f"\nComputing stats for {len(env_dfs)} environment(s):\n")
    saved_files: list[str] = []

    for env_name, dfs in sorted(env_dfs.items()):
        n_runs = len(dfs)
        n_rollouts = max(len(df) for df in dfs)
        print(f"  {env_name}: {n_runs} run(s), up to {n_rollouts} rollouts each")

        stats_df = compute_stats_for_env(dfs)

        out_path = os.path.join(output_dir, f"{env_name}_rollout_stats.csv")
        stats_df.to_csv(out_path, index=False)
        saved_files.append(out_path)
        print(f"    → saved {out_path}")

    print(f"\n✓ Saved {len(saved_files)} stats file(s) to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-rollout mean and 95 % bootstrap CIs from training logs "
            "and save one CSV per environment."
        )
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Directory containing the experiment runs (with training_log.csv files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Where to write the stats CSVs. "
            "Defaults to <experiment_path>/rollout_stats/."
        ),
    )
    args = parser.parse_args()
    run(args.experiment_path, args.output_dir)
