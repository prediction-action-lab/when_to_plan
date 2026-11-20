# When2Plan

Code for [**When to Plan: Learning to Select Between Reactive Control and Deliberative Planning**](https://rlj.cs.umass.edu/2026/papers/Paper94.pdf) (RLC 2026).

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
source .venv/bin/activate
```

## Training

Experiments are configured with [Hydra](https://hydra.cc) (`experiments/configs/`).

```bash
python experiments/train.py env=<env> experiment=<env>_<type>
```

- `<env>`: `maze`, `doorkey`, `box_push`, `navigation`, `fetch_pick_place`
- `<type>`:
  - `pretrain`: uses pretrained model-free actors (see [below](#pretraining-the-model-free-actors))
  - `joint_training`: trains the model-free actors alongside the meta-policy
  - `default`: behavior-clones the model-free actors online (not available for `fetch_pick_place`)

Results are written to the Hydra run directory (set with `hydra.run.dir=...`).

Common overrides:

| Override | Description |
|---|---|
| `seed`, `training_timesteps` | Paper runs use seeds 42–71 and `300000` steps (`1000000` for `joint_training`) |
| `experiment.num_cached_tasks` | Number of training tasks (paper: `20`; ID/OOD ablation: `10`, `20`, `50`) |
| `evaluate_heuristics=True` | Also evaluate fixed baselines (always model-free / short / medium / long plan, random) |
| `post_run_ablation=True` | After training, evaluate on held-out tasks |
| `ablation` | Method ablation: `no_history`, `no_model_free_uncertainty`, `no_distance_to_goal`, `no_option_duration`, `no_last_action`, `raw_state` |
| `plan_cost_scaling_factor` | Environment ablation on planning cost (paper: `0.1`, `1.0`, `5.0`) |
| `action_noise` | Environment ablation on action noise (paper: `0.0`–`0.3`) |

Example:

```bash
python experiments/train.py env=maze experiment=maze_pretrain \
    experiment.num_cached_tasks=20 training_timesteps=300000 seed=42
```

## Pretraining the model-free actors

The `pretrain` experiments and `post_run_ablation=True` load actors from `when2plan/actors/`. To generate them, run `notebooks/<env>_confidence.ipynb` from inside `notebooks/`. Each notebook behavior-clones an ensemble of 4 actors from planner rollouts, saves them, and plots in- vs out-of-distribution uncertainty.

Then copy them for the post-run evaluation:

```bash
cd when2plan/actors && mkdir -p post_ablation && \
for f in pretrained_*.pth; do cp "$f" "post_ablation/${f/_actor_/_actor_post_ablation_}"; done
```

## Plotting

Scripts in `when2plan/plotting/` take a directory of runs:

| Script | Output |
|---|---|
| `compute_rollout_stats.py --experiment_path DIR`<br>`plot_rollout_stats.py --stats_dir DIR/rollout_stats` | Learning curves (reward and model-free action %) |
| `plot_evaluation_results.py --experiment_path DIR` | Learned policy vs. heuristic baselines |
| `get_method_ablation_results.py --experiment_paths DIR ...` | Method ablation table |
| `get_environment_ablation_results.py --experiment_paths DIR ...`<br>`plot_env_ablation_results.py --experiment_paths DIR ...` | Environment ablation statistics and plots |
| `get_action_percentages.py --experiment_paths DIR ...` | Action-type percentages per environment |

## Citation

```bibtex
@article{labiosa2026when,
  title={When to Plan: Learning to Select Between Reactive Control and Deliberative Planning},
  author={Adam Labiosa and Josiah P. Hanna},
  journal={Reinforcement Learning Journal},
  volume={7},
  year={2026}
}
```
