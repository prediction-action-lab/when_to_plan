from when2plan.plotting.plot_helpers.plot_logged_data import plot_log_exp_data
import argparse

# parse args
parser = argparse.ArgumentParser(description="Plot experiment results")
parser.add_argument(
    "--experiment_path",
    type=str,
    required=True,
    help="Directory containing experiment outputs",
)
args = parser.parse_args()

# plot data - now returns list of saved filenames
saved_files = plot_log_exp_data(args.experiment_path)

print(f"\n✓ Successfully generated {len(saved_files)} plot file(s)")
