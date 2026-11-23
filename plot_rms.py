import os
import csv
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "brownian_results"


def load_csv(path: str):
    times, displacements = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            displacements.append(float(row["displacement"]))
    return np.array(times), np.array(displacements)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the Root Mean Square Displacement (RMSD) across all Brownian simulation runs."
    )
    parser.add_argument(
        "--dir", "-d",
        default=INPUT_DIR,
        help=f"Directory containing the CSV files (default: '{INPUT_DIR}').",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Hide the ±1 std deviation shaded band.",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Save the figure to this file path instead of showing it (e.g. plot.png).",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.dir, "run_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No CSV files found in '{args.dir}/'. Run brownian_simulation.py first.")
        return

    print(f"Found {len(csv_files)} run(s) in '{args.dir}/'.")

    # --- Load all runs ---
    all_displacements = []
    common_t = None
    for path in csv_files:
        t, d = load_csv(path)
        if common_t is None:
            common_t = t
        all_displacements.append(d)

    stacked = np.vstack(all_displacements)   # shape: (n_runs, n_steps)

    # RMSD = sqrt( mean( displacement² ) )
    rmsd = np.sqrt(np.mean(stacked ** 2, axis=0))
    std  = np.std(stacked ** 2, axis=0)      # std of d² for the shaded band
    upper = np.sqrt(np.maximum(np.mean(stacked ** 2, axis=0) + std, 0))
    lower = np.sqrt(np.maximum(np.mean(stacked ** 2, axis=0) - std, 0))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 6))

    if not args.no_std:
        ax.fill_between(
            common_t, lower, upper,
            alpha=0.25,
            color="steelblue",
            label="±1 std dev",
        )

    ax.plot(common_t, rmsd, color="steelblue", linewidth=2.2, label=f"RMSD (n={len(csv_files)})")

    ax.set_xlabel("Simulation time (s)", fontsize=13)
    ax.set_ylabel("RMSD (units)", fontsize=13)
    ax.set_title(
        f"Brownian Particle — Root Mean Square Displacement\n"
        f"{len(csv_files)} run(s) from '{args.dir}/'",
        fontsize=14,
    )

    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Figure saved to '{args.out}'.")
    else:
        plt.show()


if __name__ == "__main__":
    main()