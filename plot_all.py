import os
import csv
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        description="Plot displacement vs time for all Brownian simulation runs."
    )
    parser.add_argument(
        "--dir", "-d",
        default=INPUT_DIR,
        help=f"Directory containing the CSV files (default: '{INPUT_DIR}').",
    )
    parser.add_argument(
        "--square", "-s",
        action="store_true",
        help="Plot displacement² instead of displacement (should be linear in t).",
    )
    parser.add_argument(
        "--mean", "-m",
        action="store_true",
        help="Overlay the mean across all runs as a bold black line.",
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
    all_times, all_values = [], []
    for path in csv_files:
        t, d = load_csv(path)
        v = d ** 2 if args.square else d
        all_times.append(t)
        all_values.append(v)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = cm.plasma(np.linspace(0.1, 0.9, len(csv_files)))

    for i, (t, v, path) in enumerate(zip(all_times, all_values, csv_files)):
        label = os.path.splitext(os.path.basename(path))[0]  # e.g. "run_001"
        ax.plot(t, v, color=colors[i], linewidth=0.9, alpha=0.75, label=label)

    # Optional mean line (interpolate onto a common time axis first)
    if args.mean and len(csv_files) > 1:
        common_t = all_times[0]  # all runs share the same time axis
        stacked = np.vstack(all_values)
        mean_v = stacked.mean(axis=0)
        ax.plot(common_t, mean_v, color="black", linewidth=2.2,
                linestyle="--", label="Mean", zorder=10)

    y_label = "Displacement² (units²)" if args.square else "Displacement (units)"
    ax.set_xlabel("Simulation time (s)", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    title_suffix = "² — should grow linearly with t" if args.square else ""
    ax.set_title(f"Brownian Particle Displacement{title_suffix}\n"
                 f"{len(csv_files)} run(s) from '{args.dir}/'", fontsize=14)

    # Legend: show it only when runs are few enough to be readable
    if len(csv_files) <= 20:
        ax.legend(fontsize=8, loc="upper left", ncol=max(1, len(csv_files) // 10))
    else:
        # Just annotate the count
        ax.text(0.98, 0.02, f"n = {len(csv_files)} runs",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="gray")

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Figure saved to '{args.out}'.")
    else:
        plt.show()


if __name__ == "__main__":
    main()