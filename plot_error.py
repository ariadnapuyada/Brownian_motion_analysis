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


def fit_ballistic(t, rmsd, frac=0.05):
    n = max(2, int(len(t) * frac))
    t_fit, r_fit = t[1:n], rmsd[1:n]
    v = np.sum(t_fit * r_fit) / np.sum(t_fit ** 2)
    return v


def fit_diffusive(t, rmsd, frac=0.20):
    n = max(2, int(len(t) * frac))
    t_fit  = t[-n:]
    r2_fit = rmsd[-n:] ** 2
    D4 = np.sum(t_fit * r2_fit) / np.sum(t_fit ** 2)
    return D4 / 4.0


def main():
    parser = argparse.ArgumentParser(
        description="Plot absolute error between RMSD and the ballistic/diffusive theoretical curves."
    )
    parser.add_argument("--dir",    "-d", default=INPUT_DIR,
                        help=f"Directory with CSV files (default: '{INPUT_DIR}').")
    parser.add_argument("--ballistic-frac", type=float, default=0.05,
                        help="Fraction of early data used to fit ballistic regime (default: 0.05).")
    parser.add_argument("--diffusive-frac", type=float, default=0.20,
                        help="Fraction of late data used to fit diffusive regime (default: 0.20).")
    parser.add_argument("--out", "-o", default=None,
                        help="Save figure to this path instead of showing it.")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.dir, "run_*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{args.dir}/'. Run brownian_simulation.py first.")
        return
    print(f"Found {len(csv_files)} run(s) in '{args.dir}/'.")

    # --- Load & compute RMSD ---
    all_displacements, common_t = [], None
    for path in csv_files:
        t, d = load_csv(path)
        if common_t is None:
            common_t = t
        all_displacements.append(d)

    stacked = np.vstack(all_displacements)
    rmsd    = np.sqrt(np.mean(stacked ** 2, axis=0))

    # --- Fit & build theoretical curves ---
    v_rms = fit_ballistic(common_t, rmsd, frac=args.ballistic_frac)
    D     = fit_diffusive(common_t, rmsd, frac=args.diffusive_frac)

    ballistic_curve = v_rms * common_t
    diffusive_curve = np.sqrt(4 * D * np.maximum(common_t, 0))

    print(f"  Fitted v_rms : {v_rms:.4f}")
    print(f"  Fitted D     : {D:.6f}")

    # --- Absolute errors ---
    err_ballistic = np.abs(rmsd - ballistic_curve)
    err_diffusive = np.abs(rmsd - diffusive_curve)

    # --- Two-panel figure ---
    fig, (ax_rmsd, ax_err) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]}
    )

    # Top panel: RMSD + theoretical curves
    ax_rmsd.plot(common_t, rmsd,
                 color="steelblue", linewidth=2.2, label=f"RMSD (n={len(csv_files)})")
    ax_rmsd.plot(common_t, ballistic_curve,
                 color="tomato", linewidth=1.8, linestyle="--",
                 label=rf"Ballistic: $v_{{rms}} \cdot t$  ($v_{{rms}}={v_rms:.2f}$)")
    ax_rmsd.plot(common_t, diffusive_curve,
                 color="seagreen", linewidth=1.8, linestyle="--",
                 label=rf"Diffusive: $\sqrt{{4Dt}}$  ($D={D:.4f}$)")
    ax_rmsd.set_ylabel("RMSD (units)", fontsize=12)
    ax_rmsd.set_title(
        "Brownian Particle — RMSD and Absolute Error vs Theoretical Regimes\n"
        f"{len(csv_files)} run(s) · ballistic fit: first {args.ballistic_frac*100:.0f}% · "
        f"diffusive fit: last {args.diffusive_frac*100:.0f}%",
        fontsize=13,
    )
    ax_rmsd.legend(fontsize=10)
    ax_rmsd.grid(True, linestyle="--", alpha=0.4)

    # Bottom panel: absolute errors
    ax_err.plot(common_t, err_ballistic,
                color="tomato",  linewidth=1.8,
                label=rf"|RMSD $-$ ballistic|")
    ax_err.plot(common_t, err_diffusive,
                color="seagreen", linewidth=1.8,
                label=rf"|RMSD $-$ diffusive|")
    ax_err.set_xlabel("Simulation time (s)", fontsize=12)
    ax_err.set_ylabel("Absolute error (units)", fontsize=12)
    ax_err.legend(fontsize=10)
    ax_err.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Figure saved to '{args.out}'.")
    else:
        plt.show()


if __name__ == "__main__":
    main()