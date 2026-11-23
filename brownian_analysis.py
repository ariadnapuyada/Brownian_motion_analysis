import pymunk
import random
import math
import csv
import os
import argparse
import time

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FPS = 60                  # Physics timestep (dt = 1/FPS), NOT a real-time cap
SOLVENT_COUNT = 500
SOLVENT_RADIUS = 4
BROWNIAN_RADIUS = 25
INITIAL_IMPULSE = 100     # Sets the "Temperature"
OUTPUT_DIR = "brownian_results"


def create_particle(space, pos, radius, mass):
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 1.0   # Perfectly elastic (no energy loss)
    shape.friction = 0.0
    space.add(body, shape)
    return body


def run_simulation(sim_duration: float, run_id: int) -> str:
    """
    Run one headless simulation for `sim_duration` seconds of physics time.
    Records (time, displacement) at every physics step.
    Returns the path to the saved CSV file.
    """
    space = pymunk.Space()
    space.gravity = (0, 0)

    # --- Walls ---
    static_lines = [
        pymunk.Segment(space.static_body, (0, 0),        (WIDTH, 0),      5),
        pymunk.Segment(space.static_body, (WIDTH, 0),    (WIDTH, HEIGHT), 5),
        pymunk.Segment(space.static_body, (WIDTH, HEIGHT),(0, HEIGHT),    5),
        pymunk.Segment(space.static_body, (0, HEIGHT),   (0, 0),          5),
    ]
    for line in static_lines:
        line.elasticity = 1.0
        space.add(line)

    # --- Brownian particle (starts at centre) ---
    start_x, start_y = WIDTH / 2, HEIGHT / 2
    brownian_body = create_particle(space, (start_x, start_y), BROWNIAN_RADIUS, mass=20)

    # --- Solvent particles ---
    for _ in range(SOLVENT_COUNT):
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        body = create_particle(space, (x, y), SOLVENT_RADIUS, mass=1)
        impulse = (
            random.uniform(-INITIAL_IMPULSE, INITIAL_IMPULSE),
            random.uniform(-INITIAL_IMPULSE, INITIAL_IMPULSE),
        )
        body.apply_impulse_at_local_point(impulse)

    # --- Data collection ---
    dt = 1.0 / FPS
    total_steps = int(sim_duration * FPS)
    records = []  # list of (sim_time, displacement)

    wall_start = time.perf_counter()

    for step in range(total_steps + 1):
        sim_time = step * dt
        bx, by = brownian_body.position
        displacement = math.hypot(bx - start_x, by - start_y)
        records.append((sim_time, displacement))

        if step < total_steps:        # don't step past the end
            space.step(dt)

    wall_elapsed = time.perf_counter() - wall_start

    # --- Save to CSV ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"run_{run_id:03d}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "displacement"])
        writer.writerows(records)

    print(
        f"  Run {run_id:>3d} | "
        f"physics time: {sim_duration:.1f} s | "
        f"steps: {total_steps:,} | "
        f"wall time: {wall_elapsed:.2f} s | "
        f"saved → {csv_path}"
    )
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Headless Brownian-motion simulator — saves displacement vs time to CSV."
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Physics duration of each simulation in seconds (default: 30).",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of independent simulation runs (default: 1).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Base random seed (each run gets seed+run_id). Omit for non-deterministic runs.",
    )
    args = parser.parse_args()

    print(f"\nBrownian Motion Simulation")
    print(f"  Runs      : {args.runs}")
    print(f"  Duration  : {args.duration} s  ({int(args.duration * FPS):,} steps each)")
    print(f"  Seed      : {args.seed if args.seed is not None else 'random'}")
    print(f"  Output dir: {OUTPUT_DIR}/\n")

    all_start = time.perf_counter()

    for run_id in range(1, args.runs + 1):
        if args.seed is not None:
            random.seed(args.seed + run_id)
        run_simulation(args.duration, run_id)

    total_wall = time.perf_counter() - all_start
    print(f"\nAll {args.runs} run(s) completed in {total_wall:.2f} s total.")
    print(f"Results saved in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()