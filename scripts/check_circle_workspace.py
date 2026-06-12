#!/usr/bin/env python3
"""Check whether the lesion-centered circle is inside the PCC tip workspace."""

import argparse
import math

import numpy as np


PCC_SEGMENT_LENGTHS = [0.30, 0.30, 0.30, 0.30]
PCC_SEGMENT_WEIGHTS = [1.00, 0.95, 0.90, 0.85]
PCC_POINTS_PER_SEGMENT = 8
PCC_MAX_CURVATURE = 0.45
PCC_BASE_OFFSET = np.array([-1.10, -0.08, 0.0], dtype=np.float64)
LESION_CENTER_REF = np.array([0.08, -0.14, 0.0], dtype=np.float64)
INSERTION_LIMIT = 0.08


def generate_tip(curvature_command):
    command = np.asarray(curvature_command, dtype=np.float64).reshape(3)
    curvature = command[:2]
    insertion_offset = float(np.clip(command[2], -INSERTION_LIMIT, INSERTION_LIMIT))
    curvature = np.clip(curvature, -PCC_MAX_CURVATURE, PCC_MAX_CURVATURE)
    theta_y = 0.0
    theta_z = 0.0
    current = PCC_BASE_OFFSET + np.array([insertion_offset, 0.0, 0.0], dtype=np.float64)

    for seg_len, seg_weight in zip(PCC_SEGMENT_LENGTHS, PCC_SEGMENT_WEIGHTS):
        seg_curvature = curvature * seg_weight
        ds = seg_len / float(PCC_POINTS_PER_SEGMENT)
        for _ in range(PCC_POINTS_PER_SEGMENT):
            theta_y += seg_curvature[0] * ds
            theta_z += seg_curvature[1] * ds
            current[0] += math.cos(theta_y) * math.cos(theta_z) * ds
            current[1] += math.sin(theta_y) * ds
            current[2] += math.sin(theta_z) * ds
    return current


def circle_target(radius, phase):
    return LESION_CENTER_REF + np.array(
        [radius * math.cos(phase), 0.0, radius * math.sin(phase)],
        dtype=np.float64,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--samples", type=int, default=49)
    parser.add_argument("--grid", type=int, default=91)
    parser.add_argument("--insertion-grid", type=int, default=41)
    parser.add_argument("--tolerance", type=float, default=0.025)
    args = parser.parse_args()

    curvature_values = np.linspace(-PCC_MAX_CURVATURE, PCC_MAX_CURVATURE, args.grid)
    insertion_values = np.linspace(-INSERTION_LIMIT, INSERTION_LIMIT, args.insertion_grid)
    tips = []
    commands = []
    for ky in curvature_values:
        for kz in curvature_values:
            for insertion_offset in insertion_values:
                commands.append((ky, kz, insertion_offset))
                tips.append(generate_tip((ky, kz, insertion_offset)))
    tips = np.asarray(tips)
    commands = np.asarray(commands)

    errors = []
    best_commands = []
    for phase in np.linspace(0.0, 2.0 * math.pi, args.samples, endpoint=False):
        target = circle_target(args.radius, phase)
        distances = np.linalg.norm(tips - target, axis=1)
        best_idx = int(np.argmin(distances))
        errors.append(float(distances[best_idx]))
        best_commands.append(commands[best_idx])

    errors = np.asarray(errors)
    best_commands = np.asarray(best_commands)
    print(f"lesion_center={LESION_CENTER_REF.tolist()}")
    print(f"circle_radius={args.radius:.4f}")
    print(f"pcc_curvature_limit=+/-{PCC_MAX_CURVATURE:.4f}")
    print(f"insertion_limit=+/-{INSERTION_LIMIT:.4f}")
    print(
        f"samples={args.samples} grid={args.grid} "
        f"insertion_grid={args.insertion_grid} tolerance={args.tolerance:.4f}"
    )
    print(f"max_error={errors.max():.6f}")
    print(f"mean_error={errors.mean():.6f}")
    print(f"min_error={errors.min():.6f}")
    print(
        "best_curvature_range="
        f"ky[{best_commands[:, 0].min():.4f},{best_commands[:, 0].max():.4f}] "
        f"kz[{best_commands[:, 1].min():.4f},{best_commands[:, 1].max():.4f}] "
        f"insertion[{best_commands[:, 2].min():.4f},{best_commands[:, 2].max():.4f}]"
    )
    if errors.max() <= args.tolerance:
        print("PASS: every sampled circle target is inside the reachable workspace tolerance.")
        return
    raise SystemExit("FAIL: at least one circle target is outside the reachable workspace tolerance.")


if __name__ == "__main__":
    main()
