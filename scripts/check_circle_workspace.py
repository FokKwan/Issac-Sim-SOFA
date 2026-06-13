#!/usr/bin/env python3
"""Check whether the lesion-centered circle is inside the PCC tip workspace."""

import argparse
import math

import numpy as np


PCC_SEGMENT_LENGTHS = [0.30, 0.30, 0.30, 0.30]
PCC_SEGMENT_WEIGHTS = [1.00, 0.95, 0.90, 0.85]
PCC_POINTS_PER_SEGMENT = 8
PCC_MAX_CURVATURE = 1.20
PCC_CURVATURE_DOF = 4
PCC_INSERTION_INDEX = 4
PCC_ROLL_INDEX = 5
PCC_BASE_OFFSET = np.array([-1.10, -0.08, 0.0], dtype=np.float64)
LESION_CENTER_REF = np.array([0.09, -0.12, 0.0], dtype=np.float64)
TISSUE_GRID_MIN = np.array([-0.18, -0.22, -0.06], dtype=np.float64)
TISSUE_GRID_MAX = np.array([0.25, -0.10, 0.06], dtype=np.float64)
INSERTION_LIMIT = 0.08
ROLL_LIMIT = math.pi


def normalize_s_curve_curvature_command(curvature_command):
    curvature = np.asarray(curvature_command, dtype=np.float64).reshape(-1)
    if curvature.size == 0:
        curvature = np.zeros(PCC_CURVATURE_DOF, dtype=np.float64)
    elif curvature.size == 1:
        curvature = np.array([curvature[0], curvature[0], 0.0, 0.0], dtype=np.float64)
    elif curvature.size == 2:
        curvature = np.array([curvature[0], curvature[0], curvature[1], curvature[1]], dtype=np.float64)
    elif curvature.size < PCC_CURVATURE_DOF:
        curvature = np.pad(curvature, (0, PCC_CURVATURE_DOF - curvature.size))
    else:
        curvature = curvature[:PCC_CURVATURE_DOF]
    curvature = np.clip(curvature, -PCC_MAX_CURVATURE, PCC_MAX_CURVATURE)
    for pair_indices in ((0, 2), (1, 3)):
        pair = curvature[list(pair_indices)]
        pair_norm = float(np.linalg.norm(pair))
        if pair_norm > PCC_MAX_CURVATURE:
            curvature[list(pair_indices)] = pair * (PCC_MAX_CURVATURE / max(pair_norm, 1e-8))
    return curvature


def segment_curvature_from_s_command(curvature_command, segment_index, segment_count):
    split_index = max(1, segment_count // 2)
    if segment_index < split_index:
        return np.array([curvature_command[0], curvature_command[2]], dtype=np.float64)
    return np.array([curvature_command[1], curvature_command[3]], dtype=np.float64)


def generate_tip(curvature_command):
    command = np.asarray(curvature_command, dtype=np.float64).reshape(-1)
    if command.size < PCC_CURVATURE_DOF:
        command = np.pad(command, (0, PCC_CURVATURE_DOF - command.size))
    curvature = normalize_s_curve_curvature_command(command[:PCC_CURVATURE_DOF])
    insertion_offset = 0.0
    if command.size > PCC_INSERTION_INDEX:
        insertion_offset = float(np.clip(command[PCC_INSERTION_INDEX], -INSERTION_LIMIT, INSERTION_LIMIT))
    roll_angle = 0.0
    if command.size > PCC_ROLL_INDEX:
        roll_angle = float(np.clip(command[PCC_ROLL_INDEX], -ROLL_LIMIT, ROLL_LIMIT))
    nominal_length = max(float(np.sum(PCC_SEGMENT_LENGTHS)), 1e-8)
    length_scale = max(0.1, (nominal_length + insertion_offset) / nominal_length)
    theta_y = 0.0
    theta_z = 0.0
    current = PCC_BASE_OFFSET.copy()

    segment_count = len(PCC_SEGMENT_LENGTHS)
    for seg_idx, (seg_len, seg_weight) in enumerate(zip(PCC_SEGMENT_LENGTHS, PCC_SEGMENT_WEIGHTS)):
        seg_curvature = segment_curvature_from_s_command(curvature, seg_idx, segment_count) * seg_weight
        ds = (seg_len * length_scale) / float(PCC_POINTS_PER_SEGMENT)
        for _ in range(PCC_POINTS_PER_SEGMENT):
            theta_y += seg_curvature[0] * ds
            theta_z += seg_curvature[1] * ds
            current[0] += math.cos(theta_y) * math.cos(theta_z) * ds
            current[1] += math.sin(theta_y) * ds
            current[2] += math.sin(theta_z) * ds
    if abs(roll_angle) < 1e-12:
        return current
    relative = current - PCC_BASE_OFFSET
    cos_roll = math.cos(roll_angle)
    sin_roll = math.sin(roll_angle)
    return PCC_BASE_OFFSET + np.array(
        [
            relative[0],
            relative[1] * cos_roll - relative[2] * sin_roll,
            relative[1] * sin_roll + relative[2] * cos_roll,
        ],
        dtype=np.float64,
    )


def circle_target(radius, phase):
    return LESION_CENTER_REF + np.array(
        [radius * math.cos(phase), 0.0, radius * math.sin(phase)],
        dtype=np.float64,
    )


def point_inside_tissue(point):
    return bool(np.all(point >= TISSUE_GRID_MIN) and np.all(point <= TISSUE_GRID_MAX))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=49)
    parser.add_argument("--grid", type=int, default=25)
    parser.add_argument("--insertion-grid", type=int, default=9)
    parser.add_argument("--roll-grid", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=0.025)
    args = parser.parse_args()

    curvature_values = np.linspace(-PCC_MAX_CURVATURE, PCC_MAX_CURVATURE, args.grid)
    insertion_values = np.linspace(-INSERTION_LIMIT, INSERTION_LIMIT, args.insertion_grid)
    roll_values = np.linspace(-ROLL_LIMIT, ROLL_LIMIT, max(1, args.roll_grid))
    tips = []
    commands = []
    for ky_prox in curvature_values:
        for ky_dist in curvature_values:
            for kz_prox in curvature_values:
                for kz_dist in curvature_values:
                    for insertion_offset in insertion_values:
                        for roll_angle in roll_values:
                            command = (ky_prox, ky_dist, kz_prox, kz_dist, insertion_offset, roll_angle)
                            commands.append(command)
                            tips.append(generate_tip(command))
    tips = np.asarray(tips)
    commands = np.asarray(commands)

    errors = []
    best_commands = []
    tissue_flags = []
    for phase in np.linspace(0.0, 2.0 * math.pi, args.samples, endpoint=False):
        target = circle_target(args.radius, phase)
        tissue_flags.append(point_inside_tissue(target))
        distances = np.linalg.norm(tips - target, axis=1)
        best_idx = int(np.argmin(distances))
        errors.append(float(distances[best_idx]))
        best_commands.append(commands[best_idx])

    errors = np.asarray(errors)
    best_commands = np.asarray(best_commands)
    print(f"lesion_center={LESION_CENTER_REF.tolist()}")
    print(f"circle_radius={args.radius:.4f}")
    print(f"pcc_curvature_limit=+/-{PCC_MAX_CURVATURE:.4f}")
    print(
        f"samples={args.samples} grid={args.grid} "
        f"insertion_grid={args.insertion_grid} roll_grid={args.roll_grid} tolerance={args.tolerance:.4f}"
    )
    print(f"circle_inside_tissue={all(tissue_flags)}")
    print(f"max_error={errors.max():.6f}")
    print(f"mean_error={errors.mean():.6f}")
    print(f"min_error={errors.min():.6f}")
    print(
        "best_curvature_range="
        f"ky_prox[{best_commands[:, 0].min():.4f},{best_commands[:, 0].max():.4f}] "
        f"ky_dist[{best_commands[:, 1].min():.4f},{best_commands[:, 1].max():.4f}] "
        f"kz_prox[{best_commands[:, 2].min():.4f},{best_commands[:, 2].max():.4f}] "
        f"kz_dist[{best_commands[:, 3].min():.4f},{best_commands[:, 3].max():.4f}] "
        f"insertion[{best_commands[:, 4].min():.4f},{best_commands[:, 4].max():.4f}] "
        f"roll[{best_commands[:, 5].min():.4f},{best_commands[:, 5].max():.4f}]"
    )
    if not all(tissue_flags):
        raise SystemExit("FAIL: at least one circle target is outside the tissue volume.")
    if errors.max() <= args.tolerance:
        print("PASS: every sampled circle target is inside the reachable workspace tolerance.")
        return
    raise SystemExit("FAIL: at least one circle target is outside the reachable workspace tolerance.")


if __name__ == "__main__":
    main()
