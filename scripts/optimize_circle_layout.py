#!/usr/bin/env python3
"""Search lesion center / circle radius for all circle points <= tolerance."""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_circle_workspace import (  # noqa: E402
    INSERTION_LIMIT,
    LESION_CENTER_REF,
    PCC_BASE_OFFSET,
    PCC_MAX_CURVATURE,
    ROLL_LIMIT,
    TISSUE_GRID_MAX,
    TISSUE_GRID_MIN,
    generate_tip,
    point_inside_tissue,
)


def build_workspace(grid=13, insertion_grid=5, roll_grid=5):
    curvature_values = np.linspace(-PCC_MAX_CURVATURE, PCC_MAX_CURVATURE, grid)
    insertion_values = np.linspace(-INSERTION_LIMIT, INSERTION_LIMIT, insertion_grid)
    roll_values = np.linspace(-ROLL_LIMIT, ROLL_LIMIT, max(1, roll_grid))
    tips = []
    for ky_prox in curvature_values:
        for ky_dist in curvature_values:
            for kz_prox in curvature_values:
                for kz_dist in curvature_values:
                    for insertion_offset in insertion_values:
                        for roll_angle in roll_values:
                            tips.append(
                                generate_tip(
                                    (ky_prox, ky_dist, kz_prox, kz_dist, insertion_offset, roll_angle)
                                )
                            )
    return np.asarray(tips, dtype=np.float64)


def max_radius_at_center(tree, center, tol, n_samples=72, r_hi=0.06, r_lo=0.005):
    c = np.asarray(center, dtype=np.float64)
    r_tissue = min(
        c[0] - TISSUE_GRID_MIN[0],
        TISSUE_GRID_MAX[0] - c[0],
        c[1] - TISSUE_GRID_MIN[1],
        TISSUE_GRID_MAX[1] - c[1],
        c[2] - TISSUE_GRID_MIN[2],
        TISSUE_GRID_MAX[2] - c[2],
    )
    r_hi = min(r_hi, r_tissue - 1e-6)
    if r_hi <= r_lo:
        return None

    def ok(radius):
        phases = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
        targets = np.empty((n_samples, 3), dtype=np.float64)
        targets[:, 0] = c[0] + radius * np.cos(phases)
        targets[:, 1] = c[1]
        targets[:, 2] = c[2] + radius * np.sin(phases)
        if not np.all(
            (targets >= TISSUE_GRID_MIN).all(axis=1) & (targets <= TISSUE_GRID_MAX).all(axis=1)
        ):
            return False, float("inf")
        dists, _ = tree.query(targets, k=1)
        mx = float(np.max(dists))
        return mx <= tol, mx

    if not ok(r_lo)[0]:
        return None
    if ok(r_hi)[0]:
        _, err = ok(r_hi)
        return {"center": tuple(c), "radius": r_hi, "max_error": err}

    lo, hi = r_lo, r_hi
    best = None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        passed, err = ok(mid)
        if passed:
            best = {"center": tuple(c), "radius": mid, "max_error": err}
            lo = mid
        else:
            hi = mid
    return best


def main():
    tips = build_workspace(grid=13, insertion_grid=5, roll_grid=5)
    tree = cKDTree(tips)
    print(f"workspace_points={tips.shape[0]}", flush=True)

    tol = 0.01
    best = None

    cx_vals = np.linspace(0.06, 0.14, 17)
    cy_vals = np.linspace(-0.17, -0.12, 11)
    cz_vals = np.linspace(-0.02, 0.02, 5)

    for cx in cx_vals:
        for cy in cy_vals:
            for cz in cz_vals:
                cand = max_radius_at_center(tree, (cx, cy, cz), tol)
                if cand and (best is None or cand["radius"] > best["radius"]):
                    best = cand

    if best is None:
        print("FAIL: no layout with max_error <= 1cm", flush=True)
        sys.exit(1)

    # Fine center refine
    bc = best["center"]
    for cx in np.linspace(bc[0] - 0.012, bc[0] + 0.012, 13):
        for cy in np.linspace(bc[1] - 0.006, bc[1] + 0.006, 13):
            for cz in np.linspace(bc[2] - 0.006, bc[2] + 0.006, 7):
                cand = max_radius_at_center(tree, (cx, cy, cz), tol, r_hi=best["radius"] + 0.008)
                if cand and cand["radius"] > best["radius"]:
                    best = cand

    print("BEST (grid=13, KD-tree):", flush=True)
    print(f"  lesion_center={[round(x, 4) for x in best['center']]}", flush=True)
    print(f"  circle_radius={best['radius']:.4f}", flush=True)
    print(f"  max_error={best['max_error']:.6f}", flush=True)
    print(f"  current_center={LESION_CENTER_REF.tolist()}", flush=True)
    print(f"  base_offset={PCC_BASE_OFFSET.tolist()}", flush=True)


if __name__ == "__main__":
    main()
