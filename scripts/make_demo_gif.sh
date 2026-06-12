#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROBOT_GLOB="${ROBOT_GLOB:-${INPUT_GLOB:-sofa/vtk_output/robot/frame_*.vtk}}"
TISSUE_GLOB="${TISSUE_GLOB:-sofa/vtk_output/tissue/frame_*.vtk}"
OUTPUT_GIF="${OUTPUT_GIF:-logs/sofa_demo.gif}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
FPS="${FPS:-12}"
POINT_SIZE="${POINT_SIZE:-12}"
MOTION_SCALE="${MOTION_SCALE:-1.0}"
METRICS_CSV="${METRICS_CSV:-sofa/vtk_output/frame_metrics.csv}"
ELEVATION="${ELEVATION:-20}"
AZIMUTH="${AZIMUTH:-40}"
VENV_PATH="${VENV_PATH:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x "$VENV_PATH/bin/python" ]]; then
  PYTHON_BIN="$VENV_PATH/bin/python"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/make_demo_gif.sh [output_gif] [frame_stride] [fps]

Examples:
  scripts/make_demo_gif.sh
  scripts/make_demo_gif.sh logs/demo.gif 25 15

Environment overrides:
  ROBOT_GLOB   Default: sofa/vtk_output/robot/frame_*.vtk
  TISSUE_GLOB  Default: sofa/vtk_output/tissue/frame_*.vtk
  INPUT_GLOB   Backward-compatible alias of ROBOT_GLOB
  OUTPUT_GIF   Default: logs/sofa_demo.gif
  FRAME_STRIDE Default: 1
  FPS          Default: 12
  POINT_SIZE   Default: 12
  MOTION_SCALE Default: 1.0 (set >1.0 to amplify deformation visually)
  METRICS_CSV  Default: sofa/vtk_output/frame_metrics.csv
  ELEVATION    Default: 20
  AZIMUTH      Default: 40
  VENV_PATH    Default: .venv
  PYTHON_BIN   Default: python3 (or .venv/bin/python when available)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ge 1 ]]; then
  OUTPUT_GIF="$1"
fi
if [[ $# -ge 2 ]]; then
  FRAME_STRIDE="$2"
fi
if [[ $# -ge 3 ]]; then
  FPS="$3"
fi

if ! [[ "$FRAME_STRIDE" =~ ^[0-9]+$ ]] || [[ "$FRAME_STRIDE" -lt 1 ]]; then
  echo "[ERROR] FRAME_STRIDE must be an integer >= 1" >&2
  exit 1
fi
if ! [[ "$FPS" =~ ^[0-9]+$ ]] || [[ "$FPS" -lt 1 ]]; then
  echo "[ERROR] FPS must be an integer >= 1" >&2
  exit 1
fi
if ! [[ "$MOTION_SCALE" =~ ^[0-9]*\.?[0-9]+$ ]]; then
  echo "[ERROR] MOTION_SCALE must be a positive number" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_GIF")"

ensure_package() {
  local import_name="$1"
  local pip_name="$2"
  if ! "$PYTHON_BIN" -c "import ${import_name}" >/dev/null 2>&1; then
    echo "[INFO] Installing missing package: ${pip_name}"
    "$PYTHON_BIN" -m pip install "$pip_name"
  fi
}

ensure_package numpy numpy
ensure_package meshio meshio
ensure_package matplotlib matplotlib
ensure_package PIL pillow

export ROBOT_GLOB TISSUE_GLOB OUTPUT_GIF FRAME_STRIDE FPS POINT_SIZE MOTION_SCALE METRICS_CSV ELEVATION AZIMUTH

"$PYTHON_BIN" - <<'PY'
import glob
import os
import re
import csv

import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colors

robot_glob = os.environ["ROBOT_GLOB"]
tissue_glob = os.environ["TISSUE_GLOB"]
output_gif = os.environ["OUTPUT_GIF"]
frame_stride = int(os.environ["FRAME_STRIDE"])
fps = int(os.environ["FPS"])
point_size = float(os.environ["POINT_SIZE"])
motion_scale = float(os.environ["MOTION_SCALE"])
metrics_csv = os.environ["METRICS_CSV"]
elevation = float(os.environ["ELEVATION"])
azimuth = float(os.environ["AZIMUTH"])


def frame_id(path):
    match = re.search(r"frame_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def index_by_frame_id(paths):
    indexed = {}
    for path in sorted(paths):
        step = frame_id(path)
        if step >= 0:
            indexed[step] = path
    return indexed


def load_points(path):
    mesh = meshio.read(path)
    return np.asarray(mesh.points, dtype=np.float64)


def dominant_point_count(point_clouds):
    if not point_clouds:
        return None
    counts = {}
    for _path, points in point_clouds:
        key = int(points.shape[0])
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)


def load_frame_metrics(path):
    if not path or not os.path.exists(path):
        return {}
    metrics = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
                metrics[step] = {
                    "contact_force_peak": float(row.get("contact_force_peak", 0.0)),
                    "contact_force_mean": float(row.get("contact_force_mean", 0.0)),
                    "contact_force_total": float(row.get("contact_force_total", 0.0)),
                    "lesion_contact_force_peak": float(row.get("lesion_contact_force_peak", row.get("contact_force_peak", 0.0))),
                    "lesion_contact_force_mean": float(row.get("lesion_contact_force_mean", row.get("contact_force_mean", 0.0))),
                    "lesion_contact_force_total": float(row.get("lesion_contact_force_total", row.get("contact_force_total", 0.0))),
                    "lesion_surface_stress_peak": float(row.get("lesion_surface_stress_peak", 0.0)),
                    "lesion_nodal_reaction_peak": float(row.get("lesion_nodal_reaction_peak", 0.0)),
                    "contact_distance": float(row.get("contact_distance", 1.0)),
                    "lesion_contact_distance": float(row.get("lesion_contact_distance", row.get("contact_distance", 1.0))),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return metrics


robot_files = sorted(glob.glob(robot_glob))
if not robot_files:
    raise SystemExit(f"[ERROR] No robot VTK frames found with pattern: {robot_glob}")
tissue_files = sorted(glob.glob(tissue_glob))

robot_by_id = index_by_frame_id(robot_files)
tissue_by_id = index_by_frame_id(tissue_files)
metrics_by_id = load_frame_metrics(metrics_csv)
common_ids = sorted(set(robot_by_id) & set(tissue_by_id)) if tissue_by_id else sorted(robot_by_id)
if not common_ids:
    raise SystemExit("[ERROR] No robot/tissue frame pairs with matching frame IDs.")

selected_ids = common_ids[::frame_stride]
if len(selected_ids) < 2 and len(common_ids) > 1:
    selected_ids = [common_ids[0], common_ids[-1]]

robot_point_clouds = []
tissue_point_clouds = []
frame_contact_values = []
mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

for step in selected_ids:
    robot_path = robot_by_id[step]
    robot_points = load_points(robot_path)
    if robot_points.size == 0:
        continue
    robot_point_clouds.append((robot_path, robot_points))
    frame_contact_values.append(
        metrics_by_id.get(step, {}).get(
            "lesion_contact_force_peak",
            metrics_by_id.get(step, {}).get("contact_force_peak", 0.0),
        )
    )
    mins = np.minimum(mins, robot_points.min(axis=0))
    maxs = np.maximum(maxs, robot_points.max(axis=0))

    if tissue_by_id:
        tissue_path = tissue_by_id[step]
        tissue_points = load_points(tissue_path)
        if tissue_points.size == 0:
            tissue_point_clouds.append((tissue_path, None))
        else:
            tissue_point_clouds.append((tissue_path, tissue_points))
            mins = np.minimum(mins, tissue_points.min(axis=0))
            maxs = np.maximum(maxs, tissue_points.max(axis=0))

if len(robot_point_clouds) < 2:
    raise SystemExit("[ERROR] Need at least 2 non-empty VTK frames to render GIF.")

dominant_robot_count = dominant_point_count(robot_point_clouds)
dominant_tissue_count = dominant_point_count(
    [(path, pts) for path, pts in tissue_point_clouds if pts is not None]
)
filtered_robot = []
filtered_tissue = []
filtered_contact_values = []
skipped_topology = 0
for (robot_path, robot_points), tissue_item, contact_value in zip(
    robot_point_clouds,
    tissue_point_clouds or [None] * len(robot_point_clouds),
    frame_contact_values,
):
    if dominant_robot_count is not None and robot_points.shape[0] != dominant_robot_count:
        skipped_topology += 1
        continue
    tissue_path, tissue_points = tissue_item if tissue_item is not None else (None, None)
    if tissue_points is not None and dominant_tissue_count is not None and tissue_points.shape[0] != dominant_tissue_count:
        skipped_topology += 1
        continue
    filtered_robot.append((robot_path, robot_points))
    filtered_tissue.append((tissue_path, tissue_points))
    filtered_contact_values.append(float(contact_value))

if skipped_topology:
    print(
        f"[WARN] Skipped {skipped_topology} frame(s) with mixed VTK topology. "
        f"Keep robot={dominant_robot_count} tissue={dominant_tissue_count} points per frame."
    )
robot_point_clouds = filtered_robot
tissue_point_clouds = filtered_tissue
frame_contact_values = filtered_contact_values
if len(robot_point_clouds) < 2:
    raise SystemExit(
        "[ERROR] After topology filtering, fewer than 2 frames remain. "
        "Clear old sofa/vtk_output and rerun simulation."
    )

margin = 0.05 * np.maximum(maxs - mins, 1e-6)
mins -= margin
maxs += margin

fig = plt.figure(figsize=(8, 6), dpi=120)
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])
ax.set_box_aspect((maxs - mins).tolist())

first_robot_points = robot_point_clouds[0][1]
contact_max = max(frame_contact_values) if frame_contact_values else 0.0
use_contact_colors = bool(metrics_by_id) and contact_max > 0.0
contact_norm = colors.Normalize(vmin=0.0, vmax=max(contact_max, 1e-8))
robot_colors = (
    np.full(first_robot_points.shape[0], frame_contact_values[0])
    if use_contact_colors
    else np.linspace(0.0, 1.0, first_robot_points.shape[0])
)
robot_scatter = ax.scatter(
    first_robot_points[:, 0],
    first_robot_points[:, 1],
    first_robot_points[:, 2],
    c=robot_colors,
    cmap="inferno" if use_contact_colors else "viridis",
    norm=contact_norm if use_contact_colors else None,
    s=point_size,
    alpha=0.85,
    label="Robot",
)
if use_contact_colors:
    colorbar = fig.colorbar(robot_scatter, ax=ax, shrink=0.65, pad=0.08)
    colorbar.set_label("Lesion contact force peak")
tissue_scatter = None
first_tissue_points = None
if tissue_point_clouds:
    first_tissue_points = tissue_point_clouds[0][1]
    tissue_scatter = ax.scatter(
        first_tissue_points[:, 0],
        first_tissue_points[:, 1],
        first_tissue_points[:, 2],
        c="salmon",
        s=max(1.0, point_size * 0.45),
        alpha=0.35,
        label="Target tissue",
    )
ax.legend(loc="upper right")
ax.view_init(elev=elevation, azim=azimuth)

if motion_scale <= 0:
    raise SystemExit("[ERROR] MOTION_SCALE must be > 0")

robot_rest_by_count = {first_robot_points.shape[0]: first_robot_points.copy()}
tissue_rest_by_count = {}
if first_tissue_points is not None:
    tissue_rest_by_count[first_tissue_points.shape[0]] = first_tissue_points.copy()


def scaled_points(points, rest_by_count):
    if motion_scale == 1.0:
        return points
    rest = rest_by_count.get(points.shape[0])
    if rest is None or rest.shape != points.shape:
        return points
    return rest + motion_scale * (points - rest)


def update(frame_idx):
    _path, robot_points = robot_point_clouds[frame_idx]
    render_robot_points = scaled_points(robot_points, robot_rest_by_count)
    if use_contact_colors:
        robot_scatter.set_array(
            np.full(render_robot_points.shape[0], frame_contact_values[frame_idx])
        )
    robot_scatter._offsets3d = (
        render_robot_points[:, 0],
        render_robot_points[:, 1],
        render_robot_points[:, 2],
    )
    artists = [robot_scatter]

    if tissue_scatter is not None and frame_idx < len(tissue_point_clouds):
        _, tissue_points = tissue_point_clouds[frame_idx]
        if tissue_points is not None:
            render_tissue_points = scaled_points(tissue_points, tissue_rest_by_count)
            tissue_scatter._offsets3d = (
                render_tissue_points[:, 0],
                render_tissue_points[:, 1],
                render_tissue_points[:, 2],
            )
            artists.append(tissue_scatter)
    contact_text = ""
    if use_contact_colors:
        contact_text = f" | lesion_contact_peak={frame_contact_values[frame_idx]:.4f}"
    ax.set_title(
        f"SOFA deformation demo ({frame_idx + 1}/{len(robot_point_clouds)}) "
        f"| motion_scale={motion_scale:.2f}{contact_text}"
    )
    return tuple(artists)

ani = FuncAnimation(
    fig,
    update,
    frames=len(robot_point_clouds),
    interval=1000 / fps,
    blit=False,
)
writer = PillowWriter(fps=fps)
ani.save(output_gif, writer=writer)
plt.close(fig)

tip_displacements = [
    float(np.linalg.norm(points[-1] - first_robot_points[-1]))
    for _, points in robot_point_clouds
]
print(f"[OK] Saved GIF: {output_gif}")
print(
    f"[INFO] Robot frames: {len(robot_files)}; Tissue frames: {len(tissue_files)}; "
    f"paired IDs: {len(common_ids)} -> rendered {len(robot_point_clouds)}; stride={frame_stride}"
)
if use_contact_colors:
    print(
        "[INFO] Lesion contact force peak stats: "
        f"min={min(frame_contact_values):.6f}, max={max(frame_contact_values):.6f}, "
        f"mean={float(np.mean(frame_contact_values)):.6f}"
    )
elif metrics_by_id:
    print("[INFO] Metrics CSV found, but all lesion_contact_force_peak values are zero.")
print(
    "[INFO] Tip displacement stats (raw meters): "
    f"min={min(tip_displacements):.6f}, max={max(tip_displacements):.6f}, "
    f"mean={float(np.mean(tip_displacements)):.6f}"
)
PY
