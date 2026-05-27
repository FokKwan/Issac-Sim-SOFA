#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROBOT_GLOB="${ROBOT_GLOB:-${INPUT_GLOB:-sofa/vtk_output/robot/frame_*.vtk}}"
TISSUE_GLOB="${TISSUE_GLOB:-sofa/vtk_output/tissue/frame_*.vtk}"
OUTPUT_GIF="${OUTPUT_GIF:-logs/sofa_demo.gif}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"
FPS="${FPS:-12}"
POINT_SIZE="${POINT_SIZE:-12}"
MOTION_SCALE="${MOTION_SCALE:-1.0}"
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
  FRAME_STRIDE Default: 10
  FPS          Default: 12
  POINT_SIZE   Default: 12
  MOTION_SCALE Default: 1.0 (set >1.0 to amplify deformation visually)
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

export ROBOT_GLOB TISSUE_GLOB OUTPUT_GIF FRAME_STRIDE FPS POINT_SIZE MOTION_SCALE ELEVATION AZIMUTH

"$PYTHON_BIN" - <<'PY'
import glob
import os

import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

robot_glob = os.environ["ROBOT_GLOB"]
tissue_glob = os.environ["TISSUE_GLOB"]
output_gif = os.environ["OUTPUT_GIF"]
frame_stride = int(os.environ["FRAME_STRIDE"])
fps = int(os.environ["FPS"])
point_size = float(os.environ["POINT_SIZE"])
motion_scale = float(os.environ["MOTION_SCALE"])
elevation = float(os.environ["ELEVATION"])
azimuth = float(os.environ["AZIMUTH"])

robot_files = sorted(glob.glob(robot_glob))
if not robot_files:
    raise SystemExit(f"[ERROR] No robot VTK frames found with pattern: {robot_glob}")
tissue_files = sorted(glob.glob(tissue_glob))

selected_robot_files = robot_files[::frame_stride]
if len(selected_robot_files) < 2 and len(robot_files) > 1:
    selected_robot_files = [robot_files[0], robot_files[-1]]

selected_tissue_files = tissue_files[::frame_stride] if tissue_files else []
if len(selected_tissue_files) < 2 and len(tissue_files) > 1:
    selected_tissue_files = [tissue_files[0], tissue_files[-1]]

robot_point_clouds = []
tissue_point_clouds = []
mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
for path in selected_robot_files:
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.size == 0:
        continue
    robot_point_clouds.append((path, points))
    mins = np.minimum(mins, points.min(axis=0))
    maxs = np.maximum(maxs, points.max(axis=0))

for path in selected_tissue_files:
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.size == 0:
        continue
    tissue_point_clouds.append((path, points))
    mins = np.minimum(mins, points.min(axis=0))
    maxs = np.maximum(maxs, points.max(axis=0))

if len(robot_point_clouds) < 2:
    raise SystemExit("[ERROR] Need at least 2 non-empty VTK frames to render GIF.")

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
robot_colors = np.linspace(0.0, 1.0, first_robot_points.shape[0])
robot_scatter = ax.scatter(
    first_robot_points[:, 0],
    first_robot_points[:, 1],
    first_robot_points[:, 2],
    c=robot_colors,
    cmap="viridis",
    s=point_size,
    alpha=0.85,
    label="Robot",
)
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

def scaled_points(points):
    if motion_scale == 1.0:
        return points
    return first_robot_points + motion_scale * (points - first_robot_points)


def scaled_tissue_points(points):
    if motion_scale == 1.0 or first_tissue_points is None:
        return points
    return first_tissue_points + motion_scale * (points - first_tissue_points)

def update(frame_idx):
    _path, robot_points = robot_point_clouds[frame_idx]
    render_robot_points = scaled_points(robot_points)
    robot_scatter._offsets3d = (
        render_robot_points[:, 0],
        render_robot_points[:, 1],
        render_robot_points[:, 2],
    )
    artists = [robot_scatter]

    if tissue_scatter is not None and frame_idx < len(tissue_point_clouds):
        _, tissue_points = tissue_point_clouds[frame_idx]
        render_tissue_points = scaled_tissue_points(tissue_points)
        tissue_scatter._offsets3d = (
            render_tissue_points[:, 0],
            render_tissue_points[:, 1],
            render_tissue_points[:, 2],
        )
        artists.append(tissue_scatter)
    ax.set_title(
        f"SOFA deformation demo ({frame_idx + 1}/{len(robot_point_clouds)}) | motion_scale={motion_scale:.2f}"
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
    f"[INFO] Robot frames: {len(robot_files)} -> rendered {len(robot_point_clouds)}; "
    f"Tissue frames: {len(tissue_files)} -> rendered {len(tissue_point_clouds)}; stride={frame_stride}"
)
print(
    "[INFO] Tip displacement stats (raw meters): "
    f"min={min(tip_displacements):.6f}, max={max(tip_displacements):.6f}, "
    f"mean={float(np.mean(tip_displacements)):.6f}"
)
PY
