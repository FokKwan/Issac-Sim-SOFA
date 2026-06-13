# Isaac Sim + SOFA Simulation Pipeline

本文档说明当前仓库中 **连续体机器人-组织交互仿真** 的实现流程、参数含义与运行方式。

---

## 1. Overall Architecture

当前系统采用“训练与物理解耦”设计：

- **SOFA (`sofa/soft_cable_scene.py`)**
  - 负责连续体机器人、组织、病灶区域与接触约束的物理仿真
  - 通过 ZMQ REP 端口 `tcp://*:5555` 提供 step/reset 服务
  - 输出训练指标与可视化 VTK 帧

- **Isaac/RL Side (`issac_sim/envs/sofa_env.py`, `issac_sim/run_env.py`)**
  - 将 SOFA 包装成 Gymnasium 环境
  - 使用 PPO 训练策略并下发动作
  - 消费 SOFA 回传观测，计算 reward/done

- **Runtime Orchestration (`scripts/train_task.sh`)**
  - SOFA 端在 conda 环境（默认 `sofa_rl`）启动
  - 训练端用 Omniverse Python（默认 `~/omniverse/python.sh`）启动
  - 通过 tmux 管理长任务（断开 SSH 后仍可运行）

---

## 2. Data Flow

每个 RL step 的数据流如下：

1. PPO 输出归一化动作 `action in [-1, 1]^3`
2. `SoftSofaEnv.step()` 将动作映射为曲率/插入增量 `cable_disp=[ky_delta,kz_delta,insertion_delta]`
3. `SofaCableClient` 通过 ZMQ 发送 `{"type":"step","cable_disp":...}`
4. SOFA 执行物理步进并计算状态
5. SOFA 返回观测字典（含接触力统计、组织应变等）
6. `SoftSofaEnv` 计算 reward / terminated / truncated

`reset` 流程类似，只是消息类型为 `{"type":"reset"}`。

---

## 3. SOFA Scene (`sofa/soft_cable_scene.py`)

### 3.1 Scene Components

- Root:
  - `FreeMotionAnimationLoop`
  - `GenericConstraintSolver(computeConstraintForces=True)`
  - 接触流水线：`CollisionPipeline` / `BruteForceBroadPhase` / `BVHNarrowPhase`
  - 接触判定：`LocalMinDistance(alarmDistance=0.005, contactDistance=0.0015)`
  - 响应：`CollisionResponse(FrictionContactConstraint, mu=0.45)`

- Robot (`SoftBody`):
  - 分段常曲率（PCC）中心线 + `MechanicalObject(Vec3d)`
  - 每步按累积曲率/插入增量更新形状（固定基端）
  - `ky` 控制 X-Y 平面弯曲，`kz` 控制 X-Z 平面弯曲，`insertion` 控制沿 +X 插入/回撤
  - 曲率限制 `PCC_MAX_CURVATURE = 0.45`，每步曲率增量默认 `0.03`，避免末端甩出病灶工作区
  - `PointCollisionModel` + `LineCollisionModel`
  - 基座 `PCC_BASE_OFFSET = (-1.10, -0.08, 0)`，初始 tip 位于 `(0.10, -0.08, 0)`，中心线位于组织上方约 `0.02 m`

- Tissue (`TargetTissue`):
  - `RegularGridTopology` (10x6x5)，范围 `min=(-0.18, -0.22, -0.06)` 到 `max=(0.25, -0.10, 0.06)`
  - 病灶中心 `LESION_CENTER_REF = (0.08, -0.14, 0)`（PCC 工作空间校核可接近）
  - `HexahedronFEMForceField`（更软材料）
  - `BoxROI + FixedConstraint`（组织边界约束）

### 3.2 Key Outputs

SOFA 每步回传：

- 机器人指标：`tip_position`, `tip_displacement`, `von_mises`, `avg_strain`
- 组织/病灶指标：`lesion_distance`, `lesion_strain`, `tissue_strain`, `lesion_center`
- 接触相关：
  - `contact_distance`（几何最近距离）
  - `contact_force_mean`
  - `contact_force_peak`
  - `contact_force_total`

> 注意：接触力来自约束求解器 `constraintForces` 的统计，并在 reset 时记录基线后做扣除。

### 3.3 VTK Export

导出目录：

- `sofa/vtk_output/robot/frame_XXXX.vtk`
- `sofa/vtk_output/tissue/frame_XXXX.vtk`
- `sofa/vtk_output/frame_metrics.csv`（每帧接触力、接触距离、末端位置）

导出间隔通过环境变量控制：

```bash
SOFA_EXPORT_INTERVAL=2  # 每 2 步导出一次；需要更少文件时可调大
```

---

## 4. RL Environment (`issac_sim/envs/sofa_env.py`)

### 4.1 Action Space

- `Box(low=-1, high=1, shape=(3,))`
- 实际控制量：`cable_disp = action * action_scale`（`[ky_delta,kz_delta,insertion_delta]`，SOFA 侧逐步累积）

### 4.2 Observation Space

字典观测包含：

- `tip_pos`
- `von_mises`, `avg_strain`
- `lesion_distance`, `lesion_strain`, `tissue_strain`
- `contact_distance`
- `contact_force_mean`, `contact_force_peak`, `contact_force_total`
- `lesion_center`
- `circle_target`：当前相位下病灶周围圆轨迹的目标点
- `circle_phase`：当前圆周相位
- `circle_error`：末端到 `circle_target` 的距离

### 4.3 Reward Design

任务目标：末端围绕病灶中心在 `X-Z` 平面追踪一圈圆形轨迹。

- 圆心：`lesion_center`
- 半径：`circle_radius = 0.05 m`
- 周期：`circle_period_steps = 240`
- 工作空间检查：`python scripts/check_circle_workspace.py --radius 0.05`

`reward = -tracking - radial + progress - safety - over_contact + lap_bonus`

- tracking: `25 * circle_error + 60 * circle_error^2`，强惩罚末端偏离当前圆周目标点
- progress: `8 * clip(circle_error 改善量, ±0.05)`，奖励逐步贴近目标
- radial: `12 * |r-r0| + 30 * (r-r0)^2`，惩罚末端偏离期望圆半径
- safety: 机器人/组织/病灶应变惩罚
- over_contact: 过大接触力惩罚
- lap_bonus: 完成一圈且 `circle_error < 0.01 m` 时额外 +15

### 4.4 Termination

- 安全超限（应变/应力/过大接触力）
- 或达到成功条件：
  - `t >= circle_period_steps`
  - `circle_error < circle_target_tolerance`（当前 `0.01 m`）

---

## 5. Training & Runtime

推荐启动方式：

```bash
scripts/train_task.sh start
scripts/train_task.sh status
scripts/train_task.sh logs sofa
scripts/train_task.sh logs train
scripts/train_task.sh stop   # 训练完成后停止 SOFA/RL tmux 会话
```

默认解释器策略：

- SOFA: conda env `sofa_rl`
- Train: `~/omniverse/python.sh`（若存在）

可覆盖变量示例：

```bash
SOFA_CONDA_ENV=sofa_rl \
TRAIN_PYTHON_BIN=/home/jhuo/omniverse/python.sh \
TIMESTEPS=200000 \
scripts/train_task.sh restart
```

---

## 6. Demo Animation

生成机器人+组织同屏 GIF：

```bash
MOTION_SCALE=3 FRAME_STRIDE=1 scripts/make_demo_gif.sh logs/sofa_robot_tissue.gif
```

GIF 图例（轨迹叠加均使用**物理坐标**，`MOTION_SCALE` 仅放大组织形变）：

- 蓝色虚线圆：完整期望圆轨迹（半径 `TARGET_RADIUS=0.05`，圆心优先读 `frame_metrics.csv`）
- 绿色星标路径：RL 每步 `circle_target`（相位按 `CIRCLE_PERIOD_STEPS=240` 计算）
- 橙色实线/圆点：连续体末端实际轨迹（VTK `points[-1]`）
- 粉色点云：组织；机器人点云颜色表示接触力（若存在 metrics CSV）

同时会生成 `logs/sofa_robot_tissue_trajectories.csv`，包含期望圆轨迹采样点与各帧末端坐标。

若存在 `sofa/vtk_output/frame_metrics.csv`，GIF 会读取每帧 `contact_force_peak`，用机器人颜色和色条显示接触力大小。

可选：

- `ROBOT_GLOB=sofa/vtk_output/robot/frame_*.vtk`
- `TISSUE_GLOB=sofa/vtk_output/tissue/frame_*.vtk`

---

## 7. Current Limitations

1. `von_mises` 仍为位移驱动的代理指标，不是完整本构后处理真值。
2. 接触力统计是求解器约束层面的全局统计，后续可细化为“病灶邻域接触力”。
3. 每个 RL action 已在 SOFA 侧执行 `PHYSICS_SUBSTEPS=5` 次物理积分，提升组织响应幅度。

---

## 8. Suggested Next Steps

1. 将接触力统计按空间区域分解（病灶区域、非病灶区域）。
2. 引入组织粘弹性或更高保真材料模型。
3. 增加 curriculum（无接触 -> 轻接触 -> 病灶操作）配置化训练。
