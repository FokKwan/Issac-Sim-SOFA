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

1. PPO 输出归一化动作 `action in [-1, 1]`
2. `SoftSofaEnv.step()` 将动作映射为 `cable_disp`
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
  - 接触判定：`LocalMinDistance(alarmDistance=0.006, contactDistance=0.002)`
  - 响应：`CollisionResponse(FrictionContactConstraint, mu=0.2)`

- Robot (`SoftBody`):
  - `RegularGridTopology` (3x3x10)
  - `MechanicalObject(Vec3d)`
  - `HexahedronFEMForceField(method="large")`
  - `CableConstraint`（动作控制入口）
  - `BoxROI + FixedConstraint`（基座固定）

- Tissue (`TargetTissue`):
  - `RegularGridTopology` (8x5x6)
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

导出间隔通过环境变量控制：

```bash
SOFA_EXPORT_INTERVAL=5  # 每 5 步导出一次
```

---

## 4. RL Environment (`issac_sim/envs/sofa_env.py`)

### 4.1 Action Space

- `Box(low=-1, high=1, shape=(1,))`
- 实际控制量：`cable_disp = action * action_scale`

### 4.2 Observation Space

字典观测包含：

- `tip_pos`
- `von_mises`, `avg_strain`
- `lesion_distance`, `lesion_strain`, `tissue_strain`
- `contact_distance`
- `contact_force_mean`, `contact_force_peak`, `contact_force_total`
- `lesion_center`

### 4.3 Reward Design

`reward = task + contact - safety - over_contact + success_bonus`

- task: 负的病灶距离（越近越好）
- contact: 使用 `contact_force_peak` 的饱和激励（`tanh`）
- safety: 机器人/组织/病灶应变惩罚
- over_contact: 过大接触力惩罚
- success_bonus: 达成接近+合理接触时额外奖励

### 4.4 Termination

- 安全超限（应变/应力/过大接触力）
- 或达到成功条件：
  - `lesion_distance < success_distance`
  - `min_contact_force < contact_force_peak < max_contact_force`

---

## 5. Training & Runtime

推荐启动方式：

```bash
scripts/train_task.sh start
scripts/train_task.sh status
scripts/train_task.sh logs sofa
scripts/train_task.sh logs train
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
MOTION_SCALE=3 FRAME_STRIDE=5 scripts/make_demo_gif.sh logs/sofa_robot_tissue.gif
```

可选：

- `ROBOT_GLOB=sofa/vtk_output/robot/frame_*.vtk`
- `TISSUE_GLOB=sofa/vtk_output/tissue/frame_*.vtk`

---

## 7. Current Limitations

1. `von_mises` 仍为位移驱动的代理指标，不是完整本构后处理真值。
2. 接触力统计是求解器约束层面的全局统计，后续可细化为“病灶邻域接触力”。
3. 目前单 action 对应单物理步，后续可增加 action repeat/substeps 提升稳定性。

---

## 8. Suggested Next Steps

1. 将接触力统计按空间区域分解（病灶区域、非病灶区域）。
2. 引入组织粘弹性或更高保真材料模型。
3. 增加 curriculum（无接触 -> 轻接触 -> 病灶操作）配置化训练。
