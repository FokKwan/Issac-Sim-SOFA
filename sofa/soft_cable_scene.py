import Sofa
import Sofa.Core
import SofaRuntime
import numpy as np
import zmq
import json
import os
import meshio

# 本文件是 SOFA 物理服务器入口，负责：
# 1) 构建“连续体机器人 + 组织 + 病灶”同场景仿真
# 2) 接收 Isaac/RL 发来的控制指令并步进仿真
# 3) 回传可用于训练的状态与安全指标
#
# 运行方式：python soft_cable_scene.py
# 对外接口：ZMQ REP 监听 tcp://*:5555（与 Isaac 侧 REQ 对接）

# 确保加载核心插件
SofaRuntime.importPlugin("Sofa.Component")
SofaRuntime.importPlugin("SoftRobots")

# 机器人材料参数（线弹性近似）
ROBOT_YOUNG_MODULUS = 3000.0
ROBOT_POISSON_RATIO = 0.45
# 组织材料参数（更软）
TISSUE_YOUNG_MODULUS = 1200.0
TISSUE_POISSON_RATIO = 0.46
# 分段常曲率（PCC）机器人参数
PCC_SEGMENT_LENGTHS = [0.25, 0.25, 0.25, 0.25]
PCC_SEGMENT_WEIGHTS = [1.00, 0.85, 0.70, 0.55]
PCC_POINTS_PER_SEGMENT = 8
PCC_MAX_CURVATURE = 4.0  # 1/m
# 缆绳控制参数：限制绝对位移，避免数值发散
CABLE_DISP_LIMIT = 1.5
CABLE_DISP_SCALE = 1.0
# VTK 导出默认间隔（每 N 个仿真 step 导出一次）
DEFAULT_EXPORT_INTERVAL = 10
# 力统计距离门限：距离大于该值时视作“非接触”
CONTACT_FORCE_DISTANCE_GATE = 0.004


def build_line_edges(point_count):
    if point_count <= 1:
        return np.zeros((0, 2), dtype=np.int32)
    start_idx = np.arange(0, point_count - 1, dtype=np.int32)
    end_idx = np.arange(1, point_count, dtype=np.int32)
    return np.stack([start_idx, end_idx], axis=1)


def generate_segmented_constant_curvature_points(curvature_command):
    """
    生成固定基端的分段常曲率（PCC）中心线。

    Args:
        curvature_command: 全局控制曲率（标量），每段按权重缩放后保持常数。

    Returns:
        np.ndarray: (N, 3) 机器人中心线点集
    """
    kappa = float(np.clip(curvature_command, -PCC_MAX_CURVATURE, PCC_MAX_CURVATURE))
    points = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    theta = 0.0
    current = points[0].copy()

    for seg_len, seg_weight in zip(PCC_SEGMENT_LENGTHS, PCC_SEGMENT_WEIGHTS):
        seg_kappa = kappa * seg_weight
        ds = seg_len / float(PCC_POINTS_PER_SEGMENT)
        for _ in range(PCC_POINTS_PER_SEGMENT):
            theta += seg_kappa * ds
            # 平面常曲率：弯曲发生在 X-Z 平面
            current[0] += np.sin(theta) * ds
            current[2] += np.cos(theta) * ds
            points.append(current.copy())
    return np.asarray(points, dtype=np.float64)

def createScene(root):
    """
    构建软体机器人+目标组织+病灶区域场景。

    Args:
        root: SOFA 根节点
    """
    # 全局积分步长与重力
    root.dt = 0.01
    root.gravity = [0, -9.81, 0]
    # 约束求解循环：适用于接触/约束问题
    root.addObject("FreeMotionAnimationLoop")
    root.addObject(
        "GenericConstraintSolver",
        name="constraint_solver",
        maxIterations=300,
        tolerance=1.0e-6,
        computeConstraintForces=True,
    )

    # 基础接触流水线：
    # - LocalMinDistance 的 alarmDistance/contactDistance 控制接触检测敏感度
    # - CollisionResponse 使用摩擦接触约束，mu 为摩擦系数
    root.addObject("CollisionPipeline")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject("LocalMinDistance", alarmDistance=0.006, contactDistance=0.002, angleCone=0.0)
    root.addObject(
        "CollisionResponse",
        name="contact_response",
        response="FrictionContactConstraint",
        responseParams="mu=0.2",
    )

    # 1) 连续体机器人节点（分段常曲率 PCC 模型）
    soft = root.addChild("SoftBody")
    pcc_points = generate_segmented_constant_curvature_points(curvature_command=0.0)
    pcc_edges = build_line_edges(point_count=len(pcc_points))
    soft.addObject("EdgeSetTopologyContainer", name="line_topology", edges=pcc_edges.tolist())
    # 机器人机械状态：由控制器每步按 PCC 更新位置（固定基端）
    soft.addObject("MechanicalObject", name="dofs", template="Vec3d", position=pcc_points.tolist())
    # 同时启用点碰撞和线碰撞，增强细长体接触稳定性
    soft.addObject("PointCollisionModel", name="collision_points", group=1)
    soft.addObject("LineCollisionModel", name="collision_lines", group=1)

    # 2) 目标组织节点（包含病灶区域）
    tissue = root.addChild("TargetTissue")
    tissue.addObject("EulerImplicitSolver")
    tissue.addObject("CGLinearSolver", iterations=200, tolerance=1e-9, threshold=1e-9)
    # 组织网格参数（比机器人更密）
    tissue.addObject(
        "RegularGridTopology",
        name="grid",
        min=[-0.03, -0.02, 0.72],
        max=[0.13, 0.05, 0.94],
        nx=8,
        ny=5,
        nz=6,
    )
    tissue.addObject("MechanicalObject", name="dofs", template="Vec3d")
    # 组织材料参数（更软）
    tissue.addObject(
        "HexahedronFEMForceField",
        name="fem",
        topology="@grid",
        method="large",
        youngModulus=TISSUE_YOUNG_MODULUS,
        poissonRatio=TISSUE_POISSON_RATIO,
    )
    tissue.addObject("UniformMass", totalMass=0.8)
    # 固定组织部分边界，模拟与周围组织连接
    tissue.addObject(
        "BoxROI",
        name="fixed_roi",
        box=[-0.04, -0.021, 0.70, 0.14, -0.013, 0.96],
        drawBoxes=False,
    )
    tissue.addObject("FixedConstraint", indices="@fixed_roi.indices")
    tissue.addObject("PointCollisionModel", name="collision", group=2)
    return root


def compute_mechanics_metrics(positions, rest_positions, young_modulus):
    """
    用位移场构造代理应力/应变指标（训练信号近似）。

    Args:
        positions: 当前节点坐标 (N,3)
        rest_positions: 参考坐标 (N,3)
        young_modulus: 对应材料杨氏模量

    Returns:
        (von_mises_proxy, avg_strain)
    """
    if positions.size == 0 or rest_positions.size == 0:
        return 0.0, 0.0

    displacement = positions - rest_positions
    displacement_norm = np.linalg.norm(displacement, axis=1)
    bbox_min = np.min(rest_positions, axis=0)
    bbox_max = np.max(rest_positions, axis=0)
    char_length = max(np.linalg.norm(bbox_max - bbox_min), 1e-8)

    avg_strain = float(np.mean(displacement_norm) / char_length)
    # 当前场景使用位移比例近似应力指标，避免奖励函数失真
    von_mises_proxy = float(young_modulus * avg_strain)
    return von_mises_proxy, avg_strain


def compute_lesion_mask(rest_positions, lesion_center, lesion_radius):
    """
    在组织参考网格中生成病灶节点索引。

    Args:
        rest_positions: 组织参考坐标
        lesion_center: 病灶中心
        lesion_radius: 病灶半径
    """
    if rest_positions.size == 0:
        return np.array([], dtype=np.int64)
    dist = np.linalg.norm(rest_positions - lesion_center, axis=1)
    lesion_indices = np.where(dist <= lesion_radius)[0]
    if lesion_indices.size == 0:
        # 至少保留一个最近点，保证病灶指标可计算
        lesion_indices = np.array([int(np.argmin(dist))], dtype=np.int64)
    return lesion_indices


def compute_tissue_interaction_metrics(
    tip_position,
    tissue_positions,
    tissue_rest_positions,
    lesion_indices,
):
    """
    计算机器人末端与组织/病灶的几何与应变关系。

    Returns:
        dict:
            lesion_distance: 末端到病灶中心距离
            lesion_strain: 病灶区域应变
            tissue_strain: 组织整体应变
            contact_distance: 末端到组织最近点距离
            lesion_center: 病灶中心坐标
    """
    if tissue_positions.size == 0:
        return {
            "lesion_distance": 0.0,
            "lesion_strain": 0.0,
            "tissue_strain": 0.0,
            "contact_distance": 1.0,
            "lesion_center": [0.0, 0.0, 0.0],
        }

    _, tissue_avg_strain = compute_mechanics_metrics(
        tissue_positions,
        tissue_rest_positions,
        TISSUE_YOUNG_MODULUS,
    )
    lesion_positions = tissue_positions[lesion_indices]
    lesion_rest_positions = tissue_rest_positions[lesion_indices]
    _, lesion_avg_strain = compute_mechanics_metrics(
        lesion_positions,
        lesion_rest_positions,
        TISSUE_YOUNG_MODULUS,
    )
    lesion_center = np.mean(lesion_positions, axis=0)
    lesion_distance = float(np.linalg.norm(tip_position - lesion_center))

    min_contact_distance = float(
        np.min(np.linalg.norm(tissue_positions - tip_position, axis=1))
    )
    return {
        "lesion_distance": lesion_distance,
        "lesion_strain": float(lesion_avg_strain),
        "tissue_strain": float(tissue_avg_strain),
        "contact_distance": min_contact_distance,
        "lesion_center": lesion_center.tolist(),
    }


def extract_topology_cells(grid_component):
    """
    从 SOFA 拓扑对象提取 meshio 可写的 cells。
    优先级：hexahedron > tetra > triangle。
    """
    topo_val = grid_component.hexahedra.value
    cell_type = "hexahedron"
    if len(topo_val) == 0:
        topo_val = grid_component.tetrahedra.value
        cell_type = "tetra"
    if len(topo_val) == 0:
        topo_val = grid_component.triangles.value
        cell_type = "triangle"
    if len(topo_val) == 0:
        topo_val = grid_component.getData("tetrahedra").value
        cell_type = "tetra"
    if len(topo_val) == 0:
        return None
    return [(cell_type, np.array(topo_val))]


def compute_contact_force_stats(constraint_forces, baseline_force_level, contact_distance):
    """
    从约束求解器力向量统计接触力强度。

    Args:
        constraint_forces: 求解器约束力向量
        baseline_force_level: reset 时的基线力统计(mean, peak, total)
        contact_distance: 几何接触距离（用于剔除远距离噪声）

    Returns:
        (force_mean, force_peak, force_total)
    """
    if constraint_forces is None:
        return 0.0, 0.0, 0.0
    force_array = np.asarray(constraint_forces, dtype=np.float64).reshape(-1)
    if force_array.size == 0:
        return 0.0, 0.0, 0.0

    abs_force = np.abs(force_array)
    force_mean = float(np.mean(abs_force))
    force_peak = float(np.max(abs_force))
    force_total = float(np.sum(abs_force))

    force_mean = max(0.0, force_mean - baseline_force_level[0])
    force_peak = max(0.0, force_peak - baseline_force_level[1])
    force_total = max(0.0, force_total - baseline_force_level[2])

    if contact_distance > CONTACT_FORCE_DISTANCE_GATE:
        # 距离较远时将约束力视作非接触力（例如固定约束或数值噪声）
        return 0.0, 0.0, 0.0
    return force_mean, force_peak, force_total


def read_constraint_forces(constraint_solver):
    """安全读取 constraintForces，失败则返回空数组。"""
    try:
        raw_values = constraint_solver.constraintForces.value
    except Exception:
        return np.array([], dtype=np.float64)
    return np.asarray(raw_values, dtype=np.float64).reshape(-1)


def apply_robot_pcc_shape(robot_dofs, curvature_command):
    """
    按分段常曲率模型更新机器人中心线点位（固定基端）。

    Args:
        robot_dofs: SOFA MechanicalObject
        curvature_command: 目标曲率控制量

    Returns:
        np.ndarray: 更新后的中心线点集
    """
    points = generate_segmented_constant_curvature_points(curvature_command=curvature_command)
    robot_dofs.position.value = points.tolist()
    return points

def main():
    """SOFA headless 服务主循环。"""
    root = Sofa.Core.Node("root")
    createScene(root)
    
    # 1. 初始化仿真
    Sofa.Simulation.init(root)

    # 2. 【核心修复】由于 Headless 模式下 init 后数据可能未同步
    # 我们手动运行一个极小的 animate 来强制引擎计算并填充网格数据
    print("正在激活网格拓扑...")
    Sofa.Simulation.animate(root, 0.0001) 

    # 提取机器人和组织拓扑，供 VTK 导出
    robot_edges = np.asarray(root.SoftBody.line_topology.edges.value, dtype=np.int32)
    robot_topo_cells = [("line", robot_edges)] if robot_edges.size > 0 else None
    tissue_topo_cells = extract_topology_cells(root.TargetTissue.grid)
    if robot_topo_cells is None or tissue_topo_cells is None:
        print("[Error] 无法获取机器人或组织拓扑数据，请检查网格参数配置。")
        return
    print(f"Robot topology: {robot_topo_cells[0][0]}, count={len(robot_topo_cells[0][1])}")
    print(f"Tissue topology: {tissue_topo_cells[0][0]}, count={len(tissue_topo_cells[0][1])}")

    robot_dofs = root.SoftBody.dofs
    tissue_dofs = root.TargetTissue.dofs
    current_curvature_cmd = 0.0

    robot_rest_positions = apply_robot_pcc_shape(robot_dofs, curvature_command=0.0)
    tissue_rest_positions = np.array(tissue_dofs.position.value, copy=True)
    rest_tip_position = np.array(robot_rest_positions[-1], copy=True)
    # 病灶初始中心（可作为任务配置参数暴露给上层）
    lesion_center_ref = np.array([0.05, 0.015, 0.82])
    lesion_indices = compute_lesion_mask(
        tissue_rest_positions,
        lesion_center=lesion_center_ref,
        lesion_radius=0.025,
    )
    print(f"Lesion region nodes: {len(lesion_indices)}")
    baseline_force_level = (0.0, 0.0, 0.0)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    # ZMQ REP 端口，与 Isaac 侧 REQ 一一配对通信
    sock.bind("tcp://*:5555")
    
    robot_vtk_dir = os.path.join("vtk_output", "robot")
    tissue_vtk_dir = os.path.join("vtk_output", "tissue")
    os.makedirs(robot_vtk_dir, exist_ok=True)
    os.makedirs(tissue_vtk_dir, exist_ok=True)
    print("[SOFA] Headless Server Ready (Manual Export Mode)...")
    # SOFA_EXPORT_INTERVAL 可通过环境变量覆盖，便于平衡 IO 和可视化密度
    export_interval = max(1, int(os.environ.get("SOFA_EXPORT_INTERVAL", str(DEFAULT_EXPORT_INTERVAL))))
    print(f"[SOFA] Export interval: every {export_interval} steps")

    step_count = 0

    try:
        while True:
            # 接收来自 Isaac Sim 的指令
            msg = sock.recv_string()
            cmd = json.loads(msg)

            if cmd.get("type") == "reset":
                # reset：重置场景、控制量与参考状态，用于 episode 开始
                Sofa.Simulation.reset(root)
                step_count = 0
                current_curvature_cmd = 0.0
                robot_rest_positions = apply_robot_pcc_shape(robot_dofs, curvature_command=current_curvature_cmd)
                # reset 后推进一个小步以刷新位置缓存
                Sofa.Simulation.animate(root, 0.0001)
                tissue_rest_positions = np.array(tissue_dofs.position.value, copy=True)
                rest_tip_position = np.array(robot_rest_positions[-1], copy=True)
                # 记录 reset 基线约束力，用于运行时剔除静态约束背景噪声
                solver_forces = read_constraint_forces(root.constraint_solver)
                if solver_forces.size > 0:
                    abs_solver_forces = np.abs(solver_forces)
                    baseline_force_level = (
                        float(np.mean(abs_solver_forces)),
                        float(np.max(abs_solver_forces)),
                        float(np.sum(abs_solver_forces)),
                    )
                else:
                    baseline_force_level = (0.0, 0.0, 0.0)
            else:
                # 执行一步控制
                raw_cable_disp = float(cmd.get("cable_disp", 0.0))
                # 动作缩放+限幅，防止超大控制导致数值问题
                current_curvature_cmd = float(
                    np.clip(raw_cable_disp * CABLE_DISP_SCALE, -CABLE_DISP_LIMIT, CABLE_DISP_LIMIT)
                )
                apply_robot_pcc_shape(robot_dofs, curvature_command=current_curvature_cmd)
                
                # 物理步进
                Sofa.Simulation.animate(root, float(root.dt.value))
                step_count += 1

            # 周期导出 VTK 供 ParaView / GIF 可视化
            if step_count % export_interval == 0:
                try:
                    robot_points = np.array(robot_dofs.position.value)
                    tissue_points = np.array(tissue_dofs.position.value)
                    # 分别导出机器人与组织，便于叠加动画
                    if len(robot_points) > 0:
                        robot_mesh = meshio.Mesh(points=robot_points, cells=robot_topo_cells)
                        robot_file = os.path.join(robot_vtk_dir, f"frame_{step_count:04d}.vtk")
                        robot_mesh.write(robot_file)
                    if len(tissue_points) > 0:
                        tissue_mesh = meshio.Mesh(points=tissue_points, cells=tissue_topo_cells)
                        tissue_file = os.path.join(tissue_vtk_dir, f"frame_{step_count:04d}.vtk")
                        tissue_mesh.write(tissue_file)
                except Exception as e:
                    print(f"VTK Export Error: {e}")

            # 反馈当前状态
            robot_positions = np.array(robot_dofs.position.value)
            tissue_positions = np.array(tissue_dofs.position.value)
            if len(robot_positions) == 0:
                tip_position = [0.0, 0.0, 0.0]
                von_mises = 0.0
                avg_strain = 0.0
                lesion_metrics = {
                    "lesion_distance": 0.0,
                    "lesion_strain": 0.0,
                    "tissue_strain": 0.0,
                    "contact_distance": 1.0,
                    "lesion_center": [0.0, 0.0, 0.0],
                }
            else:
                tip_np = robot_positions[-1]
                tip_position = tip_np.tolist()
                tip_displacement = float(np.linalg.norm(tip_np - rest_tip_position))
                von_mises, avg_strain = compute_mechanics_metrics(
                    robot_positions,
                    robot_rest_positions,
                    ROBOT_YOUNG_MODULUS,
                )
                lesion_metrics = compute_tissue_interaction_metrics(
                    tip_position=tip_np,
                    tissue_positions=tissue_positions,
                    tissue_rest_positions=tissue_rest_positions,
                    lesion_indices=lesion_indices,
                )
            # 读取约束力统计作为真实接触信号
            contact_force_mean, contact_force_peak, contact_force_total = compute_contact_force_stats(
                constraint_forces=read_constraint_forces(root.constraint_solver),
                baseline_force_level=baseline_force_level,
                contact_distance=lesion_metrics["contact_distance"],
            )
            if len(robot_positions) == 0:
                tip_displacement = 0.0
                contact_force_mean = 0.0
                contact_force_peak = 0.0
                contact_force_total = 0.0

            reply = {
                # step: SOFA 物理步计数
                "step": step_count,
                "tip_position": tip_position,
                "tip_displacement": tip_displacement,
                "von_mises": von_mises,
                "avg_strain": avg_strain,
                "lesion_distance": lesion_metrics["lesion_distance"],
                "lesion_strain": lesion_metrics["lesion_strain"],
                "tissue_strain": lesion_metrics["tissue_strain"],
                "contact_distance": lesion_metrics["contact_distance"],
                "contact_force_mean": contact_force_mean,
                "contact_force_peak": contact_force_peak,
                "contact_force_total": contact_force_total,
                "lesion_center": lesion_metrics["lesion_center"],
            }
            sock.send_string(json.dumps(reply))
    finally:
        sock.close()
        ctx.term()

if __name__ == '__main__':
    main()