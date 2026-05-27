import Sofa
import Sofa.Core
import SofaRuntime
import numpy as np
import zmq
import json
import os
import meshio

# 确保加载核心插件
SofaRuntime.importPlugin("Sofa.Component")
SofaRuntime.importPlugin("SoftRobots")

ROBOT_YOUNG_MODULUS = 3000.0
ROBOT_POISSON_RATIO = 0.45
TISSUE_YOUNG_MODULUS = 1200.0
TISSUE_POISSON_RATIO = 0.46
CABLE_DISP_LIMIT = 1.5
CABLE_DISP_SCALE = 1.0
DEFAULT_EXPORT_INTERVAL = 10
CONTACT_FORCE_DISTANCE_GATE = 0.004

def createScene(root):
    """
    构建软体机器人+目标组织+病灶区域场景
    """
    root.dt = 0.01
    root.gravity = [0, -9.81, 0]
    root.addObject("FreeMotionAnimationLoop")
    root.addObject(
        "GenericConstraintSolver",
        name="constraint_solver",
        maxIterations=300,
        tolerance=1.0e-6,
        computeConstraintForces=True,
    )

    # 基础接触流水线，让软体机器人和组织在同一 SOFA 物理世界交互
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

    # 1) 连续体机器人节点
    soft = root.addChild("SoftBody")
    soft.addObject("EulerImplicitSolver")
    soft.addObject("CGLinearSolver", iterations=200, tolerance=1e-9, threshold=1e-9)
    
    # 使用 RegularGridTopology 自动生成 3x3x10 的网格 (共90个点)
    soft.addObject('RegularGridTopology', name='grid', 
                   min=[0, 0, 0], max=[0.1, 0.1, 1.0], 
                   nx=3, ny=3, nz=10)
    
    # 状态量容器
    soft.addObject("MechanicalObject", name="dofs", template="Vec3d")
    
    # RegularGridTopology 生成的是 hexa 网格，配套使用 HexahedronFEMForceField
    soft.addObject(
        "HexahedronFEMForceField",
        name="fem",
        topology="@grid",
        method="large",
        youngModulus=ROBOT_YOUNG_MODULUS,
        poissonRatio=ROBOT_POISSON_RATIO,
    )
    
    soft.addObject("UniformMass", totalMass=0.5)
    soft.addObject("PointCollisionModel", name="collision", group=1)
    # 固定机器人基座，避免整体刚体漂移导致“看起来没有弯曲运动”
    soft.addObject(
        "BoxROI",
        name="base_roi",
        box=[-0.001, -0.001, -0.001, 0.101, 0.101, 0.08],
        drawBoxes=False,
    )
    soft.addObject("FixedConstraint", indices="@base_roi.indices")

    # 缆绳约束：索引需在网格点数范围内
    soft.addObject(
        "CableConstraint",
        name="cable",
        indices=[0, 9, 18, 27, 36, 45, 54, 63, 72, 81], 
        value=[0.0],
        valueType="displacement",
        pullPoint=[0, 0, -0.1]
    )

    # 2) 目标组织节点（包含病灶区域）
    tissue = root.addChild("TargetTissue")
    tissue.addObject("EulerImplicitSolver")
    tissue.addObject("CGLinearSolver", iterations=200, tolerance=1e-9, threshold=1e-9)
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
    tissue.addObject(
        "HexahedronFEMForceField",
        name="fem",
        topology="@grid",
        method="large",
        youngModulus=TISSUE_YOUNG_MODULUS,
        poissonRatio=TISSUE_POISSON_RATIO,
    )
    tissue.addObject("UniformMass", totalMass=0.8)
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
    try:
        raw_values = constraint_solver.constraintForces.value
    except Exception:
        return np.array([], dtype=np.float64)
    return np.asarray(raw_values, dtype=np.float64).reshape(-1)

def main():
    root = Sofa.Core.Node("root")
    createScene(root)
    
    # 1. 初始化仿真
    Sofa.Simulation.init(root)

    # 2. 【核心修复】由于 Headless 模式下 init 后数据可能未同步
    # 我们手动运行一个极小的 animate 来强制引擎计算并填充网格数据
    print("正在激活网格拓扑...")
    Sofa.Simulation.animate(root, 0.0001) 

    robot_topo_cells = extract_topology_cells(root.SoftBody.grid)
    tissue_topo_cells = extract_topology_cells(root.TargetTissue.grid)
    if robot_topo_cells is None or tissue_topo_cells is None:
        print("[Error] 无法获取机器人或组织拓扑数据，请检查网格参数配置。")
        return
    print(f"Robot topology: {robot_topo_cells[0][0]}, count={len(robot_topo_cells[0][1])}")
    print(f"Tissue topology: {tissue_topo_cells[0][0]}, count={len(tissue_topo_cells[0][1])}")

    cable = root.SoftBody.cable
    robot_dofs = root.SoftBody.dofs
    tissue_dofs = root.TargetTissue.dofs

    robot_rest_positions = np.array(robot_dofs.position.value, copy=True)
    tissue_rest_positions = np.array(tissue_dofs.position.value, copy=True)
    rest_tip_position = np.array(robot_rest_positions[-1], copy=True)
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
    sock.bind("tcp://*:5555")
    
    robot_vtk_dir = os.path.join("vtk_output", "robot")
    tissue_vtk_dir = os.path.join("vtk_output", "tissue")
    os.makedirs(robot_vtk_dir, exist_ok=True)
    os.makedirs(tissue_vtk_dir, exist_ok=True)
    print("[SOFA] Headless Server Ready (Manual Export Mode)...")
    export_interval = max(1, int(os.environ.get("SOFA_EXPORT_INTERVAL", str(DEFAULT_EXPORT_INTERVAL))))
    print(f"[SOFA] Export interval: every {export_interval} steps")

    step_count = 0

    try:
        while True:
            # 接收来自 Isaac Sim 的指令
            msg = sock.recv_string()
            cmd = json.loads(msg)

            if cmd.get("type") == "reset":
                Sofa.Simulation.reset(root)
                step_count = 0
                cable.value = [0.0]
                # reset 后推进一个小步以刷新位置缓存
                Sofa.Simulation.animate(root, 0.0001)
                robot_rest_positions = np.array(robot_dofs.position.value, copy=True)
                tissue_rest_positions = np.array(tissue_dofs.position.value, copy=True)
                rest_tip_position = np.array(robot_rest_positions[-1], copy=True)
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
                cable_disp = float(np.clip(raw_cable_disp * CABLE_DISP_SCALE, -CABLE_DISP_LIMIT, CABLE_DISP_LIMIT))
                cable.value = [cable_disp]
                
                # 物理步进
                Sofa.Simulation.animate(root, float(root.dt.value))
                step_count += 1

            # 周期导出 VTK 供 ParaView / GIF 可视化
            if step_count % export_interval == 0:
                try:
                    robot_points = np.array(robot_dofs.position.value)
                    tissue_points = np.array(tissue_dofs.position.value)
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