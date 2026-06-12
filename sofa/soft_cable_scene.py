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
# 组织材料参数（更软，便于产生可见形变）
TISSUE_YOUNG_MODULUS = 800.0
TISSUE_POISSON_RATIO = 0.46
# 病灶材料参数：ESD 场景下的早期消化道病灶/纤维化黏膜下层近似。
# 当前 SOFA 网格仍为单连续组织体，病灶材料用于局部应力与节点反力导出。
LESION_YOUNG_MODULUS = 3000.0
LESION_POISSON_RATIO = 0.47
# 分段常曲率（PCC）机器人参数
PCC_SEGMENT_LENGTHS = [0.30, 0.30, 0.30, 0.30]
PCC_SEGMENT_WEIGHTS = [1.00, 0.95, 0.90, 0.85]
PCC_POINTS_PER_SEGMENT = 8
PCC_MAX_CURVATURE = 1.20  # 1/m，固定基端后需要更大曲率进入组织内部画小圆
PCC_CURVATURE_DOF = 4  # [ky_prox, ky_dist, kz_prox, kz_dist]，每个平面两段曲率可形成 S 型
PCC_ACTION_DOF = PCC_CURVATURE_DOF + 1
# 机器人基座初始偏移（左侧固定，主轴沿 +x 方向）
# 初始 tip 在原基座附近 (0.10, -0.08)，负曲率时末端向组织/病灶方向弯曲。
PCC_BASE_OFFSET = np.array([-1.10, -0.08, 0.0], dtype=np.float64)
# 组织初始位置：整体放在机器人初始直线下方，避免 reset 首帧穿模。
# 初始机器人中心线为 y=-0.08，组织上表面 y=-0.10，保留 2 cm 间隙。
TISSUE_GRID_MIN = np.array([-0.18, -0.22, -0.06], dtype=np.float64)
TISSUE_GRID_MAX = np.array([0.25, -0.10, 0.06], dtype=np.float64)
LESION_CENTER_REF = np.array([0.08, -0.14, 0.0], dtype=np.float64)
LESION_RADIUS = 0.025
LESION_SURFACE_BAND = 0.006
# 缆绳控制参数：每步曲率/插入增量（累积控制）。
# 病灶圆轨迹所需曲率约在 +/-0.35 以内，较小增量可避免几步内甩出工作区。
CURVATURE_DELTA_SCALE = 0.03
CURVATURE_DELTA_LIMIT = 0.04
INSERTION_DELTA_SCALE = 0.01
INSERTION_DELTA_LIMIT = 0.015
INSERTION_LIMIT = 0.08
# tip 端限速：单个 RL step 内最大几何位移，避免曲率/插入命令造成末端突跳。
TIP_MAX_STEP_DISPLACEMENT = 0.006
# 每个 RL step 执行的物理子步数，提升组织响应幅度
PHYSICS_SUBSTEPS = 5
# VTK 导出默认间隔（每 N 个仿真 step 导出一次）。
# 默认更密集导出，避免生成的 GIF 帧数过少；长时间训练可用环境变量调大。
DEFAULT_EXPORT_INTERVAL = 2
# 力统计距离门限：距离大于该值时视作“非接触”
CONTACT_FORCE_DISTANCE_GATE = 0.004


def build_line_edges(point_count):
    if point_count <= 1:
        return np.zeros((0, 2), dtype=np.int32)
    start_idx = np.arange(0, point_count - 1, dtype=np.int32)
    end_idx = np.arange(1, point_count, dtype=np.int32)
    return np.stack([start_idx, end_idx], axis=1)


def normalize_s_curve_curvature_command(curvature_command):
    """
    统一曲率命令格式。

    新格式为 [ky_prox, ky_dist, kz_prox, kz_dist]：
    - ky_prox/ky_dist 控制 X-Y 平面前半段/后半段弯曲
    - kz_prox/kz_dist 控制 X-Z 平面前半段/后半段弯曲

    兼容旧格式 [ky, kz]，会扩展为同向 C 型曲率 [ky, ky, kz, kz]。
    """
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

    # 每个物理段同时存在 ky/kz，限制平面合成曲率，避免对角方向超过最大曲率。
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


def generate_segmented_constant_curvature_points(curvature_command, insertion_offset=0.0):
    """
    生成固定基端的分段常曲率（PCC）中心线。

    Args:
        curvature_command: S 型曲率向量 [ky_prox, ky_dist, kz_prox, kz_dist]。
        insertion_offset: 固定基端下的有效伸出长度变化，正值让 tip 端伸出。

    Returns:
        np.ndarray: (N, 3) 机器人中心线点集
    """
    curvature = normalize_s_curve_curvature_command(curvature_command)
    insertion_offset = float(np.clip(insertion_offset, -INSERTION_LIMIT, INSERTION_LIMIT))
    nominal_length = max(float(np.sum(PCC_SEGMENT_LENGTHS)), 1e-8)
    length_scale = max(0.1, (nominal_length + insertion_offset) / nominal_length)
    base_position = PCC_BASE_OFFSET.copy()
    points = [base_position.copy()]
    theta_y = 0.0
    theta_z = 0.0
    current = points[0].copy()

    segment_count = len(PCC_SEGMENT_LENGTHS)
    for seg_idx, (seg_len, seg_weight) in enumerate(zip(PCC_SEGMENT_LENGTHS, PCC_SEGMENT_WEIGHTS)):
        seg_curvature = segment_curvature_from_s_command(curvature, seg_idx, segment_count) * seg_weight
        ds = (seg_len * length_scale) / float(PCC_POINTS_PER_SEGMENT)
        for _ in range(PCC_POINTS_PER_SEGMENT):
            theta_y += seg_curvature[0] * ds
            theta_z += seg_curvature[1] * ds
            # 主轴沿 +X。前/后半段分别使用独立曲率，允许同一平面 S 型弯曲。
            current[0] += np.cos(theta_y) * np.cos(theta_z) * ds
            current[1] += np.sin(theta_y) * ds
            current[2] += np.sin(theta_z) * ds
            points.append(current.copy())
    return np.asarray(points, dtype=np.float64)


def compute_point_cloud_aabb_clearance(points, bbox_min, bbox_max):
    """
    计算点云到轴对齐包围盒的最小外部距离；若点在盒内则距离为 0。
    """
    if points.size == 0:
        return float("inf")
    lower_gap = np.maximum(bbox_min - points, 0.0)
    upper_gap = np.maximum(points - bbox_max, 0.0)
    outside_delta = lower_gap + upper_gap
    return float(np.min(np.linalg.norm(outside_delta, axis=1)))

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
    root.addObject("LocalMinDistance", alarmDistance=0.005, contactDistance=0.0015, angleCone=0.0)
    root.addObject(
        "CollisionResponse",
        name="contact_response",
        response="FrictionContactConstraint",
        responseParams="mu=0.45",
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
        min=TISSUE_GRID_MIN.tolist(),
        max=TISSUE_GRID_MAX.tolist(),
        nx=10,
        ny=6,
        nz=5,
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
        box=[
            TISSUE_GRID_MIN[0] - 0.01,
            TISSUE_GRID_MIN[1] - 0.001,
            TISSUE_GRID_MIN[2] - 0.01,
            TISSUE_GRID_MAX[0] + 0.01,
            TISSUE_GRID_MIN[1] + 0.009,
            TISSUE_GRID_MAX[2] + 0.01,
        ],
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


def compute_lesion_surface_indices(rest_positions, lesion_center, lesion_radius, lesion_indices):
    """
    从病灶 ROI 中提取靠近球面外壳的节点，用于表面局部应力/反力导出。
    """
    if rest_positions.size == 0 or lesion_indices.size == 0:
        return np.array([], dtype=np.int64)
    lesion_rest_positions = rest_positions[lesion_indices]
    dist = np.linalg.norm(lesion_rest_positions - lesion_center, axis=1)
    inner_radius = max(0.0, lesion_radius - LESION_SURFACE_BAND)
    surface_indices = lesion_indices[dist >= inner_radius]
    if surface_indices.size == 0:
        surface_indices = lesion_indices
    return surface_indices.astype(np.int64)


def build_lesion_point_data(
    tissue_positions,
    tissue_rest_positions,
    lesion_indices,
    lesion_surface_indices,
):
    """
    构造写入 tissue VTK 的病灶局部场。

    lesion_surface_stress_proxy:
        病灶表面位移/半径得到的局部应变，再乘病灶材料杨氏模量。
    lesion_nodal_reaction_proxy:
        病灶节点等效刚度乘节点位移，用于估计病灶区域反力。
    """
    point_count = int(tissue_positions.shape[0])
    lesion_roi = np.zeros(point_count, dtype=np.float64)
    lesion_surface_roi = np.zeros(point_count, dtype=np.float64)
    lesion_surface_stress_proxy = np.zeros(point_count, dtype=np.float64)
    lesion_nodal_reaction_proxy = np.zeros(point_count, dtype=np.float64)

    if (
        point_count == 0
        or tissue_rest_positions.size == 0
        or lesion_indices.size == 0
    ):
        return {
            "lesion_roi": lesion_roi,
            "lesion_surface_roi": lesion_surface_roi,
            "lesion_surface_stress_proxy": lesion_surface_stress_proxy,
            "lesion_nodal_reaction_proxy": lesion_nodal_reaction_proxy,
        }

    lesion_indices = lesion_indices[lesion_indices < point_count]
    lesion_surface_indices = lesion_surface_indices[lesion_surface_indices < point_count]
    displacement_norm = np.linalg.norm(tissue_positions - tissue_rest_positions, axis=1)
    char_length = max(LESION_RADIUS, 1e-8)
    lesion_node_stiffness = LESION_YOUNG_MODULUS / char_length

    lesion_roi[lesion_indices] = 1.0
    lesion_surface_roi[lesion_surface_indices] = 1.0
    lesion_surface_stress_proxy[lesion_surface_indices] = (
        LESION_YOUNG_MODULUS * displacement_norm[lesion_surface_indices] / char_length
    )
    lesion_nodal_reaction_proxy[lesion_indices] = (
        lesion_node_stiffness * displacement_norm[lesion_indices]
    )

    return {
        "lesion_roi": lesion_roi,
        "lesion_surface_roi": lesion_surface_roi,
        "lesion_surface_stress_proxy": lesion_surface_stress_proxy,
        "lesion_nodal_reaction_proxy": lesion_nodal_reaction_proxy,
    }


def default_lesion_local_metrics():
    return {
        "lesion_surface_stress_mean": 0.0,
        "lesion_surface_stress_peak": 0.0,
        "lesion_nodal_reaction_mean": 0.0,
        "lesion_nodal_reaction_peak": 0.0,
        "lesion_nodal_reaction_total": 0.0,
        "lesion_contact_distance": 1.0,
        "lesion_contact_force_mean": 0.0,
        "lesion_contact_force_peak": 0.0,
        "lesion_contact_force_total": 0.0,
    }


def compute_lesion_local_metrics(
    tip_position,
    tissue_positions,
    tissue_rest_positions,
    lesion_indices,
    lesion_surface_indices,
    contact_force_mean,
    contact_force_peak,
    contact_force_total,
):
    """
    根据病灶表面局部应力和节点反力估计病灶区域接触力。
    """
    if (
        tissue_positions.size == 0
        or tissue_rest_positions.size == 0
        or lesion_indices.size == 0
    ):
        return default_lesion_local_metrics()

    point_data = build_lesion_point_data(
        tissue_positions=tissue_positions,
        tissue_rest_positions=tissue_rest_positions,
        lesion_indices=lesion_indices,
        lesion_surface_indices=lesion_surface_indices,
    )
    surface_indices = lesion_surface_indices
    if surface_indices.size == 0:
        surface_indices = lesion_indices

    surface_stress = point_data["lesion_surface_stress_proxy"][surface_indices]
    nodal_reaction = point_data["lesion_nodal_reaction_proxy"][lesion_indices]
    surface_positions = tissue_positions[surface_indices]
    surface_distances = np.linalg.norm(surface_positions - tip_position, axis=1)
    lesion_contact_distance = float(np.min(surface_distances)) if surface_distances.size else 1.0

    char_length = max(LESION_RADIUS, 1e-8)
    weights = np.exp(-np.square(surface_distances / char_length))
    surface_reaction = point_data["lesion_nodal_reaction_proxy"][surface_indices]
    weight_mean = float(np.mean(weights)) if weights.size else 0.0
    weight_peak = float(np.max(weights)) if weights.size else 0.0
    lesion_contact_force_mean = max(
        float(contact_force_mean) * weight_mean,
        float(np.mean(surface_reaction)) if surface_reaction.size else 0.0,
    )
    lesion_contact_force_peak = max(
        float(contact_force_peak) * weight_peak,
        float(np.max(surface_reaction)) if surface_reaction.size else 0.0,
    )
    lesion_contact_force_total = max(
        float(contact_force_total) * weight_mean,
        float(np.sum(surface_reaction)) if surface_reaction.size else 0.0,
    )

    return {
        "lesion_surface_stress_mean": float(np.mean(surface_stress)) if surface_stress.size else 0.0,
        "lesion_surface_stress_peak": float(np.max(surface_stress)) if surface_stress.size else 0.0,
        "lesion_nodal_reaction_mean": float(np.mean(nodal_reaction)) if nodal_reaction.size else 0.0,
        "lesion_nodal_reaction_peak": float(np.max(nodal_reaction)) if nodal_reaction.size else 0.0,
        "lesion_nodal_reaction_total": float(np.sum(nodal_reaction)) if nodal_reaction.size else 0.0,
        "lesion_contact_distance": lesion_contact_distance,
        "lesion_contact_force_mean": lesion_contact_force_mean,
        "lesion_contact_force_peak": lesion_contact_force_peak,
        "lesion_contact_force_total": lesion_contact_force_total,
    }


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


def apply_robot_pcc_shape(robot_dofs, curvature_command, insertion_offset=0.0):
    """
    按分段常曲率模型更新机器人中心线点位（固定基端）。

    Args:
        robot_dofs: SOFA MechanicalObject
        curvature_command: 目标曲率控制量
        insertion_offset: 固定基端下的有效伸出长度变化

    Returns:
        np.ndarray: 更新后的中心线点集
    """
    points = generate_segmented_constant_curvature_points(
        curvature_command=curvature_command,
        insertion_offset=insertion_offset,
    )
    robot_dofs.position.value = points.tolist()
    return points


def limit_tip_step_motion(
    previous_curvature,
    previous_insertion,
    target_curvature,
    target_insertion,
    current_tip_position,
):
    """
    限制单个控制 step 内的 tip 几何位移，避免速度突变。
    """
    target_points = generate_segmented_constant_curvature_points(
        curvature_command=target_curvature,
        insertion_offset=target_insertion,
    )
    target_tip_step = float(np.linalg.norm(target_points[-1] - current_tip_position))
    if target_tip_step <= TIP_MAX_STEP_DISPLACEMENT:
        return target_curvature, target_insertion, target_tip_step, False

    low = 0.0
    high = 1.0
    limited_curvature = np.asarray(previous_curvature, dtype=np.float64)
    limited_insertion = float(previous_insertion)
    limited_tip_step = 0.0
    for _ in range(12):
        alpha = 0.5 * (low + high)
        candidate_curvature = normalize_s_curve_curvature_command(
            previous_curvature + alpha * (target_curvature - previous_curvature)
        )
        candidate_insertion = float(
            np.clip(
                previous_insertion + alpha * (target_insertion - previous_insertion),
                -INSERTION_LIMIT,
                INSERTION_LIMIT,
            )
        )
        candidate_points = generate_segmented_constant_curvature_points(
            curvature_command=candidate_curvature,
            insertion_offset=candidate_insertion,
        )
        candidate_tip_step = float(np.linalg.norm(candidate_points[-1] - current_tip_position))
        if candidate_tip_step <= TIP_MAX_STEP_DISPLACEMENT:
            low = alpha
            limited_curvature = candidate_curvature
            limited_insertion = candidate_insertion
            limited_tip_step = candidate_tip_step
        else:
            high = alpha
    return limited_curvature, limited_insertion, limited_tip_step, True


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
    current_curvature_cmd = np.zeros(PCC_CURVATURE_DOF, dtype=np.float64)
    current_insertion_cmd = 0.0

    robot_rest_positions = apply_robot_pcc_shape(
        robot_dofs,
        curvature_command=0.0,
        insertion_offset=0.0,
    )
    tissue_rest_positions = np.array(tissue_dofs.position.value, copy=True)
    rest_tip_position = np.array(robot_rest_positions[-1], copy=True)
    initial_clearance = compute_point_cloud_aabb_clearance(
        robot_rest_positions,
        TISSUE_GRID_MIN,
        TISSUE_GRID_MAX,
    )
    print(
        "[SOFA] Initial robot-tissue AABB clearance: "
        f"{initial_clearance:.4f} m"
    )
    if initial_clearance <= 0.0:
        print("[WARN] Initial robot centerline overlaps the tissue AABB.")

    # 病灶初始中心（可作为任务配置参数暴露给上层）
    lesion_center_ref = LESION_CENTER_REF.copy()
    lesion_indices = compute_lesion_mask(
        tissue_rest_positions,
        lesion_center=lesion_center_ref,
        lesion_radius=LESION_RADIUS,
    )
    lesion_surface_indices = compute_lesion_surface_indices(
        tissue_rest_positions,
        lesion_center=lesion_center_ref,
        lesion_radius=LESION_RADIUS,
        lesion_indices=lesion_indices,
    )
    print(f"Lesion region nodes: {len(lesion_indices)}")
    print(f"Lesion surface nodes: {len(lesion_surface_indices)}")
    print(
        "[SOFA] Lesion material proxy: "
        f"E={LESION_YOUNG_MODULUS:.1f}, nu={LESION_POISSON_RATIO:.2f}; "
        f"background tissue E={TISSUE_YOUNG_MODULUS:.1f}, nu={TISSUE_POISSON_RATIO:.2f}"
    )
    baseline_force_level = (0.0, 0.0, 0.0)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    # ZMQ REP 端口，与 Isaac 侧 REQ 一一配对通信
    sock.bind("tcp://*:5555")
    
    robot_vtk_dir = os.path.join("vtk_output", "robot")
    tissue_vtk_dir = os.path.join("vtk_output", "tissue")
    os.makedirs(robot_vtk_dir, exist_ok=True)
    os.makedirs(tissue_vtk_dir, exist_ok=True)
    metrics_file = os.path.join("vtk_output", "frame_metrics.csv")
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(
            "step,tip_x,tip_y,tip_z,lesion_distance,contact_distance,"
            "contact_force_mean,contact_force_peak,contact_force_total,"
            "lesion_contact_distance,lesion_contact_force_mean,"
            "lesion_contact_force_peak,lesion_contact_force_total,"
            "lesion_surface_stress_mean,lesion_surface_stress_peak,"
            "lesion_nodal_reaction_mean,lesion_nodal_reaction_peak,"
            "lesion_nodal_reaction_total\n"
        )
    print("[SOFA] Headless Server Ready (Manual Export Mode)...")
    # SOFA_EXPORT_INTERVAL 可通过环境变量覆盖，便于平衡 IO 和可视化密度
    export_interval = max(1, int(os.environ.get("SOFA_EXPORT_INTERVAL", str(DEFAULT_EXPORT_INTERVAL))))
    print(f"[SOFA] Export interval: every {export_interval} steps")

    step_count = 0
    tip_step_displacement = 0.0
    tip_step_speed = 0.0
    tip_velocity_limited = False

    try:
        while True:
            # 接收来自 Isaac Sim 的指令
            msg = sock.recv_string()
            cmd = json.loads(msg)

            if cmd.get("type") == "reset":
                # reset：重置场景、控制量与参考状态，用于 episode 开始
                Sofa.Simulation.reset(root)
                step_count = 0
                tip_step_displacement = 0.0
                tip_step_speed = 0.0
                tip_velocity_limited = False
                current_curvature_cmd = np.zeros(PCC_CURVATURE_DOF, dtype=np.float64)
                current_insertion_cmd = 0.0
                robot_rest_positions = apply_robot_pcc_shape(
                    robot_dofs,
                    curvature_command=current_curvature_cmd,
                    insertion_offset=current_insertion_cmd,
                )
                # reset 后推进一个小步以刷新位置缓存
                Sofa.Simulation.animate(root, 0.0001)
                tissue_rest_positions = np.array(tissue_dofs.position.value, copy=True)
                rest_tip_position = np.array(robot_rest_positions[-1], copy=True)
                lesion_indices = compute_lesion_mask(
                    tissue_rest_positions,
                    lesion_center=lesion_center_ref,
                    lesion_radius=LESION_RADIUS,
                )
                lesion_surface_indices = compute_lesion_surface_indices(
                    tissue_rest_positions,
                    lesion_center=lesion_center_ref,
                    lesion_radius=LESION_RADIUS,
                    lesion_indices=lesion_indices,
                )
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
                # 执行一步控制：cable_disp 为曲率/插入增量，累积后限幅。
                # 新格式：[ky_prox_delta, ky_dist_delta, kz_prox_delta, kz_dist_delta, insertion_delta]
                raw_cable_disp = np.asarray(
                    cmd.get("cable_disp", np.zeros(PCC_ACTION_DOF, dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
                if raw_cable_disp.size == 0:
                    raw_cable_disp = np.zeros(PCC_ACTION_DOF, dtype=np.float64)
                elif raw_cable_disp.size < PCC_ACTION_DOF:
                    raw_cable_disp = np.pad(raw_cable_disp, (0, PCC_ACTION_DOF - raw_cable_disp.size))
                else:
                    raw_cable_disp = raw_cable_disp[:PCC_ACTION_DOF]
                curvature_delta = np.clip(
                    raw_cable_disp[:PCC_CURVATURE_DOF] * CURVATURE_DELTA_SCALE,
                    -CURVATURE_DELTA_LIMIT,
                    CURVATURE_DELTA_LIMIT,
                )
                insertion_delta = float(
                    np.clip(
                        raw_cable_disp[PCC_CURVATURE_DOF] * INSERTION_DELTA_SCALE,
                        -INSERTION_DELTA_LIMIT,
                        INSERTION_DELTA_LIMIT,
                    )
                )
                target_curvature_cmd = normalize_s_curve_curvature_command(
                    current_curvature_cmd + curvature_delta
                )
                target_insertion_cmd = float(
                    np.clip(
                        current_insertion_cmd + insertion_delta,
                        -INSERTION_LIMIT,
                        INSERTION_LIMIT,
                    )
                )
                current_tip_position = np.array(robot_dofs.position.value, dtype=np.float64)[-1]
                (
                    current_curvature_cmd,
                    current_insertion_cmd,
                    tip_step_displacement,
                    tip_velocity_limited,
                ) = limit_tip_step_motion(
                    previous_curvature=current_curvature_cmd,
                    previous_insertion=current_insertion_cmd,
                    target_curvature=target_curvature_cmd,
                    target_insertion=target_insertion_cmd,
                    current_tip_position=current_tip_position,
                )
                apply_robot_pcc_shape(
                    robot_dofs,
                    curvature_command=current_curvature_cmd,
                    insertion_offset=current_insertion_cmd,
                )

                # 多子步物理积分，使组织形变更充分
                dt = float(root.dt.value)
                for _ in range(PHYSICS_SUBSTEPS):
                    Sofa.Simulation.animate(root, dt)
                tip_step_speed = tip_step_displacement / max(dt * PHYSICS_SUBSTEPS, 1e-8)
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
                        tissue_point_data = build_lesion_point_data(
                            tissue_positions=tissue_points,
                            tissue_rest_positions=tissue_rest_positions,
                            lesion_indices=lesion_indices,
                            lesion_surface_indices=lesion_surface_indices,
                        )
                        tissue_mesh = meshio.Mesh(
                            points=tissue_points,
                            cells=tissue_topo_cells,
                            point_data=tissue_point_data,
                        )
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
                lesion_local_metrics = default_lesion_local_metrics()
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
                tip_step_displacement = 0.0
                tip_step_speed = 0.0
                tip_velocity_limited = False
            else:
                lesion_local_metrics = compute_lesion_local_metrics(
                    tip_position=robot_positions[-1],
                    tissue_positions=tissue_positions,
                    tissue_rest_positions=tissue_rest_positions,
                    lesion_indices=lesion_indices,
                    lesion_surface_indices=lesion_surface_indices,
                    contact_force_mean=contact_force_mean,
                    contact_force_peak=contact_force_peak,
                    contact_force_total=contact_force_total,
                )

            if step_count % export_interval == 0:
                try:
                    with open(metrics_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"{step_count},"
                            f"{float(tip_position[0]):.8f},"
                            f"{float(tip_position[1]):.8f},"
                            f"{float(tip_position[2]):.8f},"
                            f"{float(lesion_metrics['lesion_distance']):.8f},"
                            f"{float(lesion_metrics['contact_distance']):.8f},"
                            f"{contact_force_mean:.8f},"
                            f"{contact_force_peak:.8f},"
                            f"{contact_force_total:.8f},"
                            f"{lesion_local_metrics['lesion_contact_distance']:.8f},"
                            f"{lesion_local_metrics['lesion_contact_force_mean']:.8f},"
                            f"{lesion_local_metrics['lesion_contact_force_peak']:.8f},"
                            f"{lesion_local_metrics['lesion_contact_force_total']:.8f},"
                            f"{lesion_local_metrics['lesion_surface_stress_mean']:.8f},"
                            f"{lesion_local_metrics['lesion_surface_stress_peak']:.8f},"
                            f"{lesion_local_metrics['lesion_nodal_reaction_mean']:.8f},"
                            f"{lesion_local_metrics['lesion_nodal_reaction_peak']:.8f},"
                            f"{lesion_local_metrics['lesion_nodal_reaction_total']:.8f}\n"
                        )
                except Exception as e:
                    print(f"Frame Metrics Export Error: {e}")

            reply = {
                # step: SOFA 物理步计数
                "step": step_count,
                "tip_position": tip_position,
                "tip_displacement": tip_displacement,
                "tip_step_displacement": tip_step_displacement,
                "tip_step_speed": tip_step_speed,
                "tip_velocity_limited": tip_velocity_limited,
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
                "lesion_surface_stress_mean": lesion_local_metrics["lesion_surface_stress_mean"],
                "lesion_surface_stress_peak": lesion_local_metrics["lesion_surface_stress_peak"],
                "lesion_nodal_reaction_mean": lesion_local_metrics["lesion_nodal_reaction_mean"],
                "lesion_nodal_reaction_peak": lesion_local_metrics["lesion_nodal_reaction_peak"],
                "lesion_nodal_reaction_total": lesion_local_metrics["lesion_nodal_reaction_total"],
                "lesion_contact_distance": lesion_local_metrics["lesion_contact_distance"],
                "lesion_contact_force_mean": lesion_local_metrics["lesion_contact_force_mean"],
                "lesion_contact_force_peak": lesion_local_metrics["lesion_contact_force_peak"],
                "lesion_contact_force_total": lesion_local_metrics["lesion_contact_force_total"],
            }
            sock.send_string(json.dumps(reply))
    finally:
        sock.close()
        ctx.term()

if __name__ == '__main__':
    main()
