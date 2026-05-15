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

YOUNG_MODULUS = 3000.0
POISSON_RATIO = 0.45

def createScene(root):
    """
    构建软体机器人仿真场景
    """
    root.dt = 0.01
    root.gravity = [0, -9.81, 0]
    root.addObject("DefaultAnimationLoop")

    # 1. 物理模型节点
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
        youngModulus=YOUNG_MODULUS,
        poissonRatio=POISSON_RATIO,
    )
    
    soft.addObject("UniformMass", totalMass=0.5)

    # 缆绳约束：索引需在网格点数范围内
    soft.addObject(
        "CableConstraint",
        name="cable",
        indices=[0, 9, 18, 27, 36, 45, 54, 63, 72, 81], 
        value=[0.0],
        valueType="displacement",
        pullPoint=[0, 0, -0.1]
    )
    return root


def compute_mechanics_metrics(positions, rest_positions):
    if positions.size == 0 or rest_positions.size == 0:
        return 0.0, 0.0

    displacement = positions - rest_positions
    displacement_norm = np.linalg.norm(displacement, axis=1)
    bbox_min = np.min(rest_positions, axis=0)
    bbox_max = np.max(rest_positions, axis=0)
    char_length = max(np.linalg.norm(bbox_max - bbox_min), 1e-8)

    avg_strain = float(np.mean(displacement_norm) / char_length)
    # 当前场景使用位移比例近似应力指标，避免奖励函数失真
    von_mises_proxy = float(YOUNG_MODULUS * avg_strain)
    return von_mises_proxy, avg_strain

def main():
    root = Sofa.Core.Node("root")
    createScene(root)
    
    # 1. 初始化仿真
    Sofa.Simulation.init(root)

    # 2. 【核心修复】由于 Headless 模式下 init 后数据可能未同步
    # 我们手动运行一个极小的 animate 来强制引擎计算并填充网格数据
    print("正在激活网格拓扑...")
    Sofa.Simulation.animate(root, 0.0001) 

    # 3. 获取组件引用并提取拓扑
    grid_component = root.SoftBody.grid
    
    # 优先尝试 hexa 数据，确保与当前 FEM 力场一致
    topo_val = grid_component.hexahedra.value
    cell_type = "hexahedron"

    # 备选导出单元：tetra / triangle
    if len(topo_val) == 0:
        topo_val = grid_component.tetrahedra.value
        cell_type = "tetra"
    if len(topo_val) == 0:
        topo_val = grid_component.triangles.value
        cell_type = "triangle"

    if len(topo_val) == 0:
        # 如果还是空，尝试直接从 Data 字段强制读取
        topo_val = grid_component.getData("tetrahedra").value
        if len(topo_val) == 0:
            print("[Error] 无法获取拓扑数据！请检查 createScene 中的 nx, ny, nz 是否都 >= 2")
            return

    # 封装给 meshio 使用
    topo_cells = [(cell_type, np.array(topo_val))]
    print(f"成功获取拓扑！类型: {cell_type}, 数量: {len(topo_val)}")

    # --- 以下 ZMQ 和循环逻辑保持不变 ---
    cable = root.SoftBody.cable
    dofs = root.SoftBody.dofs
    rest_positions = np.array(dofs.position.value, copy=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:5555")
    
    if not os.path.exists('vtk_output'): os.makedirs('vtk_output')
    print("[SOFA] Headless Server Ready (Manual Export Mode)...")

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
                rest_positions = np.array(dofs.position.value, copy=True)
            else:
                # 执行一步控制
                cable_disp = float(cmd.get("cable_disp", 0.0))
                cable.value = [cable_disp]
                
                # 物理步进
                Sofa.Simulation.animate(root, float(root.dt.value))
                step_count += 1

            # 每 10 步导出一次 VTK 供 ParaView 查看
            if step_count % 10 == 0:
                try:
                    target_dir = 'vtk_output'
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    
                    points = np.array(dofs.position.value)
                    if len(points) > 0 and len(topo_cells) > 0:
                        mesh = meshio.Mesh(points=points, cells=topo_cells)
                        vtk_file = os.path.join(target_dir, f"frame_{step_count:04d}.vtk")
                        mesh.write(vtk_file)
                except Exception as e:
                    print(f"VTK Export Error: {e}")

            # 反馈当前状态
            positions = np.array(dofs.position.value)
            if len(positions) == 0:
                tip_position = [0.0, 0.0, 0.0]
                von_mises = 0.0
                avg_strain = 0.0
            else:
                tip_position = positions[-1].tolist()
                von_mises, avg_strain = compute_mechanics_metrics(positions, rest_positions)

            reply = {
                "step": step_count,
                "tip_position": tip_position,
                "von_mises": von_mises,
                "avg_strain": avg_strain,
            }
            sock.send_string(json.dumps(reply))
    finally:
        sock.close()
        ctx.term()

if __name__ == '__main__':
    main()