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

def createScene(root):
    """
    构建软体机器人仿真场景
    """
    root.dt = 0.01
    root.gravity = [0, -9.81, 0]
    root.addObject("DefaultAnimationLoop")

    # 1. 物理模型节点
    soft = root.addChild("SoftBody")
    
    # 使用 RegularGridTopology 自动生成 3x3x10 的网格 (共90个点)
    soft.addObject('RegularGridTopology', name='grid', 
                   min=[0, 0, 0], max=[0.1, 0.1, 1.0], 
                   nx=3, ny=3, nz=10)
    
    # 状态量容器
    soft.addObject("MechanicalObject", name="dofs", template="Vec3d")
    
    # 有限元组件：显式关联到上面的 grid 拓扑
    soft.addObject("TetrahedronFEMForceField", name="fem", 
                   topology="@grid", 
                   method="large", youngModulus=3000, poissonRatio=0.45)
    
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
    
    # 尝试获取四面体数据
    topo_val = grid_component.tetrahedra.value
    cell_type = "tetra"
    
    # 如果四面体为空，尝试获取三角形（作为备选，防止崩溃）
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

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:5555")
    
    if not os.path.exists('vtk_output'): os.makedirs('vtk_output')
    print("[SOFA] Headless Server Ready (Manual Export Mode)...")

    step_count = 0

    while True:
        # 接收来自 Isaac Sim 的指令
        msg = sock.recv_string()
        cmd = json.loads(msg)

        if cmd.get("type") == "reset":
            Sofa.Simulation.reset(root)
            step_count = 0
            cable.value = [0.0]
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
                
                # 显式定义路径，防止变量未定义错误
                target_dir = 'vtk_output' 
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                points = np.array(dofs.position.value)
                if len(points) > 0 and len(topo_cells) > 0:
                    mesh = meshio.Mesh(points=points, cells=topo_cells)
                    # 构造完整的文件路径
                    vtk_file = os.path.join(target_dir, f"frame_{step_count:04d}.vtk")
                    mesh.write(vtk_file)
                    # print(f"Saved: {vtk_file}") # 如果想看保存记录可以取消注释
            except Exception as e:
                print(f"VTK Export Error: {e}")

        # 反馈当前状态
        positions = np.array(dofs.position.value)
        reply = {
            "step": step_count,
            "tip_position": positions[-1].tolist(),
            "von_mises": 0.0, # 这里的计算逻辑可按需补全
            "avg_strain": 0.0
        }
        sock.send_string(json.dumps(reply))

if __name__ == '__main__':
    main()