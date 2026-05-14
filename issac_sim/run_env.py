import sys
import os

# 将当前文件的上一级目录强行加入 Python 搜索路径，防止报找不到 envs 的错
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from envs.sofa_env import SoftSofaEnv
from stable_baselines3 import PPO

# 1. 初始化你的 SOFA 环境
env = SoftSofaEnv()

# 2. 实例化 PPO 算法
# 核心点：因为你的环境返回的是字典格式的 obs，所以必须用 "MultiInputPolicy"
# verbose=1 会在终端打印训练进度、奖励和 loss
model = PPO(
    "MultiInputPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_sofa_tensorboard/"  # 日志存放路径
)

# 3. 开始训练！(SB3 自动在后台替你运行与 SOFA 的交互循环)
print("🚀 开始强化学习训练...")
model.learn(total_timesteps=100000)

# 4. 训练完成后保存模型权重
model.save("ppo_soft_sofa")
print("✅ 训练完成并保存模型！")

# ==========================================
# 5. 测试阶段 (可选：看看训练出来的模型表现如何)
print("开始测试验证模型效果...")
obs, info = env.reset()
for i in range(100):
    # predict() 替代了你自己写的 act()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()