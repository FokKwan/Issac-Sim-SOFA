import gymnasium as gym
from gymnasium import spaces
import numpy as np
from communication.sofa_client import SofaCableClient

class SoftSofaEnv(gym.Env):
    """
    符合 Stable-Baselines3 / Gymnasium 标准接口的 SOFA 环境
    """
    def __init__(self):
        super().__init__()
        self.sofa = SofaCableClient()

        # 1. 定义动作空间 (Action Space)
        # 假设你的缆绳位移 (cable_disp) 是一个介于 -0.5 到 0.5 之间的连续浮点数
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

        # 2. 定义观测空间 (Observation Space)
        # 必须与 _build_obs 返回的字典格式严丝合缝
        self.observation_space = spaces.Dict({
            "tip_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "von_mises": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "avg_strain": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        # 兼容最新版 gymnasium 规范，必须带 seed
        super().reset(seed=seed)
        self.t = 0
        
        # 调用之前写的 client 里的 reset 逻辑
        sofa_obs = self.sofa.reset()
        
        # 必须要返回 obs 和 info 两个变量
        return self._build_obs(sofa_obs), {}

    def step(self, action):
        self.t += 1

        # 注意：SB3 传进来的 action 是一个 numpy 数组形如 [0.12]
        # 但我们发给 SOFA 的只需要标量，所以要取 action[0] 并转为 float
        cable_disp = float(action[0])
        sofa_obs = self.sofa.step(cable_disp=cable_disp)

        obs = self._build_obs(sofa_obs)
        reward = float(self._compute_reward(obs))
        
        terminated = self._check_done(obs) # 是否达到危险终止条件
        truncated = False                  # 可以在这里设置最大步数截断 (比如 self.t > 1000)

        info = {
            "von_mises": obs["von_mises"][0],
            "strain": obs["avg_strain"][0]
        }

        # 最新 gymnasium 标准必须返回 5 个值
        return obs, reward, terminated, truncated, info

    def _build_obs(self, sofa_obs):
        # 包装成 numpy 数组以匹配 spaces.Box
        return {
            "tip_pos": np.array(sofa_obs["tip_position"], dtype=np.float32),
            "von_mises": np.array([sofa_obs["von_mises"]], dtype=np.float32),
            "avg_strain": np.array([sofa_obs["avg_strain"]], dtype=np.float32)
        }

    def _compute_reward(self, obs):
        task_term = -np.linalg.norm(obs["tip_pos"] - np.array([0.0, 0.1, 0.0]))
        safety_penalty = (
            0.1 * obs["von_mises"][0]
            + 0.05 * obs["avg_strain"][0]
        )
        return task_term - safety_penalty

    def _check_done(self, obs):
        if obs["von_mises"][0] > 1e6:
            return True
        return False