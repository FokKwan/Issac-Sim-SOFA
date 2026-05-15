import gymnasium as gym
from gymnasium import spaces
import numpy as np
from issac_sim.communication.sofa_client import SofaCableClient

class SoftSofaEnv(gym.Env):
    """
    符合 Stable-Baselines3 / Gymnasium 标准接口的 SOFA 环境
    """
    def __init__(self, max_episode_steps=1000):
        super().__init__()
        self.sofa = SofaCableClient()
        self.max_episode_steps = int(max_episode_steps)
        self.target_tip = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        self.max_von_mises = 2500.0
        self.max_avg_strain = 0.45

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
        self._last_obs = self._build_obs(self._default_sofa_obs())

    def reset(self, seed=None, options=None):
        # 兼容最新版 gymnasium 规范，必须带 seed
        super().reset(seed=seed)
        self.t = 0
        
        # 调用之前写的 client 里的 reset 逻辑
        sofa_obs = self.sofa.reset()
        info = {}
        if not self._is_valid_sofa_obs(sofa_obs):
            info["communication_error"] = True
            sofa_obs = self._default_sofa_obs()

        obs = self._build_obs(sofa_obs)
        self._last_obs = obs

        # 必须要返回 obs 和 info 两个变量
        return obs, info

    def step(self, action):
        self.t += 1

        # 注意：SB3 传进来的 action 是一个 numpy 数组形如 [0.12]
        # 但我们发给 SOFA 的只需要标量，所以要取 action[0] 并转为 float
        cable_disp = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        sofa_obs = self.sofa.step(cable_disp=cable_disp)
        if not self._is_valid_sofa_obs(sofa_obs):
            info = {
                "communication_error": True,
                "von_mises": float(self._last_obs["von_mises"][0]),
                "strain": float(self._last_obs["avg_strain"][0]),
            }
            return self._last_obs, -100.0, True, False, info

        obs = self._build_obs(sofa_obs)
        self._last_obs = obs
        reward = float(self._compute_reward(obs))
        
        terminated = self._check_done(obs)
        truncated = self.t >= self.max_episode_steps

        info = {
            "von_mises": float(obs["von_mises"][0]),
            "strain": float(obs["avg_strain"][0]),
            "communication_error": False,
        }

        # 最新 gymnasium 标准必须返回 5 个值
        return obs, reward, terminated, truncated, info

    def _build_obs(self, sofa_obs):
        # 包装成 numpy 数组以匹配 spaces.Box
        tip_pos = np.asarray(sofa_obs.get("tip_position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if tip_pos.shape[0] != 3:
            tip_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return {
            "tip_pos": tip_pos,
            "von_mises": np.array([float(sofa_obs.get("von_mises", 0.0))], dtype=np.float32),
            "avg_strain": np.array([float(sofa_obs.get("avg_strain", 0.0))], dtype=np.float32),
        }

    def _compute_reward(self, obs):
        task_term = -np.linalg.norm(obs["tip_pos"] - self.target_tip)
        safety_penalty = (
            0.1 * obs["von_mises"][0]
            + 0.05 * obs["avg_strain"][0]
        )
        return task_term - safety_penalty

    def _check_done(self, obs):
        return bool(
            obs["von_mises"][0] > self.max_von_mises
            or obs["avg_strain"][0] > self.max_avg_strain
        )

    def _default_sofa_obs(self):
        return {
            "tip_position": [0.0, 0.0, 0.0],
            "von_mises": 0.0,
            "avg_strain": 0.0,
        }

    def _is_valid_sofa_obs(self, sofa_obs):
        if not isinstance(sofa_obs, dict):
            return False
        required_keys = ("tip_position", "von_mises", "avg_strain")
        return all(key in sofa_obs for key in required_keys)

    def close(self):
        self.sofa.close()