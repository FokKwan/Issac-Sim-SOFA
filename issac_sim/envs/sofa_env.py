import gymnasium as gym
from gymnasium import spaces
import numpy as np
from issac_sim.communication.sofa_client import SofaCableClient


# 该文件作用：
# - 把 SOFA 的 ZMQ 接口封装成 Gymnasium 环境
# - 供 Stable-Baselines3/Isaac 训练流程直接调用
# - 在 step/reset 中完成动作映射、状态解析、奖励与终止判定
class SoftSofaEnv(gym.Env):
    """
    符合 Stable-Baselines3 / Gymnasium 标准接口的 SOFA 环境
    """
    def __init__(self, max_episode_steps=400):
        """
        Args:
            max_episode_steps: 单个 episode 最大步数，超过后 truncated=True。
        """
        super().__init__()
        self.sofa = SofaCableClient()
        self.max_episode_steps = int(max_episode_steps)
        # 归一化动作到物理控制量的缩放系数：
        # SOFA 实际控制量 cable_disp = action * action_scale
        self.action_scale = 1.2
        # 安全阈值参数（用于 done 判定）
        self.max_von_mises = 2500.0
        self.max_avg_strain = 0.45
        self.max_tissue_strain = 0.35
        self.max_lesion_strain = 0.30
        # 接触力阈值参数：
        # - min_contact_force: 成功接触的下限（避免“无接触假成功”）
        # - max_contact_force: 成功接触的上限（避免过压）
        self.max_contact_force = 1.4
        self.min_contact_force = 0.02
        # 成功距离阈值（tip 到病灶中心）
        self.success_distance = 0.035
        # 奖励整形参数
        self.desired_contact_force = 0.40
        self.contact_force_sigma = 0.20
        self.progress_gain = 6.0
        self.time_penalty = 0.002

        # 1. 定义动作空间 (Action Space)
        # 标准化动作空间，发送前再做尺度映射
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 2. 定义观测空间 (Observation Space)
        # 每个键都需与 _build_obs 完全对应。
        # 参数说明：
        # - tip_pos: 机器人末端位置 (x,y,z)
        # - von_mises/avg_strain: 机器人本体代理应力/应变
        # - lesion_distance: 末端到病灶中心距离
        # - lesion_strain/tissue_strain: 病灶/组织应变
        # - contact_distance: 末端到组织最近距离
        # - contact_force_mean/peak/total: 约束求解器接触力统计
        # - lesion_center: 病灶中心位置
        self.observation_space = spaces.Dict({
            "tip_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "von_mises": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "avg_strain": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_distance": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_strain": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "tissue_strain": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "contact_distance": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "contact_force_mean": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "contact_force_peak": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "contact_force_total": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_center": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        # 通信异常兜底：保留上一帧有效观测，避免训练循环崩溃
        self._last_obs = self._build_obs(self._default_sofa_obs())
        self._prev_lesion_distance = float(self._last_obs["lesion_distance"][0])
        self._last_progress = 0.0

    def reset(self, seed=None, options=None):
        # 兼容最新版 gymnasium 规范，必须带 seed
        super().reset(seed=seed)
        self.t = 0
        
        # 调用之前写的 client 里的 reset 逻辑
        sofa_obs = self.sofa.reset()
        info = {}
        if not self._is_valid_sofa_obs(sofa_obs):
            # reset 回复异常时，退化为默认观测并标记通信错误
            info["communication_error"] = True
            sofa_obs = self._default_sofa_obs()

        obs = self._build_obs(sofa_obs)
        self._last_obs = obs
        self._prev_lesion_distance = float(obs["lesion_distance"][0])
        self._last_progress = 0.0

        # 必须要返回 obs 和 info 两个变量
        return obs, info

    def step(self, action):
        self.t += 1

        # SB3 输出一般是 ndarray，这里统一为标量并裁剪到 [-1, 1]
        normalized_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        normalized_action = float(np.clip(normalized_action, -1.0, 1.0))
        # 动作映射到 SOFA 缆绳位移
        cable_disp = normalized_action * self.action_scale
        sofa_obs = self.sofa.step(cable_disp=cable_disp)
        if not self._is_valid_sofa_obs(sofa_obs):
            # 通信失败：给强惩罚并终止当前回合
            info = {
                "communication_error": True,
                "von_mises": float(self._last_obs["von_mises"][0]),
                "strain": float(self._last_obs["avg_strain"][0]),
                "lesion_strain": float(self._last_obs["lesion_strain"][0]),
                "tissue_strain": float(self._last_obs["tissue_strain"][0]),
                "lesion_distance": float(self._last_obs["lesion_distance"][0]),
                "contact_force_peak": float(self._last_obs["contact_force_peak"][0]),
                "success": False,
            }
            return self._last_obs, -100.0, True, False, info

        obs = self._build_obs(sofa_obs)
        self._last_obs = obs
        reward = float(self._compute_reward(obs))
        self._prev_lesion_distance = float(obs["lesion_distance"][0])
        
        # terminated: 安全超限或任务成功；truncated: 达到步数上限
        terminated = self._check_done(obs)
        success = self._is_success(obs)
        terminated = terminated or success
        truncated = self.t >= self.max_episode_steps

        info = {
            "von_mises": float(obs["von_mises"][0]),
            "strain": float(obs["avg_strain"][0]),
            "lesion_strain": float(obs["lesion_strain"][0]),
            "tissue_strain": float(obs["tissue_strain"][0]),
            "lesion_distance": float(obs["lesion_distance"][0]),
            "contact_force_mean": float(obs["contact_force_mean"][0]),
            "contact_force_peak": float(obs["contact_force_peak"][0]),
            "contact_force_total": float(obs["contact_force_total"][0]),
            "success": success,
            "distance_progress": self._last_progress,
            "tip_displacement": float(sofa_obs.get("tip_displacement", 0.0)),
            "communication_error": False,
        }

        # 最新 gymnasium 标准必须返回 5 个值
        return obs, reward, terminated, truncated, info

    def _build_obs(self, sofa_obs):
        # 将 SOFA 字段强制转换为 numpy，确保与 observation_space 兼容
        tip_pos = np.asarray(sofa_obs.get("tip_position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if tip_pos.shape[0] != 3:
            tip_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        lesion_center = np.asarray(
            sofa_obs.get("lesion_center", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(-1)
        if lesion_center.shape[0] != 3:
            lesion_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return {
            "tip_pos": tip_pos,
            "von_mises": np.array([float(sofa_obs.get("von_mises", 0.0))], dtype=np.float32),
            "avg_strain": np.array([float(sofa_obs.get("avg_strain", 0.0))], dtype=np.float32),
            "lesion_distance": np.array([float(sofa_obs.get("lesion_distance", 0.0))], dtype=np.float32),
            "lesion_strain": np.array([float(sofa_obs.get("lesion_strain", 0.0))], dtype=np.float32),
            "tissue_strain": np.array([float(sofa_obs.get("tissue_strain", 0.0))], dtype=np.float32),
            "contact_distance": np.array([float(sofa_obs.get("contact_distance", 1.0))], dtype=np.float32),
            "contact_force_mean": np.array([float(sofa_obs.get("contact_force_mean", 0.0))], dtype=np.float32),
            "contact_force_peak": np.array([float(sofa_obs.get("contact_force_peak", 0.0))], dtype=np.float32),
            "contact_force_total": np.array([float(sofa_obs.get("contact_force_total", 0.0))], dtype=np.float32),
            "lesion_center": lesion_center,
        }

    def _compute_reward(self, obs):
        # 奖励设计：
        # 1) task_term: 鼓励靠近病灶（距离越小越高）
        lesion_distance = float(obs["lesion_distance"][0])
        task_term = -1.2 * lesion_distance
        # 2) progress_term: 奖励相对上一步的距离改善，降低“原地抖动”
        progress = self._prev_lesion_distance - lesion_distance
        self._last_progress = float(progress)
        progress_term = self.progress_gain * float(np.clip(progress, -0.05, 0.05))
        contact_force_peak = float(obs["contact_force_peak"][0])
        # 3) contact_term: 鼓励接触力接近期望区间，且距离病灶越近权重越高
        contact_alignment = np.exp(
            -((contact_force_peak - self.desired_contact_force) ** 2)
            / (2.0 * self.contact_force_sigma ** 2)
        )
        contact_term = 0.8 * contact_alignment * np.exp(-2.0 * lesion_distance)
        # 4) safety_penalty: 惩罚机器人/组织/病灶应变
        safety_penalty = (
            0.03 * float(obs["von_mises"][0])
            + 0.05 * float(obs["avg_strain"][0])
            + 0.10 * float(obs["tissue_strain"][0])
            + 0.14 * float(obs["lesion_strain"][0])
        )
        # 5) excessive_contact_penalty: 对过大接触力进行二次惩罚
        excessive_contact_penalty = 0.5 * max(contact_force_peak - self.max_contact_force, 0.0) ** 2
        # 6) success_bonus: 达成任务时给一次额外奖励
        success_bonus = 8.0 if self._is_success(obs) else 0.0
        return (
            task_term
            + progress_term
            + contact_term
            - safety_penalty
            - excessive_contact_penalty
            - self.time_penalty
            + success_bonus
        )

    def _check_done(self, obs):
        # 任何安全指标超阈值即终止，避免策略学习到危险操作
        return bool(
            obs["von_mises"][0] > self.max_von_mises
            or obs["avg_strain"][0] > self.max_avg_strain
            or obs["tissue_strain"][0] > self.max_tissue_strain
            or obs["lesion_strain"][0] > self.max_lesion_strain
            or obs["contact_force_peak"][0] > 3.0 * self.max_contact_force
        )

    def _is_success(self, obs):
        # 成功定义：到达病灶附近 + 接触力处于可接受区间
        return bool(
            obs["lesion_distance"][0] < self.success_distance
            and self.min_contact_force < obs["contact_force_peak"][0] < self.max_contact_force
        )

    def _default_sofa_obs(self):
        # 通信失败或初始化阶段的兜底观测（键与 shape 必须完整）
        return {
            "tip_position": [0.0, 0.0, 0.0],
            "von_mises": 0.0,
            "avg_strain": 0.0,
            "lesion_distance": 1.0,
            "lesion_strain": 0.0,
            "tissue_strain": 0.0,
            "contact_distance": 1.0,
            "contact_force_mean": 0.0,
            "contact_force_peak": 0.0,
            "contact_force_total": 0.0,
            "lesion_center": [0.0, 0.0, 0.0],
        }

    def _is_valid_sofa_obs(self, sofa_obs):
        # SOFA 响应校验：必须是 dict 且包含所有必需键
        if not isinstance(sofa_obs, dict):
            return False
        required_keys = (
            "tip_position",
            "von_mises",
            "avg_strain",
            "lesion_distance",
            "lesion_strain",
            "tissue_strain",
            "contact_distance",
            "contact_force_mean",
            "contact_force_peak",
            "contact_force_total",
            "lesion_center",
        )
        return all(key in sofa_obs for key in required_keys)

    def close(self):
        self.sofa.close()