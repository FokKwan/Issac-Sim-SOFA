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
        self.t = 0
        # 归一化动作到曲率/插入增量的缩放系数：
        # SOFA 每步累积 cable_disp =
        # [ky_prox_delta, ky_dist_delta, kz_prox_delta, kz_dist_delta, insertion_delta] * action_scale。
        self.action_scale = 1.0
        # 安全阈值参数（用于 done 判定，略放宽以允许更大动作）
        self.max_von_mises = 3200.0
        self.max_avg_strain = 0.55
        self.max_tissue_strain = 0.50
        self.max_lesion_strain = 0.42
        # 接触力阈值参数：
        # - min_contact_force: 成功接触的下限（避免“无接触假成功”）
        # - max_contact_force: 成功接触的上限（避免过压）
        self.max_contact_force = 1.4
        self.min_contact_force = 0.02
        # 绕病灶画圆任务参数：圆在 X-Z 平面内，圆心为病灶中心。
        self.circle_radius = 0.06
        self.circle_period_steps = 240
        self.circle_target_tolerance = 0.025
        # 奖励整形参数
        self.tracking_gain = 8.0
        self.radial_gain = 2.0
        self.progress_gain = 3.0
        self.time_penalty = 0.002

        # 1. 定义动作空间 (Action Space)
        # 标准化五维动作：XY/XZ 平面各两个曲率段，可在每个平面形成 S 型弯曲。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # 2. 定义观测空间 (Observation Space)
        # 每个键都需与 _build_obs 完全对应。
        # 参数说明：
        # - tip_pos: 机器人末端位置 (x,y,z)
        # - von_mises/avg_strain: 机器人本体代理应力/应变
        # - lesion_distance: 末端到病灶中心距离
        # - lesion_strain/tissue_strain: 病灶/组织应变
        # - contact_distance: 末端到组织最近距离
        # - contact_force_mean/peak/total: 约束求解器接触力统计
        # - lesion_contact_force_*: 由病灶表面应力和节点反力估计的局部接触力
        # - lesion_surface_stress_* / lesion_nodal_reaction_*: 病灶表面局部场统计
        # - lesion_center: 病灶中心位置
        # - circle_target: 当前相位下末端应追踪的圆周目标点
        # - circle_phase: 当前圆周相位 [0, 2pi]
        # - circle_error: 末端到 circle_target 的距离
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
            "lesion_contact_distance": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_contact_force_mean": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_contact_force_peak": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_contact_force_total": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_surface_stress_mean": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_surface_stress_peak": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_nodal_reaction_mean": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_nodal_reaction_peak": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_nodal_reaction_total": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "lesion_center": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "circle_target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "circle_phase": spaces.Box(low=0.0, high=2.0 * np.pi, shape=(1,), dtype=np.float32),
            "circle_error": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })
        # 通信异常兜底：保留上一帧有效观测，避免训练循环崩溃
        self._last_obs = self._build_obs(self._default_sofa_obs())
        self._prev_circle_error = float(self._last_obs["circle_error"][0])
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
        self._prev_circle_error = float(obs["circle_error"][0])
        self._last_progress = 0.0

        # 必须要返回 obs 和 info 两个变量
        return obs, info

    def step(self, action):
        self.t += 1

        # SB3 输出一般是 ndarray，这里统一为五维向量并裁剪到 [-1, 1]
        normalized_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if normalized_action.size < 5:
            normalized_action = np.pad(normalized_action, (0, 5 - normalized_action.size))
        normalized_action = np.clip(normalized_action[:5], -1.0, 1.0)
        # 动作映射到 SOFA 曲率/插入增量（累积控制）
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
                "circle_error": float(self._last_obs["circle_error"][0]),
                "contact_force_peak": float(self._last_obs["contact_force_peak"][0]),
                "lesion_contact_force_peak": float(self._last_obs["lesion_contact_force_peak"][0]),
                "success": False,
            }
            return self._last_obs, -100.0, True, False, info

        obs = self._build_obs(sofa_obs)
        self._last_obs = obs
        reward = float(self._compute_reward(obs))
        self._prev_circle_error = float(obs["circle_error"][0])
        
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
            "circle_phase": float(obs["circle_phase"][0]),
            "circle_error": float(obs["circle_error"][0]),
            "circle_target": obs["circle_target"].tolist(),
            "contact_force_mean": float(obs["contact_force_mean"][0]),
            "contact_force_peak": float(obs["contact_force_peak"][0]),
            "contact_force_total": float(obs["contact_force_total"][0]),
            "lesion_contact_distance": float(obs["lesion_contact_distance"][0]),
            "lesion_contact_force_mean": float(obs["lesion_contact_force_mean"][0]),
            "lesion_contact_force_peak": float(obs["lesion_contact_force_peak"][0]),
            "lesion_contact_force_total": float(obs["lesion_contact_force_total"][0]),
            "lesion_surface_stress_mean": float(obs["lesion_surface_stress_mean"][0]),
            "lesion_surface_stress_peak": float(obs["lesion_surface_stress_peak"][0]),
            "lesion_nodal_reaction_mean": float(obs["lesion_nodal_reaction_mean"][0]),
            "lesion_nodal_reaction_peak": float(obs["lesion_nodal_reaction_peak"][0]),
            "lesion_nodal_reaction_total": float(obs["lesion_nodal_reaction_total"][0]),
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

        circle_phase = self._circle_phase()
        circle_target = self._circle_target(lesion_center, circle_phase)
        circle_error = float(np.linalg.norm(tip_pos - circle_target))

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
            "lesion_contact_distance": np.array([float(sofa_obs.get("lesion_contact_distance", 1.0))], dtype=np.float32),
            "lesion_contact_force_mean": np.array([float(sofa_obs.get("lesion_contact_force_mean", 0.0))], dtype=np.float32),
            "lesion_contact_force_peak": np.array([float(sofa_obs.get("lesion_contact_force_peak", 0.0))], dtype=np.float32),
            "lesion_contact_force_total": np.array([float(sofa_obs.get("lesion_contact_force_total", 0.0))], dtype=np.float32),
            "lesion_surface_stress_mean": np.array([float(sofa_obs.get("lesion_surface_stress_mean", 0.0))], dtype=np.float32),
            "lesion_surface_stress_peak": np.array([float(sofa_obs.get("lesion_surface_stress_peak", 0.0))], dtype=np.float32),
            "lesion_nodal_reaction_mean": np.array([float(sofa_obs.get("lesion_nodal_reaction_mean", 0.0))], dtype=np.float32),
            "lesion_nodal_reaction_peak": np.array([float(sofa_obs.get("lesion_nodal_reaction_peak", 0.0))], dtype=np.float32),
            "lesion_nodal_reaction_total": np.array([float(sofa_obs.get("lesion_nodal_reaction_total", 0.0))], dtype=np.float32),
            "lesion_center": lesion_center,
            "circle_target": circle_target.astype(np.float32),
            "circle_phase": np.array([circle_phase], dtype=np.float32),
            "circle_error": np.array([circle_error], dtype=np.float32),
        }

    def _compute_reward(self, obs):
        # 奖励设计：追踪病灶周围圆形目标点，同时惩罚偏离圆半径和危险形变。
        circle_error = float(obs["circle_error"][0])
        task_term = -self.tracking_gain * circle_error
        progress = self._prev_circle_error - circle_error
        self._last_progress = float(progress)
        progress_term = self.progress_gain * float(np.clip(progress, -0.08, 0.08))
        lesion_center = obs["lesion_center"]
        tip_pos = obs["tip_pos"]
        radial_distance = float(np.linalg.norm(tip_pos[[0, 2]] - lesion_center[[0, 2]]))
        radial_error = abs(radial_distance - self.circle_radius)
        radial_penalty = self.radial_gain * radial_error
        contact_force_peak = max(
            float(obs["contact_force_peak"][0]),
            float(obs["lesion_contact_force_peak"][0]),
        )
        safety_penalty = (
            0.03 * float(obs["von_mises"][0])
            + 0.05 * float(obs["avg_strain"][0])
            + 0.10 * float(obs["tissue_strain"][0])
            + 0.14 * float(obs["lesion_strain"][0])
        )
        excessive_contact_penalty = 0.5 * max(contact_force_peak - self.max_contact_force, 0.0) ** 2
        lap_bonus = 8.0 if self._is_success(obs) else 0.0
        return (
            task_term
            + progress_term
            - radial_penalty
            - safety_penalty
            - excessive_contact_penalty
            - self.time_penalty
            + lap_bonus
        )

    def _check_done(self, obs):
        # 任何安全指标超阈值即终止，避免策略学习到危险操作
        return bool(
            obs["von_mises"][0] > self.max_von_mises
            or obs["avg_strain"][0] > self.max_avg_strain
            or obs["tissue_strain"][0] > self.max_tissue_strain
            or obs["lesion_strain"][0] > self.max_lesion_strain
            or max(
                obs["contact_force_peak"][0],
                obs["lesion_contact_force_peak"][0],
            ) > 3.0 * self.max_contact_force
        )

    def _is_success(self, obs):
        # 成功定义：完成一圈后仍贴近当前圆周目标点。
        return bool(
            self.t >= self.circle_period_steps
            and obs["circle_error"][0] < self.circle_target_tolerance
        )

    def _circle_phase(self):
        progress = min(max(self.t, 0), self.circle_period_steps) / float(self.circle_period_steps)
        return float(2.0 * np.pi * progress)

    def _circle_target(self, lesion_center, phase):
        target = np.array(lesion_center, dtype=np.float32)
        target[0] = lesion_center[0] + self.circle_radius * np.cos(phase)
        target[1] = lesion_center[1]
        target[2] = lesion_center[2] + self.circle_radius * np.sin(phase)
        return target

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
            "lesion_contact_distance": 1.0,
            "lesion_contact_force_mean": 0.0,
            "lesion_contact_force_peak": 0.0,
            "lesion_contact_force_total": 0.0,
            "lesion_surface_stress_mean": 0.0,
            "lesion_surface_stress_peak": 0.0,
            "lesion_nodal_reaction_mean": 0.0,
            "lesion_nodal_reaction_peak": 0.0,
            "lesion_nodal_reaction_total": 0.0,
            "lesion_center": [0.08, -0.14, 0.0],
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
            "lesion_contact_distance",
            "lesion_contact_force_mean",
            "lesion_contact_force_peak",
            "lesion_contact_force_total",
            "lesion_surface_stress_mean",
            "lesion_surface_stress_peak",
            "lesion_nodal_reaction_mean",
            "lesion_nodal_reaction_peak",
            "lesion_nodal_reaction_total",
            "lesion_center",
        )
        return all(key in sofa_obs for key in required_keys)

    def close(self):
        self.sofa.close()
