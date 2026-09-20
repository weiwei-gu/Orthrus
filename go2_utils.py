"""Go2 辅助模块: 坐标转换 / 步态调度 / 摆动轨迹 / 落脚点规划 / 腿部 IK"""
import numpy as np
import mujoco

FOOT_NAMES = ["FL", "FR", "RL", "RR"]
# 对角小跑分组: (FL,RR) 与 (FR,RL) 交替摆动
DIAG_A = ("FL", "RR")
DIAG_B = ("FR", "RL")


def quat_to_rpy(q):
    """四元数 (w,x,y,z) -> ZYX 欧拉角 [roll, pitch, yaw]"""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------- #
# 步态调度
# ---------------------------------------------------------------------- #
class TrotGait:
    """对角小跑 (trot): 对角足对交替摆动。

    相位 φ ∈ [0,1) 随时间推进, 周期 T。
    DIAG_A (FL,RR) 在 φ∈[0, sf) 摆动, DIAG_B (FR,RL) 在 φ∈[0.5, 0.5+sf) 摆动,
    sf = swing_frac。两组窗口等长且严格落在 [0,1) 内 —— 若 B 组窗口
    不设上界 (φ≥0.5 直到回绕), 摆动进度会超过 1, 足端目标被外插到
    地面以下, 腿向地里踩。
    """

    def __init__(self, period=0.5, swing_frac=0.4):
        """swing_frac < 0.5 保证两组对角足有同时支撑的重叠窗口 (占空比 > 0.5)。
        纯 50% 占空比时任何时刻仅两足着地, 绕对角轴的力矩不可控, 侧滚发散。"""
        self.T = period
        self.swing_frac = swing_frac
        self.phase = 0.0
        self._prev_stance = {f: True for f in FOOT_NAMES}

    def advance(self, dt):
        self.phase = (self.phase + dt / self.T) % 1.0

    def in_group_A(self, foot):
        return foot in DIAG_A

    @staticmethod
    def _swing_s_of(foot, phi, sf):
        """给定相位, 返回该足摆动进度 s∈[0,1); 支撑相返回 None"""
        if foot in DIAG_A:
            if phi < sf:
                return phi / sf
        elif 0.5 <= phi < 0.5 + sf:
            return (phi - 0.5) / sf
        return None

    def is_stance(self, foot):
        return self._swing_s_of(foot, self.phase, self.swing_frac) is None

    def swing_progress(self, foot):
        """摆动进度 s∈[0,1); 支撑相返回 None"""
        return self._swing_s_of(foot, self.phase, self.swing_frac)

    def stance_at(self, t_offset):
        """从当前相位前推 t_offset 秒后的支撑矩阵预测 (4,) bool"""
        phi = (self.phase + t_offset / self.T) % 1.0
        return np.array([self._swing_s_of(f, phi, self.swing_frac) is None
                         for f in FOOT_NAMES])

    def swing_start_events(self):
        """返回本步内刚从支撑转入摆动的足 (用于记录起摆/落点)"""
        events = []
        for foot in FOOT_NAMES:
            st = self.is_stance(foot)
            if self._prev_stance[foot] and not st:
                events.append(foot)
            self._prev_stance[foot] = st
        return events


# ---------------------------------------------------------------------- #
# 摆动腿轨迹 + Raibert 落脚点
# ---------------------------------------------------------------------- #
def _smoothstep(s):
    """五次多项式平滑插值 (起末速度/加速度为零)"""
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def swing_position(p0, p1, s, height, descend_by=0.9):
    """摆动足目标位置: p0 起摆点 -> p1 落点, 抬腿高度 height

    z 用 sin³(πs) 拱线: 起摆/触地两端的速度和加速度均为零。
    剖面两端的加速度决定摆动腿对机身的反作用力 (SRB-MPC 未建模),
    sin/sin² 剖面在端点加速度达峰值 (~4g), 会把机身周期性踹下沉。

    descend_by: z 剖面在 s=descend_by 时已回到地面, 之后停等 ——
    足端若因重力超前于剖面, 目标已在地面等它, 避免带速度撞地。
    """
    sigma = _smoothstep(s)
    s_z = min(s / descend_by, 1.0)
    pos = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * sigma
    pos[2] = (1.0 - sigma) * p0[2] + sigma * p1[2] + height * np.sin(np.pi * s_z) ** 3
    return pos


def raibert_foothold(hip_xy, v_xy, v_cmd_xy, stance_time, k_gain=0.05):
    """Raibert 启发式落脚点 (xy 平面, world 系)

    p = 髋关节投影 + (T_st/2)·v_cmd + k·(v - v_cmd)
    """
    hip_xy = np.asarray(hip_xy, dtype=float)
    v_xy = np.asarray(v_xy, dtype=float)
    v_cmd_xy = np.asarray(v_cmd_xy, dtype=float)
    return hip_xy + 0.5 * stance_time * v_cmd_xy + k_gain * (v_xy - v_cmd_xy)


# ---------------------------------------------------------------------- #
# 腿部 IK (DLS 数值解, 与具体模型参数解耦)
# ---------------------------------------------------------------------- #
class Go2IK:
    """逐腿 3-DoF 逆运动学: 目标足端 world 位置 -> 关节角

    用独立的 MjData 做正运动学/雅可比, 不污染仿真状态。
    """

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.foot_geom = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f)
                          for f in FOOT_NAMES}
        self.leg_joints = {}
        for f in FOOT_NAMES:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{part}_joint")
                   for part in ("hip", "thigh", "calf")]
            self.leg_joints[f] = {
                "qadr": [model.jnt_qposadr[i] for i in ids],
                "vadr": [model.jnt_dofadr[i] for i in ids],
                "range": np.array([model.jnt_range[i] for i in ids]),
            }
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(self, foot, base_quat, base_pos, target_world, q_init, iters=3, damping=0.01):
        """返回 (关节角 (3,), 收敛误差)"""
        leg = self.leg_joints[foot]
        q = np.clip(np.asarray(q_init, dtype=float).copy(), leg["range"][:, 0], leg["range"][:, 1])
        d = self.data
        d.qpos[3:7] = base_quat
        d.qpos[0:3] = base_pos
        vadr = leg["vadr"]
        err_norm = 0.0
        for _ in range(iters):
            for a in range(3):
                d.qpos[leg["qadr"][a]] = q[a]
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            gid = self.foot_geom[foot]
            err = np.asarray(target_world) - d.geom_xpos[gid]
            err_norm = np.linalg.norm(err)
            if err_norm < 1e-5:
                break
            mujoco.mj_jacGeom(self.model, d, self._jacp, self._jacr, gid)
            J = self._jacp[:, vadr]  # 3×3
            # 阻尼最小二乘 (DLS)
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = np.clip(q + dq, leg["range"][:, 0], leg["range"][:, 1])
        return q, err_norm
