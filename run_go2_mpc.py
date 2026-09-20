"""Go2 convex MPC 主控制器

控制架构 (MIT Cheetah 风格):
  50 Hz   convex MPC   —— 单刚体模型滚动优化, 输出 4 足期望接触力 (world 系)
  500 Hz  低层控制     —— 支撑腿 τ = Jᵀ·f ; 摆动腿 PD 跟踪 IK 关节轨迹

测试脚本 (无窗口):
  .venv/bin/python run_go2_mpc.py
"""
import sys
import numpy as np
import mujoco

from convex_mpc import ConvexMPC
from go2_utils import (FOOT_NAMES, TrotGait, swing_position, raibert_foothold,
                       quat_to_rpy, rot_z, Go2IK)

SCENE = "models/go2/go2_mpc_scene.xml"

# ---- 时间与频率 ----
DT = 0.002          # 仿真步长 500 Hz
MPC_DT = 0.02       # MPC knot 间隔
MPC_STEPS = 5       # 每 5 步重解一次 MPC (100 Hz)
MPC_H = 15          # MPC horizon = 15 × 0.02 = 0.3 s
# (试验记录: 0.5s 视野 + 摆动中落点重规划 + 倾角捕获步的组合在推力测试中
#  未显示任何收益, 且让行走工况 10N·s 恢复率从 5/8 崩到 0/8, 已全部回退。
#  保留的只有"原地模式步态时序自适应" —— 摆动加速落地/四足冻结。)

# ---- 演示脚本时间轴 ----
T_STAND = 4.0       # 0~4s   纯站立 (验证静态平衡)
T_TROT = 8.0        # 4~8s   原地对角小跑
T_END = 16.0        # 8~16s  以 0.5 m/s 前进

# ---- 步态/低层参数 ----
GAIT_PERIOD = 0.6
SWING_HEIGHT = 0.05
K_RAIBERT = 0.18    # 落脚点速度增益: 扰动恢复靠"迈步接住" (捕获点思想)
K_ROLL_CATCH = 0.30   # 倾角捕获增益 (m/rad): 向倾倒方向跨步
K_PITCH_CATCH = 0.20
STANCE_HALF_MIN = 0.10  # 落点最小侧向半宽 (防远侧腿越过中线交叉)
STANCE_HALF_MAX = 0.24  # 落点最大侧向半宽 (工作空间边界)
Z_DES = 0.27        # 目定站立高度 (与 home keyframe 一致)
KP_SWING = 80.0
KV_SWING = 4.0
KP_STANCE = 30.0    # 支撑腿弱 PD: 锚定"足端钉在落地点"的运动学构型
KV_STANCE = 1.5
K_CART = 40.0       # 近地笛卡尔阻尼 (N·s/m), 抑制触地冲击
WBC_FF = 0.0        # 惯性前馈增益 (M·q̈_ref, 仅摆动腿) — A/B 测试未显示收益, 暂关
QDD_EMA = 0.85      # 参考加速度 EMA 滤波系数 (重滤波压 IK 数值噪声)
QDD_MAX = 100.0     # 参考加速度限幅 (rad/s²)
V_WALK = 0.5        # 前进速度指令


class Go2Controller:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # ---- 模型索引 ----
        self.foot_geom = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f)
                          for f in FOOT_NAMES]

        # 加硬足端接触: menagerie 默认 solimp (0.015, 1, 0.022) 是"软橡胶脚",
        # 承载下陷 ~1cm, 起摆时弹性能会把机身蹬飞。改为近刚性 (下陷 <1mm)。
        for g in self.foot_geom:
            model.geom_solimp[g] = [0.9, 0.95, 0.001, 0.5, 2.0]
        self.hip_jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_hip_joint")
                        for f in FOOT_NAMES]
        # Raibert 落脚点参考: 大腿体原点 (腿实际悬挂处, y=±0.142)。
        # 若用髋关节锚点 (y=±0.0465) 会把落脚点内收 9.5cm, 腿交叉导致侧翻。
        self.thigh_bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{f}_thigh")
                          for f in FOOT_NAMES]
        self.leg = []
        act_joint = model.actuator_trnid[:, 0]   # 每个执行器驱动的关节 id
        for f in FOOT_NAMES:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{p}_joint")
                   for p in ("hip", "thigh", "calf")]
            self.leg.append({
                "qadr": [model.jnt_qposadr[i] for i in ids],
                "vadr": [model.jnt_dofadr[i] for i in ids],
                # 驱动这些关节的执行器索引 (tau 按执行器编号写入 ctrl)
                "aadr": [int(np.where(act_joint == jid)[0][0]) for jid in ids],
            })
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

        # ---- 传感器地址 ----
        def sens(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return model.sensor_adr[i]
        self.adr_com = sens("com")
        self.adr_v = sens("com_vel")

        # ---- 复合惯量 (以 home 姿态计算, 供单刚体 MPC 使用) ----
        m_tot, com, I = self._composite_inertia()
        self.mass = m_tot
        self.I_diag = np.diag(I)

        # ---- 控制器组件 ----
        # μ=0.4 (真实摩擦 0.8): 留实现误差余量; f_max=130 给扰动恢复留爆发力
        # q_pos 水平权重 12: 推力位移后 MPC 主动拉回 (原 2 太弱, 恢复率低)
        self.mpc = ConvexMPC(m_tot, self.I_diag, mu=0.4, f_max=130.0,
                             horizon=MPC_H, dt=MPC_DT,
                             q_rpy=(60.0, 120.0, 15.0),
                             q_pos=(12.0, 12.0, 100.0),
                             q_vel=(6.0, 6.0, 8.0))
        self.ik = Go2IK(model)
        self.gait = TrotGait(GAIT_PERIOD)
        self.gait_enabled = False
        self.stand_until = T_STAND   # 此时刻前纯站立
        self.get_command = self._default_command

        # ---- 运行时状态 ----
        self.step = 0
        self.forces = np.zeros((4, 3))
        self.liftoff = {}
        self.landing = {}
        self.planted = {f: None for f in FOOT_NAMES}   # 支撑足的"钉地"位置
        self._prev_swing = {f: False for f in FOOT_NAMES}
        # 参考轨迹历史: WBC 惯性前馈 (M·q̈) 需要参考的一阶/二阶导数
        self._ref = {f: {"q": None, "qd": None, "qdd": np.zeros(3)} for f in FOOT_NAMES}
        self._Mbuf = np.zeros((model.nv, model.nv))
        self._recovery = False   # 自适应步态时序: 恢复模式 (迟滞)
        self.ground_z = float(np.mean([data.geom_xpos[g][2] for g in self.foot_geom]))
        self.yaw0 = quat_to_rpy(data.qpos[3:7])[2]
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def _composite_inertia(self):
        """整机器人关于总质心的复合惯量 (world 系, home 姿态基座为单位姿态)"""
        d = self.data
        m_tot, com = 0.0, np.zeros(3)
        for b in range(1, self.model.nbody):
            m = self.model.body_mass[b]
            m_tot += m
            com += m * d.xipos[b]
        com /= m_tot
        I = np.zeros((3, 3))
        for b in range(1, self.model.nbody):
            m = self.model.body_mass[b]
            R = d.xmat[b].reshape(3, 3)
            I_w = R @ np.diag(self.model.body_inertia[b]) @ R.T
            dd = d.xipos[b] - com
            I += I_w + m * (np.dot(dd, dd) * np.eye(3) - np.outer(dd, dd))
        return m_tot, com, I

    # ------------------------------------------------------------------ #
    def _landing_target(self, i, d, v, v_cmd, rpy, omega, catch):
        """Raibert 落脚点; catch=True 时叠加倾角捕获项 (向倾倒方向跨步接住)"""
        hip_xy = d.xpos[self.thigh_bid[i]][:2]
        stance_time = self.gait.T * (1 - self.gait.swing_frac)
        land = raibert_foothold(hip_xy, v[:2], v_cmd[:2],
                                stance_time=stance_time, k_gain=K_RAIBERT)
        if catch:
            # roll>0 = 左侧抬起 = 向右倒 -> 落点向 -y 跨; pitch>0 = 低头 = 向前倒 -> 向 +x 跨
            # 捕获增量限幅 ±0.15m: 大扰动下原始捕获位移可达 0.5m, 超出
            # 工作空间后 IK 钳位到关节极限, 腿伸直撑地反而加速翻倒
            dx = np.clip(K_PITCH_CATCH * (rpy[1] + 0.15 * omega[1]), -0.15, 0.15)
            dy = np.clip(-K_ROLL_CATCH * (rpy[0] + 0.15 * omega[0]), -0.15, 0.15)
            land[0] += dx
            land[1] += dy
            side = np.sign(hip_xy[1])
            land[1] = side * np.clip(side * land[1], STANCE_HALF_MIN, STANCE_HALF_MAX)
            # 总落点偏移限幅 (相对大腿锚点)
            off = land - hip_xy
            n = np.hypot(off[0], off[1])
            if n > 0.28:
                land = hip_xy + off * (0.28 / n)
        return land

    def _ref_derivs(self, foot, q_ref):
        """参考关节轨迹的一阶/二阶差分 (EMA 滤波 + 限幅), 供 M·q̈ 前馈

        支撑腿参考 = 钉地 IK (随机身运动连续变化), 摆动腿参考 = 摆动轨迹 IK。
        起摆时调用 reset_ref 截断历史, 避免轨迹切换产生的差分尖峰。
        """
        h = self._ref[foot]
        qd = np.zeros(3) if h["q"] is None else (q_ref - h["q"]) / DT
        qdd_raw = np.zeros(3) if h["qd"] is None else (qd - h["qd"]) / DT
        h["q"], h["qd"] = q_ref.copy(), qd
        h["qdd"] = np.clip(QDD_EMA * h["qdd"] + (1.0 - QDD_EMA) * qdd_raw,
                           -QDD_MAX, QDD_MAX)
        return qd, h["qdd"]

    def reset_ref(self, foot):
        self._ref[foot] = {"q": None, "qd": None, "qdd": np.zeros(3)}

    # ------------------------------------------------------------------ #
    def _default_command(self, t):
        """机身坐标系速度指令 [vx, vy] (m/s), 演示脚本可覆盖 self.get_command"""
        if t < T_STAND:
            return np.zeros(2)
        if t < T_TROT:
            return np.zeros(2)
        return np.array([V_WALK, 0.0])

    def update(self, t):
        """读取状态 -> 指令 -> 步态 -> (50Hz)MPC -> (500Hz)低层力矩"""
        d, model = self.data, self.model

        # ---- 1. 状态 ----
        com = d.sensordata[self.adr_com:self.adr_com + 3].copy()
        v = d.sensordata[self.adr_v:self.adr_v + 3].copy()
        omega = d.qvel[3:6].copy()
        rpy = quat_to_rpy(d.qpos[3:7])
        yaw = rpy[2]
        x0 = np.concatenate([rpy, com, omega, v])
        foot_pos = np.array([d.geom_xpos[g] for g in self.foot_geom])

        # ---- 2. 速度指令 ----
        v_cmd_body = self.get_command(t)
        gait_on = t >= self.stand_until
        yaw_cmd = self.yaw0
        v_cmd = rot_z(yaw_cmd) @ np.array([v_cmd_body[0], v_cmd_body[1], 0.0])

        # 位置参考锚点: 随指令速度积分; 指令变化时重置到当前位置 (阶跃参考)
        if not hasattr(self, "v_cmd_prev"):
            self.v_cmd_prev = v_cmd.copy()
            self.p_cmd = com[:2].copy()
        if np.any(v_cmd[:2] != self.v_cmd_prev[:2]):
            self.p_cmd = com[:2].copy()
            self.v_cmd_prev = v_cmd.copy()
        self.p_cmd = self.p_cmd + v_cmd[:2] * DT

        # ---- 3. 步态推进 (含自适应恢复时序) ----
        if gait_on and not self.gait_enabled:
            self.gait_enabled = True
            self.gait.phase = 0.0
        frozen = False
        if self.gait_enabled:
            # 恢复模式 (迟滞): 进入阈值必须高于正常行走的姿态瞬态
            # (行走俯仰标称 5-9°, 尖峰 ~12°), 否则误触发步态冻结反而摔倒。
            # 步态冻结/加速只用于原地模式 —— 行走中冻结步态会破坏节奏,
            # 实测让行走工况推力恢复率从 62% 崩到 0%
            in_place = np.hypot(v_cmd[0], v_cmd[1]) < 0.05
            tilt = max(abs(rpy[0]), abs(rpy[1]))
            if not self._recovery:
                if tilt > np.deg2rad(18.0):
                    self._recovery = True
            elif tilt < np.deg2rad(10.0):
                self._recovery = False
            rate = 1.0
            if self._recovery and in_place:
                any_swing = not all(self.gait.is_stance(f) for f in FOOT_NAMES)
                if any_swing:
                    rate = 1.8
                elif np.hypot(v[0], v[1]) < 0.5:
                    # 已停住但姿态仍歪: 冻结保持四足;
                    # 仍在滑移 (大动量): 继续迈步用捕获步接住
                    rate, frozen = 0.0, True
            self.gait.advance(DT * rate)
            for foot in self.gait.swing_start_events():
                i = FOOT_NAMES.index(foot)
                self.liftoff[foot] = foot_pos[i].copy()
                # 捕获步试验未显示收益 (trot 持平, 行走恶化), 默认关闭;
                # 仅原地模式且大倾角时保留弱捕获
                in_place = np.hypot(v_cmd[0], v_cmd[1]) < 0.05
                gate = in_place and (abs(rpy[0]) > np.deg2rad(8.0)
                                     or abs(rpy[1]) > np.deg2rad(18.0))
                land_xy = self._landing_target(i, d, v, v_cmd, rpy, omega, catch=gate)
                self.landing[foot] = np.array([land_xy[0], land_xy[1], self.ground_z])
                # 起摆: 截断参考历史, 避免轨迹切换产生差分尖峰
                self.reset_ref(foot)
        stance = np.array([self.gait.is_stance(f) if self.gait_enabled else True
                           for f in FOOT_NAMES])

        # 摆动中落点重规划: 推力来时正在摆动的腿, 其落点仍是起摆前规划的
        # 标称位置 —— 等下一组起摆往往已经晚了。倾角超阈值 (roll 8°/pitch
        # 18°, 均高于正常行走的瞬态) 时对 s<0.6 的摆动足持续重规划,
        # 叠加向倾倒方向的捕获跨步。
        # [bisect] 暂时禁用, 定位行走回归
        if False and self.gait_enabled:
            gate = (abs(rpy[0]) > np.deg2rad(8.0)) or (abs(rpy[1]) > np.deg2rad(18.0))
            if gate:
                for i, f in enumerate(FOOT_NAMES):
                    s = self.gait.swing_progress(f)
                    if s is not None and s < 0.6 and f in self.landing:
                        land_xy = self._landing_target(i, d, v, v_cmd, rpy, omega, catch=True)
                        self.landing[f] = np.array([land_xy[0], land_xy[1], self.ground_z])

        # 触地检测: 摆动->支撑转变时记录该足的"钉地"位置 (支撑 PD 的锚点)
        for i, f in enumerate(FOOT_NAMES):
            is_swing = (not stance[i]) if self.gait_enabled else False
            if self._prev_swing[f] and not is_swing:
                self.planted[f] = foot_pos[i].copy()
            elif self.planted[f] is None:
                self.planted[f] = foot_pos[i].copy()
            self._prev_swing[f] = is_swing

        # ---- 4. convex MPC (50 Hz) ----
        if self.step % MPC_STEPS == 0:
            stance_pred = np.zeros((MPC_H, 4), dtype=bool)
            for k in range(MPC_H):
                if self.gait_enabled and not frozen:
                    stance_pred[k] = self.gait.stance_at(MPC_DT * (k + 1))
                else:
                    stance_pred[k] = True
            feet_pos = foot_pos.copy()
            for i, f in enumerate(FOOT_NAMES):
                if not stance[i] and f in self.landing:
                    feet_pos[i] = self.landing[f]
            x_ref = np.zeros((MPC_H, 12))
            for k in range(MPC_H):
                tk = MPC_DT * (k + 1)
                p_ref = self.p_cmd + v_cmd[:2] * tk
                x_ref[k] = [0.0, 0.0, yaw_cmd,
                            p_ref[0], p_ref[1], Z_DES,
                            0.0, 0.0, 0.0,
                            v_cmd[0], v_cmd[1], 0.0]
            self.forces = self.mpc.solve(x0, yaw, com, feet_pos, stance_pred, x_ref)

        # ---- 5. 低层: WBC 风格 ----
        # 支撑: -Jᵀ·f_MPC + 弱PD(钉地构型) + 重力补偿 + M·q̈ 惯性前馈
        # 摆动: PD(轨迹IK) + 重力补偿 + M·q̈ 惯性前馈 + 近地笛卡尔阻尼
        mujoco.mj_fullM(model, d, self._Mbuf)
        tau = np.zeros(model.nu)
        for i, f in enumerate(FOOT_NAMES):
            vadr = self.leg[i]["vadr"]
            aadr = self.leg[i]["aadr"]
            q = np.array([d.qpos[a] for a in self.leg[i]["qadr"]])
            qd = np.array([d.qvel[a] for a in vadr])
            # 腿自身重力/科氏力前馈 (SRB 模型忽略的部分)
            tau_grav = np.array([d.qfrc_bias[a] for a in vadr])
            if stance[i]:
                mujoco.mj_jacGeom(model, d, self._jacp, self._jacr, self.foot_geom[i])
                J = self._jacp[:, vadr]
                # 支撑不做惯性前馈: 钉地 IK 数值解二阶差分噪声过大, 且其
                # 科氏项已含于 qfrc_bias, 惯性项本身很小
                target = self.planted[f] if self.planted[f] is not None else foot_pos[i]
                q_ref, _ = self.ik.solve(f, d.qpos[3:7], d.qpos[0:3], target, q, iters=3)
                qd_ref, qdd_ref = self._ref_derivs(f, q_ref)
                tau_leg = (-(J.T @ self.forces[i])
                           + KP_STANCE * (q_ref - q) + KV_STANCE * (0.0 - qd)
                           + tau_grav)
            else:
                s = self.gait.swing_progress(f)
                target = swing_position(self.liftoff[f], self.landing[f], s, SWING_HEIGHT)
                q_ref, _ = self.ik.solve(f, d.qpos[3:7], d.qpos[0:3], target, q, iters=4)
                qd_ref, qdd_ref = self._ref_derivs(f, q_ref)
                tau_leg = (KP_SWING * (q_ref - q) + KV_SWING * (qd_ref - qd)
                           + WBC_FF * (self._Mbuf[np.ix_(vadr, vadr)] @ qdd_ref)
                           + tau_grav)
                # 近地笛卡尔阻尼: 足端进入地面 3cm 内时直接耗散垂直动量,
                # 抑制"足端超前剖面 + 撞地弹飞机身"的冲击环
                if d.geom_xpos[self.foot_geom[i]][2] < self.ground_z + 0.03:
                    mujoco.mj_jacGeom(model, d, self._jacp, self._jacr, self.foot_geom[i])
                    J_sw = self._jacp[:, vadr]
                    v_foot = self._jacp @ d.qvel
                    tau_leg += -J_sw.T @ (K_CART * v_foot)
            for a in range(3):
                tau[aadr[a]] = tau_leg[a]

        self.step += 1
        self.state = {"t": t, "com": com, "v": v, "rpy": rpy, "stance": stance}
        return np.clip(tau, self.ctrl_lo, self.ctrl_hi)


# ---------------------------------------------------------------------- #
def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # home 站立姿态
    mujoco.mj_forward(model, data)

    # menagerie 的 home keyframe 足端嵌入地板 ~1.8cm, 会被近刚性接触弹飞。
    # 抬高基座使足端球恰好触地。
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
    foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
    foot_r = model.geom_size[foot_geoms[0]][0]
    data.qpos[2] += foot_r - foot_z
    mujoco.mj_forward(model, data)

    ctrl = Go2Controller(model, data)
    print(f"模型: 总质量 {ctrl.mass:.2f} kg, 复合惯量对角 {np.round(ctrl.I_diag, 5)}")
    print(f"时间轴: 0~{T_STAND}s 站立 | {T_STAND}~{T_TROT}s 原地trot | {T_TROT}~{T_END}s 前进 {V_WALK} m/s")

    log = {k: [] for k in ("t", "z", "roll", "pitch", "vx", "vy", "yaw", "fz_sum")}
    com0 = data.sensordata[ctrl.adr_com:ctrl.adr_com + 3].copy()
    fallen = False

    while data.time < T_END:
        mujoco.mj_forward(model, data)
        data.ctrl[:] = ctrl.update(data.time)
        mujoco.mj_step(model, data)

        st = ctrl.state
        if ctrl.step % 50 == 0:   # 100 Hz 记录
            log["t"].append(st["t"]); log["z"].append(st["com"][2])
            log["roll"].append(st["rpy"][0]); log["pitch"].append(st["rpy"][1])
            log["yaw"].append(st["rpy"][2])
            log["vx"].append(st["v"][0]); log["vy"].append(st["v"][1])
            log["fz_sum"].append(float(ctrl.forces[:, 2].sum()))
        if ctrl.step % 1000 == 0:
            print(f"t={st['t']:5.2f}s  z={st['com'][2]:.3f}  roll={np.degrees(st['rpy'][0]):+6.1f}°"
                  f"  pitch={np.degrees(st['rpy'][1]):+6.1f}°  vx={st['v'][0]:+.2f} m/s"
                  f"  MPC[{ctrl.mpc.last_status}]")
        if st["com"][2] < 0.16 or abs(st["rpy"][0]) > 1.0 or abs(st["rpy"][1]) > 1.0:
            fallen = True
            print(f"!! t={st['t']:.2f}s 摔倒 (z={st['com'][2]:.3f}, rpy={np.degrees(st['rpy'])})")
            break

    if fallen:
        print("\n结果: 失败 —— 机器人摔倒")
        sys.exit(1)

    # ---- 统计 ----
    log = {k: np.array(v) for k, v in log.items()}
    com_end = data.sensordata[ctrl.adr_com:ctrl.adr_com + 3]
    walk = log["t"] >= T_TROT
    dist = com_end[0] - com0[0]
    print("\n===== 结果 =====")
    print(f"全程前进距离: {dist:.2f} m  (期望 {V_WALK * (T_END - T_TROT):.2f} m)")
    print(f"行走段平均速度: {log['vx'][walk].mean():+.2f} m/s (指令 +{V_WALK})")
    print(f"行走段横向漂移速度: {log['vy'][walk].mean():+.2f} m/s")
    print(f"高度 z: 均值 {log['z'][walk].mean():.3f} m, 最小 {log['z'][walk].min():.3f} m")
    print(f"roll: |max| {np.degrees(np.abs(log['roll'][walk]).max()):.1f}°, "
          f"pitch: |max| {np.degrees(np.abs(log['pitch'][walk]).max()):.1f}°")
    print(f"MPC 求解 {ctrl.mpc.solve_count} 次, 末次状态 {ctrl.mpc.last_status}")
    np.savez("results/go2_mpc_log.npz", **log)
    print("曲线数据已存 results/go2_mpc_log.npz")


if __name__ == "__main__":
    main()
