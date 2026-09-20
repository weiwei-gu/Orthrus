"""Go2 混合控制器 — MPC 管行走, RL 策略管救援 (MPC+RL 融合的本地实现)

架构:
  平时 (MPC 模式): convex MPC 全栈行走 (速度跟踪 100%), 输出关节力矩,
    经代数转换到位置伺服目标: q_des = q + (τ + kd·qd) / kp
    (kp/kd 从 go2_mjx 位置伺服读出 —— 等价于"通过位置接口做力矩控制",
     MPC 全栈一行不改; Go2Controller 原有的 ctrlrange 裁剪是力矩域被
     位置域范围误用, 实例化后覆写为 ±inf 禁用)
  被推 (交棒): 仲裁器检测 质心速度偏离指令 >0.45 m/s / 倾角 >14° /
    高度 <0.19 → 40ms 淡化交棒 RL 策略 (bc_init, 推力恢复 60% 冠军)
  稳住 (接回): 判据是"脱离危机"而非"完美姿态" (bc_init 标称俯仰即有
    5~12°, 不适合做退出线): 高度/倾角/滑移回稳 0.3s 且救援 ≥0.8s;
    或 2.5s 超时强制接回。接回时重锚 MPC 步态相位/钉地点/位置参考
    (p_cmd 锚到当前质心, 不往回拽), 残余姿态交给 MPC 自己修
  RL 全程影子运行 (50 Hz), obs-action 链连续 —— 交棒无跳变

场景: hybrid_scene.xml = scene_mjx + com/com_vel 传感器;
      timestep 运行时覆写 0.002 (MPC 500Hz 低层; RL 策略仍 50Hz)

用法:
  .venv/bin/python run_go2_hybrid.py            # 16s 剧本 + 行走中踹一脚
  .venv/bin/python run_go2_hybrid.py --push     # 72 试验推力套件
"""
import argparse
import json
import pickle

import numpy as np
import mujoco

from go2_utils import FOOT_NAMES, quat_to_rpy, rot_z
from run_go2_mpc import Go2Controller, T_STAND, T_TROT, T_END, V_WALK
from eval_go2_rl import PolicyController

HYBRID_SCENE = "models/go2/hybrid_scene.xml"
RL_POLICY = "policies/bc_init.pkl"

# ---- 仲裁阈值 ----
ENTER_DV = 0.45                 # m/s, 质心速度偏离指令 (推力最可靠的信号)
ENTER_TILT = np.deg2rad(14.0)   # rad, 高于 MPC 行走俯仰瞬态 (标称 5~12°)
ENTER_Z = 0.19                  # m, 高度塌陷
HANDBACK_GRACE = 0.6            # s, 接回后此时间内不重复交棒 (防抖)
MIN_RESCUE = 0.8                # s, 最短救援时长
MAX_RESCUE = 2.5                # s, 超时强制接回 (只要还活着)
EXIT_TILT = np.deg2rad(15.0)    # rad, 脱离危机线 (非完美姿态)
EXIT_Z = 0.20
EXIT_V = 0.40                   # m/s, 仍在滑移则不接回
CALM_STEPS = 150                # 0.3s @500Hz 稳定窗
FADE_STEP = 0.05                # 每步淡化增量 (20 步 = 40ms 过渡)
CMD_SLEW = 0.8                  # m/s², 仲裁指令斜坡 (阶跃指令的加速期不误判为扰动)


class HybridController:
    """MPC+RL 仲裁控制器, 与 Go2Controller/PolicyController 同接口"""

    def __init__(self, model, data, rl_params=None, verbose=False,
                 restore_soft_feet=False):
        self.model, self.data = model, data
        self.verbose = verbose
        self.step = 0
        self.mode = "mpc"        # 'mpc' 行走 | 'rl' 救援
        self.fade = 1.0          # 位置目标混合系数: 1=MPC, 0=RL
        self.stand_until = 0.0
        self.get_command = lambda t: np.zeros(2)
        self.state = {"t": 0.0, "com": np.zeros(3), "v": np.zeros(3),
                      "rpy": np.zeros(3), "stance": np.ones(4, bool)}

        # 位置伺服参数 (τ→位置转换): kp = gainprm[0], kd = -biasprm[2]
        self.kp_vec = model.actuator_gainprm[:, 0].copy()
        self.kd_vec = -model.actuator_biasprm[:, 2].copy()
        assert self.kp_vec.min() > 30.0 and 0.2 < self.kd_vec.min() < 1.5, \
            "执行器不是位置伺服 (gainprm/biasprm 不符)"
        self.kp, self.kd = float(self.kp_vec[0]), float(self.kd_vec[0])
        # 位置目标的合法范围 (ctrlrange; 0,0 表示无限位)
        lo = model.actuator_ctrlrange[:, 0].copy()
        hi = model.actuator_ctrlrange[:, 1].copy()
        unl = hi <= lo
        lo[unl], hi[unl] = -np.inf, np.inf
        self.pos_lo, self.pos_hi = lo, hi
        # 执行器 -> 关节地址 (读该关节 q/qd 做转换)
        self._act_qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]].copy()
        self._act_vadr = model.jnt_dofadr[model.actuator_trnid[:, 0]].copy()

        # MPC 全栈 (内部会加硬足端接触)
        foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f)
                      for f in FOOT_NAMES]
        solimp0 = model.geom_solimp[foot_geoms].copy()
        self.mpc = Go2Controller(model, data)
        if restore_soft_feet:      # 调试用: 恢复 scene_mjx 原接触 (bc_init 标定物理)
            model.geom_solimp[foot_geoms] = solimp0
        self.mpc.ctrl_lo[:] = -np.inf     # 其 clip 是力矩被位置范围误裁, 禁用
        self.mpc.ctrl_hi[:] = np.inf
        self.mpc.get_command = lambda t: self.get_command(t)
        self.adr_com = self.mpc.adr_com
        self.adr_v = self.mpc.adr_v

        # RL 策略 (bc_init — 推力恢复冠军)
        if rl_params is None:
            with open(RL_POLICY, "rb") as f:
                rl_params = pickle.load(f)
        self.rl = PolicyController(model, data, rl_params)
        self.rl.get_command = lambda t: self.get_command(t)

        self._last_mpc_ctrl = data.qpos[self._act_qadr].copy()
        self._rescue_t0 = None
        self._calm = 0
        self._handback_t = -10.0
        self._v_cmd_flt = np.zeros(3)   # 斜坡滤波后的指令 (仅仲裁用)
        self.rescues = 0

    # ------------------------------------------------------------------ #
    def _tau_to_pos(self, tau):
        """MPC 关节力矩 -> 位置伺服目标: q_des = q + (τ + kd·qd)/kp

        位置伺服输出 τ_s = kp·(q_des − q) − kd·qd, 令 τ_s = τ_mpc 解出 q_des,
        等价于通过位置接口做力矩控制 (无需改动 MPC 全栈)。
        """
        d = self.data
        q = d.qpos[self._act_qadr]
        qd = d.qvel[self._act_vadr]
        return q + (tau + self.kd_vec * qd) / self.kp_vec

    def _handback(self, com):
        """RL 救援结束后 MPC 重新接管: 重锚内部状态, 残余姿态交给 MPC 修"""
        m = self.mpc
        m.gait.phase = 0.0                        # 对角组从相位 0 重新起拍
        m.planted = {f: None for f in FOOT_NAMES}  # 触地检测重新锚定钉地点
        m._recovery = False
        if hasattr(m, "p_cmd"):
            m.p_cmd = com[:2].copy()              # 位置参考锚到当前质心, 不往回拽

    # ------------------------------------------------------------------ #
    def update(self, t):
        d = self.data
        self.mpc.stand_until = self.stand_until   # 套件从外部设置

        # 状态 (传感器, 与 MPC 同源)
        com = d.sensordata[self.adr_com:self.adr_com + 3].copy()
        v = d.sensordata[self.adr_v:self.adr_v + 3].copy()
        rpy = quat_to_rpy(d.qpos[3:7])

        # RL 每步都跑 (影子), 保持 obs-action 链连续 —— 交棒时无跳变
        rl_ctrl = self.rl.update(t)

        # ---- 仲裁 ----
        cmd = np.asarray(self.get_command(t), dtype=float)
        v_cmd = rot_z(self.mpc.yaw0) @ np.array([cmd[0], cmd[1], 0.0])
        # 指令斜坡滤波 (仲裁专用): 阶跃指令 0->0.5 的加速期 dv 瞬时超阈值,
        # 实测造成一次假交棒。按 CMD_SLEW 逼近真实指令 —— MPC 跟得上斜坡,
        # 真推力是瞬时的速度跳变, 依然立刻超阈值。
        step_max = CMD_SLEW * 0.002
        dcmd = v_cmd[:2] - self._v_cmd_flt[:2]
        ncmd = np.hypot(dcmd[0], dcmd[1])
        if ncmd > step_max:
            self._v_cmd_flt[:2] += dcmd * (step_max / ncmd)
        else:
            self._v_cmd_flt[:2] = v_cmd[:2]
        dv = np.hypot(v[0] - self._v_cmd_flt[0], v[1] - self._v_cmd_flt[1])
        tilt = max(abs(rpy[0]), abs(rpy[1]))
        z = com[2]

        if self.mode == "mpc":
            graced = (t - self._handback_t) > HANDBACK_GRACE
            if z < 0.17 or (graced and (dv > ENTER_DV or tilt > ENTER_TILT
                                         or z < ENTER_Z)):
                self.mode, self._rescue_t0, self._calm = "rl", t, 0
                self.rescues += 1
                if self.verbose:
                    print(f"t={t:5.2f}s 💥 交棒 RL 救援 #{self.rescues} "
                          f"(dv={dv:.2f} tilt={np.degrees(tilt):.0f}° z={z:.3f})",
                          flush=True)
        else:
            rescue_t = t - self._rescue_t0
            ok = tilt < EXIT_TILT and z > EXIT_Z and np.hypot(v[0], v[1]) < EXIT_V
            self._calm = self._calm + 1 if ok else 0
            timeout_ok = rescue_t > MAX_RESCUE and z > ENTER_Z
            if rescue_t >= MIN_RESCUE and (self._calm >= CALM_STEPS or timeout_ok):
                self.mode = "mpc"
                self._handback_t = t
                self._handback(com)
                if self.verbose:
                    print(f"t={t:5.2f}s ✅ MPC 接回 (救援 {rescue_t:.2f}s, "
                          f"残余 tilt={np.degrees(tilt):.0f}°)", flush=True)

        # ---- 混合输出 (40ms 交叉淡化) ----
        target = 1.0 if self.mode == "mpc" else 0.0
        self.fade += float(np.clip(target - self.fade, -FADE_STEP, FADE_STEP))
        mpc_ctrl = self._last_mpc_ctrl
        if self.fade > 0.0:
            tau = self.mpc.update(t)
            mpc_ctrl = self._tau_to_pos(tau)
            self._last_mpc_ctrl = mpc_ctrl
        ctrl = self.fade * mpc_ctrl + (1.0 - self.fade) * rl_ctrl

        self.step += 1
        self.state = {"t": t, "com": com, "v": v, "rpy": rpy,
                      "stance": self.mpc.state.get("stance", np.ones(4, bool))}
        return np.clip(ctrl, self.pos_lo, self.pos_hi)


# ---------------------------------------------------------------------- #
def load_model():
    """混合场景 + timestep 覆写 (MPC 500Hz 低层; RL 策略仍 50Hz)"""
    model = mujoco.MjModel.from_xml_path(HYBRID_SCENE)
    model.opt.timestep = 0.002
    return model


def reset_stance(model, data):
    """home keyframe 起步 + 足端恰好触地 (与套件同款)"""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
    foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
    data.qpos[2] += model.geom_size[foot_geoms[0]][0] - foot_z
    mujoco.mj_forward(model, data)


def standard_test(push=True):
    """16s 剧本: 站立 -> trot -> 前进 0.5, 行进中从 90° 踹 18 N·s"""
    model = load_model()
    data = mujoco.MjData(model)
    reset_stance(model, data)

    ctrl = HybridController(model, data, verbose=True)
    ctrl.stand_until = T_STAND

    def script(t):
        if t < T_TROT:
            return np.zeros(2)
        return np.array([V_WALK, 0.0])
    ctrl.get_command = script

    print(f"位置伺服 kp={ctrl.kp:.0f} kd={ctrl.kd:.2f} | 质量 {ctrl.mpc.mass:.2f} kg")
    print(f"时间轴: 0~{T_STAND}s 站立 | {T_STAND}~{T_TROT}s trot | {T_TROT}~{T_END}s 前进")
    push_t, push_dur = (T_TROT + 2.5) if push else 1e9, 0.1
    if push:
        print(f"t={push_t}s 从 90° (侧向) 踹 18 N·s —— MPC 单独在此方向恢复率 ~0%\n")

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    fvec = np.array([np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90)), 0.0]) * (18.0 / push_dur)
    pushed = push_done = False
    com0 = data.sensordata[ctrl.adr_com:ctrl.adr_com + 3].copy()

    while data.time < T_END:
        if not pushed and data.time >= push_t:
            data.xfrc_applied[base_id, :3] = fvec
            pushed = True
        if pushed and not push_done and data.time >= push_t + push_dur:
            data.xfrc_applied[base_id, :] = 0.0
            push_done = True

        mujoco.mj_forward(model, data)
        data.ctrl[:] = ctrl.update(data.time)
        mujoco.mj_step(model, data)

        st = ctrl.state
        if ctrl.step % 500 == 0:
            print(f"t={st['t']:5.2f}s  z={st['com'][2]:.3f}  roll={np.degrees(st['rpy'][0]):+6.1f}°"
                  f"  pitch={np.degrees(st['rpy'][1]):+6.1f}°  vx={st['v'][0]:+.2f} m/s"
                  f"  [{'MPC' if ctrl.fade > 0.5 else 'RL 救援'}]", flush=True)
        if st["com"][2] < 0.16 or max(abs(st["rpy"][0]), abs(st["rpy"][1])) > 1.0:
            print(f"\n!! t={st['t']:.2f}s 摔倒")
            return 1

    com_end = data.sensordata[ctrl.adr_com:ctrl.adr_com + 3]
    print("\n===== 结果 =====")
    print(f"前进距离: {com_end[0] - com0[0]:.2f} m (MPC 单独参考 ~3.9 m)")
    print(f"救援交棒: {ctrl.rescues} 次")
    if not push:
        print("✅ 无扰动对照通过" if ctrl.rescues == 0
              else f"⚠️ 无扰动却交棒 {ctrl.rescues} 次 (仲裁误触发, 需修)")
    elif ctrl.rescues >= 1:
        print("✅ 混合控制器通过: 行走中被踹 -> RL 救援 -> MPC 接回继续走")
    else:
        print("⚠️ 未触发救援 (推力可能未生效, 需检查)")
    return 0


def push_suite():
    """72 试验推力套件 (同 MPC/bc_init 完全一致的判定)"""
    from test_push_recovery import run_trial, MAGS, DIR_ANGLES, CONDITIONS

    model = load_model()
    with open(RL_POLICY, "rb") as f:
        rl_params = pickle.load(f)
    factory = lambda m, d: HybridController(m, d, rl_params)

    trials = []
    for cond in CONDITIONS:
        r = run_trial(model, cond, 0.0, 0.0, 99, no_push=True, ctrl_factory=factory)
        r["angle_deg"] = -1
        trials.append(r)
        print(f"[对照/{cond}] pass={r['pass']}", flush=True)

    total = 0
    for cond in CONDITIONS:
        for mag in MAGS:
            for angle in DIR_ANGLES:
                for seed in (1, 2) if cond == "trot" else (3,):
                    r = run_trial(model, cond, mag, angle, seed, ctrl_factory=factory)
                    trials.append(r)
                    total += 1
                    print(f"[{cond} {mag:4.0f}N·s {r['angle_deg']:5.1f}°] "
                          f"{'✓' if r['pass'] else '✗'} rec={r['recover_s']}", flush=True)

    with open("results/push_recovery_hybrid.json", "w") as f:
        json.dump(trials, f, indent=1)

    print("\n===== 混合控制器推力恢复汇总 =====")
    print("基线: MPC trot 88/44/12 walk 62/0/13 总40% | RL-BC trot 88/56/25 walk 88/75/38 总60%")
    for cond in CONDITIONS:
        for mag in MAGS:
            sel = [t for t in trials if t["cond"] == cond and t["mag"] == mag]
            n_pass = sum(t["pass"] for t in sel)
            print(f"{cond:<6}{mag:>5.0f}N·s  {n_pass}/{len(sel)}")
    sel = [t for t in trials if t["angle_deg"] >= 0]
    n_pass = sum(t["pass"] for t in sel)
    print(f"\n总计: {n_pass}/{len(sel)} ({100 * n_pass / len(sel):.0f}%)  详细数据已存 results/push_recovery_hybrid.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true", help="跑 72 试验推力套件")
    p.add_argument("--no-push", action="store_true", help="标准剧本但不踹 (无扰动对照)")
    args = p.parse_args()
    if args.push:
        push_suite()
    else:
        standard_test(push=not args.no_push)


if __name__ == "__main__":
    main()
