"""Go2 混合控制器交互演示 (用 mjpython 运行!)

  .venv/bin/mjpython show_go2_hybrid.py

演示剧本 (每 24s 循环): 站立 4s -> 前进 0.5 m/s 16s -> 减速 4s
外加: 每 3.5~5.5s 随机方向踹一脚 (15~25 N·s, 比 RL 演示更狠)

看点 (机身颜色 = 当前"大脑"):
  原色   -> MPC 在开车 (对角小跑, 速度跟踪)
  橙红色 -> RL 在救援 (被踹后交棒, 蹲低碎步稳住)
  原色   -> MPC 接回 (继续走)
摔倒趴 2s 自动重置; 镜头跟随; 控制台实时打印踹踢/交棒/接回事件。
"""
import pickle
import time

import jax
import numpy as np
import mujoco
import mujoco.viewer

from eval_go2_rl import ACT_SIZE
from go2_utils import FOOT_NAMES, quat_to_rpy
from run_go2_hybrid import HYBRID_SCENE, HybridController, RL_POLICY

CYCLE = 24.0
KICK_MAG = (15.0, 25.0)    # N·s (比 RL 演示更狠 —— 混合版 20 N·s 恢复率 100%)
KICK_EVERY = (3.5, 5.5)    # s
KICK_DUR = 0.1
FALL_Z = 0.16
RGBA_RESCUE = np.array([0.95, 0.35, 0.15, 1.0])


def demo_command(t):
    tc = t % CYCLE
    if tc < 4.0:
        return np.zeros(2)
    if tc < 20.0:
        return np.array([0.5, 0.0])
    return np.zeros(2)


def phase_name(t):
    tc = t % CYCLE
    if tc < 4.0:
        return "站立"
    if tc < 20.0:
        return "前进 0.5 m/s"
    return "减速"


def reset_all(model, data, ctrl):
    """摔倒后完整重置 (物理 + MPC 内部状态 + RL 内部状态 + 仲裁器)"""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
    foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
    data.qpos[2] += model.geom_size[foot_geoms[0]][0] - foot_z
    mujoco.mj_forward(model, data)

    m = ctrl.mpc
    m.gait_enabled = False
    m.gait.phase = 0.0
    m.planted = {f: None for f in FOOT_NAMES}
    m._prev_swing = {f: False for f in FOOT_NAMES}
    m.liftoff, m.landing = {}, {}
    m._recovery = False
    m.step = 0
    m.forces = np.zeros((4, 3))
    for f in FOOT_NAMES:
        m.reset_ref(f)
    if hasattr(m, "p_cmd"):
        del m.p_cmd
    if hasattr(m, "v_cmd_prev"):
        del m.v_cmd_prev
    m.ground_z = float(np.mean([data.geom_xpos[g][2] for g in m.foot_geom]))
    m.yaw0 = quat_to_rpy(data.qpos[3:7])[2]

    r = ctrl.rl
    r.step = 0
    r.action = np.zeros(ACT_SIZE)
    r.rng = jax.random.PRNGKey(0)
    r._cmd3 = np.zeros(3)

    ctrl.mode = "mpc"
    ctrl.fade = 1.0
    ctrl._v_cmd_flt = np.zeros(3)
    ctrl._rescue_t0 = None
    ctrl._calm = 0
    ctrl._handback_t = -10.0
    ctrl._last_mpc_ctrl = data.qpos[ctrl._act_qadr].copy()


def main():
    from run_go2_hybrid import load_model, reset_stance
    model = load_model()          # hybrid_scene + timestep 0.002
    data = mujoco.MjData(model)
    reset_stance(model, data)

    with open(RL_POLICY, "rb") as f:
        rl_params = pickle.load(f)
    ctrl = HybridController(model, data, rl_params, verbose=True)
    ctrl.get_command = demo_command
    ctrl.stand_until = 4.0

    # 预热 RL 策略 (把 jax 编译挪到窗口打开前), 然后复位其内部状态
    ctrl.rl.update(0.0)
    ctrl.rl.step = 0
    ctrl.rl.action = np.zeros(ACT_SIZE)
    ctrl.rl.rng = jax.random.PRNGKey(0)
    ctrl.rl._cmd3 = np.zeros(3)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    g0, gn = model.body_geomadr[base_id], model.body_geomnum[base_id]
    base_geoms = list(range(g0, g0 + gn))
    rgba0 = model.geom_rgba[base_geoms].copy()

    print(f"混合控制器演示: MPC 行走 + RL (bc_init) 救援 | 机身变色 = 换脑", flush=True)
    print(f"剧本: 站立 4s -> 前进 16s -> 减速 4s (循环), 每 {KICK_EVERY[0]}~{KICK_EVERY[1]}s "
          f"踹 {KICK_MAG[0]}~{KICK_MAG[1]} N·s", flush=True)

    rng = np.random.default_rng(7)
    next_kick, kick_until, fall_until = 6.5, -1.0, -1.0
    kicks = falls = 0
    last_phase, last_mode = "", True   # True=MPC 色

    with mujoco.viewer.launch_passive(model, data) as viewer:
        follow_cam = getattr(viewer, "cam", None)
        t0_wall = time.perf_counter()
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            data.ctrl[:] = ctrl.update(data.time)

            # ---- 随机踹一脚 (xfrc 冲量, 与套件同标度) ----
            if fall_until < 0 and kick_until < 0 and data.time >= next_kick:
                mag = rng.uniform(*KICK_MAG)
                ang = rng.uniform(0, 2 * np.pi)
                data.xfrc_applied[base_id, :3] = np.array(
                    [np.cos(ang), np.sin(ang), 0.0]) * (mag / KICK_DUR)
                kick_until = data.time + KICK_DUR
                kicks += 1
                print(f"t={data.time:6.1f}s 💥 第{kicks}脚: {mag:.0f} N·s @ "
                      f"{np.degrees(ang):5.0f}° [{phase_name(data.time)}]", flush=True)
            if 0 <= kick_until <= data.time:
                data.xfrc_applied[base_id, :] = 0.0
                kick_until = -1.0
                next_kick = data.time + rng.uniform(*KICK_EVERY)

            mujoco.mj_step(model, data)

            # ---- 摔倒检测: 趴 2s 后完整重置续演 ----
            if fall_until < 0 and data.time > 1.0 and ctrl.state["com"][2] < FALL_Z:
                fall_until = data.time + 2.0
                falls += 1
                print(f"t={data.time:6.1f}s ❌ 摔倒 (第{falls}次), 2s 后重置", flush=True)
            if 0 <= fall_until <= data.time:
                reset_all(model, data, ctrl)
                next_kick, kick_until, fall_until = 6.5, -1.0, -1.0
                t0_wall = time.perf_counter()
                model.geom_rgba[base_geoms] = rgba0
                last_mode = True
                print("— 已重置, 继续演示 —", flush=True)

            # ---- 机身颜色 = 当前大脑 (MPC 原色 / RL 救援橙红) ----
            mpc_mode = ctrl.fade > 0.5
            if mpc_mode != last_mode:
                model.geom_rgba[base_geoms] = rgba0 if mpc_mode else RGBA_RESCUE
                last_mode = mpc_mode

            ph = phase_name(data.time)
            if ph != last_phase:
                st = ctrl.state
                print(f"t={data.time:6.1f}s {ph} | z={st['com'][2]:.3f} "
                      f"pitch={np.degrees(st['rpy'][1]):+.1f}° vx={st['v'][0]:+.2f}",
                      flush=True)
                last_phase = ph

            if ctrl.step % 10 == 0:
                if follow_cam is not None:
                    follow_cam.lookat[0] = data.qpos[0]
                    follow_cam.lookat[1] = data.qpos[1]
                    follow_cam.lookat[2] = 0.25
                viewer.sync()
            ahead = data.time - (time.perf_counter() - t0_wall)
            if ahead > 0:
                time.sleep(ahead)

    print(f"窗口已关闭 | 被踹 {kicks} 次, 摔倒 {falls} 次, "
          f"RL 救援交棒 {ctrl.rescues} 次", flush=True)


if __name__ == "__main__":
    main()
