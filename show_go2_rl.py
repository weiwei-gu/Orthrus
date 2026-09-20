"""RL 策略交互演示 (用 mjpython 运行!)

  .venv/bin/mjpython show_go2_rl.py                          # 默认 bc_init + 踹踢
  .venv/bin/mjpython show_go2_rl.py <策略.pkl>                # 指定策略 + 踹踢
  .venv/bin/mjpython show_go2_rl.py <策略.pkl> --no-kicks     # 纯行走, 镜头跟随

演示剧本 (每 24s 循环): 站立 4s -> 前进 0.5 m/s 12s -> 斜行 4s -> 减速
默认每 3.5~5s 随机方向踹一脚 (12~20 N·s, 与推力套件同标度);
摔倒则趴 2s 自动重置续演; 镜头自动跟随机身 (行走不走出画面)。

(斜行和持续行走是 MPC 版没有的能力; 踹踢是 MPC 版演示没有的)
"""
import pickle
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from eval_go2_rl import ACT_SIZE, PolicyController, RL_SCENE

CYCLE = 24.0
KICK_MAG = (12.0, 20.0)   # N·s, 与推力套件同标度 (bc_init 在此区间高概率恢复)
KICK_EVERY = (3.5, 5.0)   # s, 两次踹踢间隔
KICK_DUR = 0.15           # s, 力作用时长 (套件同款: 力 = 冲量/时长)
FALL_Z = 0.16             # com_z 低于此值判定摔倒 (套件同款)


def demo_command(t):
    tc = t % CYCLE
    if tc < 4.0:
        return np.array([0.0, 0.0])
    if tc < 16.0:
        return np.array([0.5, 0.0])
    if tc < 20.0:
        return np.array([0.3, 0.3])
    return np.array([0.0, 0.0])


def phase_name(t):
    tc = t % CYCLE
    if tc < 4.0:
        return "站立"
    if tc < 16.0:
        return "前进 0.5 m/s"
    if tc < 20.0:
        return "斜行 0.3+0.3 m/s"
    return "减速"


def reset_scene(model, data, ctrl):
    """摔倒后重置: 复位物理状态 + 控制器内部状态 (不重建网络, 避免重新编译卡顿)"""
    import jax
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] += 0.005               # 微抬避免初始穿透 (与 eval 同款)
    mujoco.mj_forward(model, data)
    ctrl.step = 0
    ctrl.action = np.zeros(ACT_SIZE)
    ctrl.rng = jax.random.PRNGKey(0)
    ctrl._cmd3 = np.zeros(3)
    return ctrl


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_kicks = "--no-kicks" in sys.argv
    policy_path = args[0] if args else "policies/bc_init.pkl"
    import jax

    with open(policy_path, "rb") as f:
        params = pickle.load(f)

    model = mujoco.MjModel.from_xml_path(RL_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # 初始必须从 home 姿态复位 (否则全零状态生成, 开场即假摔)
    data.qpos[2] += 0.005
    mujoco.mj_forward(model, data)
    ctrl = PolicyController(model, data, params)
    ctrl.get_command = demo_command

    print(f"RL 策略演示: {policy_path}", flush=True)
    print(f"JAX 设备: {jax.devices()}", flush=True)
    mode = "纯行走 (无踹踢)" if no_kicks else \
        f"抗扰模式: 每 {KICK_EVERY[0]}~{KICK_EVERY[1]}s 踹一脚 {KICK_MAG[0]}~{KICK_MAG[1]} N·s"
    print(f"剧本: 站立 4s -> 前进 12s -> 斜行 4s -> 减速 4s (循环) | {mode} | 镜头跟随",
          flush=True)

    rng = np.random.default_rng(42)
    base_id = ctrl._base_id
    next_kick, kick_until, fall_until = 2.5, -1.0, -1.0
    kicks = falls = 0
    last_phase = ""

    with mujoco.viewer.launch_passive(model, data) as viewer:
        follow_cam = getattr(viewer, "cam", None)   # 旧版本无 cam 属性时跳过跟随
        t0_wall = time.perf_counter()
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            data.ctrl[:] = ctrl.update(data.time)

            # ---- 随机踹一脚 (xfrc 冲量, 与推力套件同标度) ----
            if not no_kicks and fall_until < 0 and kick_until < 0 and data.time >= next_kick:
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

            # ---- 摔倒检测: 趴 2s 后自动重置续演 (落地后 1s 内不判定, 防初始下落误报) ----
            if fall_until < 0 and data.time > 1.0 and ctrl.state["com"][2] < FALL_Z:
                fall_until = data.time + 2.0
                falls += 1
                print(f"t={data.time:6.1f}s ❌ 摔倒 (第{falls}次), 2s 后重置", flush=True)
            if 0 <= fall_until <= data.time:
                reset_scene(model, data, ctrl)
                next_kick, kick_until, fall_until = 2.5, -1.0, -1.0
                t0_wall = time.perf_counter()
                print("— 已重置, 继续演示 —", flush=True)

            ph = phase_name(data.time)
            if ph != last_phase:
                st = ctrl.state
                print(f"t={data.time:6.1f}s {ph} | z={st['com'][2]:.3f} "
                      f"pitch={np.degrees(st['rpy'][1]):+.1f}° vx={st['v'][0]:+.2f}",
                      flush=True)
                last_phase = ph

            if ctrl.step % 10 == 0:
                if follow_cam is not None:      # 镜头跟随机身, 只动目标点不打扰缩放/视角
                    follow_cam.lookat[0] = data.qpos[0]
                    follow_cam.lookat[1] = data.qpos[1]
                    follow_cam.lookat[2] = 0.24
                viewer.sync()
            ahead = data.time - (time.perf_counter() - t0_wall)
            if ahead > 0:
                time.sleep(ahead)

    tail = f"共被踹 {kicks} 次, 摔倒 {falls} 次" if kicks else "纯行走模式"
    print(f"窗口已关闭 | {tail}", flush=True)


if __name__ == "__main__":
    main()
