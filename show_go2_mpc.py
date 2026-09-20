"""Go2 convex MPC 交互式演示 (用 mjpython 运行!)

  .venv/bin/mjpython show_go2_mpc.py

演示剧本 (每 24s 循环): 站立 4s -> 原地对角小跑 4s -> 前进 0.5 m/s 12s
                         -> 减速回原地小跑 -> 站立

窗口操作: 左键旋转 / 右键平移 / 滚轮缩放 / 空格暂停 / 关闭窗口退出
"""
import time

import numpy as np
import mujoco
import mujoco.viewer

from run_go2_mpc import Go2Controller, SCENE, DT, FOOT_NAMES

CYCLE = 24.0     # 剧本周期


def demo_command(t):
    """循环剧本: 机身系速度指令 [vx, vy]"""
    tc = t % CYCLE
    if tc < 4.0:
        return np.zeros(2)                    # 站立 (步态未开)
    if tc < 8.0:
        return np.zeros(2)                    # 原地 trot
    if tc < 20.0:
        return np.array([0.5, 0.0])           # 前进
    return np.zeros(2)                        # 减速停回原地


def phase_name(t):
    tc = t % CYCLE
    if tc < 4.0:
        # 仅第一循环是纯站立; 之后步态保持开启, v_cmd=0 即原地小跑
        return "站立 (MPC 静态平衡)" if t < 4.0 else "原地小跑 (新循环)"
    if tc < 8.0:
        return "原地对角小跑 (trot)"
    if tc < 20.0:
        return "前进 0.5 m/s (convex MPC + Raibert 落脚点)"
    return "减速停止"


model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# 足端恰好触地 (keyframe 嵌入地板修正)
foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
foot_r = model.geom_size[foot_geoms[0]][0]
data.qpos[2] += foot_r - foot_z
mujoco.mj_forward(model, data)

ctrl = Go2Controller(model, data)
ctrl.get_command = demo_command
ctrl.stand_until = 4.0

print("Go2 convex MPC 演示: 站立 -> 原地小跑 -> 前进 -> 减速, 循环")
print(f"模型质量 {ctrl.mass:.1f} kg | MPC 100 Hz (OSQP) | 低层 500 Hz")

last_phase = ""
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0_wall = time.perf_counter()
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        data.ctrl[:] = ctrl.update(data.time)
        mujoco.mj_step(model, data)

        ph = phase_name(data.time)
        if ph != last_phase:
            print(f"t={data.time:6.1f}s  {ph}")
            last_phase = ph

        if ctrl.step % 10 == 0:
            viewer.sync()

        # 实时节流 (1x)
        ahead = data.time - (time.perf_counter() - t0_wall)
        if ahead > 0:
            time.sleep(ahead)

print("窗口已关闭, 演示结束")
