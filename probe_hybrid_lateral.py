"""混合版 30N·s 纯侧向失败格复现探针: 打印交棒/接回事件时间线 vs 摔倒时刻
(判定失败发生在 RL 救援期还是接回之后 —— 两种签名对应不同修法)
"""
import sys

import numpy as np
import mujoco

from run_go2_hybrid import (HybridController, RL_POLICY, load_model,
                             reset_stance)
from test_push_recovery import CONDITIONS, PUSH_DUR

cond = sys.argv[1] if len(sys.argv) > 1 else "trot"
angle_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

model = load_model()
data = mujoco.MjData(model)
reset_stance(model, data)

import pickle
with open(RL_POLICY, "rb") as f:
    rl_params = pickle.load(f)
ctrl = HybridController(model, data, rl_params, verbose=True)

t_push0, t_end, cmd_fn = CONDITIONS[cond]
ctrl.get_command = cmd_fn
ctrl.stand_until = 4.0

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
mag = 30.0
angle = np.deg2rad(angle_deg)
rng = np.random.default_rng(seed)             # 与套件同 seed
t_push = t_push0 + rng.uniform(0.0, 1.0)
fvec = np.array([np.cos(angle), np.sin(angle), 0.0]) * (mag / PUSH_DUR)
print(f"工况 {cond}, 推力时刻 t={t_push:.2f}s, {mag:.0f}N·s @{angle_deg:+.0f}°", flush=True)

pushed = push_done = False
while data.time < t_end:
    if not pushed and data.time >= t_push:
        data.xfrc_applied[base_id, :3] = fvec
        pushed = True
        print(f"---- 推力施加 t={data.time:.2f}s ----", flush=True)
    if pushed and not push_done and data.time >= t_push + PUSH_DUR:
        data.xfrc_applied[base_id, :] = 0.0
        push_done = True
    mujoco.mj_forward(model, data)
    data.ctrl[:] = ctrl.update(data.time)
    mujoco.mj_step(model, data)
    st = ctrl.state
    if st["com"][2] < 0.16:
        print(f"!! 摔倒 t={data.time:.2f}s roll={np.degrees(st['rpy'][0]):.0f}° "
              f"(推力后 {data.time - t_push:.2f}s)", flush=True)
        break
    if ctrl.step % 250 == 0:
        print(f"    t={data.time:5.2f} mode={ctrl.mode:3s} fade={ctrl.fade:.2f} "
              f"z={st['com'][2]:.3f} roll={np.degrees(st['rpy'][0]):+5.1f}° "
              f"v=({st['v'][0]:+.2f},{st['v'][1]:+.2f})", flush=True)
else:
    print(f"未摔, 结束 z={ctrl.state['com'][2]:.3f} "
          f"roll={np.degrees(ctrl.state['rpy'][0]):.1f}° 交棒 {ctrl.rescues} 次", flush=True)
