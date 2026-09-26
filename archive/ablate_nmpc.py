"""NMPC 消融实验: 单独开关 轨迹力臂 / 全旋转惯量, 跑同一 16s 剧本

用法: .venv/bin/python ablate_nmpc.py <lever:T/F> <inertia:T/F>
定位哪个建模改动导致行走段发散 (E 运动学恒开, 已数值验证)
"""
import sys

import numpy as np
import mujoco

from nmpc import NMPC
from run_go2_mpc import (Go2Controller, MPC_DT, MPC_H, SCENE, T_END,
                         T_STAND)
from go2_utils import FOOT_NAMES

lever = sys.argv[1] == "T"
inertia = sys.argv[2] == "T"
print(f"消融: lever_traj={lever} inertia_rot={inertia}", flush=True)

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
data.qpos[2] += model.geom_size[foot_geoms[0]][0] - foot_z
mujoco.mj_forward(model, data)

ctrl = Go2Controller(model, data)          # 凸版, 然后换脑
ctrl.mpc = NMPC(ctrl.mpc.m, np.diagonal(ctrl.mpc.I_body),
                mu=0.4, f_max=130.0, horizon=MPC_H, dt=MPC_DT,
                q_rpy=(60.0, 120.0, 15.0), q_pos=(12.0, 12.0, 100.0),
                q_vel=(6.0, 6.0, 8.0), lever_traj=lever, inertia_rot=inertia)
ctrl.stand_until = T_STAND

fallen = False
while data.time < T_END:
    mujoco.mj_forward(model, data)
    data.ctrl[:] = ctrl.update(data.time)
    mujoco.mj_step(model, data)
    st = ctrl.state
    if st["com"][2] < 0.16 or abs(st["rpy"][0]) > 1.0 or abs(st["rpy"][1]) > 1.0:
        print(f"!! t={st['t']:.2f} 摔倒 (z={st['com'][2]:.3f}, "
              f"rpy={np.degrees(st['rpy'])})", flush=True)
        fallen = True
        break

walk = data.time >= 12.0
ms = np.array(ctrl.mpc.solve_ms) * 1e3
print(f"结果: {'失败' if fallen else ('通过(全程)' if walk else '中途结束')} | "
      f"末态 t={data.time:.1f}s z={ctrl.state['com'][2]:.3f} "
      f"pitch={np.degrees(ctrl.state['rpy'][1]):+.1f}° | "
      f"solve 均值 {ms.mean():.1f}ms", flush=True)
sys.exit(1 if fallen else 0)
