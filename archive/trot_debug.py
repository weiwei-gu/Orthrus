"""触地瞬间足端速度检测: 验证摆动足是否带着速度砸地"""
import numpy as np
import mujoco

from run_go2_mpc import Go2Controller, SCENE
from go2_utils import FOOT_NAMES

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
foot_r = model.geom_size[foot_geoms[0]][0]
data.qpos[2] += foot_r - foot_z
mujoco.mj_forward(model, data)

ctrl = Go2Controller(model, data)
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

touching = [False] * 4
print("每次触地事件: 足端速度 (world) 和冲击力")
while data.time < 6.5:
    mujoco.mj_forward(model, data)
    data.ctrl[:] = ctrl.update(data.time)

    for i, g in enumerate(foot_geoms):
        touch = data.geom_xpos[g][2] < foot_r + 0.004
        if touch and not touching[i]:
            mujoco.mj_jacGeom(model, data, jacp, jacr, g)
            v = jacp @ data.qvel
            print(f"t={data.time:5.3f} {FOOT_NAMES[i]} 触地: "
                  f"v=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) m/s  "
                  f"|vz|={abs(v[2]):.2f}  (软着陆应 <0.1)")
        touching[i] = touch
    mujoco.mj_step(model, data)

    if ctrl.state["com"][2] < 0.16:
        print("!! 摔倒")
        break
