"""最小接触测试: 无控制纯物理, 从 2cm 高处落下, 看哪些几何体产生接触"""
import numpy as np
import mujoco

from go2_utils import FOOT_NAMES

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/go2_mpc_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
foot_r = model.geom_size[foot_geoms[0]][0]
data.qpos[2] += foot_r - foot_z + 0.02   # 抬高 2cm 自由落下
mujoco.mj_forward(model, data)
data.ctrl[:] = 0.0

print(f"foot 半径 {foot_r}, 初始足心高度 {foot_z:.4f} -> 调整后抬起 2cm")
print(f"floor: contype={model.geom_contype[0]} conaffinity={model.geom_conaffinity[0]} "
      f"margin={model.geom_margin[0]:.4f}")
print(f"foot : contype={model.geom_contype[foot_geoms[0]]} "
      f"conaffinity={model.geom_conaffinity[foot_geoms[0]]} "
      f"margin={model.geom_margin[foot_geoms[0]]:.4f} condim={model.geom_condim[foot_geoms[0]]}")

for step in range(400):   # 0.8s
    mujoco.mj_step(model, data)
    if step % 50 == 0 or step == 10:
        pairs = {}
        for c in range(data.ncon):
            con = data.contact[c]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            pairs[f"{n1}|{n2}"] = round(float(con.dist), 5)
        fz = [round(float(data.geom_xpos[g][2]), 4) for g in foot_geoms]
        print(f"t={data.time:.2f} ncon={data.ncon} com_z={data.sensordata[2]:.3f} "
              f"足心z={fz} 接触={pairs}")
