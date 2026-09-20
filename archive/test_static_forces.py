"""剥离测试: 不用 MPC, 直接 Jᵀ·(0,0,mg/4) 静态支撑力, 看能否站稳"""
import numpy as np
import mujoco

from go2_utils import FOOT_NAMES, quat_to_rpy

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/go2_mpc_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
foot_r = model.geom_size[foot_geoms[0]][0]
data.qpos[2] += foot_r - foot_z
mujoco.mj_forward(model, data)

# 执行器 -> 关节映射
act_joint = model.actuator_trnid[:, 0]
leg = {}
for f in FOOT_NAMES:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{p}_joint")
           for p in ("hip", "thigh", "calf")]
    leg[f] = {
        "vadr": [model.jnt_dofadr[i] for i in ids],
        "aadr": [int(np.where(act_joint == j)[0][0]) for j in ids],
    }

m_tot = sum(model.body_mass[b] for b in range(1, model.nbody))
f_static = np.array([0.0, 0.0, m_tot * 9.81 / 4])

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
tau = np.zeros(model.nu)

print(f"静态支撑测试: 每足 f={f_static[2]:.1f} N, 总 {f_static[2]*4:.1f} N (体重 {m_tot*9.81:.1f})")
print(f"初始 com_z = {data.sensordata[1]:.4f}")

for step in range(1000):   # 2 秒
    mujoco.mj_forward(model, data)
    for i, f in enumerate(FOOT_NAMES):
        mujoco.mj_jacGeom(model, data, jacp, jacr, foot_geoms[i])
        J = jacp[:, leg[f]["vadr"]]
        t3 = -(J.T @ f_static)   # 压足方向 (符号与主控制器一致)
        for a in range(3):
            tau[leg[f]["aadr"][a]] = t3[a]
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

    if step % 100 == 0:
        rpy = quat_to_rpy(data.qpos[3:7])
        # 接触法向力
        fn = np.zeros(4)
        ncon = data.ncon
        for c in range(ncon):
            con = data.contact[c]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, f6)
            for i, g in enumerate(foot_geoms):
                if con.geom1 == g or con.geom2 == g:
                    fn[i] = max(fn[i], f6[0])
        print(f"t={data.time:.2f}  z={data.sensordata[2]:.4f}  pitch={np.degrees(rpy[1]):+7.2f}°  "
              f"接触法向力 {np.round(fn, 1)}  ctrl_max={np.abs(tau).max():.1f}")

rpy = quat_to_rpy(data.qpos[3:7])
print(f"\n最终: z={data.sensordata[2]:.4f}, roll={np.degrees(rpy[0]):+.1f}°, pitch={np.degrees(rpy[1]):+.1f}°")
print("结论:", "站稳 ✓" if data.sensordata[2] > 0.2 and abs(rpy[1]) < 0.3 else "失败 ✗")
