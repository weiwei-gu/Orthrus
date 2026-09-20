"""组件测试: 模型加载 / IK 精度 / MPC 静态站立解"""
import time
import numpy as np
import mujoco

from go2_utils import Go2IK, quat_to_rpy, FOOT_NAMES
from convex_mpc import ConvexMPC

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/go2_mpc_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
print(f"模型加载 OK: nq={model.nq} nv={model.nv} nu={model.nu} nsensor={model.nsensor}")

foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
leg_qadr = {}
for f in FOOT_NAMES:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{p}_joint")
           for p in ("hip", "thigh", "calf")]
    leg_qadr[f] = [model.jnt_qposadr[j] for j in ids]

# --- IK 测试 1: 目标=当前足位, 应返回当前关节角且误差极小 ---
ik = Go2IK(model)
for i, f in enumerate(FOOT_NAMES):
    q_cur = np.array([data.qpos[a] for a in leg_qadr[f]])
    target = data.geom_xpos[foot_geoms[i]].copy()
    q_sol, err = ik.solve(f, data.qpos[3:7], data.qpos[0:3], target, q_cur)
    print(f"IK[{f}]: pos_err={err:.2e}  q_err={np.abs(q_sol - q_cur).max():.2e}")

# --- IK 测试 2: 随机姿态 FK->IK 往返 ---
rng = np.random.default_rng(1)
maxerr = 0.0
for _ in range(50):
    for i, f in enumerate(FOOT_NAMES):
        q_cur = np.array([data.qpos[a] for a in leg_qadr[f]])
        q_rand = q_cur + rng.uniform(-0.3, 0.3, 3)
        d2 = mujoco.MjData(model)
        d2.qpos[:] = data.qpos
        for a, qv in zip(leg_qadr[f], q_rand):
            d2.qpos[a] = qv
        mujoco.mj_forward(model, d2)
        target = d2.geom_xpos[foot_geoms[i]].copy()
        q_sol, err = ik.solve(f, data.qpos[3:7], data.qpos[0:3], target, q_cur, iters=6)
        maxerr = max(maxerr, np.abs(q_sol - q_rand).max())
print(f"IK 随机往返测试: 最大关节角误差 {maxerr:.2e} rad")

# --- 复合惯量 + MPC 静态站立解 ---
m_tot, com = 0.0, np.zeros(3)
for b in range(1, model.nbody):
    m = model.body_mass[b]
    m_tot += m
    com += m * data.xipos[b]
com /= m_tot
I = np.zeros((3, 3))
for b in range(1, model.nbody):
    m = model.body_mass[b]
    R = data.xmat[b].reshape(3, 3)
    Iw = R @ np.diag(model.body_inertia[b]) @ R.T
    dd = data.xipos[b] - com
    I += Iw + m * (np.dot(dd, dd) * np.eye(3) - np.outer(dd, dd))
print(f"总质量 {m_tot:.3f} kg, 复合惯量对角 {np.round(np.diag(I), 5)}")

mpc = ConvexMPC(m_tot, np.diag(I))
rpy = quat_to_rpy(data.qpos[3:7])
x0 = np.concatenate([rpy, com, np.zeros(3), np.zeros(3)])
fp = np.array([data.geom_xpos[g] for g in foot_geoms])
x_ref = np.tile([0, 0, rpy[2], com[0], com[1], 0.27, 0, 0, 0, 0, 0, 0], (15, 1))
sp = np.ones((15, 4), bool)

t0 = time.perf_counter()
F = mpc.solve(x0, rpy[2], com, fp, sp, x_ref)
dt_ms = (time.perf_counter() - t0) * 1e3
print(f"MPC 站立解: fz 总和 = {F[:, 2].sum():.1f} N  (体重 = {m_tot * 9.81:.1f} N)")
print(f"  每足 f_z: {np.round(F[:, 2], 1)}")
print(f"  水平力最大分量: {np.abs(F[:, :2]).max():.3f} N (应接近 0)")
print(f"  单次求解耗时 {dt_ms:.1f} ms (50Hz 预算 20ms), 状态: {mpc.last_status}")

# --- 求解 20 次看平均耗时 (含矩阵构建) ---
t0 = time.perf_counter()
for _ in range(20):
    mpc.solve(x0, rpy[2], com, fp, sp, x_ref)
print(f"  20 次连续求解平均 {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms/次")
