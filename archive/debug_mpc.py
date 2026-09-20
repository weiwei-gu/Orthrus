"""诊断 MPC 站立解: 检查 QP 是否返回真正的最优解"""
import numpy as np
import mujoco

from go2_utils import quat_to_rpy, FOOT_NAMES
from convex_mpc import ConvexMPC

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/go2_mpc_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]

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
    dd = data.xipos[b] - com
    I += R @ np.diag(model.body_inertia[b]) @ R.T + m * (np.dot(dd, dd) * np.eye(3) - np.outer(dd, dd))
print(f"com_z = {com[2]:.4f} (参考 0.27), 体重 {m_tot * 9.81:.1f} N")

mpc = ConvexMPC(m_tot, np.diag(I))
rpy = quat_to_rpy(data.qpos[3:7])
x0 = np.concatenate([rpy, com, np.zeros(3), np.zeros(3)])
fp = np.array([data.geom_xpos[g] for g in foot_geoms])
N = mpc.N
x_ref = np.tile([0, 0, rpy[2], com[0], com[1], 0.27, 0, 0, 0, 0, 0, 0], (N, 1))
sp = np.ones((N, 4), bool)

F = mpc.solve(x0, rpy[2], com, fp, sp, x_ref)
print(f"MPC 返回力 f_z 总和: {F[:, 2].sum():.1f} N")

dbg = mpc.dbg
S, G, d, H, g = dbg["S"], dbg["G"], dbg["d"], dbg["H"], dbg["g"]
Qbar, Rbar = mpc.Qbar, mpc.Rbar


def cost(u):
    X = G @ u + S @ dbg["x0"] + d
    e = X - dbg["x_ref"].reshape(-1)
    return float(e @ Qbar @ e + u @ Rbar @ u), X


def feasibility(u):
    A_con = mpc.A_con.toarray()
    viol = A_con @ u
    return int(((viol < dbg["l"] - 1e-6) | (viol > dbg["u"] + 1e-6)).sum())


# 无约束最优 (若可行即真解)
u_unc = -np.linalg.solve(H, g)
c_unc, X_unc = cost(u_unc)
z_unc = X_unc.reshape(N, 12)[:, 5]
print(f"\n无约束最优: 代价 {c_unc:.4f}, 违反约束 {feasibility(u_unc)} 行, "
      f"fz 首段总和 {u_unc[:12].reshape(4, 3)[:, 2].sum():.1f} N")
print(f"  预测 z: {z_unc[0]:.4f} -> {z_unc[-1]:.4f} (参考 0.27)")
print(f"  H 特征值范围: [{np.linalg.eigvalsh(H).min():.2e}, {np.linalg.eigvalsh(H).max():.2e}]")

u_sol = np.tile(F.reshape(-1), N)
c_sol, X_sol = cost(u_sol)
print(f"OSQP 返回解: 代价 {c_sol:.4f} (差 {c_sol - c_unc:.4f}), 违反 {feasibility(u_sol)} 行")

# ---- OSQP 收紧精度重试 ----
import osqp
from scipy import sparse

l, u = mpc._bounds(sp)
P = sparse.csc_matrix(np.triu(H))
for tag, kw in [("默认", {}),
                ("polish+eps1e-6", dict(polishing=True, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000))]:
    prob = osqp.OSQP()
    prob.setup(P, g, mpc.A_con, l, u, verbose=False, **kw)
    res = prob.solve()
    if res.x is None:
        print(f"\nOSQP[{tag}]: 失败 {res.info.status}")
        continue
    uu = res.x
    cc, _ = cost(uu)
    print(f"\nOSQP[{tag}]: 状态={res.info.status}, 迭代={res.info.iter}, "
          f"代价={cc:.4f} (最优 {c_unc:.4f}), fz总和={uu[:12].reshape(4, 3)[:, 2].sum():.1f} N")
