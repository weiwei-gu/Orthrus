"""单刚体 NMPC — 逐次凸化 (successive convexification, SCvx)

与凸版 ConvexMPC 的建模差异 (默认配置):
  1. 欧拉运动学精确化: rpy' = E(φ,θ)·ω_body —— body 系角速度 → ZYX 欧拉率
     (凸版近似 rpy' = Rz(ψ)·ω; 已对 MuJoCo 无接触单步真值数值验证,
      中等姿态下凸版运动学误差是精确映射的 ~64 倍)
  2. 惯量全旋转: I_w = R(φ,θ,ψ)·I_body·Rᵀ  (凸版只旋转 yaw)

SCvx 迭代 (scvx_iters 次, 默认 2 —— 实测每 solve 平均仅接受 ~1 次,
  第 2 轮起基本拒绝=已收敛):
  非线性前向滚动 → ATV 线性化 (仿射时变: x⁺ = A_k x + B_k u + c_k,
  c_k 含重力; 沿自身轨迹线性化时缺陷严格为零, 非线性全被状态依赖矩阵捕获)
  → 沿轨迹堆叠预测 → QP (与凸版同一套权重/约束)
  → 非线性代价改善才接受, 否则 0.5 阻尼回退一次, 再差则终止。

负结果入档 (消融实验 ablate_nmpc.py, 2026-09-26):
  力臂沿预测质心轨迹更新 (lever_traj=True) —— 几何上更真实, 但闭环发散:
  模型相信质心前移使后足力臂在时域内变长 → "未来修正更便宜" → 每次重解
  把俯仰修正推迟到下一结点 (MPC 拖延病理), 俯仰从 0.2° 单调爬到 15°+,
  SCvx 代价全程很低 (模型自洽地满意), t=11.2s 横向爆炸摔倒。
  冻结力臂 (lever_traj=False, 与凸版一致) 后病理消失。
  故默认 lever_traj=False —— 单步决策上更"错"的模型在闭环里更稳。

实现注记 (标准工程折衷, 如实声明):
  - A_k 中省略 ∂E/∂θ 与 ∂I_w/∂θ 二阶项 (0.3s 时域内姿态变化小)
  - 摆动足的力由 QP 边界强制为零; 滚动前再按 stance 掩码清零一次
  - 每次调用内 QP 逐次重建 OSQP (与凸版同款模式); 跨 solve 热启动存 _u_warm

用法: 与 ConvexMPC 完全同接口 (drop-in), 见 run_go2_mpc.py --nmpc。
"""
import time

import numpy as np
import osqp
from scipy import sparse

from convex_mpc import ConvexMPC, GRAVITY, skew

SCVX_ITERS = 2
DAMPING = 0.5          # 代价变差时的回退混合系数
THETA_CLAMP = 1.45     # rad (~83°): 欧拉运动学钳位, 超出即已摔倒


def euler_rates_map(rpy):
    """[φ̇, θ̇, ψ̇]ᵀ = E(φ,θ)·[p,q,r]ᵀ —— body 系角速度 → ZYX 欧拉率

    E(0,0) = I。θ 钳位 ±THETA_CLAMP 防奇异 (cosθ 下限 0.12)。
    """
    phi, theta = float(rpy[0]), float(rpy[1])
    theta = np.clip(theta, -THETA_CLAMP, THETA_CLAMP)
    cphi, sphi = np.cos(phi), np.sin(phi)
    ct = max(np.cos(theta), 0.12)
    tt = np.sin(theta) / ct
    return np.array([
        [1.0, sphi * tt, cphi * tt],
        [0.0, cphi, -sphi],
        [0.0, sphi / ct, cphi / ct],
    ])


def _rotz(yaw):
    """只绕 yaw 的旋转 (凸版惯量约定, 消融开关 inertia_rot=False 时用)"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_zyx(rpy):
    """R = Rz(ψ)·Ry(θ)·Rx(φ) (world ← body)"""
    phi, th, ps = rpy
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(th), np.sin(th)
    cp, sp = np.cos(ps), np.sin(ps)
    return np.array([
        [cp * ct, cp * st * sf - sp, cp * st * cf + sp * sf],
        [sp * ct, sp * st * sf + cp, sp * st * cf - cp * sf],
        [-st,    ct * sf,            ct * cf],
    ])


class NMPC(ConvexMPC):
    """逐次凸化 NMPC (继承凸版的权重/约束模式/边界生成)

    因子开关 (消融用, 定位各建模改动的闭环贡献):
      lever_traj: 力臂沿预测质心轨迹更新 (False = 冻结在当前质心, 与凸版一致)
      inertia_rot: 惯量全旋转 R(φ,θ,ψ)·I·Rᵀ (False = 只旋转 yaw, 与凸版一致)
      E 运动学 (精确欧拉率映射) 恒开 —— 已对 MuJoCo 真值数值验证到 1e-5
    """

    def __init__(self, *args, scvx_iters=SCVX_ITERS, lever_traj=False,
                 inertia_rot=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.scvx_iters = int(scvx_iters)
        self.lever_traj = bool(lever_traj)
        self.inertia_rot = bool(inertia_rot)
        self._u_warm = None
        self._f_ext = np.zeros(3)      # 扰动观测器前馈 (solve 时更新, _f 消费)
        self._tau_ext = np.zeros(3)
        # 统计 (诊断用)
        self.solve_ms = []
        self.accepted = 0
        self.damped = 0
        self.rejected = 0

    # ------------------------------------------------------------------ #
    # 非线性模型一步: x_{k+1} = f(x_k, u_k)  (feet_pos 随周期给定)
    # ------------------------------------------------------------------ #
    def _iw_inv(self, rpy):
        R = rot_zyx(rpy) if self.inertia_rot else _rotz(rpy[2])
        return np.linalg.inv(R @ self.I_body @ R.T)

    def _lever_p(self, x, p_anchor):
        """力臂参考质心: 轨迹预测值 (lever_traj) 或冻结的当前质心"""
        return x[3:6] if self.lever_traj else p_anchor

    def _f(self, x, u, feet_pos, p_anchor):
        rpy, p, w, v = x[0:3], x[3:6], x[6:9], x[9:12]
        E = euler_rates_map(rpy)
        Iw_inv = self._iw_inv(rpy)
        p_ref = self._lever_p(x, p_anchor)
        torque = np.zeros(3)
        fsum = np.zeros(3)
        for i in range(4):
            f = u[3 * i:3 * i + 3]
            torque += np.cross(feet_pos[i] - p_ref, f)
            fsum += f
        xnext = x.copy()
        xnext[0:3] = rpy + self.dt * (E @ w)
        xnext[3:6] = p + self.dt * v
        xnext[6:9] = w + self.dt * (Iw_inv @ (torque + self._tau_ext))
        xnext[9:12] = v + self.dt * (GRAVITY + (fsum + self._f_ext) / self.m)
        return xnext

    def _rollout(self, x0, U, feet_pos):
        X = np.empty((self.N + 1, 12))
        X[0] = x0
        p_anchor = x0[3:6]          # 冻结力臂时的锚 (当前质心)
        for k in range(self.N):
            X[k + 1] = self._f(X[k], U[12 * k:12 * k + 12], feet_pos, p_anchor)
        return X

    # ------------------------------------------------------------------ #
    # ATV 线性化: 沿轨迹的 A_k, B_k, 缺陷偏移 c_k (重力含在 c_k 内)
    # ------------------------------------------------------------------ #
    def _atv(self, Xbar, U, feet_pos):
        p_anchor = Xbar[0][3:6]
        As, Bs, cs = [], [], []
        for k in range(self.N):
            xk, uk = Xbar[k], U[12 * k:12 * k + 12]
            rpy, p = xk[0:3], xk[3:6]
            E = euler_rates_map(rpy)
            Iw_inv = self._iw_inv(rpy)
            p_ref = self._lever_p(xk, p_anchor)
            A = np.eye(12)
            A[0:3, 6:9] = self.dt * E
            A[3:6, 9:12] = self.dt * np.eye(3)
            B = np.zeros((12, 12))
            for i in range(4):
                B[6:9, 3 * i:3 * i + 3] = self.dt * (Iw_inv @ skew(feet_pos[i] - p_ref))
                B[9:12, 3 * i:3 * i + 3] = self.dt / self.m * np.eye(3)
            c = self._f(xk, uk, feet_pos, p_anchor) - A @ xk - B @ uk
            As.append(A)
            Bs.append(B)
            cs.append(c)
        return As, Bs, cs

    # ------------------------------------------------------------------ #
    # 堆叠预测 (变 A): 行 k ↔ 状态 x_{k+1}
    #   S_k = A_k···A_0;  G_{k,j} = (A_k···A_{j+1})·B_j;  d_k = Σ_j (A_k···A_{j+1})·c_j
    # ------------------------------------------------------------------ #
    def _stack(self, As, Bs, cs):
        N = self.N
        S = np.zeros((N * 12, 12))
        G = np.zeros((N * 12, N * 12))
        d = np.zeros(N * 12)
        for k in range(N):
            acc = np.eye(12)
            r0 = 12 * k
            for j in range(k, -1, -1):
                G[r0:r0 + 12, 12 * j:12 * j + 12] = acc @ Bs[j]
                d[r0:r0 + 12] += acc @ cs[j]
                acc = acc @ As[j]
            S[r0:r0 + 12] = acc
        return S, G, d

    # ------------------------------------------------------------------ #
    def solve(self, x0, yaw, com, feet_pos, stance_pred, x_ref,
              f_ext=None, tau_ext=None):
        """接口与 ConvexMPC.solve 完全一致 (yaw/com 被 E(·) 与轨迹力臂取代, 仅保留参数位)。
        f_ext/tau_ext: 扰动观测器估计 (world 系), 进非线性模型 —— 与重力同路径,
        由缺陷偏移机制自动传播进 QP 的 d 项。"""
        t0 = time.perf_counter()
        self._f_ext = np.asarray(f_ext, dtype=float) if f_ext is not None else np.zeros(3)
        self._tau_ext = np.asarray(tau_ext, dtype=float) if tau_ext is not None else np.zeros(3)
        N, nu = self.N, 12
        x0 = np.asarray(x0, dtype=float)
        xr = np.asarray(x_ref, dtype=float).reshape(-1)
        feet_pos = np.asarray(feet_pos, dtype=float)
        l, u = self._bounds(stance_pred)
        mask = np.repeat(stance_pred.astype(float), 3, axis=1).reshape(-1)   # (N*12,)

        # ---- 冷/热启动 ----
        if self._u_warm is None:
            U = np.zeros(N * nu)
            for k in range(N):
                nst = max(1, stance_pred[k].sum())
                for i in range(4):
                    if stance_pred[k, i]:
                        U[12 * k + 3 * i + 2] = -self.m * GRAVITY[2] / nst
        else:
            U = self._u_warm * mask

        def rollout_cost(Uv):
            Xk = self._rollout(x0, Uv, feet_pos)[1:]           # (N,12) ↔ 参考结点
            e = Xk.reshape(-1) - xr
            return float(e @ self.Qbar @ e + Uv @ self.Rbar @ Uv)

        best_U = U.copy()
        best_c = rollout_cost(best_U)
        qp_ok = 0

        for _ in range(self.scvx_iters):
            Xbar = self._rollout(x0, best_U * mask, feet_pos)
            As, Bs, cs = self._atv(Xbar, best_U * mask, feet_pos)
            S, G, d = self._stack(As, Bs, cs)
            H = 2.0 * (G.T @ self.Qbar @ G + self.Rbar)
            g = 2.0 * G.T @ self.Qbar @ (S @ x0 + d - xr)
            P = sparse.csc_matrix(np.triu(H))
            prob = osqp.OSQP()
            prob.setup(P, g, self.A_con, l, u, verbose=False,
                       eps_abs=1e-3, eps_rel=1e-3, max_iter=2000,
                       polishing=False, adaptive_rho=True)
            res = prob.solve()
            if res.x is None:
                self.rejected += 1
                break
            qp_ok += 1
            U_new = np.asarray(res.x, dtype=float) * mask
            c_new = rollout_cost(U_new)
            if c_new < best_c:
                best_U, best_c = U_new.copy(), c_new
                self.accepted += 1
                continue
            # 阻尼回退一次
            U_damp = DAMPING * (best_U * mask) + (1.0 - DAMPING) * U_new
            c_damp = rollout_cost(U_damp)
            if c_damp < best_c:
                best_U, best_c = U_damp.copy(), c_damp
                self.damped += 1
            else:
                self.rejected += 1
                break                             # SCvx 收敛/卡住

        self.solve_ms.append(time.perf_counter() - t0)
        self.solve_count += 1

        if qp_ok == 0:
            # 全部 QP 失败: 退化为静态重力分配 (与凸版同款回退)
            self.last_status = "FALLBACK"
            forces = np.zeros((4, 3))
            n_stance = max(1, stance_pred[0].sum())
            for i in range(4):
                if stance_pred[0, i]:
                    forces[i, 2] = -self.m * GRAVITY[2] / n_stance
            return forces

        self._u_warm = best_U.copy()
        self.last_cost = best_c
        self.last_status = f"SCvx ok={qp_ok}"
        return best_U[:12].reshape(4, 3)
