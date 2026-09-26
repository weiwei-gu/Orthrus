"""单刚体 convex MPC — Di Carlo et al. (ICRA 2018) 风格实现

状态 x = [rpy, p, ω, v] (12 维):
  rpy: 机身欧拉角 (world)      p: 质心位置 (world)
  ω:   角速度 (body 系)         v: 质心线速度 (world)

单刚体动力学 (yaw 线性化 / "凸化"):
  rpy' = Rz(yaw) · ω
  p'   = v
  ω'   = I⁻¹ Σ (r_i × f_i)          r_i = 足 i 相对质心的位置
  v'   = g + (1/m) Σ f_i

决策变量: N 个 knot 上 4 足 × 3D 足端力 (world 系, 共 12N 维)
约束: 单边接触 + 摩擦金字塔 + 力上限; 步态预测中处于摆动相的足力为零
求解: OSQP
"""
import numpy as np
import osqp
from scipy import sparse

GRAVITY = np.array([0.0, 0.0, -9.81])


def skew(v):
    """反对称矩阵, skew(v) @ f = v × f"""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


class ConvexMPC:
    def __init__(self, mass, inertia_diag, mu=0.6, f_max=80.0,
                 horizon=15, dt=0.02,
                 q_rpy=(60.0, 60.0, 15.0), q_pos=(2.0, 2.0, 100.0),
                 q_omega=(2.0, 2.0, 0.5), q_vel=(3.0, 3.0, 6.0),
                 r_force=2e-5, terminal_mult=3.0):
        self.m = float(mass)
        self.I_body = np.diag(np.asarray(inertia_diag, dtype=float))
        self.mu = float(mu)
        self.f_max = float(f_max)
        self.N = int(horizon)   # knot 数
        self.dt = float(dt)     # knot 间隔

        # 逐 knot 状态权重 (最后一个 knot 放大, 近似终端代价)
        Q = np.diag(np.concatenate([q_rpy, q_pos, q_omega, q_vel]).astype(float))
        self.Qbar = np.zeros((self.N * 12, self.N * 12))
        for k in range(self.N):
            mult = terminal_mult if k == self.N - 1 else 1.0
            self.Qbar[12 * k:12 * (k + 1), 12 * k:12 * (k + 1)] = mult * Q
        self.Rbar = np.eye(self.N * 12) * r_force

        self._setup_constraints_pattern()
        self.solve_count = 0
        self.last_status = None

    # ------------------------------------------------------------------ #
    # 约束矩阵模式固定 (9 行/足/knot), 只切换上下界:
    #   前 3 行: 单位阵 —— 摆动足 l=u=0 (力为零), 支撑足 l=-inf u=+inf (不激活)
    #   后 6 行: 摩擦金字塔 + 单边 + 上限 —— 支撑足激活, 摆动足放开
    # ------------------------------------------------------------------ #
    def _setup_constraints_pattern(self):
        N, nu = self.N, 12
        rows, cols, vals = [], [], []
        self.eq_row_of = np.zeros((N, 4), dtype=int)    # 每足 eq 块首行
        self.inq_row_of = np.zeros((N, 4), dtype=int)
        for k in range(N):
            for i in range(4):
                r0 = (k * 4 + i) * 9
                c0 = k * 12 + 3 * i
                self.eq_row_of[k, i] = r0
                self.inq_row_of[k, i] = r0 + 3
                # 3 行等式 (I3)
                for a in range(3):
                    rows.append(r0 + a)
                    cols.append(c0 + a)
                    vals.append(1.0)
                # 6 行不等式
                ineq = [
                    (1.0, 0.0, -self.mu),   #  fx - μ fz ≤ 0
                    (-1.0, 0.0, -self.mu),  # -fx - μ fz ≤ 0
                    (0.0, 1.0, -self.mu),   #  fy - μ fz ≤ 0
                    (0.0, -1.0, -self.mu),  # -fy - μ fz ≤ 0
                    (0.0, 0.0, -1.0),       # -fz ≤ 0  (单边)
                    (0.0, 0.0, 1.0),        #  fz ≤ f_max
                ]
                for a, w in enumerate(ineq):
                    for b in range(3):
                        if w[b] != 0.0:
                            rows.append(r0 + 3 + a)
                            cols.append(c0 + b)
                            vals.append(w[b])
        self.A_con = sparse.csc_matrix(
            (vals, (rows, cols)), shape=(N * 36, N * nu))
        self.n_rows = N * 36

    def _bounds(self, stance_pred):
        """由步态预测生成上下界"""
        N = self.N
        big = 1e6  # OSQP 不接受 ±inf 混合? 用大数代替"不激活"
        l = np.full(self.n_rows, -big)
        u = np.full(self.n_rows, big)
        for k in range(N):
            for i in range(4):
                r0 = self.eq_row_of[k, i]
                q0 = self.inq_row_of[k, i]
                if stance_pred[k, i]:  # 支撑足
                    # eq 行不激活
                    # 摩擦/单边/上限约束激活
                    u[q0 + 0] = 0.0
                    u[q0 + 1] = 0.0
                    u[q0 + 2] = 0.0
                    u[q0 + 3] = 0.0
                    u[q0 + 4] = 0.0
                    u[q0 + 5] = self.f_max
                else:                  # 摆动足: 力强制为零
                    l[r0:r0 + 3] = 0.0
                    u[r0:r0 + 3] = 0.0
        return l, u

    # ------------------------------------------------------------------ #
    def solve(self, x0, yaw, com, feet_pos, stance_pred, x_ref,
              f_ext=None, tau_ext=None):
        """求当前时刻的足端力。

        参数:
          x0:          (12,) 当前状态 [rpy, p, ω, v]
          yaw:         当前偏航角 (线性化点)
          com:         (3,)  质心 world —— 用于 r_i = feet_pos[i] - com
          feet_pos:    (4,3) 足端位置 world (支撑足=当前位置, 摆动足=预计落点)
          stance_pred: (N,4) bool, knot k 时刻足 i 是否支撑
          x_ref:       (N,12) 参考状态轨迹
          f_ext/tau_ext: (3,) 可选, 扰动观测器估计的外力/外力矩 (world 系),
                        作为已知常值输入进入预测 (与接触力同路径, ω 行经 I_w⁻¹)
        返回:
          forces: (4,3) 第一段足端力 (world 系)
        """
        N, dt = self.N, self.dt

        # ---- 离散动力学 (yaw 线性化) ----
        A = np.eye(12)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        A[0:3, 6:9] = dt * Rz          # rpy' = Rz(yaw) ω
        A[3:6, 9:12] = dt * np.eye(3)  # p' = v

        I_w = Rz @ self.I_body @ Rz.T
        I_w_inv = np.linalg.inv(I_w)

        B = np.zeros((N, 12, 12))
        for k in range(N):
            for i in range(4):
                r = np.asarray(feet_pos[i]) - np.asarray(com)
                B[k][6:9, 3 * i:3 * i + 3] = dt * (I_w_inv @ skew(r))
                B[k][9:12, 3 * i:3 * i + 3] = dt / self.m * np.eye(3)

        cvec = np.zeros(12)
        cvec[9:12] = dt * GRAVITY
        # 扰动观测器前馈: 已知外力/外力矩作为常值输入 (与接触力同路径)
        if f_ext is not None:
            cvec[9:12] += dt * np.asarray(f_ext, dtype=float) / self.m
        if tau_ext is not None:
            cvec[6:9] += dt * (I_w_inv @ np.asarray(tau_ext, dtype=float))

        # ---- 预测矩阵: X = [x_1;...;x_N] = S x0 + G u + d ----
        Apow = [np.eye(12)]
        for _ in range(N):
            Apow.append(Apow[-1] @ A)
        S = np.zeros((N * 12, 12))
        G = np.zeros((N * 12, N * 12))
        d = np.zeros(N * 12)
        for k in range(N):
            # 行块 k 对应状态 x_{k+1}
            S[12 * k:12 * (k + 1)] = Apow[k + 1]
            for j in range(k + 1):
                G[12 * k:12 * (k + 1), 12 * j:12 * (j + 1)] = Apow[k - j] @ B[j]
            for j in range(k + 1):
                d[12 * k:12 * (k + 1)] += Apow[k - j] @ cvec

        # ---- QP: min 0.5 u'Pu + q'u ----
        pred_const = S @ x0 + d - x_ref.reshape(-1)
        H = 2.0 * (G.T @ self.Qbar @ G + self.Rbar)
        g = 2.0 * G.T @ self.Qbar @ pred_const
        l, u = self._bounds(stance_pred)

        # 调试钩子: 保存预测矩阵, 便于外部验证
        self.dbg = {"S": S, "G": G, "d": d, "H": H, "g": g, "l": l, "u": u,
                    "x0": x0.copy(), "x_ref": x_ref.copy(), "com": np.asarray(com).copy()}

        P = sparse.csc_matrix(np.triu(H))
        prob = osqp.OSQP()
        prob.setup(P, g, self.A_con, l, u, verbose=False,
                   eps_abs=1e-3, eps_rel=1e-3, max_iter=2000,
                   polishing=False, adaptive_rho=True)
        res = prob.solve()
        self.solve_count += 1

        forces = np.zeros((4, 3))
        if res.x is not None:
            self.last_status = res.info.status
            forces = res.x[:12].reshape(4, 3)
        else:
            # 求解失败: 退化为静态重力分配 (支撑足平摊体重)
            self.last_status = "FALLBACK"
            n_stance = max(1, stance_pred[0].sum())
            for i in range(4):
                if stance_pred[0, i]:
                    forces[i, 2] = -self.m * GRAVITY[2] / n_stance
        return forces
