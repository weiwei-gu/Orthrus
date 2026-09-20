"""用 MPC 控制器采集步态示范 (obs -> 关节目标), 供行为克隆引导 RL

示范来源: run_go2_mpc.Go2Controller 在 MuJoCo 中行走 (站立/trot/前进),
以 50 Hz 记录 RL 环境同款 45 维观察 + 动作 = (q_ref - 默认)/0.25。
q_ref 为 MPC 的摆动轨迹 IK / 支撑钉地 IK 关节目标 —— 恰是位置执行器
兼容的运动学步态, 克隆后可直接被位置执行器复现。
"""
import pickle

import numpy as np
import mujoco

from run_go2_mpc import Go2Controller, SCENE, FOOT_NAMES, DT
from eval_go2_rl import quat_to_grav, DEFAULT_Q

ACTION_SCALE = 0.4
OBS_NOISE = 0.02


def collect(n_episodes=12, t_end=22.0):
    model = mujoco.MjModel.from_xml_path(SCENE)
    rng = np.random.default_rng(0)
    obs_all, act_all, ret_all = [], [], []

    for ep in range(n_episodes):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
        foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
        data.qpos[2] += model.geom_size[foot_geoms[0]][0] - foot_z
        mujoco.mj_forward(model, data)

        ctrl = Go2Controller(model, data)
        # 每回合随机速度曲线: 站立 -> 随机前进速度 (含加减速段)
        v1 = rng.uniform(0.2, 0.6)
        t_switch = rng.uniform(6.0, 10.0)
        v2 = rng.uniform(0.0, 0.5)
        stand_T = rng.uniform(2.0, 4.0)

        def cmd(t):
            if t < stand_T:
                return np.zeros(2)
            return np.array([v1 if t < t_switch else v2, 0.0])

        ctrl.get_command = cmd
        ctrl.stand_until = stand_T

        prev_action = np.zeros(12)
        q_ref_all = DEFAULT_Q.copy()
        ep_obs, ep_act, ep_ret = [], [], []
        # 拦截 IK 解: 每步记录各腿 q_ref (摆动=轨迹 IK, 支撑=钉地 IK)
        ik_solve = ctrl.ik.solve
        ik_calls = {}

        def wrapped_solve(foot, base_quat, base_pos, target, q_init, iters=3):
            q, err = ik_solve(foot, base_quat, base_pos, target, q_init, iters=iters)
            ik_calls[foot] = q
            return q, err

        ctrl.ik.solve = wrapped_solve
        fell = False

        while data.time < t_end:
            ik_calls.clear()
            mujoco.mj_forward(model, data)
            # 步态相位锁定到自由时钟 (部署时钟从 t=0 自由跑, 与 gait 相位
            # 确定对应 —— 否则随机偏移让网络无法建立 时钟->摆腿 的映射)
            ctrl.gait.phase = (data.time / 0.6) % 1.0
            data.ctrl[:] = ctrl.update(data.time)
            # update 内部已调用 ik.solve -> ik_calls 记录了本步各腿 q_ref
            for f, q in ik_calls.items():
                i = FOOT_NAMES.index(f)
                q_ref_all[3 * i:3 * i + 3] = q

            if ctrl.step % 10 == 0:   # 50 Hz 采样
                c = cmd(data.time)
                # 时钟用 MPC 的真实步态相位 (部署时自由运行, 相位偏移任意)
                phase = ctrl.gait.phase if ctrl.gait_enabled else 0.0
                clock = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
                obs = np.concatenate([
                    clock,
                    quat_to_grav(data.qpos[3:7]),
                    [c[0], c[1], 0.0],
                    data.qvel[3:6],
                    data.qpos[7:19] - DEFAULT_Q,
                    data.qvel[6:18],
                    prev_action,
                ])
                # 动作钳位到 ±1 (tanh 分布的可达域; 超出部分策略永远无法输出)
                action = np.clip((q_ref_all - DEFAULT_Q) / ACTION_SCALE, -1.0, 1.0)
                # 与训练环境一致的即时奖励 (供价值网络预训练算回报)
                q4 = data.qpos[3:7]
                w, x, y, z = q4
                vw = data.qvel[0:3]
                vbx = ((1 - 2 * (y * y + z * z)) * vw[0] + 2 * (x * y - w * z) * vw[1]
                       + 2 * (x * z + w * y) * vw[2])
                vby = (2 * (x * y + w * z) * vw[0] + (1 - 2 * (x * x + z * z)) * vw[1]
                       + 2 * (y * z - w * x) * vw[2])
                g = quat_to_grav(q4)
                z_h = data.qpos[2]
                qd = data.qvel[6:18]
                c = cmd(data.time)
                rew = (2.5 * np.exp(-((vbx - c[0]) ** 2) / 0.12)
                       + 0.5 * np.exp(-((vby - c[1]) ** 2) / 0.12)
                       + 0.5 * np.clip(1.0 - abs(vbx - c[0]), 0, 1)
                       + 0.25 * np.exp(-((data.qvel[5] - 0.0) ** 2) / 0.25)
                       + 0.35 * np.exp(-100.0 * (z_h - 0.28) ** 2)
                       + 0.5 * np.exp(-10.0 * (g[0] ** 2 + g[1] ** 2))
                       + 0.2
                       - 0.01 * np.sum((action - prev_action) ** 2)
                       - 2e-4 * np.sum(qd ** 2)
                       - 0.02 * np.sum((data.qpos[7:19] - DEFAULT_Q) ** 2))
                ep_obs.append(obs)
                ep_act.append(action)
                ep_ret.append(rew)
                prev_action = action.copy()

            mujoco.mj_step(model, data)
            if ctrl.state["com"][2] < 0.16:
                fell = True
                break

        # 回报 (return-to-go), γ=0.97 与 PPO discounting 一致; 摔倒回合末尾
        # 追加终止罚, 让价值网学到"接近失稳 = 低价值"
        rets = ep_ret + ([-5.0] if fell else [])
        G = 0.0
        to_go = []
        for r in reversed(rets):
            G = r + 0.97 * G
            to_go.append(G)
        to_go.reverse()
        n_keep = len(ep_ret)
        obs_all.extend(ep_obs)
        act_all.extend(ep_act)
        ret_all.extend(to_go[:n_keep])

        print(f"回合 {ep}: {len(obs_all)} 样本累计, {'摔倒!' if fell else '完成'}"
              f" (v1={v1:.2f} v2={v2:.2f})")

    return np.array(obs_all), np.array(act_all), np.array(ret_all)


if __name__ == "__main__":
    obs, act, ret = collect()
    # 剔除姿态偏大 (>17°) 的失稳段样本 (obs 布局: 时钟2 + 重力3 + ...)
    tilt_ok = np.linalg.norm(obs[:, 2:4], axis=1) < 0.3
    obs, act, ret = obs[tilt_ok], act[tilt_ok], ret[tilt_ok]
    # 加与训练同款的观察噪声
    obs = obs + np.random.default_rng(1).normal(0, OBS_NOISE, obs.shape)
    with open("policies/bc_demos.pkl", "wb") as f:
        pickle.dump({"obs": obs, "act": act, "ret": ret}, f)
    print(f"\n共 {len(obs)} 个示范 (过滤后), 已存 policies/bc_demos.pkl")
    print(f"回报范围 [{ret.min():.1f}, {ret.max():.1f}], 均值 {ret.mean():.1f}")
    print(f"obs {obs.shape}, act {act.shape}, act 范围 [{act.min():.2f}, {act.max():.2f}]")
