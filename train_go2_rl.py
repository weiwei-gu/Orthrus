"""Go2 RL 行走策略训练 — MJX + Brax PPO (CPU 即可)

配方 (对标 legged_gym / unitree_rl_gym):
  观察 (47): 步态时钟(2) + 重力方向(3) + 速度指令(3) + 机身角速度(3) + 关节角偏差(12)
            + 关节角速度(12) + 上一动作(12)   —— 全本体感知, 无全局真值
  动作 (12): ctrl = 默认姿态 + 0.4·clip(a, ±1.5)    (位置执行器 kp=50/kd=0.5)
  奖励: 速度跟踪 + 高度 + 姿态 + 存活 − 动作/关节惩罚 − 终止罚
  域随机化: 初始状态 + 观察噪声 + 速度冲量推力 (0~2 m/s ≈ 0~30 N·s, 核心!)

用法:
  .venv/bin/python train_go2_rl.py --smoke     # 快速验证训练管线
  .venv/bin/python train_go2_rl.py             # 完整训练
"""
import argparse
import functools
import json
import pickle
import time
from datetime import datetime

import jax
import jax.numpy as jp
import mujoco
from brax import envs
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

SCENE = "models/go2/scene_mjx.xml"

# ---- 环境/奖励超参 ----
N_FRAMES = 4             # 0.005 × 4 = 0.02 s 策略周期 (50 Hz)
ACTION_SCALE = 0.4
CLOCK_T = 0.6            # 步态时钟周期 (与 MPC 演示的步态周期一致)
CMD_X = (-0.3, 1.0)
CMD_Y = (-0.3, 0.3)
CMD_W = (-0.8, 0.8)
PUSH_VEL_MAX = 2.0       # Δv 上限 ≈ 30 N·s (与 MPC 测试同标度)
PUSH_EVERY = (100, 250)  # 每 2~5 s 一次推力
OBS_NOISE = 0.02
Z_TARGET = 0.28
Z_MIN, Z_MAX = 0.16, 0.45
GRAVITY_DONE = 0.5       # 上方向量 z 分量 < 0.5 (倾斜 > 60°) 终止

DEFAULT_QPOS = jp.array([0.0, 0.0, 0.27, 1, 0, 0, 0] + [0.0, 0.9, -1.8] * 4)
DEFAULT_JOINTS = DEFAULT_QPOS[7:19]


def quat_to_grav(q):
    """四元数 (w,x,y,z) -> 重力方向在机体系投影 Rᵀ·[0,0,1]"""
    w, x, y, z = q
    return jp.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class Go2Env(PipelineEnv):
    """Go2 行走环境 (50 Hz 决策, 含速度冲量推力随机化)"""

    def __init__(self, push_max=PUSH_VEL_MAX, push_every=PUSH_EVERY,
                 cmd_x=CMD_X, **kwargs):
        mj_model = mujoco.MjModel.from_xml_path(SCENE)
        sys = mjcf.load_model(mj_model)
        self._push_max = push_max
        self._push_every = push_every
        self._cmd_x = cmd_x
        kwargs.setdefault("n_frames", N_FRAMES)
        kwargs.setdefault("backend", "mjx")
        super().__init__(sys=sys, **kwargs)

    # ------------------------------------------------------------------ #
    def _obs(self, ps, cmd, prev_action, phase):
        """47 维: 步态时钟(2) + 重力(3) + 指令(3) + 角速度(3) + 关节(12+12) + 上一动作(12)

        时钟是关键: 周期性步态目标若无相位信号, MSE 最优解是相位平均
        (静止姿态) —— 行为克隆必须有时钟才能复现步态极限环。
        """
        clock = jp.array([jp.sin(2 * jp.pi * phase), jp.cos(2 * jp.pi * phase)])
        grav = quat_to_grav(ps.qpos[3:7])
        q = ps.qpos[7:19] - DEFAULT_JOINTS
        qd = ps.qvel[6:18]
        return jp.concatenate([clock, grav, cmd, ps.qvel[3:6], q, qd, prev_action])

    def reset(self, rng):
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        qpos = DEFAULT_QPOS + jp.concatenate([
            jax.random.uniform(rng1, (2,), minval=-0.1, maxval=0.1),
            jax.random.uniform(rng2, (1,), minval=-0.01, maxval=0.05),
            jax.random.normal(rng1, (4,)) * 0.02,            # 姿态扰动
            jax.random.normal(rng2, (12,)) * 0.15,           # 关节扰动
        ])
        qvel = jp.concatenate([
            jax.random.normal(rng3, (6,)) * 0.2,
            jax.random.normal(rng3, (12,)) * 0.3,
        ])
        ps = self.pipeline_init(qpos, qvel)

        cmd = jp.array([
            jax.random.uniform(rng3, (), minval=self._cmd_x[0], maxval=self._cmd_x[1]),
            jax.random.uniform(rng3, (), minval=CMD_Y[0], maxval=CMD_Y[1]),
            jax.random.uniform(rng3, (), minval=CMD_W[0], maxval=CMD_W[1]),
        ])
        info = {"rng": rng3, "steps": jp.zeros(()),
                "next_push": jax.random.uniform(rng3, (),
                                                minval=self._push_every[0],
                                                maxval=self._push_every[1]),
                "cmd": cmd, "prev_action": jp.zeros(12),
                "phase": jax.random.uniform(rng3, ())}   # 时钟随机初相
        obs = self._obs(ps, cmd, jp.zeros(12), info["phase"])
        reward, done = jp.zeros(2)
        # metrics 键必须在 reset 声明且与 step 完全一致:
        # EpisodeWrapper 会快照键集做回合聚合, 结构漂移会破坏 scan
        zero = jp.zeros(())
        metrics = {"vel": zero, "height": zero, "done": zero}
        return State(ps, obs, reward, done, metrics, info=info)

    # ------------------------------------------------------------------ #
    def step(self, state, action):
        action = jp.clip(action, -1.5, 1.5)
        # 速度冲量推力 (域随机化; 强度/频率由课程阶段决定)
        rng, rng_push, rng_sched, rng_obs = jax.random.split(state.info["rng"], 4)
        do_push = state.info["steps"] >= state.info["next_push"]
        dv = jax.random.uniform(rng_push, (2,),
                                minval=-self._push_max, maxval=self._push_max)
        ps0 = state.pipeline_state
        qvel = ps0.qvel.at[:2].add(jp.where(do_push, dv, jp.zeros(2)))
        ps0 = ps0.replace(qvel=qvel)
        next_push = jp.where(
            do_push,
            state.info["steps"] + jax.random.uniform(rng_sched, (),
                                                      minval=self._push_every[0],
                                                      maxval=self._push_every[1]),
            state.info["next_push"])

        ctrl = DEFAULT_JOINTS + ACTION_SCALE * action
        ps = self.pipeline_step(ps0, ctrl)

        cmd = state.info["cmd"]
        quat = ps.qpos[3:7]
        grav = quat_to_grav(quat)
        # 机体系线速度 (奖励用): v_body = Rᵀ v_world
        w, x, y, z = quat
        v_world = ps.qvel[:3]
        v_body_x = ((1 - 2 * (y * y + z * z)) * v_world[0]
                    + 2 * (x * y - w * z) * v_world[1]
                    + 2 * (x * z + w * y) * v_world[2])
        v_body_y = (2 * (x * y + w * z) * v_world[0]
                    + (1 - 2 * (x * x + z * z)) * v_world[1]
                    + 2 * (y * z - w * x) * v_world[2])
        z_h = ps.qpos[2]
        qd = ps.qvel[6:18]
        prev_a = state.info["prev_action"]

        # ---- 奖励 ----
        # 速度: 高斯核 + 线性项 (远处的稠密梯度), 权重 2.5 —— "站着不动"
        # 在 cx>=0.4 的指令区间收益趋零, 逼策略发现行走
        r_vel = (2.5 * jp.exp(-((v_body_x - cmd[0]) ** 2) / 0.12)
                 + 0.5 * jp.exp(-((v_body_y - cmd[1]) ** 2) / 0.12)
                 + 0.5 * jp.clip(1.0 - jp.abs(v_body_x - cmd[0]), 0.0, 1.0))
        r_ang = 0.25 * jp.exp(-((ps.qvel[5] - cmd[2]) ** 2) / 0.25)
        r_z = 0.35 * jp.exp(-100.0 * (z_h - Z_TARGET) ** 2)
        r_att = 0.5 * jp.exp(-10.0 * (grav[0] ** 2 + grav[1] ** 2))
        c_action = -0.01 * jp.sum((action - prev_a) ** 2)
        c_qvel = -2e-4 * jp.sum(qd ** 2)
        c_qdev = -0.02 * jp.sum((ps.qpos[7:19] - DEFAULT_JOINTS) ** 2)
        reward = r_vel + r_ang + r_z + r_att + 0.2 + c_action + c_qvel + c_qdev

        # done 用 float (与 reset 的 jp.zeros 一致, scan 要求 dtype 稳定)
        # 注意 grav 是"上方向量" Rᵀ·[0,0,1]: 直立时 z 分量 ≈ +1
        done = jp.where((z_h < Z_MIN) | (z_h > Z_MAX) | (grav[2] < GRAVITY_DONE),
                        1.0, 0.0)
        reward = reward + jp.where(done > 0, -5.0, 0.0)

        obs = self._obs(ps, cmd, action, state.info["phase"])
        obs = obs + jax.random.normal(rng_obs, obs.shape) * OBS_NOISE

        # 保留 wrapper 注入的 info/metrics 键 (EpisodeWrapper 的回合追踪),
        # 只更新自己的键 —— 整体替换会破坏 scan 的 pytree 结构一致性
        state.info.update(rng=rng, steps=state.info["steps"] + 1,
                           next_push=next_push, cmd=cmd, prev_action=action,
                           phase=(state.info["phase"] + self.dt / CLOCK_T) % 1.0)
        state.metrics.update(vel=r_vel, height=z_h, done=done)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done)


# ---------------------------------------------------------------------- #
def network_factory(obs_size, action_size, preprocess_observations_fn):
    return ppo_networks.make_ppo_networks(
        obs_size, action_size, preprocess_observations_fn,
        policy_hidden_layer_sizes=(128, 128),
        value_hidden_layer_sizes=(256, 256))


# ---- 推力课程: 一上来就用目标难度会让策略陷进"蹲下保命"局部最优
# (实测: 速度奖励低于原地站立基线). 先学走, 再逐步加推力.
STAGES = {
    1: dict(steps=12e6, push_max=0.0, push_every=(10**9, 10**9 + 1),
            cmd_x=(0.4, 1.0)),                                          # 逼出行走
    2: dict(steps=15e6, push_max=0.0, push_every=(10**9, 10**9 + 1),
            cmd_x=(-0.3, 1.0)),                                         # 全指令域
    3: dict(steps=8e6, push_max=0.9, push_every=(250, 500),
            cmd_x=(-0.3, 1.0)),                                         # 中度推力
    4: dict(steps=8e6, push_max=1.8, push_every=(150, 350),
            cmd_x=(-0.3, 1.0)),                                         # 目标难度
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="快速管线验证")
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--entropy", type=float, default=None, help="覆盖 entropy_cost")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 learning_rate")
    parser.add_argument("--steps", type=float, default=None, help="覆盖阶段步数")
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--resume", default=None, help="上一阶段的参数 pkl")
    args = parser.parse_args()

    cfg = STAGES[args.stage]
    num_timesteps = int(200e3 if args.smoke else (args.steps or cfg["steps"]))
    num_envs = 64 if args.smoke else args.envs

    restore_params = None
    if args.resume:
        with open(args.resume, "rb") as f:
            restore_params = pickle.load(f)
        print(f"从 {args.resume} 续训")

    print(f"JAX 设备: {jax.devices()} | 阶段 {args.stage} "
          f"(push_max={cfg['push_max']}, cmd_x={cfg['cmd_x']}) | "
          f"步数 {num_timesteps:,} | 并行环境 {num_envs}")
    env = Go2Env(push_max=cfg["push_max"], push_every=cfg["push_every"],
                 cmd_x=cfg["cmd_x"])

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=10 if args.smoke else 10,
        episode_length=1000,
        normalize_observations=True,
        unroll_length=20,          # 每次 update 的数据量 ×4 (原 5 导致梯度噪声大)
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=args.lr if args.lr is not None else 2e-4,
        entropy_cost=args.entropy if args.entropy is not None else 2e-3,
        num_envs=num_envs,
        batch_size=num_envs,
        network_factory=network_factory,
        eval_env=env,
        num_eval_envs=16,
        deterministic_eval=True,
        restore_params=restore_params,
        seed=0,
    )

    t0 = time.perf_counter()
    log = []

    def progress(num_steps, metrics):
        elapsed = time.perf_counter() - t0
        sps = num_steps / elapsed if elapsed > 0 else 0.0
        entry = {"steps": int(num_steps), "elapsed_s": round(elapsed, 1),
                 "sps": round(sps),
                 **{k: float(v) for k, v in metrics.items()}}
        log.append(entry)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {num_steps:>10,} 步 "
              f"{sps:>8,.0f} sps | "
              f"reward={metrics.get('eval/episode_reward', 0):+.1f} | "
              f"长度={metrics.get('eval/avg_episode_length', 0):.0f} | "
              f"vel={metrics.get('eval/episode_vel', 0):.2f} | "
              f"loss={metrics.get('training/total_loss', 0):.4f}", flush=True)

    def save_snapshot(current_step, make_policy, params):
        # 文件名带阶段号, 避免跨阶段覆盖
        with open(f"policies/rl_policy_snap_s{args.stage}_{int(current_step):d}.pkl", "wb") as f:
            pickle.dump(params, f)

    make_policy, params, _ = train_fn(environment=env, progress_fn=progress,
                                      policy_params_fn=save_snapshot)

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    param_path = f"policies/rl_policy_s{args.stage}_{stamp}.pkl"
    with open(param_path, "wb") as f:
        pickle.dump(params, f)
    with open(f"results/rl_train_log_s{args.stage}_{stamp}.json", "w") as f:
        json.dump(log, f, indent=1)
    print(f"\n阶段 {args.stage} 训练完成, 参数已存 {param_path}")


if __name__ == "__main__":
    main()
