"""RL 策略 sim2sim 评估: 训练好的策略在标准 MuJoCo 中运行, 复用推力测试套件

用法:
  .venv/bin/python eval_go2_rl.py policies/rl_policy_XXXX.pkl            # 行走冒烟测试
  .venv/bin/python eval_go2_rl.py policies/rl_policy_XXXX.pkl --push     # 推力恢复全套
"""
import argparse
import pickle

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from brax.training.agents.ppo import networks as ppo_networks
from go2_utils import quat_to_rpy

RL_SCENE = "models/go2/scene_mjx.xml"
OBS_SIZE, ACT_SIZE = 47, 12
ACTION_SCALE = 0.4
CLOCK_T = 0.6
DEFAULT_Q = np.array([0.0, 0.9, -1.8] * 4)   # 关节顺序 FL,FR,RL,RR


def quat_to_grav(q):
    """四元数 -> 重力在机体系的投影 Rᵀ·[0,0,1]"""
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class PolicyController:
    """与 Go2Controller 同接口的 RL 策略控制器 (50 Hz 策略, 500 Hz 保持)"""

    def __init__(self, model, data, params=None):
        self.model, self.data = model, data
        self.step = 0
        self.action = np.zeros(ACT_SIZE)
        self.rng = jax.random.PRNGKey(0)
        self.get_command = lambda t: np.zeros(2)
        self.stand_until = 0.0          # 兼容测试套件接口 (策略无步态概念)
        self.state = {"t": 0.0, "com": np.zeros(3), "v": np.zeros(3),
                      "rpy": np.zeros(3), "stance": np.ones(4, bool)}
        self._base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if params is not None:
            net = ppo_networks.make_ppo_networks(
                OBS_SIZE, ACT_SIZE,
                policy_hidden_layer_sizes=(128, 128),
                value_hidden_layer_sizes=(256, 256))
            make_policy = ppo_networks.make_inference_fn(net)
            try:
                policy = make_policy(params, deterministic=True)
            except TypeError:
                policy = make_policy(params)
            self._act = jax.jit(policy)
        self._cmd3 = np.zeros(3)

    def _obs(self, d):
        phase = (d.time / CLOCK_T) % 1.0
        clock = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        grav = quat_to_grav(d.qpos[3:7])
        q = d.qpos[7:19] - DEFAULT_Q
        qd = d.qvel[6:18]
        return np.concatenate([clock, grav, self._cmd3, d.qvel[3:6], q, qd, self.action])

    def update(self, t):
        d = self.data
        cmd = self.get_command(t)
        self._cmd3 = np.array([cmd[0], cmd[1], 0.0])

        if self.step % 10 == 0:          # 50 Hz 策略
            self.rng, key = jax.random.split(self.rng)
            act, _ = self._act(jp.array(self._obs(d)), key)
            self.action = np.asarray(act)
        ctrl = DEFAULT_Q + ACTION_SCALE * np.clip(self.action, -1.5, 1.5)

        self.step += 1
        self.state = {"t": t,
                      "com": d.subtree_com[self._base_id].copy(),
                      "v": d.qvel[0:3].copy(),
                      "rpy": quat_to_rpy(d.qpos[3:7]),
                      "stance": np.ones(4, bool)}
        return ctrl


# ---------------------------------------------------------------------- #
def sanity_walk(params, t_end=14.0):
    """行走冒烟测试: 站 3s -> 前进 0.5 m/s"""
    model = mujoco.MjModel.from_xml_path(RL_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] += 0.005               # 微抬避免初始穿透
    mujoco.mj_forward(model, data)

    ctrl = PolicyController(model, data, params)
    ctrl.get_command = lambda t: np.array([0.0, 0.0]) if t < 3.0 else np.array([0.5, 0.0])

    log = []
    while data.time < t_end:
        mujoco.mj_forward(model, data)
        data.ctrl[:] = ctrl.update(data.time)
        mujoco.mj_step(model, data)
        if ctrl.step % 500 == 0:
            st = ctrl.state
            print(f"t={st['t']:5.1f}  z={st['com'][2]:.3f}  "
                  f"roll={np.degrees(st['rpy'][0]):+5.1f}°  pitch={np.degrees(st['rpy'][1]):+5.1f}°  "
                  f"vx={st['v'][0]:+.2f} m/s")
            log.append((st["t"], st["com"][2], st["v"][0]))
    walk = np.array([l for l in log if l[0] >= 5.0])
    print(f"\n行走段: 平均 vx = {walk[:, 2].mean():+.2f} m/s (指令 +0.5), "
          f"z 均值 {walk[:, 1].mean():.3f} m")
    return not (walk[:, 1].mean() < 0.18)


def push_suite(params):
    """推力恢复全套 (与 MPC 同一套件)"""
    from test_push_recovery import run_trial, MAGS, DIR_ANGLES, CONDITIONS
    model = mujoco.MjModel.from_xml_path(RL_SCENE)
    factory = lambda m, d: PolicyController(m, d, params)
    trials = []
    for cond in CONDITIONS:
        for mag in MAGS:
            for angle in DIR_ANGLES:
                for seed in (1, 2) if cond == "trot" else (3,):
                    r = run_trial(model, cond, mag, angle, seed, ctrl_factory=factory)
                    trials.append(r)
                    print(f"[{cond} {mag:.0f}N·s {r['angle_deg']:5.1f}°] "
                          f"{'✓' if r['pass'] else '✗'} rec={r['recover_s']}")
    print("\n===== RL 推力恢复汇总 (vs MPC: trot 88%/44%/12%, walk 62%/0%/13%) =====")
    for cond in CONDITIONS:
        for mag in MAGS:
            sel = [t for t in trials if t["cond"] == cond and t["mag"] == mag]
            n_pass = sum(t["pass"] for t in sel)
            print(f"{cond:<6}{mag:>5.0f}N·s  {n_pass}/{len(sel)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    with open(args.policy, "rb") as f:
        params = pickle.load(f)
    print(f"策略已加载: {args.policy}")
    if args.push:
        push_suite(params)
    else:
        sanity_walk(params)


if __name__ == "__main__":
    main()
