"""行为克隆引导: 用 MPC 步态示范训练 PPO 策略网络, 供 PPO 续训

策略网络的 Dense 权重看到的是 rs.normalize(state, obs) 后的观察
(与 ppo.train normalize_observations=True 的部署管线完全一致),
损失 = MSE(分布模式, 示范动作)。输出 (normalizer, policy, value)
参数元组, 直接作为 ppo.train(restore_params=...) 的起点。
"""
import pickle

import jax
import jax.numpy as jp
import numpy as np
import optax

from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks as ppo_networks
from train_go2_rl import network_factory  # 同款网络结构 (128,128)/(256,256)

OBS_SIZE, ACT_SIZE = 47, 12


def main():
    with open("policies/bc_demos.pkl", "rb") as f:
        demos = pickle.load(f)
    obs = jp.array(demos["obs"])
    act = jp.array(demos["act"])
    n = len(obs)
    print(f"示范: {n} 样本, act 范围 [{np.asarray(act.min()):.2f}, {np.asarray(act.max()):.2f}]")

    # ---- 归一化状态 (与部署管线一致) ----
    norm_state = rs.update(rs.init_state(jp.zeros(OBS_SIZE)), obs)
    obs_n = rs.normalize(obs, norm_state)

    # ---- 网络 (identity preprocess; 归一化已在数据侧完成) ----
    # 注意 preprocess 签名是 fn(obs, processor_params) -> obs
    def _identity(obs, processor_params):
        return obs

    key = jax.random.PRNGKey(0)
    key_p, key_v, key_s = jax.random.split(key, 3)
    networks = network_factory(OBS_SIZE, ACT_SIZE, _identity)
    policy_net = networks.policy_network
    value_params = networks.value_network.init(key_v)
    policy_params = policy_net.init(key_p)
    dist = networks.parametric_action_distribution

    # ---- BC 训练: MSE(mode(logits), demo) ----
    def loss_fn(params, o, a):
        logits = policy_net.apply(None, params, o)   # identity 处理器占位
        pred = dist.mode(logits)
        return jp.mean((pred - a) ** 2)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(policy_params)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    idx = np.arange(n)
    rng = np.random.default_rng(0)
    batch = 256
    best_loss, best_params = 1e9, policy_params
    for epoch in range(150):
        rng.shuffle(idx)
        losses = []
        for i in range(0, n, batch):
            sel = idx[i:i + batch]
            l, g = grad_fn(policy_params, obs_n[sel], act[sel])
            updates, opt_state = optimizer.update(g, opt_state)
            policy_params = optax.apply_updates(policy_params, updates)
            losses.append(float(l))
        ml = float(np.mean(losses))
        if ml < best_loss:
            best_loss, best_params = ml, policy_params
        if epoch % 15 == 0:
            print(f"epoch {epoch:3d}  MSE {ml:.5f}  (动作幅度 mse→0.01 ≈ 0.1 rad)")

    policy_params = best_params
    # 训练集上的行为对比
    logits = policy_net.apply(None, policy_params, obs_n[:2000])
    pred = dist.mode(logits)
    err = np.abs(np.asarray(pred) - np.asarray(act[:2000]))
    print(f"最终 MSE {best_loss:.5f}, 平均动作误差 {err.mean():.3f} (0.4 尺度), "
          f"最大 {err.max():.2f}")

    # ---- 价值网络预训练: 用示范回报 (return-to-go) 监督 ----
    # 随机初始化的价值网会让 PPO 早期优势估计全错, 大步更新摧毁 BC 精度
    if "ret" in demos:
        ret = jp.array(demos["ret"])
        value_net = networks.value_network

        def v_loss(vp, o, r):
            v = value_net.apply(None, vp, o).squeeze()
            return jp.mean((v - r) ** 2)

        v_opt = optax.adam(1e-3)
        v_state = v_opt.init(value_params)
        v_grad = jax.jit(jax.value_and_grad(v_loss))
        for epoch in range(100):
            rng.shuffle(idx)
            losses = []
            for i in range(0, n, batch):
                sel = idx[i:i + batch]
                lv, gv = v_grad(value_params, obs_n[sel], ret[sel])
                upd, v_state = v_opt.update(gv, v_state)
                value_params = optax.apply_updates(value_params, upd)
                losses.append(float(lv))
            if epoch % 25 == 0:
                print(f"value epoch {epoch:3d}  MSE {np.mean(losses):.1f}")
        print(f"价值网预训完成 (回报量级 {np.asarray(ret).std():.0f})")

    out = (norm_state, policy_params, value_params)
    with open("policies/bc_init.pkl", "wb") as f:
        pickle.dump(out, f)
    print("已存 policies/bc_init.pkl —— 可作为 ppo.train 的 restore_params")


if __name__ == "__main__":
    main()
