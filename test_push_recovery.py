"""推力恢复测试套件 — 量化 MPC 控制器抗扰动能力

每次试验: 站立 4s -> trot/行走 -> 在随机时机向机身施加 0.1s 脉冲力
         -> 观察恢复情况, 记录指标

判定:
  失败   = com_z < 0.16 (趴地) 或 结束时姿态未恢复 (|roll|,|pitch| > 0.3 或 z < 0.2)
  恢复时间 = 推力结束后, |v_xy| < 0.15 且 |roll|,|pitch| < 0.15° 持续 0.2s 的时刻

用法:
  .venv/bin/python test_push_recovery.py [json 输出路径]
"""
import json
import sys

import numpy as np
import mujoco

from run_go2_mpc import Go2Controller, SCENE, FOOT_NAMES

PUSH_DUR = 0.1                       # 推力持续时间 (s)
MAGS = [10.0, 20.0, 30.0]            # 冲量档位 N·s (Δv ≈ 0.66/1.3/2.0 m/s)
DIR_ANGLES = np.deg2rad(np.arange(0, 360, 45))   # 8 个方向
STAND_T = 4.0

# 工况: (名称, 推力窗口起点, 观察终点, 速度指令函数)
CONDITIONS = {
    "trot": (5.5, 11.0, lambda t: np.zeros(2)),
    "walk": (10.0, 14.0, lambda t: np.array([0.5, 0.0]) if t >= 8.0 else np.zeros(2)),
}
SEEDS = {"trot": (1, 2), "walk": (3,)}


def run_trial(model, cond, mag, angle, seed, no_push=False, ctrl_factory=Go2Controller):
    rng = np.random.default_rng(seed)
    t_push0, t_end, cmd_fn = CONDITIONS[cond]

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
    foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
    data.qpos[2] += model.geom_size[foot_geoms[0]][0] - foot_z
    mujoco.mj_forward(model, data)

    ctrl = ctrl_factory(model, data)
    ctrl.get_command = cmd_fn
    ctrl.stand_until = STAND_T

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    fvec = np.array([np.cos(angle), np.sin(angle), 0.0]) * (mag / PUSH_DUR)
    t_push = t_push0 + rng.uniform(0.0, 1.0)

    pushed = push_done = False
    com_at_push = None
    min_z, max_roll, max_pitch = 9.0, 0.0, 0.0
    drift = 0.0
    fall_t = None
    recover_t = None          # 恢复时刻
    _steady = 0               # 连续稳定计数

    while data.time < t_end:
        if not no_push and not pushed and data.time >= t_push:
            data.xfrc_applied[base_id, :3] = fvec
            pushed = True
            com_at_push = ctrl.state["com"].copy()
        if pushed and not push_done and data.time >= t_push + PUSH_DUR:
            data.xfrc_applied[base_id, :] = 0.0
            push_done = True

        mujoco.mj_forward(model, data)
        data.ctrl[:] = ctrl.update(data.time)
        mujoco.mj_step(model, data)

        st = ctrl.state
        com, v, rpy = st["com"], st["v"], st["rpy"]
        min_z = min(min_z, com[2])
        max_roll = max(max_roll, abs(rpy[0]))
        max_pitch = max(max_pitch, abs(rpy[1]))
        if pushed and com_at_push is not None:
            drift = max(drift, np.hypot(*(com[:2] - com_at_push[:2])))

        if com[2] < 0.16:
            fall_t = data.time
            break

        # 恢复检测 (推力结束后): 速度与姿态回稳并持续 0.2s
        if push_done and recover_t is None:
            ok = (np.hypot(v[0], v[1]) < 0.15
                  and abs(rpy[0]) < np.deg2rad(15) and abs(rpy[1]) < np.deg2rad(15))
            _steady = _steady + 1 if ok else 0
            if _steady >= 100:          # 0.2s @500Hz
                recover_t = data.time - (t_push + PUSH_DUR)

    end_ok = (fall_t is None and ctrl.state["com"][2] > 0.20
              and abs(ctrl.state["rpy"][0]) < np.deg2rad(20)
              and abs(ctrl.state["rpy"][1]) < np.deg2rad(20))
    return {
        "cond": cond, "mag": mag, "angle_deg": float(np.degrees(angle)),
        "seed": seed, "pass": bool(end_ok), "fall_t": fall_t,
        "recover_s": None if recover_t is None else round(recover_t, 2),
        "min_z": round(min_z, 3), "max_roll_deg": round(np.degrees(max_roll), 1),
        "max_pitch_deg": round(np.degrees(max_pitch), 1),
        "drift_m": round(drift, 3),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=None, help="json 输出路径")
    parser.add_argument("--nmpc", action="store_true",
                        help="用逐次凸化 NMPC (nmpc.py) 替换凸 MPC (默认凸 MPC)")
    args = parser.parse_args()

    out_path = args.out or ("results/push_recovery_nmpc.json" if args.nmpc
                            else "results/push_recovery_suite.json")
    model = mujoco.MjModel.from_xml_path(SCENE)
    ctrl_factory = (lambda m, d: Go2Controller(m, d, use_nmpc=True)) \
        if args.nmpc else Go2Controller
    if args.nmpc:
        print("===== NMPC (SCvx 逐次凸化) 推力恢复套件 =====", flush=True)

    trials = []

    # 对照组 (无推力, 应 100% 通过)
    for cond in CONDITIONS:
        r = run_trial(model, cond, 0.0, 0.0, 99, no_push=True,
                      ctrl_factory=ctrl_factory)
        r["angle_deg"] = -1
        trials.append(r)
        print(f"[对照/{cond}] pass={r['pass']}")

    total = 0
    for cond, (t0, t1, _) in CONDITIONS.items():
        for mag in MAGS:
            for angle in DIR_ANGLES:
                for seed in SEEDS[cond]:
                    r = run_trial(model, cond, mag, angle, seed,
                                  ctrl_factory=ctrl_factory)
                    trials.append(r)
                    total += 1
                    mark = "✓" if r["pass"] else f"✗ fall@{r['fall_t']:.1f}s" if r["fall_t"] else "✗ 未恢复"
                    print(f"[{total:3d}] {cond} {mag:4.0f}N·s {r['angle_deg']:5.1f}°  "
                          f"{mark}  rec={r['recover_s']}s  pitch_max={r['max_pitch_deg']}°")

    with open(out_path, "w") as f:
        json.dump(trials, f, indent=1)

    # ---- 汇总 ----
    print("\n" + "=" * 64)
    print(f"{'工况':<6}{'冲量':>8}{'恢复率':>10}{'平均恢复时间':>14}{'最大俯仰':>10}{'最大漂移':>10}")
    for cond in CONDITIONS:
        for mag in MAGS:
            sel = [t for t in trials if t["cond"] == cond and t["mag"] == mag]
            n_pass = sum(t["pass"] for t in sel)
            recs = [t["recover_s"] for t in sel if t["recover_s"] is not None]
            rec_avg = np.mean(recs) if recs else float("nan")
            print(f"{cond:<6}{mag:>6.0f}N·s{n_pass}/{len(sel):>7}"
                  f"{rec_avg:>11.1f}s{max(t['max_pitch_deg'] for t in sel):>9.1f}°"
                  f"{max(t['drift_m'] for t in sel):>9.2f}m")
    print(f"\n详细数据已存 {out_path}")


if __name__ == "__main__":
    main()
