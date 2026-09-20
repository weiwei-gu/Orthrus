"""离屏渲染录制混合控制器演示 → mp4 (无需屏幕录制, 可复现)

  .venv/bin/python record_video.py [输出路径]

剧本 (~26 s): 站立 4s → 前进 0.5 m/s → 3 脚确定性踢踏 (侧 18 / 前 20 / 侧 22 N·s)
→ 减速 4s。机身橙红 = RL 救援接管, 原色 = MPC 行走。摔倒则告警退出 (非零码)。
渲染: MuJoCo offscreen 1280x720 @ 50 fps, 原始帧管道喂系统 ffmpeg (libx264)。
"""
import os
import pickle
import subprocess
import sys

import jax
import mujoco
import numpy as np

from eval_go2_rl import ACT_SIZE
from go2_utils import FOOT_NAMES, quat_to_rpy
from run_go2_hybrid import RL_POLICY, HybridController, load_model, reset_stance
from show_go2_hybrid import FALL_Z, RGBA_RESCUE, demo_command, reset_all

W, H, FPS = 1280, 720, 50
EVERY = 10                      # 每 10 × 2ms = 0.02 s 渲一帧 (50 fps)
T_END = 26.0
KICK_DUR = 0.1
KICKS = [                        # (t, 角度°, 冲量 N·s) —— 均落在行走段, 方向侧/前/侧
    (8.0, 90.0, 18.0),
    (13.0, 0.0, 20.0),
    (18.5, -90.0, 22.0),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/demo.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    model = load_model()
    model.vis.global_.offwidth = W          # 离屏帧缓冲默认 640x480, 按需扩到视频分辨率
    model.vis.global_.offheight = H
    data = mujoco.MjData(model)
    reset_stance(model, data)

    with open(RL_POLICY, "rb") as f:
        rl_params = pickle.load(f)
    ctrl = HybridController(model, data, rl_params, verbose=True)
    ctrl.get_command = demo_command
    ctrl.stand_until = 4.0

    # 预热 RL 策略 (jax 编译), 然后复位其内部状态
    ctrl.rl.update(0.0)
    ctrl.rl.step = 0
    ctrl.rl.action = np.zeros(ACT_SIZE)
    ctrl.rl.rng = jax.random.PRNGKey(0)
    ctrl.rl._cmd3 = np.zeros(3)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    g0, gn = model.body_geomadr[base_id], model.body_geomnum[base_id]
    base_geoms = list(range(g0, g0 + gn))
    rgba0 = model.geom_rgba[base_geoms].copy()

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.25]
    cam.distance = 2.9
    cam.azimuth = 132.0
    cam.elevation = -17.0

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-",
         # 注意: mujoco.Renderer.render() 返回的帧已是正常方向 (行 0 = 顶部), 不需要 vflip
         "-c:v", "libx264", "-preset", "medium", "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)

    kicks_done = [False] * len(KICKS)
    kick_until = -1.0
    frames = falls = 0
    last_mode = True

    while data.time < T_END:
        mujoco.mj_forward(model, data)
        data.ctrl[:] = ctrl.update(data.time)

        # ---- 确定性踢踏 ----
        if kick_until < 0:
            for i, (tk, ang, mag) in enumerate(KICKS):
                if not kicks_done[i] and data.time >= tk:
                    data.xfrc_applied[base_id, :3] = np.array(
                        [np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang)), 0.0]) * (mag / KICK_DUR)
                    kick_until = data.time + KICK_DUR
                    kicks_done[i] = True
                    print(f"t={data.time:5.1f}s 💥 踢 {mag:.0f} N·s @ {ang:+.0f}°", flush=True)
                    break
        elif kick_until <= data.time:
            data.xfrc_applied[base_id, :] = 0.0
            kick_until = -1.0

        mujoco.mj_step(model, data)

        if data.time > 1.0 and ctrl.state["com"][2] < FALL_Z:
            falls += 1
            print(f"!! t={data.time:.1f}s 摔倒 — 视频剧本失败, 退出重录", flush=True)
            ff.stdin.close()
            ff.wait()
            return 1

        # ---- 机身颜色 = 当前大脑 ----
        mpc_mode = ctrl.fade > 0.5
        if mpc_mode != last_mode:
            model.geom_rgba[base_geoms] = rgba0 if mpc_mode else RGBA_RESCUE
            last_mode = mpc_mode

        # ---- 渲染帧 ----
        if ctrl.step % EVERY == 0:
            cam.lookat[0] = data.qpos[0]
            cam.lookat[1] = data.qpos[1]
            cam.lookat[2] = 0.25
            renderer.update_scene(data, camera=cam)
            ff.stdin.write(renderer.render().tobytes())
            frames += 1

    ff.stdin.close()
    ff.wait()
    size = os.path.getsize(out) / 1e6
    print(f"\n完成: {out} | {frames} 帧 @ {FPS} fps | {size:.1f} MB | "
          f"摔倒 {falls} 次, RL 救援 {ctrl.rescues} 次", flush=True)
    return 0 if falls == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
