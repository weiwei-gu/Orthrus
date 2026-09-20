"""MuJoCo 交互式 3D 演示: 彩色五连杆混沌摆 + 弹力球

窗口操作:
  左键拖拽     旋转视角
  右键拖拽     平移视角
  滚轮         缩放
  空格         暂停/继续
  Backspace    重置
  Ctrl+拖拽    对物体施加力
  关闭窗口     退出
"""
import time

import numpy as np
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="3 3 4" dir="-1 -1 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.35 0.38 0.42 1"/>

    <!-- 悬挂的五连杆摆, 每节不同颜色 -->
    <body name="link1" pos="0 0 2.4">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.06" rgba="0.9 0.2 0.2 1"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.06" rgba="0.95 0.6 0.1 1"/>
        <body name="link3" pos="0.5 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.06" rgba="0.2 0.8 0.3 1"/>
          <body name="link4" pos="0.5 0 0">
            <joint name="j4" type="hinge" axis="0 1 0" damping="0.05"/>
            <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.06" rgba="0.2 0.5 0.9 1"/>
            <body name="link5" pos="0.5 0 0">
              <joint name="j5" type="hinge" axis="0 1 0" damping="0.05"/>
              <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.06" rgba="0.7 0.3 0.9 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- 地面上的弹力球, 会被摆链抽飞 -->
    <body name="ball1" pos="1.3 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.15" rgba="0.2 0.7 0.9 1" mass="0.5"/>
    </body>
    <body name="ball2" pos="1.6 0.4 0.15">
      <freejoint/>
      <geom type="sphere" size="0.15" rgba="0.95 0.85 0.2 1" mass="0.5"/>
    </body>
    <body name="ball3" pos="-1.4 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.15" rgba="0.6 0.3 0.9 1" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
rng = np.random.default_rng(0)

# 初始随机速度, 一开场就甩起来
data.qvel[:5] = rng.uniform(-4, 4, 5)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0_wall = time.perf_counter()
    t0_sim = data.time
    next_nudge = data.time + 3.0
    step = 0

    while viewer.is_running():
        mujoco.mj_step(model, data)
        step += 1

        # 每 3 秒随机踢摆链两脚, 保持持续混沌运动
        if data.time >= next_nudge:
            data.qvel[0] += rng.uniform(-6, 6)
            data.qvel[1] += rng.uniform(-6, 6)
            next_nudge = data.time + 3.0

        # 每 10 步刷新一帧画面 (约 50 FPS)
        if step % 10 == 0:
            viewer.sync()

        # 实时节流: 虚拟时间与墙钟时间对齐 (1 倍速)
        ahead = (data.time - t0_sim) - (time.perf_counter() - t0_wall)
        if ahead > 0:
            time.sleep(ahead)

print("窗口已关闭, 演示结束")
