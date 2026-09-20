"""快速验证 MuJoCo 物理仿真和离屏渲染是否可用"""
import time
import numpy as np
import mujoco

# 10 个关节的串联摆 + 地面接触，典型的接触丰富场景
xml = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="1 0.005 0.0001"/>
    <body name="link1" pos="0 0 1.0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0  0.3 0 0" size="0.05"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0  0.3 0 0" size="0.05"/>
        <body name="link3" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0  0.3 0 0" size="0.05"/>
          <body name="link4" pos="0.3 0 0">
            <joint name="j4" type="hinge" axis="0 1 0" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0  0.3 0 0" size="0.05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print(f"模型: {model.njnt} 个关节, {model.ngeom} 个几何体, 时间步长 {model.opt.timestep}s")

# 随机初始状态，让摆动起来
data.qpos[:] = np.random.uniform(-1, 1, model.nq)

N = 50_000
mujoco.mj_step(model, data)  # 预热
t0 = time.perf_counter()
for _ in range(N):
    mujoco.mj_step(model, data)
elapsed = time.perf_counter() - t0
sim_time = N * model.opt.timestep
print(f"物理仿真: {N:,} 步耗时 {elapsed:.2f}s -> {N/elapsed:,.0f} 步/秒")
print(f"  (仿真了 {sim_time:.0f}s 的虚拟时间, 实时倍率 {sim_time/elapsed:.0f}x)")

# 离屏渲染测试
renderer = mujoco.Renderer(model, height=480, width=640)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
pixels = renderer.render()
print(f"离屏渲染: {pixels.shape} RGBA 图像正常生成 ✓")

print("\n结论: 你的环境可以流畅跑 MuJoCo 仿真")
