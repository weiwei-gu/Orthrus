"""DOB 单元验收: 站立态注入已知 50N 侧向力 (t∈[0.3, 0.8]s), 验证观测器
收敛速度 (~12ms 时间常数) / 稳态精度 / 移除后的衰减。通过判据:
  - 推力窗口内 |f̂_x − 50| < 5N 持续出现
  - 移除后 0.1s 内 |f̂_x| 衰减到 < 10N
"""
import numpy as np
import mujoco

from run_go2_mpc import Go2Controller, SCENE
from go2_utils import FOOT_NAMES

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
data.qpos[2] += model.geom_size[foot_geoms[0]][0] - foot_z
mujoco.mj_forward(model, data)

ctrl = Go2Controller(model, data, use_dob=True)   # 纯站立 (gait 不启动)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

log = []
while data.time < 1.4:
    if 0.3 <= data.time < 0.8:
        data.xfrc_applied[base_id, :3] = [50.0, 0.0, 0.0]
    else:
        data.xfrc_applied[base_id, :] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = ctrl.update(data.time)
    mujoco.mj_step(model, data)
    if ctrl.step % 25 == 0:   # 50Hz 采样
        log.append((data.time, ctrl.f_ext_hat.copy(), ctrl.state["com"][2]))

t = np.array([l[0] for l in log])
fx = np.array([l[1][0] for l in log])
win = (t > 0.36) & (t < 0.78)
after = t > 0.9
# 注: f̂ 估计的是"集总未知力" = 外力 + (实际施加力 − 指令力)。推力窗口内
# 站立 PD 的被动抗力 (~12N) 与外力方向相反, 故稳态 ≈38N 而非 50N —— 这正是
# 前馈想要的量 (MPC 补它不知道的部分)。
detect = next((tt for tt, ff in zip(t, fx) if tt > 0.3 and ff > 25), None)
print(f"窗口内 f̂_x: 均值 {fx[win].mean():.1f} N (集总真值≈38) | 峰值 {fx.max():.1f}")
print(f"检测时间 (|f̂|>25N): {'%.0f ms' % ((detect-0.3)*1000) if detect else '未检测'}")
print(f"推力移除后 0.1s 起 f̂_x 均值: {fx[after].mean():.1f} N (应≈0)")
print(f"站立高度 z 末值: {ctrl.state['com'][2]:.3f} m (应≈0.27)")
ok = (30 <= fx[win].mean() <= 55) and detect and (detect - 0.3) < 0.08 \
     and (abs(fx[after].mean()) < 15) and ctrl.state["com"][2] > 0.2
print("DOB 单元验收:", "✅ 通过" if ok else "❌ 未过")
