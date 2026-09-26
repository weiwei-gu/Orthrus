"""NMPC 行走段发散诊断: 16s 剧本 + t>8s 每 0.1s 打印状态与 SCvx 诊断
(定位 walk 段摔倒的发散动力学; 不入库, 调试用)
"""
import numpy as np
import mujoco

from run_go2_mpc import Go2Controller, SCENE, T_STAND, T_TROT, T_END, V_WALK
from go2_utils import FOOT_NAMES, quat_to_rpy


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
    foot_z = np.mean([data.geom_xpos[g][2] for g in foot_geoms])
    data.qpos[2] += model.geom_size[foot_geoms[0]][0] - foot_z
    mujoco.mj_forward(model, data)

    ctrl = Go2Controller(model, data, use_nmpc=True)
    ctrl.stand_until = T_STAND

    print("t      z     roll    pitch   yaw     vx    cost     acc/damp/rej", flush=True)
    last = 0.0
    while data.time < T_END:
        mujoco.mj_forward(model, data)
        data.ctrl[:] = ctrl.update(data.time)
        mujoco.mj_step(model, data)
        st = ctrl.state
        if st["t"] - last >= 0.1 and st["t"] > 7.5:
            m = ctrl.mpc
            print(f"{st['t']:5.2f} {st['com'][2]:.3f} "
                  f"{np.degrees(st['rpy'][0]):+6.1f} {np.degrees(st['rpy'][1]):+6.1f} "
                  f"{np.degrees(st['rpy'][2]):+6.1f} {st['v'][0]:+.2f} "
                  f"{m.last_cost:9.1f}  {m.accepted}/{m.damped}/{m.rejected}",
                  flush=True)
            last = st["t"]
        if st["com"][2] < 0.16:
            print(f"!! t={st['t']:.2f} 摔倒 rpy={np.degrees(st['rpy'])}", flush=True)
            break
    ms = np.array(ctrl.mpc.solve_ms) * 1e3
    print(f"solve: 均值 {ms.mean():.2f}ms 最大 {ms.max():.2f}ms | "
          f"SCvx 接受 {ctrl.mpc.accepted} 阻尼 {ctrl.mpc.damped} 拒绝 {ctrl.mpc.rejected}")


if __name__ == "__main__":
    main()
