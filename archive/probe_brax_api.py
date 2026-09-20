"""探针: 确认 brax/MJX API 形状 + CPU 批量吞吐基准"""
import inspect
import time

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

print("== 版本 ==")
import brax
print("brax:", getattr(brax, "__version__", "?"))
print("jax:", jax.__version__, jax.devices())

from brax import envs
from brax.envs import PipelineEnv, State
from brax.io import mjcf, model
from brax.training.agents.ppo import train as ppo

print("\n== API 形状 ==")
print("PipelineEnv.__init__:", inspect.signature(PipelineEnv.__init__))
print("pipeline_step:", inspect.signature(PipelineEnv.pipeline_step))
print("pipeline_init:", inspect.signature(PipelineEnv.pipeline_init))
print("State 字段:", State.__dataclass_fields__.keys() if hasattr(State, "__dataclass_fields__") else "?")
print("ppo.train:", inspect.signature(ppo))

print("\n== MJX Go2 加载 ==")
mj_model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene_mjx.xml")
sys = mjcf.load_model(mj_model)
print("加载 OK: nq =", sys.qdim0 if hasattr(sys, "qdim0") else mj_model.nq,
      " nu =", mj_model.nu, " dt =", mj_model.opt.timestep)

print("\n== CPU 批量吞吐 ==")
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.make_data(mjx_model)

N = 512
rng = jax.random.PRNGKey(0)

def step_batch(m, d, ctrl):
    def one(d, c):
        d = d.replace(ctrl=c)
        return mjx.step(m, d), None
    d, _ = jax.lax.scan(lambda d, c: one(d, c), d, ctrl)
    return d.qpos[:, 2]

ctrl = jax.random.uniform(rng, (N, 50, mj_model.nu), minval=-0.5, maxval=0.5)
data_batch = jax.vmap(lambda _: mjx_data)(jp.zeros(N))
step_jit = jax.jit(jax.vmap(step_batch, in_axes=(None, 0, 0)))

t0 = time.perf_counter()
z = step_jit(mjx_model, data_batch, ctrl)
z.block_until_ready()
print(f"编译+首跑: {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
for _ in range(3):
    z = step_jit(mjx_model, data_batch, ctrl)
    z.block_until_ready()
t_run = (time.perf_counter() - t0) / 3
sps = N * 50 / t_run
print(f"512 env × 50 步: {t_run*1000:.0f} ms/run")
print(f"吞吐: {sps:,.0f} 物理步/秒 | 策略步(÷4): {sps/4:,.0f}/秒")
print(f"20M 策略步预计: {20e6/(sps/4)/3600:.2f} 小时")
