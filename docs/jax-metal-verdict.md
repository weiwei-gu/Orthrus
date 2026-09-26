# jax-metal 实测判定：Mac GPU 跑 MJX/RL 训练 = 死路（2026-09-26）

> 问题：能否在这台 MacBook（M5 Pro）上用 GPU 加速 RL 训练（MJX + Brax PPO）？
> 结论：**不能**——Apple 的 jax-metal 插件（闭源、2024-10 后停更）缺少 MJX 物理必需的
> XLA kernel（mhlo.cholesky）。本文件记录完整实测链，防止未来重蹈。

## 实测版本矩阵（.venv-metal，已 gitignore）

可复现配方（能走到的最远点）：

```
python3.12 -m venv .venv-metal
.venv-metal/bin/pip install jax-metal            # 拉入 jax 0.11.x，需降级 ↓
.venv-metal/bin/pip install "jax==0.4.34" "jaxlib==0.4.34"
.venv-metal/bin/pip install "flax==0.8.5" "orbax-checkpoint==0.5.7"   # 时代匹配
.venv-metal/bin/pip install "mujoco==3.2.7" "mujoco-mjx==3.2.7" "brax==0.10.5"
ENABLE_PJRT_COMPATIBILITY=1 ...   # 插件官方建议的跨版本兼容开关
```

## 断点链（每层都实测）

1. **插件加载**：jax 0.11.2 下 `devices: [METAL(id=0)]` ✅——但首个计算崩于
   `StableHLO_v1.16.2 bytecode: unknown attribute code 22`：插件编译器只认识
   2024 年代的 IR。降级 jax 0.4.34 解决。
2. **基础算术**：Metal 上元素运算/矩阵乘正确且快（256×256×50 次 2.8ms）✅
3. **MJX 3.1.4**：`NotImplementedError: only condim=3`——menagerie 现代 go2_mjx.xml
   用 condim=1（MJX 提速优化）。升级 mujoco-mjx 3.2.7（有 condim=1）解决。
4. **MJX 3.2.7 API 漂移**：`make_data(m, device)`、`step(mx, dx)` 新签名。修正后：
5. **终局**：`jax.scipy.linalg.cho_factor` → `mhlo.cholesky` ——**Metal XLA 编译器
   无此 kernel**，MJX 每步都要对质量矩阵做 Cholesky 分解。Python 层无解。

## 各路线的判决

| 路线 | 判决 |
|---|---|
| jax-metal + MJX（本文件） | ❌ 算子缺失（mhlo.cholesky），闭源无法自行维护 |
| jax-metal + 纯 JAX 算术 | ✅ 但对 RL 训练无用（训练必需 MJX 物理） |
| MLX | ❌ 无 MJX 移植，重写物理 = 重写项目 |
| mujoco-warp (Metal) | 只加速裸 MuJoCo 物理，不覆盖 Brax PPO 训练管线 |
| **云 CUDA（如 RTX 4090, ~¥2/h）** | ✅ **唯一现实路线**：同量训练 CPU 2h10m → 预计 10~20min |

## 若未来 Apple 更新插件

检查 `python -c "import jax, jaxlib; ..."` 后跑本文的断点 5 探针即可立刻判定。
