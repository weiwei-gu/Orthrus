# Orthrus · 双头犬

> 一条狗，两个脑：MPC 管"怎么走"，RL 管"摔了怎么办"。

Unitree Go2 四足机器人运动控制项目（MuJoCo 仿真，全程在一台 MacBook Pro 的 CPU 上完成）。
从零实现了三种控制范式，用同一套 **72 试验推力恢复测试套件**正面对比，最终融合为双脑混合控制器：

| 控制器 | 总恢复率 | 行走速度（指令 +0.5 m/s） |
|---|---|---|
| Convex MPC（MIT Cheetah 风格） | 40% (29/72) | +0.42~0.50 m/s |
| RL 行为克隆（MPC 步态蒸馏，`bc_init`） | 60% (43/72) | -0.06 m/s |
| RL 温柔精调（11.8M 甜点快照） | 51% (37/72) | +0.14 m/s |
| **Orthrus 混合（MPC 行走 + RL 救援）** | **94% (68/72)** | ~+0.3 m/s |

推力 = 0.1s 脉冲冲量，10/20/30 N·s ≈ 轻推 / 中踢 / 重踹（8 方向 × 3 档 × trot/walk 两工况）。
混合控制器在**每一个格子**都不输给两个"父母"——walk@20 N·s 从 MPC 的 0% 提到 100%。

## 为什么叫 Orthrus

Orthrus（俄耳托斯）是希腊神话中的**双头犬**。这个项目的最终形态正是如此：
**MPC 是理性脑**——单刚体模型 + 凸优化 + 滚动时域，管标称行走的速度跟踪与约束；
**RL 是反射脑**——MPC 步态蒸馏出的神经网络，管被踹后的极限恢复；
**仲裁器**决定每个瞬间谁在"开车"。救援时机身变色（橙红），接回后恢复原色。

## 架构

```
                ┌──────────────── 仲裁器 (500 Hz) ────────────────┐
 速度指令 ──┬──▶│ 速度偏差>0.45 m/s / 倾角>14° / 高度<0.19 → 交棒 RL │
            │    │ 安静 0.3s 且救援≥0.8s → 接回 MPC (重锚步态/钉地点)  │
            │    │ 接回后 0.6s 宽限期防抖; 指令斜坡滤波防阶跃误判      │
            │    └───────────────────────────────────────────────┘
            │              │ 位置目标空间 40ms 交叉淡化, RL 全程影子运行
  ┌─────────▼──────┐  ┌────▼──────────────────┐
  │ MPC 脑          │  │ RL 脑 (bc_init)       │
  │ 100Hz convex QP │  │ 50Hz MLP (128×2)     │
  │ (OSQP, 0.3s 时域)│  │ MPC 步态行为克隆       │
  │ 500Hz τ=-Jᵀ·f   │  │ 12 维关节位置目标      │
  │ Raibert + trot  │  │ 47 维本体感知观察      │
  │ sin³ 摆动 + IK  │  │ (含步态时钟 sin/cos)  │
  └───────┬────────┘  └────┬────────────────┘
          └── τ→q_des 代数转换: q_des = q + (τ + kd·qd)/kp ──┘
                          （位置伺服 kp=50, MPC 全栈一行不改）
```

## 目录结构

```
control/
├── convex_mpc.py          单刚体 convex MPC（12 维状态 QP，OSQP）
├── go2_utils.py           trot 步态调度 / sin³ 摆动轨迹 / Raibert 落脚点 / 腿部 IK
├── run_go2_mpc.py         MPC 主控制器（站立 / 原地 trot / 前进，500Hz 力矩）
├── train_go2_rl.py        RL 训练（MJX + Brax PPO，4 阶段指令/推力课程）
├── collect_demos.py       MPC 示范采集（含 return-to-go，供价值网预训练）
├── bc_bootstrap.py        行为克隆 + 价值网预训练 → 产出 RL 脑
├── eval_go2_rl.py         RL 策略 sim2sim 评估（MuJoCo 验证 + 推力套件入口）
├── run_go2_hybrid.py      ★ 混合控制器（MPC+RL 仲裁）+ 推力套件入口
├── test_push_recovery.py  推力恢复测试套件（72 试验，三方共用同一判定）
├── compare_push.py        两版结果对比
├── show_go2_mpc.py        演示：MPC 行走（交互窗口）
├── show_go2_rl.py         演示：RL 边走边挨踹 / 纯行走（镜头跟随）
├── show_go2_hybrid.py     演示：★ 混合控制器（机身变色 = 换脑）
├── run_gentle_finetune.sh RL 精调链（采集 → BC → 三阶段 PPO）
├── policies/              全部策略 pkl（bc_init.pkl = RL 救援脑；bc_demos.pkl = 示范数据）
├── results/               试验数据（push_recovery_*.json / rl_train_log_*.json / npz）
├── archive/               调试脚本与已废弃训练链（留档）
├── docs/                  完整开发会话历史（claude-history.txt，7800 行）
└── models/go2/            Go2 模型（vendor 自 mujoco_menagerie 的 unitree_go2，含本项目两个自建场景）
```

## 快速开始

```bash
# 所有命令在项目根目录执行（Go2 模型已内置 models/go2/，克隆即用）

# ① 看冠军表演：MPC 牵狗走 → 被随机踹 → 变色(橙红)换 RL 脑救场 → 变回原色继续走
.venv/bin/mjpython show_go2_hybrid.py

# ② 复现决赛数据（各 ~15 分钟）
.venv/bin/python run_go2_hybrid.py --push                          # 混合: 94%
.venv/bin/python run_go2_mpc.py                                    # MPC 单独 (行走 16s 测试)
.venv/bin/python eval_go2_rl.py policies/bc_init.pkl --push       # RL 单独: 60%

# ③ 其他演示
.venv/bin/mjpython show_go2_mpc.py        # MPC 行走（最快，0.5 m/s 小跑）
.venv/bin/mjpython show_go2_rl.py         # RL 版（默认 bc_init + 随机踹踢）
.venv/bin/mjpython show_go2_rl.py policies/rl_walk_sweet_spot.pkl --no-kicks  # 精调版纯行走

# ④ （可选）重新训练 RL 脑：MPC 示范 → BC → 温柔精调，CPU 约 3 小时
bash run_gentle_finetune.sh
```

## 结果明细（`results/push_recovery_hybrid.json` 等）

| 工况 / 冲量 | MPC | RL-BC | 混合 |
|---|---|---|---|
| trot 10 N·s | 88% | 88% | **100%** |
| trot 20 N·s | 44% | 56% | **100%** |
| trot 30 N·s | 12% | 25% | **81%** |
| walk 10 N·s | 62% | 88% | **100%** |
| walk 20 N·s | 0% | 75% | **100%** |
| walk 30 N·s | 13% | 38% | **88%** |

混合版恢复时间均值 0.66s，与单控制器持平（融合无代价）；无扰动对照 2/2 通过（零误报）。

## 技术要点（都写进了代码注释，防重蹈）

- **BC 引导三要素**：步态时钟进观察（sin/cos 相位，否则 MSE 最优解是"相位平均 = 静止"）、
  示范步态相位与部署时钟锁定、动作钳位到 tanh 可达域 ±1
- **从零 PPO 发现不了步态**："站着"是近最优解（本机 CPU 40M 步无效）；
  用自己写的 MPC 当老师蒸馏，一步跨过步态发现难关——学生还在推力恢复上赢了老师
- **温柔精调三剂药**（修上轮"精调毁步态"）：entropy 1e-4 / lr 5e-5 / 价值网示范预训练
- **sim2sim 漂移**：MJX 训练越过 ~12M 步后迁移性单调退化（s3/s4 产物在 MuJoCo 全躺平），
  根因缺物理域随机化——故 RL 救援脑最终采用纯 BC 的 `bc_init`
- **力矩→位置接口**：`q_des = q + (τ + kd·qd)/kp` 让 MPC 力控栈一行不改地驱动位置伺服
- **仲裁阈值**全部高于 MPC 标称瞬态（行走俯仰 5~12° → 触发线 14°），
  救援退出判据用"脱离危机"而非"完美姿态"（bc_init 标称俯仰即有 5~12°）

## 已知局限与下一步

- 混合版行走 ~0.3 m/s（MPC 单独 0.5）：go2_mjx 足端半径较小（0.0175 vs 0.022），步态参数待适配
- 30 N·s 纯侧向（±90°）4 个失败：捕获点超出工作空间，突破需多步恢复规划
- RL 侧未做物理域随机化（摩擦/质量/执行器增益）——修复后 s3/s4 的推力训练成果有望迁移
- 状态直接用仿真真值，无 IMU 噪声模型与状态估计器
- 技能仅有运动控制：挥手/作揖等技能需另训（MPC+IK 脚本化或 RL 技能配方）

---
*模型：Unitree Go2（mujoco_menagerie）· 仿真：MuJoCo 3.13 · 训练：JAX 0.9.2 + Brax 0.14 + MJX*
