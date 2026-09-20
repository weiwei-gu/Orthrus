#!/bin/bash
# 温柔精调链: BC(含价值网预训) -> PPO 阶段2/3/4 (低 entropy/低 lr, 保住克隆步态)
#
# 用法:
#   bash run_gentle_finetune.sh            # 完整链: 重采示范 -> BC -> 阶段2/3/4
#   bash run_gentle_finetune.sh --skip-bc  # 跳过采集/BC, 直接用现有 policies/bc_init.pkl
#
# 守卫: 每阶段结束后校验产出了"新的"最终策略文件 —— 若训练中途崩溃,
# 防止把上一轮运行的旧策略 (mtime 更旧) 当作本阶段产物继续链式续训。
set -u
cd /Users/weiwei/Documents/control

SKIP_BC=0
[ "${1:-}" = "--skip-bc" ] && SKIP_BC=1

if [ $SKIP_BC -eq 0 ]; then
  rm -f policies/rl_policy_snap_s*.pkl policies/bc_init.pkl
  echo "=== [1/3] 采集 MPC 示范 (含 return-to-go) ==="
  .venv/bin/python collect_demos.py || { echo "!! collect_demos 失败"; exit 1; }
  echo "=== [2/3] 行为克隆 + 价值网预训练 ==="
  .venv/bin/python bc_bootstrap.py || { echo "!! bc_bootstrap 失败"; exit 1; }
else
  [ -f policies/bc_init.pkl ] || { echo "!! policies/bc_init.pkl 不存在, 无法 --skip-bc"; exit 1; }
  echo "=== 跳过采集/BC, 使用现有 policies/bc_init.pkl ==="
fi

P=policies/bc_init.pkl
for S in 2 3 4; do
  BEFORE=$(ls -t policies/rl_policy_s${S}_*.pkl 2>/dev/null | head -1)
  echo "=== [3/3] 阶段$S 温柔精调 (entropy 1e-4, lr 5e-5, 从 $P 续训) ==="
  .venv/bin/python train_go2_rl.py --stage $S --resume "$P" \
      --entropy 1e-4 --lr 5e-5 || { echo "!! 阶段$S 训练失败"; exit 1; }
  NEW=$(ls -t policies/rl_policy_s${S}_*.pkl 2>/dev/null | head -1)
  if [ -z "$NEW" ] || [ "$NEW" = "$BEFORE" ]; then
    echo "!! 阶段$S 未产出新策略文件, 链中止"
    exit 1
  fi
  P=$NEW
done
echo "=== 温柔精调全部完成 ==="
echo "最终策略: $P"
