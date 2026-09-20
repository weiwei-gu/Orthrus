#!/bin/bash
# BC 引导后的 PPO 精调链: 全指令域 -> 中推力 -> 目标推力
cd /Users/weiwei/Documents/control
FILT='UserWarning|warnings.warn|Failed to import|experimental|DeprecationWarning'
rm -f rl_policy_snap_*.pkl
P=bc_init.pkl
for S in 2 3 4; do
  echo "=== 阶段$S 从 $P 续训 ==="
  .venv/bin/python train_go2_rl.py --stage $S --resume "$P" 2>&1 | grep -vE "$FILT"
  P=$(ls -t rl_policy_s${S}_*.pkl | head -1)
done
echo "=== BC+PPO 精调全部完成 ==="
