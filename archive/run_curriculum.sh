#!/bin/bash
# 四阶段课程训练链 (指令课程 -> 推力课程)
cd /Users/weiwei/Documents/control
FILT='UserWarning|warnings.warn|Failed to import|experimental|DeprecationWarning'
rm -f rl_policy_snap_*.pkl
.venv/bin/python train_go2_rl.py --stage 1 2>&1 | grep -vE "$FILT"
for S in 2 3 4; do
  P=$(ls -t rl_policy_s$((S-1))_*.pkl | head -1)
  echo "=== 阶段$((S-1))完成: $P ==="
  .venv/bin/python train_go2_rl.py --stage $S --resume "$P" 2>&1 | grep -vE "$FILT"
done
echo "=== 四阶段全部完成 ==="
