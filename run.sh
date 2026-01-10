#!/bin/bash

# --- 配置 ---
# 使用 4 张卡 (请根据实际情况调整)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH

# 清理旧数据 (可选)
# rm -rf results/*

echo "=== 启动训练 (DDP Mode: SFT + RL) ==="

# 注意：
# 1. --batch_size 是单卡 Batch
# 2. --sft_steps 2000: 先进行 2000 步监督微调
# 3. --sft_target_deg 50: SFT 阶段尝试将球推开到至少 50 度

python main.py \
    --dim 4 \
    --num_points 23 \
    --batch_size 4096 \
    --max_steps 10000 \
    --lr 0.0003 \
    --sft_steps 2000 \
    --sft_target_deg 50.0 \
    --save_dir ./results