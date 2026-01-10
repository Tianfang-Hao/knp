#!/bin/bash

# --- 配置 ---
# 使用 4 张卡
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH

# 清理旧数据 (可选)
# rm -rf results/*

echo "=== 启动训练 (DDP Mode) ==="

# 注意：Batch Size 4096 是单卡数量
# 总 Batch = 16384 (4张卡)
python main.py \
    --dim 4 \
    --num_points 23 \
    --batch_size 4096 \
    --max_steps 10000 \
    --lr 0.0003 \
    --save_dir ./results