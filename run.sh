#!/bin/bash

# --- 硬件配置 ---
# 80G A100: 火力全开
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p results
mkdir -p knp/models

echo "=== 启动 PPO-ResNet (80G VRAM Edition) ==="
echo "Batch Size: 4096 | Hidden Dim: 1024"

# 注意：Batch Size 设为 4096，LR 设为 0.0003
# 这样每次 Update 都会处理 4096 * 128 = 52万个样本
python main.py \
    --dim 4 \
    --num_points 24 \
    --batch_size 256 \
    --max_steps 500000 \
    --lr 0.003 \
    --gpus 0,1,2,3 \
    --save_dir ./results