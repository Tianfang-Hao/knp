#!/bin/bash

# --- 硬件配置 ---
# 物理卡 4,5,6,7 -> 逻辑卡 0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH

# 创建必要的目录
mkdir -p results
mkdir -p models

# --- 实验配置 ---
# 4D, 24球 (经典问题)
# Batch=256: 既然是 A100，开大 Batch 增加采样效率
echo "=== 启动 PPO 课程学习求解器 ==="
echo "目标: 从 40度 逐步训练至 60度"
echo "模型保存路径: knp/models/"

python main.py \
    --dim 4 \
    --num_points 24 \
    --batch_size 2048 \
    --max_steps 200000 \
    --lr 0.0001 \
    --save_dir ./results
    # --gpus 0,1,2,3,4,5,6,7 \