#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p results
mkdir -p models

echo "=== 启动 PPO KNP Solver (Large Transformer) ==="
echo "配置: Dim 512 | Layers 6 | Heads 8"

python main.py \
    --dim 4 \
    --num_points 24 \
    --batch_size 2048 \
    --hidden_dim 512 \
    --num_layers 6 \
    --nhead 8 \
    --max_steps 100000 \
    --lr 0.0001 \
    --sft_steps 600 \
    --sft_target_deg 55.0 \
    --save_dir ./results