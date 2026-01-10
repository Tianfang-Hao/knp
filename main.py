import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from solver import run_solver

# === 性能优化配置 ===
# 1. 移除全局 float64 (回退到 float32，速度提升巨大)
# torch.set_default_dtype(torch.float64) <--- 删除或注释这行

# 2. 开启 A100 TF32 加速 (MatMul 和 CuDNN)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main():
    parser = argparse.ArgumentParser(description="PPO KNP Solver (Fast DDP)")
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=24)
    # 既然回到了 FP32，显存占用减半，可以尝试进一步加大 Batch
    parser.add_argument('--batch_size', type=int, default=4096) 
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--save_dir', type=str, default='./results')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    world_size = torch.cuda.device_count()
    print(f"=== KNP Solver 启动 (极速版) ===")
    print(f"GPU数量: {world_size} | 精度: Float32 (TF32 on)")
    print(f"总并行环境数: {args.batch_size * world_size}")

    if world_size > 0:
        mp.spawn(run_wrapper, args=(world_size, args), nprocs=world_size)
    else:
        print("错误: 未检测到 GPU")

def run_wrapper(rank, world_size, args):
    ddp_setup(rank, world_size)
    try:
        run_solver(rank, args)
    finally:
        destroy_process_group()

if __name__ == '__main__':
    main()