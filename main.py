import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from solver import run_solver

# === 性能优化配置 ===
# 1. 移除全局 float64 (回退到 float32，速度提升巨大)
# torch.set_default_dtype(torch.float64) 

# 2. 开启 A100 TF32 加速 (MatMul 和 CuDNN)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main():
    parser = argparse.ArgumentParser(description="PPO KNP Solver (Fast DDP + SFT)")
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=4096) 
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--save_dir', type=str, default='./results')
    
    # === [新增] SFT 预训练参数 ===
    parser.add_argument('--sft_steps', type=int, default=2000, help="SFT预训练步数，0表示关闭")
    parser.add_argument('--sft_target_deg', type=float, default=50.0, help="SFT阶段的目标角度（建议比最终目标低，如50度）")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    world_size = torch.cuda.device_count()
    print(f"=== KNP Solver 启动 (极速版 + SFT) ===")
    print(f"GPU数量: {world_size} | 精度: Float32 (TF32 on)")
    print(f"总并行环境数: {args.batch_size * world_size}")
    if args.sft_steps > 0:
        print(f"预训练模式: SFT开启 ({args.sft_steps}步, 目标{args.sft_target_deg}°)")

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