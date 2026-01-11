import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from solver import run_solver

# === 性能优化配置 ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def ddp_setup(rank, world_size):
    """初始化 DDP"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main():
    parser = argparse.ArgumentParser(description="PPO KNP Solver (Large Model)")
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=24)
    
    # === 模型参数 (Large Config) ===
    parser.add_argument('--hidden_dim', type=int, default=512, help="Transformer 隐藏层维度")
    parser.add_argument('--num_layers', type=int, default=6, help="Transformer 层数")
    parser.add_argument('--nhead', type=int, default=8, help="Transformer 注意力头数")
    
    # === 训练参数 ===
    parser.add_argument('--batch_size', type=int, default=4096) 
    parser.add_argument('--lr', type=float, default=1e-4) # 大模型可以用稍小的 LR，或者保持不变
    parser.add_argument('--max_steps', type=int, default=500000)
    parser.add_argument('--save_dir', type=str, default='./results')
    
    # SFT
    parser.add_argument('--sft_steps', type=int, default=1000)
    parser.add_argument('--sft_target_deg', type=float, default=50.0)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    world_size = torch.cuda.device_count()
    print(f"=== KNP Solver (Large Model) ===")
    print(f"GPUs: {world_size} | Model: Dim={args.hidden_dim}, Layers={args.num_layers}, Heads={args.nhead}")
    print(f"Batch/GPU: {args.batch_size} | Total Batch: {args.batch_size * world_size}")
    
    if world_size > 0:
        mp.spawn(run_wrapper, args=(world_size, args), nprocs=world_size)
    else:
        print("Error: No GPU detected.")

def run_wrapper(rank, world_size, args):
    ddp_setup(rank, world_size)
    try:
        run_solver(rank, args)
    finally:
        destroy_process_group()

if __name__ == '__main__':
    main()