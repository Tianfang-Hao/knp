import argparse
import os
import torch.multiprocessing as mp
from solver import run_solver
import torch

def main():
    parser = argparse.ArgumentParser(description="PPO KNP Solver")
    
    # 几何参数
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=24)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_steps', type=int, default=200000)
    
    # 系统参数
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--gpus', type=str, default='0')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    gpu_list = [int(x) for x in args.gpus.split(',')]
    visible_devices = torch.cuda.device_count()
    print(f"检测到 {visible_devices} 个可见 GPU, 将使用: {gpu_list}")
    
    if len(gpu_list) > 1:
        mp.spawn(run_wrapper, args=(args, gpu_list), nprocs=len(gpu_list))
    else:
        run_solver(args, gpu_list[0])

def run_wrapper(rank, args, gpu_list):
    gpu_id = gpu_list[rank]
    run_solver(args, gpu_id)

if __name__ == '__main__':
    main()