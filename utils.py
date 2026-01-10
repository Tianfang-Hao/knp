import torch
import os
import json
from datetime import datetime

def normalize_sphere(x):
    """强制将所有向量投影回单位球面"""
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def save_result(save_dir, run_id, x_final, n_points, dim, gpu_id):
    """保存成功的构型"""
    # 确保目录存在 (多进程下 safe)
    os.makedirs(save_dir, exist_ok=True)

    # 保存 Tensor
    path = os.path.join(save_dir, f"solution_N{n_points}_D{dim}_GPU{gpu_id}_{run_id}.pt")
    torch.save(x_final.cpu(), path)
    
    meta = {
        "timestamp": str(datetime.now()),
        "n_points": n_points,
        "dim": dim,
        "gpu": gpu_id,
        "path": path
    }
    
    # [修复] 每个 GPU 写自己的日志文件，避免文件锁冲突
    log_file = os.path.join(save_dir, f"success_log_gpu{gpu_id}.jsonl")
    
    with open(log_file, "a") as f:
        f.write(json.dumps(meta) + "\n")
        
    # 只在本地打印简单的提示，防止日志混乱
    # (主日志由 logger 在 rank 0 处理)
    # print(f"[GPU {gpu_id}] Saved to {path}")
    
def init_coordinates(batch_size, n_points, dim, device):
    """随机初始化"""
    x = torch.randn(batch_size, n_points, dim, device=device)
    return normalize_sphere(x)