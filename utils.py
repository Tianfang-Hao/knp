import torch
import os
import json
from datetime import datetime

def normalize_sphere(x):
    """强制将所有向量投影回单位球面"""
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def save_result(save_dir, run_id, x_final, n_points, dim, gpu_id):
    """保存成功的构型"""
    path = os.path.join(save_dir, f"solution_N{n_points}_D{dim}_GPU{gpu_id}_{run_id}.pt")
    torch.save(x_final.cpu(), path)
    
    meta = {
        "timestamp": str(datetime.now()),
        "n_points": n_points,
        "dim": dim,
        "path": path
    }
    with open(os.path.join(save_dir, "success_log.jsonl"), "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(f"[GPU {gpu_id}] *** FOUND SOLUTION! Saved to {path} ***")

def init_coordinates(batch_size, n_points, dim, device):
    """随机初始化"""
    x = torch.randn(batch_size, n_points, dim, device=device)
    return normalize_sphere(x)