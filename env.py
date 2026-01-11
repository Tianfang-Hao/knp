import torch
import math
from utils import normalize_sphere

class KNPEnvironment:
    def __init__(self, args, device, loss_fn):
        self.args = args
        self.device = device
        self.loss_fn = loss_fn 
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.num_points = args.num_points
        
        self.state = None
        self.last_loss = None
        
        self.target_deg = 30.0 
        self.target_threshold = math.cos(math.radians(self.target_deg))
        
    def set_target_angle(self, deg):
        """设置课程学习的目标角度"""
        self.target_deg = deg
        new_threshold = math.cos(math.radians(deg))
        
        # 限制最高难度阈值 (60度 -> 0.5)
        if new_threshold < 0.5: 
            new_threshold = 0.5
            
        self.target_threshold = new_threshold
        # [Fix] 不再修改 loss_fn.threshold，而是在 step 中传递

    def reset(self):
        self.state = torch.randn(self.batch_size, self.num_points, self.dim, device=self.device)
        self.state = normalize_sphere(self.state)
        
        with torch.no_grad():
            # 计算初始 Loss
            loss_vec, _ = self.loss_fn.compute(self.state, threshold=self.target_threshold)
            self.last_loss = loss_vec.detach()
            
        return self.state

    def reset_indices(self, indices):
        if len(indices) == 0: return
        
        new_data = torch.randn(len(indices), self.num_points, self.dim, device=self.device)
        new_data = normalize_sphere(new_data)
        self.state[indices] = new_data
        
        with torch.no_grad():
            loss_vec, _ = self.loss_fn.compute(new_data, threshold=self.target_threshold)
            self.last_loss[indices] = loss_vec.detach()

    def step(self, action):
        # 1. 截断
        action = torch.clamp(action, -0.1, 0.1)
        # 2. 移动
        x_proposed = self.state + action
        self.state = normalize_sphere(x_proposed)
        
        # 3. 计算 Loss [Fix] 显式传递当前的 threshold
        with torch.no_grad():
            current_loss_vec, max_cos_vec = self.loss_fn.compute(self.state, threshold=self.target_threshold)
        
        # 4. 奖励设计
        # 势能差奖励
        reward = (self.last_loss - current_loss_vec) * 10.0
        
        # 成功判定 (max_cos <= threshold)
        is_success = (max_cos_vec <= self.target_threshold)
        
        # 动态成功奖励 (随难度增加)
        difficulty_factor = max(0, (self.target_deg - 20.0) / 40.0)
        bonus = 20.0 + 80.0 * difficulty_factor
        reward += is_success.float() * bonus
        
        # 惩罚微小移动 (防止懒惰)
        move_dist = torch.norm(action, dim=-1).mean(dim=-1)
        reward -= 0.01 * (move_dist < 1e-6).float()

        self.last_loss = current_loss_vec
        done = is_success
        
        return self.state.clone(), reward, done, max_cos_vec