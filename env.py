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
        
        # 初始课程目标：从 25 度开始
        self.target_deg = 25.0
        self.target_threshold = math.cos(math.radians(self.target_deg))
        
    def set_target_angle(self, deg):
        """设置课程学习的目标角度，并同步松弛 Loss"""
        self.target_deg = deg
        new_threshold = math.cos(math.radians(deg))
        
        # 1. 更新成功判定阈值
        self.target_threshold = new_threshold
        if self.target_threshold < 0.5: self.target_threshold = 0.5
            
        # 2. 同步更新 Loss 函数的阈值 (让梯度不再死锁)
        self.loss_fn.threshold = self.target_threshold
            
    def reset(self):
        """完全重置"""
        self.state = torch.randn(self.batch_size, self.num_points, self.dim, device=self.device)
        self.state = normalize_sphere(self.state)
        
        with torch.no_grad():
            loss_vec, _ = self.loss_fn.compute(self.state)
            self.last_loss = loss_vec.detach()
            
        return self.state

    def reset_indices(self, indices):
        """局部重置"""
        if len(indices) == 0: return
        
        new_data = torch.randn(len(indices), self.num_points, self.dim, device=self.device)
        new_data = normalize_sphere(new_data)
        self.state[indices] = new_data
        
        with torch.no_grad():
            loss_vec, _ = self.loss_fn.compute(new_data)
            self.last_loss[indices] = loss_vec.detach()

    def step(self, action):
        # 1. 物理动作截断 (放宽到 0.2，让大模型发挥能力)
        action = torch.clamp(action, -0.2, 0.2)
        
        # 2. 状态更新
        x_proposed = self.state + action
        self.state = normalize_sphere(x_proposed)
        
        # 3. 计算 Loss (基于当前课程难度)
        with torch.no_grad():
            current_loss_vec, max_cos_vec = self.loss_fn.compute(self.state)
        
        # 4. 奖励设计 (Reward Scaling x100)
        # 基础奖励：势能下降
        reward = (self.last_loss - current_loss_vec) * 100.0
        
        # 成功判定
        is_success = (max_cos_vec <= self.target_threshold)
        
        # 课程大奖 (动态调整)
        # 随着角度增加，奖励倍增
        difficulty_factor = (self.target_deg - 20.0) / 40.0 
        bonus = 50.0 + 150.0 * difficulty_factor
        reward += is_success.float() * bonus
        
        # 惩罚不动
        move_dist = torch.norm(action, dim=-1).mean(dim=-1)
        reward -= 0.1 * (move_dist < 1e-6).float()

        self.last_loss = current_loss_vec
        done = is_success
        
        return self.state.clone(), reward, done, max_cos_vec