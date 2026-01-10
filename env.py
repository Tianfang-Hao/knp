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
        
        # 初始课程目标
        self.target_deg = 20.0
        self.target_threshold = math.cos(math.radians(self.target_deg))
        
    def set_target_angle(self, deg):
        """设置课程学习的目标角度，并同步松弛 Loss"""
        self.target_deg = deg
        new_threshold = math.cos(math.radians(deg))
        
        # 1. 更新成功判定阈值
        self.target_threshold = new_threshold
        # 硬性下限：哪怕目标再简单，也不会让阈值低于 60 度 (0.5)
        # 这保证了物理判断的一致性
        if self.target_threshold < 0.5: self.target_threshold = 0.5
            
        # 2. 同步更新 Loss 函数的阈值
        self.loss_fn.threshold = self.target_threshold
        
        # 3. 立即重算 last_loss，防止 Reward 震荡
        if self.state is not None:
            with torch.no_grad():
                loss_vec, _ = self.loss_fn.compute(self.state)
                self.last_loss = loss_vec.detach()
            
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
        # 1. 物理动作截断
        action = torch.clamp(action, -0.2, 0.2)
        
        # 2. 状态更新
        x_proposed = self.state + action
        self.state = normalize_sphere(x_proposed)
        
        # 3. 计算 Loss
        with torch.no_grad():
            current_loss_vec, max_cos_vec = self.loss_fn.compute(self.state)
        
        # 4. 奖励设计
        # 基础奖励：势能下降
        reward = (self.last_loss - current_loss_vec) * 100.0
        
        # [优化] 增加针对最大冲突的直接惩罚
        # 不管当前的 Curriculum 目标是多少，始终对任何小于 60 度 (cos > 0.5) 的重叠进行惩罚
        # 这迫使 Agent 优先解决最糟糕的拥挤区域
        reward -= (max_cos_vec - 0.5).clamp(min=0) * 10.0
        
        # 成功判定
        is_success = (max_cos_vec <= self.target_threshold)
        
        # 课程大奖
        difficulty_factor = (self.target_deg - 20.0) / 40.0 
        bonus = 50.0 + 150.0 * difficulty_factor
        reward += is_success.float() * bonus
        
        # 惩罚不动 (仅当没有成功时惩罚)
        move_dist = torch.norm(action, dim=-1).mean(dim=-1)
        not_done_mask = (~is_success).float()
        reward -= 0.1 * (move_dist < 1e-6).float() * not_done_mask

        self.last_loss = current_loss_vec
        done = is_success
        
        return self.state.clone(), reward, done, max_cos_vec