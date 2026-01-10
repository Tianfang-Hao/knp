import torch
import math
from utils import normalize_sphere

class KNPEnvironment:
    def __init__(self, args, device, loss_fn):
        self.args = args
        self.device = device
        self.loss_fn = loss_fn # <--- 引用传入的 KissingLoss 对象
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.num_points = args.num_points
        
        self.state = None
        self.last_loss = None
        
        self.target_deg = 30.0 # 建议初始降到 30 度
        self.target_threshold = math.cos(math.radians(self.target_deg))
        
    def set_target_angle(self, deg):
        """设置课程学习的目标角度"""
        self.target_deg = deg
        # 计算余弦阈值
        new_threshold = math.cos(math.radians(deg))
        
        # 1. 更新环境判定的阈值
        self.target_threshold = new_threshold
        if self.target_threshold < 0.5: # 限制最高只到 60度
            self.target_threshold = 0.5

        # 2. 【核心修复】同步更新 Loss 函数的阈值！
        # 这就像把"教材"也变简单了，梯度场会变得非常平滑，不再死锁
        self.loss_fn.threshold = self.target_threshold
            
    def reset(self):
        self.state = torch.randn(self.batch_size, self.num_points, self.dim, device=self.device)
        self.state = normalize_sphere(self.state)
        
        with torch.no_grad():
            loss_vec, _ = self.loss_fn.compute(self.state)
            self.last_loss = loss_vec.detach()
            
        return self.state

    def reset_indices(self, indices):
        if len(indices) == 0: return
        
        # 重新随机生成
        new_data = torch.randn(len(indices), self.num_points, self.dim, device=self.device)
        new_data = normalize_sphere(new_data)
        self.state[indices] = new_data
        
        with torch.no_grad():
            loss_vec, _ = self.loss_fn.compute(new_data)
            self.last_loss[indices] = loss_vec.detach()

    def step(self, action):
        # ... (和之前保持一致，不需要变) ...
        # 1. 截断
        action = torch.clamp(action, -0.1, 0.1)
        # 2. 移动
        x_proposed = self.state + action
        self.state = normalize_sphere(x_proposed)
        
        # 3. 计算 Loss (此时 loss_fn.threshold 已经是当前课程的难度了)
        with torch.no_grad():
            current_loss_vec, max_cos_vec = self.loss_fn.compute(self.state)
        
        # 4. 奖励
        reward = (self.last_loss - current_loss_vec) * 10.0
        
        # 判定成功 (使用严格的 < )
        # 注意：max_cos_vec 越小越好 (角度越大越好)
        is_success = (max_cos_vec <= self.target_threshold)
        
        # 动态奖励
        difficulty_factor = (self.target_deg - 20.0) / 40.0 
        bonus = 20.0 + 80.0 * difficulty_factor
        reward += is_success.float() * bonus
        
        # 惩罚不动
        move_dist = torch.norm(action, dim=-1).mean(dim=-1)
        reward -= 0.01 * (move_dist < 1e-6).float()

        self.last_loss = current_loss_vec
        done = is_success
        
        return self.state.clone(), reward, done, max_cos_vec