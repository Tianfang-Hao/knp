import torch
import torch.nn as nn
from torch.distributions import Normal

class ResidualBlock(nn.Module):
    """残差块：让网络可以很深而不丢失梯度"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out += residual # 残差连接
        return self.relu(out)

class ActorCritic(nn.Module):
    def __init__(self, dim_n, hidden_dim=1024): # <--- 默认宽度提升到 1024
        super().__init__()
        
        # === 1. 特征提取 (Shared Encoder) ===
        # Input -> Linear -> [ResBlock x 3] -> Features
        self.embedding = nn.Sequential(
            nn.Linear(dim_n, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 堆叠 3 个残差块，增加深度和非线性表达能力
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        # === 2. Actor Head (策略) ===
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_n),
            nn.Tanh()
        )
        
        # 初始噪声：设为 0.0 (即 std=1.0)，利用大显存进行强力探索
        self.actor_log_std = nn.Parameter(torch.zeros(1, dim_n) + 0.0) 

        # === 3. Critic Head (价值) ===
        # 输入维度翻倍，因为我们要拼接 Mean 和 Max
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim), # Critic 也加深一点
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, N, D)
        
        # 1. 提取特征
        h = self.embedding(x)
        h = self.res_blocks(h) # (B, N, 1024)
        
        # 2. Actor 分布
        mu = self.actor_mu(h) * 0.1 # 缩放初始动作
        std = self.actor_log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        # 3. Critic 价值评估 (关键修改)
        # 混合池化：同时看"平均情况"和"最差情况"
        h_mean = h.mean(dim=1)      # 全局概览
        h_max = h.max(dim=1)[0]     # 捕捉最强烈的冲突/特征
        
        # (B, 2048)
        h_global = torch.cat([h_mean, h_max], dim=-1)
        
        value = self.critic_head(h_global)
        
        return dist, value