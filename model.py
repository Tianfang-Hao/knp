import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, dim_n, hidden_dim=128):
        super().__init__()
        
        # 1. 特征提取 (Encoder)
        # 使用 MLP 独立处理每个粒子，保证显存效率和通用性
        self.embedding = nn.Sequential(
            nn.Linear(dim_n, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 可选：如果显存够大 (N较小)，可以加一层 Attention 增强交互感知
        # self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 2. Actor Head
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_n),
            nn.Tanh()
        )
        
        # 修正：必须是 nn.Parameter 才能被训练
        self.actor_log_std = nn.Parameter(torch.zeros(1, dim_n) - 0.5) 

        # 3. Critic Head (全局价值)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, N, D)
        h = self.embedding(x)
        
        # 如果启用了 Attention:
        # h_attn, _ = self.attn(h, h, h)
        # h = h + h_attn
        
        # Actor
        mu = self.actor_mu(h) * 0.1 # 缩放输出
        std = self.actor_log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        # Critic (Mean Pooling 聚合全局信息)
        h_global = h.mean(dim=1)
        value = self.critic_head(h_global)
        
        return dist, value