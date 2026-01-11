import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, dim_n, hidden_dim=512, num_layers=6, nhead=8):
        """
        Args:
            dim_n: 输入维度 (例如 4)
            hidden_dim: 隐藏层特征维度 (建议 256 或 512)
            num_layers: Transformer 层数 (建议 3-6)
            nhead: 注意力头数 (必须能整除 hidden_dim)
        """
        super().__init__()
        
        # 1. 特征提取 (Embedding)
        self.embedding = nn.Sequential(
            nn.Linear(dim_n, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. 核心交互层 (Transformer Encoder)
        # 增大模型：增加层数 (num_layers) 和 宽度 (hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, # 自动设置为 2048 (如果 hidden=512)
            batch_first=True,
            norm_first=True # Pre-Norm 有助于深层网络收敛
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Actor Head
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_n),
            nn.Tanh()
        )
        
        # 初始标准差
        self.actor_log_std = nn.Parameter(torch.zeros(1, dim_n) - 0.5) 

        # 4. Critic Head (Global Value)
        # 增加一点 Critic 的容量
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, N, D)
        h = self.embedding(x)
        h = self.transformer(h) 
        
        # Actor
        mu = self.actor_mu(h) * 0.1 
        std = self.actor_log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        # Critic
        h_global = h.mean(dim=1)
        value = self.critic_head(h_global)
        
        return dist, value