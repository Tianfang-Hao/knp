import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, dim_n, hidden_dim=512):
        super().__init__()
        
        # === 1. 特征提取 (Shared Encoder) ===
        self.embedding = nn.Sequential(
            nn.Linear(dim_n, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # === 2. Transformer 交互层 (DDP 修复版) ===
        # 关键修改：
        # 不要将 encoder_layer_proto 绑定为 self.layer，
        # 否则 DDP 会认为它是模型参数但未被使用，从而报错。
        # 我们只把它作为临时变量传给 TransformerEncoder。
        encoder_layer_proto = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim*2, 
            batch_first=True
        )
        
        # TransformerEncoder 会自动 clone 原型层，所以模型参数只在 self.transformer 里
        self.transformer = nn.TransformerEncoder(encoder_layer_proto, num_layers=3)
        
        # === 3. Actor Head (策略) ===
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_n),
            nn.Tanh() # 输出 [-1, 1]
        )
        
        # 初始噪声设置: 
        # exp(-1.6) ≈ 0.2，匹配环境中的动作截断范围 [-0.2, 0.2]
        # 避免初始梯度消失或不匹配
        self.actor_log_std = nn.Parameter(torch.zeros(1, dim_n) - 1.6) 

        # === 4. Critic Head (价值) ===
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (Batch, Num_Points, Dim)
        
        # 1. Embedding
        h = self.embedding(x)
        
        # 2. Transformer Interaction (让点与点之间交换位置信息)
        h = self.transformer(h) # (B, N, Hidden)
        
        # 3. Actor 分布
        # 动作幅度缩放：0.1 * Tanh -> [-0.1, 0.1] (在此基础上叠加噪声)
        mu = self.actor_mu(h) * 0.1 
        
        # === 关键修复: 限制最小方差 (防止探索坍缩) ===
        # min=-5.0 对应 std ≈ 0.0067。
        # 保证在后期高难度课程中，Agent 依然保留微小的随机探索能力，
        # 防止 std 降为 0 导致死锁在局部最优。
        log_std = torch.clamp(self.actor_log_std, min=-5.0)
        std = log_std.exp().expand_as(mu)
        
        dist = Normal(mu, std)
        
        # 4. Critic 价值评估
        # 混合池化：Mean (全局平均势能) + Max (最严重的冲突/重叠)
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1)[0]
        h_global = torch.cat([h_mean, h_max], dim=-1)
        
        value = self.critic_head(h_global)
        
        return dist, value