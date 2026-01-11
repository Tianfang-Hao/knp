import torch
import math

class KissingLoss:
    def __init__(self, default_target_deg=60.0):
        # 默认目标 (60度)
        self.default_threshold = math.cos(math.radians(default_target_deg))

    def compute(self, x, threshold=None):
        """
        x: (Batch, N, dim) 已归一化的球坐标
        threshold: 动态余弦阈值。如果为 None，使用默认值。
        """
        if threshold is None:
            threshold = self.default_threshold
            
        # 1. 批量计算 Gram 矩阵
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        gram_matrix = torch.bmm(x, x.transpose(1, 2))
        
        # 2. 移除对角线
        batch_size, n_points, _ = x.shape
        eye = torch.eye(n_points, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 减去对角线的大数值(避免干扰)
        g_no_diag = gram_matrix - eye * 2.0 
        
        # 3. 计算违规 (Violations)
        # 任何大于 threshold 的余弦值都是违规的（即角度太小）
        violations = torch.relu(g_no_diag - threshold)
        
        # 4. 损失函数：平方惩罚
        loss = (violations ** 2).sum(dim=(1, 2))
        
        # 5. 记录每个环境中最糟糕的那个夹角余弦 (用于判断成功)
        max_cosine = g_no_diag.max(dim=1)[0].max(dim=1)[0]
        
        return loss, max_cosine