import torch

class KissingLoss:
    def __init__(self, target_angle_deg=60.0):
        # 目标是夹角 >= 60度
        # cos(60) = 0.5
        # 如果你想算其他问题，可以修改这里
        import math
        self.threshold = math.cos(math.radians(target_angle_deg)) # 0.5

    def compute(self, x):
        """
        x: (Batch, N, dim) 已归一化的球坐标
        返回: 
            loss: (Batch,) 每个环境的损失
            max_cosine: (Batch,) 每个环境中最糟糕的那个夹角余弦
        """
        # 1. 批量计算 Gram 矩阵
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        gram_matrix = torch.bmm(x, x.transpose(1, 2))
        
        # 2. 移除对角线 (自己和自己永远是 1，不需要惩罚)
        # 创建一个跟 gram_matrix 一样的掩码
        batch_size, n_points, _ = x.shape
        eye = torch.eye(n_points, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. 计算违规 (Violations)
        # 我们只关心 > 0.5 的部分 (即夹角 < 60度)
        # 减去对角线的大数值(避免干扰)
        g_no_diag = gram_matrix - eye * 2.0 
        
        # 违规量：所有大于 0.5 的值
        violations = torch.relu(g_no_diag - self.threshold)
        
        # 4. 损失函数：平方惩罚 (比线性惩罚收敛更快)
        # 对每个环境求和
        loss = (violations ** 2).sum(dim=(1, 2))
        
        # 5. 记录统计信息 (每个Batch中最严重的重叠)
        max_cosine = g_no_diag.max(dim=1)[0].max(dim=1)[0]
        
        return loss, max_cosine