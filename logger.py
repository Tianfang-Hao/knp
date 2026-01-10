import os
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import time

class DetailedLogger:
    def __init__(self, base_dir, gpu_id, run_name=None):
        self.gpu_id = gpu_id
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = run_name if run_name else f"run_{timestamp}"
        
        # 1. 路径设置
        # log_dir: 存放 TensorBoard 数据
        # txt_path: 存放纯文本日志
        self.log_dir = os.path.join(base_dir, "tensorboard", f"gpu{gpu_id}_{run_name}")
        self.txt_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.txt_dir, exist_ok=True)
        
        # 2. 初始化 TensorBoard Writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 3. 初始化 Python Logger (用于存文本文件)
        self.logger = logging.getLogger(f"GPU_{gpu_id}")
        self.logger.setLevel(logging.INFO)
        # 防止重复添加 handler
        if not self.logger.handlers:
            # 文件 Handler
            fh = logging.FileHandler(os.path.join(self.txt_dir, f"gpu{gpu_id}.log"))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)
            # 控制台 Handler (可选，如果 main.py 已经打印了可以关掉)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(f'[GPU {gpu_id}] %(message)s'))
            self.logger.addHandler(ch)

    def log_scalar(self, tag, value, step):
        """记录标量曲线 (如 Loss, Learning Rate)"""
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        """记录直方图 (如 所有球的夹角分布)"""
        self.writer.add_histogram(tag, values, step)
    
    def info(self, msg):
        """打印文本日志"""
        self.logger.info(msg)
        
    def close(self):
        self.writer.close()