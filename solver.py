import torch
import torch.optim as optim
import time
import os
import math
from datetime import datetime

# DDP & Distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, ReduceOp

from model import ActorCritic
from loss import KissingLoss
from env import KNPEnvironment
from ppo import PPOAgent
from logger import DetailedLogger
from utils import save_result

def get_required_success_rate(deg):
    if deg <= 40: return 0.80  
    elif deg >= 55: return 0.01
    else:
        ratio = (deg - 40) / (55 - 40)
        return 0.80 - ratio * (0.80 - 0.01)

def refine_solution(state, loss_fn, steps=100, lr=0.01):
    """
    SGD Refinement: 局部切换到 FP64 (Double) 进行高精度微调
    [优化] 默认步数从 1000 降为 100，避免长时间阻塞训练
    """
    
    # === 关键修改：转换为 double 精度 ===
    # 训练用 float32 跑得快，最后微调用 float64 算得准
    x = state.clone().detach().double().requires_grad_(True)
    
    optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
    
    best_x = None
    min_max_cos = 1.0
    
    for i in range(steps):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        # loss_fn 计算时会自动广播类型
        # 注意：为了确保最佳精度，建议检查 loss.py 中的 eye 矩阵是否匹配 dtype，
        # 但通常 PyTorch 会自动处理 float32 eye 和 double x 的运算。
        loss, max_cos = loss_fn.compute(x_norm.unsqueeze(0)) 
        loss = loss.sum()
        
        current_max_cos = max_cos.item()
        if current_max_cos < min_max_cos:
            min_max_cos = current_max_cos
            best_x = x_norm.detach().clone() # 保持 double
            
        # 严格的 60度 阈值 (cos(60) = 0.5)
        # 容差设为 1e-9
        if current_max_cos <= 0.500000001: 
            return best_x, True
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            x.data = torch.nn.functional.normalize(x.data, p=2, dim=-1)
            
    return best_x, (min_max_cos <= 0.5000001)

def run_solver(rank, args):
    seed = int(time.time()) + rank * 10000 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device(f"cuda:{rank}")
    
    logger = None
    if rank == 0:
        logger = DetailedLogger(args.save_dir, rank)
        os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)
    
    # 模型初始化 (默认 Float32)
    raw_model = ActorCritic(args.dim, hidden_dim=512).to(device)
    
    # === 性能优化: 编译模型 ===
    try:
        raw_model = torch.compile(raw_model)
    except:
        if rank == 0: print("Warning: torch.compile failed, using eager mode.")

    model = DDP(raw_model, device_ids=[rank], find_unused_parameters=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = KissingLoss()
    env = KNPEnvironment(args, device, loss_fn)
    
    # PPO Agent
    # 注意：mini_batch_size=4096 比较大，请确保 run.sh 中的 --batch_size 设置为 4096 或更大，
    # 否则 mini_batch 可能会大于采集的数据总量，导致报错或更新效率极低。
    agent = PPOAgent(model, optimizer, 
                    entropy_coef=0.01, 
                    ppo_epoch=4, 
                    # [修改前] 4096
                    # [修改后] 改为 16384 或 32768 (A100/4090 可以设更大)
                    mini_batch_size=16384)
    
    # [优化] 课程目标从 40 度开始，跳过过简单的阶段
    curriculum_goals = list(range(25, 61, 1)) 
    curr_idx = 0
    env.set_target_angle(curriculum_goals[curr_idx])
    
    rollout_steps = 32
    total_updates = args.max_steps // rollout_steps
    max_steps_per_episode = 300
    
    state = env.reset()
    env_steps_count = torch.zeros(args.batch_size, device=device)
    
    level_finished = torch.tensor(0.0, device=device)
    level_success = torch.tensor(0.0, device=device)
    
    if rank == 0:
        logger.info(f"--- 初始目标: {curriculum_goals[curr_idx]} 度 ---")
    
    for update_step in range(total_updates):
        model.eval()
        rollouts = []
        
        batch_finished = torch.tensor(0.0, device=device)
        batch_success = torch.tensor(0.0, device=device)
        
        for step in range(rollout_steps):
            with torch.no_grad():
                dist, value = model(state)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=(-1, -2))
            
            next_state, reward, done, max_cos_vec = env.step(action)
            
            env_steps_count += 1
            is_timeout = (env_steps_count >= max_steps_per_episode)
            finished_mask = done | is_timeout
            mask = 1.0 - done.float()
            
            rollouts.append({
                'state': state, 'action': action, 'log_prob': log_prob,
                'reward': reward, 'value': value, 'mask': mask
            })
            
            finished_indices = torch.nonzero(finished_mask).squeeze(-1)
            
            if len(finished_indices) > 0:
                num_finished = len(finished_indices)
                real_success_mask = done[finished_indices]
                num_success = real_success_mask.sum()
                
                batch_finished += num_finished
                batch_success += num_success
                
                # === 终极成功微调 ===
                if num_success > 0 and curriculum_goals[curr_idx] >= 50:
                    success_indices = finished_indices[real_success_mask]
                    # [优化] 限制数量，并且使用 reduced steps=100
                    for idx in success_indices[:5]: 
                        refined_x, is_perfect = refine_solution(next_state[idx], loss_fn, steps=100)
                        if is_perfect and rank == 0:
                            logger.info(f"*** FOUND PERFECT SOLUTION (Refined)! ***")
                            save_result(args.save_dir, f"solved_perfect_u{update_step}", 
                                        refined_x.float(), args.num_points, args.dim, rank)

                env.reset_indices(finished_indices)
                next_state[finished_indices] = env.state[finished_indices]
                env_steps_count[finished_indices] = 0

            state = next_state
        
        with torch.no_grad():
            _, next_value = model(state)
            next_value = next_value.squeeze(-1)
            
        all_reduce(batch_finished, op=ReduceOp.SUM)
        all_reduce(batch_success, op=ReduceOp.SUM)
        
        level_finished += batch_finished
        level_success += batch_success
        
        current_rate = 0.0
        if level_finished > 0:
            current_rate = (level_success / level_finished).item()
            
        required_rate = get_required_success_rate(curriculum_goals[curr_idx])
        
        if level_finished >= 10000:
            if current_rate >= required_rate:
                if curr_idx < len(curriculum_goals) - 1:
                    curr_idx += 1
                    new_goal = curriculum_goals[curr_idx]
                    env.set_target_angle(new_goal)
                    if rank == 0:
                        logger.info(f"--- LEVEL UP! {new_goal} 度 | Rate {current_rate*100:.2f}% ---")
            
            level_finished.zero_()
            level_success.zero_()
        
        model.train()
        loss, v_loss, entropy = agent.update(rollouts, next_value)
        
        if rank == 0 and update_step % 1 == 0:
            std = model.module.actor_log_std.exp().mean().item()
            logger.info(
                f"Step {update_step} | Goal {curriculum_goals[curr_idx]} | "
                f"Rate {current_rate*100:.1f}% | Loss {loss:.3f} | Ent {entropy:.2f} | Std {std:.3f}"
            )

    if rank == 0:
        logger.close()