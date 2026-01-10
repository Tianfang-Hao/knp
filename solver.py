import torch
import torch.optim as optim
import time
import os
import math
from datetime import datetime

from model import ActorCritic
from loss import KissingLoss
from env import KNPEnvironment
from ppo import PPOAgent
from logger import DetailedLogger
from utils import save_result

def get_required_success_rate(deg):
    """魔鬼训练版阈值"""
    if deg <= 40: return 0.80  
    elif deg >= 55: return 0.01
    else:
        ratio = (deg - 40) / (55 - 40)
        return 0.80 - ratio * (0.80 - 0.01)

def run_solver(args, gpu_id):
    # --- 1. 初始化 ---
    seed = int(time.time()) + gpu_id * 1000 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{gpu_id}")
    
    logger = DetailedLogger(args.save_dir, gpu_id)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 大模型
    model = ActorCritic(args.dim, hidden_dim=1024).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 加载模型
    date_str = datetime.now().strftime("%Y%m%d")
    model_name = f"model_{date_str}_gpu{gpu_id}.pth"
    model_path = os.path.join(models_dir, model_name)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from {model_path}")
        except: pass

    loss_fn = KissingLoss()
    env = KNPEnvironment(args, device, loss_fn)
    agent = PPOAgent(model, optimizer, entropy_coef=0.02)
    
    # --- 2. 课程配置 ---
    curriculum_goals = list(range(20, 61, 1)) 
    curr_idx = 0
    env.set_target_angle(curriculum_goals[curr_idx])
    
    rollout_steps = 128
    total_updates = args.max_steps // rollout_steps
    max_steps_per_episode = 300 
    env_steps_count = torch.zeros(args.batch_size, device=device)
    
    state = env.reset()
    
    level_finished_episodes = 0
    level_success_episodes = 0
    
    logger.info(f"--- 初始目标: {curriculum_goals[curr_idx]} 度 (要求胜率 > 80%) ---")
    
    for update_step in range(total_updates):
        model.eval()
        rollouts = []
        
        batch_finished_count = 0
        batch_success_count = 0
        
        # === 3. 采样循环 ===
        for step in range(rollout_steps):
            with torch.no_grad():
                dist, value = model(state)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=(-1, -2))
            
            next_state, reward, done, max_cos_vec = env.step(action)
            env_steps_count += 1
            
            is_timeout = (env_steps_count >= max_steps_per_episode)
            finished_mask = done | is_timeout
            finished_indices = torch.nonzero(finished_mask).squeeze(-1)
            
            mask = 1.0 - done.float()
            
            # 正常记录
            rollouts.append({
                'state': state, 'action': action, 'log_prob': log_prob,
                'reward': reward, 'value': value, 'mask': mask
            })
            
            # 处理结束环境
            if len(finished_indices) > 0:
                num_finished = len(finished_indices)
                real_success_mask = done[finished_indices]
                num_success = real_success_mask.sum().item()
                
                batch_finished_count += num_finished
                batch_success_count += num_success
                
                # 终极成功检查
                if num_success > 0:
                    real_succ_idx = finished_indices[real_success_mask]
                    ultimate_success = (max_cos_vec[real_succ_idx] <= 0.500001)
                    if ultimate_success.any():
                        final_idx = real_succ_idx[ultimate_success]
                        for idx in final_idx:
                            logger.info(f"*** ULTIMATE SUCCESS at update {update_step} ***")
                            # 这里可以不频繁save，或者仅打印
                            save_result(args.save_dir, f"solved_deg60_u{update_step}", 
                                        next_state[idx], args.num_points, args.dim, gpu_id)

                # 重置
                env.reset_indices(finished_indices)
                env_steps_count[finished_indices] = 0
                next_state[finished_indices] = env.state[finished_indices]

            state = next_state
            
        # === 4. 进阶判定 ===
        level_finished_episodes += batch_finished_count
        level_success_episodes += batch_success_count
        
        current_success_rate = 0.0
        if level_finished_episodes > 0:
            current_success_rate = level_success_episodes / level_finished_episodes
            
        current_deg = curriculum_goals[curr_idx]
        required_rate = get_required_success_rate(current_deg)
        
        # 严格条件: 必须样本足够 且 胜率达标
        if level_finished_episodes >= 100 and current_success_rate >= required_rate:
            if curr_idx < len(curriculum_goals) - 1:
                curr_idx += 1
                new_goal = curriculum_goals[curr_idx]
                env.set_target_angle(new_goal)
                
                logger.info(f"--- LEVEL UP! {current_deg} -> {new_goal} 度 | 真实胜率 {current_success_rate*100:.1f}% ---")
                
                level_finished_episodes = 0
                level_success_episodes = 0
        
        # === 5. PPO Update (Oversampling 在内部处理) ===
        model.train()
        loss, v_loss, entropy = agent.update(rollouts)
        
        if update_step % 10 == 0:
            current_std = model.actor_log_std.exp().mean().item()
            logger.info(
                f"Upd {update_step} | Goal {current_deg}deg | "
                f"Rate {current_success_rate*100:.1f}% (Req {required_rate*100:.0f}%) | "
                f"Ent {entropy:.2f} | Std {current_std:.3f}"
            )
            
        if update_step % 1000 == 0:
            try:
                tmp_path = model_path + ".tmp"
                torch.save(model.state_dict(), tmp_path)
                os.replace(tmp_path, model_path)
            except: pass
            
    logger.close()