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
from utils import save_result, normalize_sphere

def get_required_success_rate(deg):
    if deg <= 40: return 0.80  
    elif deg >= 55: return 0.01
    else:
        ratio = (deg - 40) / (55 - 40)
        return 0.80 - ratio * (0.80 - 0.01)

def refine_solution(state, _, steps=100, lr=0.01):
    """SGD Refinement: 强制使用最终目标 (60度) 进行高精度微调"""
    target_loss_fn = KissingLoss(target_angle_deg=60.0)
    
    x = state.clone().detach().double().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
    best_x = None
    min_max_cos = 1.0
    
    for i in range(steps):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        loss, max_cos = target_loss_fn.compute(x_norm.unsqueeze(0)) 
        loss = loss.sum()
        
        current_max_cos = max_cos.item()
        if current_max_cos < min_max_cos:
            min_max_cos = current_max_cos
            best_x = x_norm.detach().clone()
            
        if current_max_cos <= 0.500000001: 
            return best_x, True
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            x.data = torch.nn.functional.normalize(x.data, p=2, dim=-1)
            
    return best_x, (min_max_cos <= 0.5000001)

def run_sft(rank, args, model, optimizer, device, logger):
    """SFT 改良版：行为克隆 (Behavior Cloning)"""
    if args.sft_steps <= 0: return
    sft_target = args.sft_target_deg
    if rank == 0:
        logger.info(f"=== 启动 SFT (行为克隆模式) | Steps: {args.sft_steps} | Target: {sft_target}° ===")
    
    sft_loss_fn = KissingLoss(target_angle_deg=sft_target)
    model.train()
    t0 = time.time()
    
    state = torch.randn(args.batch_size, args.num_points, args.dim, device=device)
    state = normalize_sphere(state)
    
    for step in range(args.sft_steps):
        state.requires_grad_(True)
        loss_physics, max_cos_vec = sft_loss_fn.compute(state)
        grad_sum = loss_physics.sum()
        grad_x = torch.autograd.grad(grad_sum, state)[0]
        state.requires_grad_(False)
        
        grad_norm = torch.norm(grad_x, dim=-1, keepdim=True) + 1e-8
        target_dir = -grad_x / grad_norm
        target_action = target_dir * 0.1 
        
        dist, value = model(state)
        pred_action = dist.loc 
        
        loss_bc = torch.nn.functional.mse_loss(pred_action, target_action)
        loss_bc += 0.0 * value.sum() + 0.0 * dist.scale.sum()
        
        optimizer.zero_grad()
        loss_bc.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            state = normalize_sphere(state + target_action)
            
        if step % 500 == 0 and step > 0:
            state = torch.randn(args.batch_size, args.num_points, args.dim, device=device)
            state = normalize_sphere(state)

        if rank == 0 and step % 200 == 0:
            avg_cos = max_cos_vec.mean().item()
            avg_deg = math.degrees(math.acos(min(avg_cos, 1.0)))
            logger.info(f"[SFT-BC] Step {step:4d} | MSE Loss {loss_bc.item():.6f} | Physics Conf {avg_deg:.1f}°")

    if rank == 0: logger.info(">>> SFT 完成，重置 Optimizer 并降低噪声...")
    optimizer.state.clear()
    with torch.no_grad():
        model.module.actor_log_std.fill_(-3.0)

def run_solver(rank, args):
    seed = int(time.time()) + rank * 10000 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{rank}")
    
    logger = None
    if rank == 0:
        logger = DetailedLogger(args.save_dir, rank)
        os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)
    
    raw_model = ActorCritic(args.dim, hidden_dim=512).to(device)
    try:
        raw_model = torch.compile(raw_model)
    except:
        if rank == 0: print("Warning: torch.compile failed, using eager mode.")

    model = DDP(raw_model, device_ids=[rank], find_unused_parameters=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = KissingLoss()
    env = KNPEnvironment(args, device, loss_fn)
    
    # 1. SFT
    if args.sft_steps > 0:
        run_sft(rank, args, model, optimizer, device, logger)

    # 2. PPO Init
    rollout_steps = 32
    total_buffer_size = args.batch_size * rollout_steps
    calculated_mini_batch = 4096 
    
    if rank == 0:
        logger.info(f"PPO Configuration: Buffer {total_buffer_size} | MiniBatch {calculated_mini_batch}")

    agent = PPOAgent(model, optimizer, 
                    entropy_coef=0.01, 
                    ppo_epoch=4, 
                    mini_batch_size=calculated_mini_batch)
    
    # 3. RL Loop
    curriculum_goals = list(range(25, 61, 1)) 
    curr_idx = 0
    
    # [优化] 自动跳过过于简单的课程，从接近 SFT 目标的难度开始
    if args.sft_steps > 0:
        start_goal = max(25, args.sft_target_deg - 5.0)
        for i, g in enumerate(curriculum_goals):
            if g >= start_goal:
                curr_idx = i
                break
    
    env.set_target_angle(curriculum_goals[curr_idx])
    
    total_updates = args.max_steps // rollout_steps
    max_steps_per_episode = 300
    state = env.reset()
    env_steps_count = torch.zeros(args.batch_size, device=device)
    level_finished = torch.tensor(0.0, device=device)
    level_success = torch.tensor(0.0, device=device)
    
    if rank == 0:
        logger.info(f"--- RL 初始目标: {curriculum_goals[curr_idx]} 度 ---")
    
    for update_step in range(total_updates):
        # === Warmup 判断 ===
        is_warmup = (update_step < 100)
        
        if rank == 0:
            if update_step == 0:
                logger.info("[Warmup] Freezing Actor parameters (via Zero-Grad). Training ONLY Critic for 100 steps...")
            elif update_step == 100:
                logger.info("[Warmup] Warmup done. Full PPO training starts...")

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
                
                if num_success > 0 and curriculum_goals[curr_idx] >= 50:
                    success_indices = finished_indices[real_success_mask]
                    for idx in success_indices[:3]: 
                        refined_x, is_perfect = refine_solution(next_state[idx], loss_fn, steps=100)
                        if is_perfect:
                            if rank == 0:
                                logger.info(f"*** FOUND PERFECT SOLUTION (Refined) on Rank {rank}! ***")
                            save_result(args.save_dir, f"solved_perfect_u{update_step}_r{rank}", 
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
        if level_finished >= 20000:
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
        
        # === 调用 PPO Update (带 Warmup 参数) ===
        # 如果 is_warmup 为 True，则 update_actor 为 False，Actor 梯度为 0
        with torch.no_grad():
            _, next_value = model(state)
            next_value = next_value.squeeze(-1)

        # 2. 传递 next_value 给 update 函数
        # 注意：如果你保留了之前提到的 update_actor 参数，这里也需要根据情况传递
        loss, v_loss, entropy = agent.update(rollouts, next_value)
        if rank == 0 and update_step % 1 == 0:
            std = model.module.actor_log_std.exp().mean().item()
            logger.info(
                f"Step {update_step} | Goal {curriculum_goals[curr_idx]} | Rate {current_rate*100:.1f}% | "
                f"ActLoss {loss:.3f} | ValLoss {v_loss:.3f} | Ent {entropy:.2f} | Std {std:.3f}"
            )

    if rank == 0:
        logger.close()