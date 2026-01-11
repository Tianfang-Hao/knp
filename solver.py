import torch
import torch.optim as optim
import time
import os
import math
from datetime import datetime

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, ReduceOp

from model import ActorCritic
from loss import KissingLoss
from env import KNPEnvironment
from ppo import PPOAgent
from logger import DetailedLogger
from utils import save_result, normalize_sphere

def get_required_success_rate(deg):
    if deg <= 40: return 0.85 
    elif deg >= 55: return 0.005
    else:
        ratio = (deg - 40) / (55 - 40)
        return 0.85 - ratio * (0.85 - 0.005)

def refine_solution(state, loss_fn, steps=200, lr=0.01):
    """SGD Refinement: 强制使用最终目标 (60度/0.5) 进行微调"""
    final_threshold = 0.5 
    
    x = state.clone().detach().double().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
    best_x = None
    min_max_cos = 1.0
    
    for i in range(steps):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        loss, max_cos = loss_fn.compute(x_norm.unsqueeze(0), threshold=final_threshold)
        loss = loss.sum()
        
        current_max_cos = max_cos.item()
        if current_max_cos < min_max_cos:
            min_max_cos = current_max_cos
            best_x = x_norm.detach().clone().float()
            
        if current_max_cos <= 0.500000001: 
            return best_x, True
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return best_x, (min_max_cos <= 0.5000001)

def run_sft(rank, args, model, optimizer, device, logger):
    """SFT (Behavior Cloning)"""
    if args.sft_steps <= 0: return
    sft_target = args.sft_target_deg
    if rank == 0:
        logger.info(f"=== 启动 SFT | Steps: {args.sft_steps} | Target: {sft_target}° ===")
    
    sft_threshold = math.cos(math.radians(sft_target))
    loss_fn = KissingLoss()
    
    model.train()
    
    state = torch.randn(args.batch_size, args.num_points, args.dim, device=device)
    state = normalize_sphere(state)
    
    for step in range(args.sft_steps):
        state.requires_grad_(True)
        loss_physics, max_cos_vec = loss_fn.compute(state, threshold=sft_threshold)
        grad_sum = loss_physics.sum()
        grad_x = torch.autograd.grad(grad_sum, state)[0]
        state.requires_grad_(False)
        
        grad_norm = torch.norm(grad_x, dim=-1, keepdim=True) + 1e-8
        target_dir = -grad_x / grad_norm
        target_action = target_dir * 0.1 
        
        dist, value = model(state)
        pred_action = dist.loc 
        
        loss_bc = torch.nn.functional.mse_loss(pred_action, target_action)
        
        # [Fix] 关键修改：加上 0 * value.sum()
        # 这会让 Critic 的参数参与反向传播图（梯度为0），解决 DDP 报错 "parameters not used"
        loss_bc += 0.0 * value.sum()
        
        optimizer.zero_grad()
        loss_bc.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            state = normalize_sphere(state + target_action)
            
        if step % 500 == 0 and step > 0:
            state = torch.randn(args.batch_size, args.num_points, args.dim, device=device)
            state = normalize_sphere(state)

        if rank == 0 and step % 100 == 0:
            avg_cos = max_cos_vec.mean().item()
            avg_deg = math.degrees(math.acos(min(avg_cos, 1.0)))
            logger.info(f"[SFT] Step {step} | Loss {loss_bc.item():.5f} | Phys {avg_deg:.1f}°")

    if rank == 0: logger.info(">>> SFT 完成，重置 Optimizer ...")
    optimizer.state.clear()
    with torch.no_grad():
        model.module.actor_log_std.fill_(-0.5)

def run_solver(rank, args):
    seed = int(time.time()) + rank * 10000 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{rank}")
    
    logger = None
    if rank == 0:
        logger = DetailedLogger(args.save_dir, rank)
        os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)
    
    raw_model = ActorCritic(
        args.dim, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers,
        nhead=args.nhead
    ).to(device)
    
    try:
        raw_model = torch.compile(raw_model)
    except Exception as e:
        if rank == 0: print(f"Warning: torch.compile failed: {e}")

    # 不需要 find_unused_parameters=True，因为我们在 SFT 里用了 hack
    model = DDP(raw_model, device_ids=[rank], find_unused_parameters=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = KissingLoss()
    env = KNPEnvironment(args, device, loss_fn)
    
    # 1. SFT
    run_sft(rank, args, model, optimizer, device, logger)

    # 2. PPO Agent
    rollout_steps = 64
    total_buffer_size = args.batch_size * rollout_steps
    mini_batch_size = min(total_buffer_size, 8192) 

    agent = PPOAgent(model, optimizer, 
                    entropy_coef=0.01, 
                    ppo_epoch=4, 
                    mini_batch_size=mini_batch_size)
    
    # 3. RL Loop
    curriculum_goals = list(range(25, 61, 1)) 
    curr_idx = 0
    
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
        logger.info(f"--- RL Start Goal: {curriculum_goals[curr_idx]} deg ---")
    
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
                
                if num_success > 0 and curriculum_goals[curr_idx] >= 50:
                    success_indices = finished_indices[real_success_mask]
                    for idx in success_indices[:2]: 
                        refined_x, is_perfect = refine_solution(next_state[idx], loss_fn, steps=200)
                        if is_perfect:
                            if rank == 0:
                                logger.info(f"*** FOUND PERFECT SOLUTION (Rank {rank}) ***")
                            save_result(args.save_dir, f"solved_refine_u{update_step}_r{rank}", 
                                        refined_x, args.num_points, args.dim, rank)

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
        
        if level_finished >= 10000:
            current_rate = (level_success / level_finished).item()
            required_rate = get_required_success_rate(curriculum_goals[curr_idx])
            if current_rate >= required_rate:
                if curr_idx < len(curriculum_goals) - 1:
                    curr_idx += 1
                    new_goal = curriculum_goals[curr_idx]
                    env.set_target_angle(new_goal)
                    if rank == 0:
                        logger.info(f"--- LEVEL UP! {new_goal} deg | Rate {current_rate*100:.2f}% ---")
            level_finished.zero_()
            level_success.zero_()
        
        model.train()
        loss, v_loss, entropy = agent.update(rollouts, next_value)
        if rank == 0 and update_step % 2 == 0:
            std = model.module.actor_log_std.exp().mean().item()
            logger.info(
                f"Step {update_step} | Goal {curriculum_goals[curr_idx]} | "
                f"Loss {loss:.3f}/{v_loss:.3f} | Ent {entropy:.3f} | Std {std:.3f}"
            )

    if rank == 0:
        logger.close()