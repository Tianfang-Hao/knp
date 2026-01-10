import torch
import torch.nn as nn

class PPOAgent:
    def __init__(self, model, optimizer, gamma=0.99, gae_lambda=0.95, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def update(self, rollouts):
        # 1. 整理数据 (保持时间维度，计算 GAE)
        states = torch.stack([r['state'] for r in rollouts]) # (T, B, ...)
        actions = torch.stack([r['action'] for r in rollouts])
        old_log_probs = torch.stack([r['log_prob'] for r in rollouts]).detach()
        rewards = torch.stack([r['reward'] for r in rollouts]) 
        old_values = torch.stack([r['value'] for r in rollouts]).squeeze(-1).detach()
        masks = torch.stack([r['mask'] for r in rollouts])

        # 2. GAE 计算
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rollouts))):
            if t == len(rollouts) - 1:
                next_non_terminal = 1.0
                next_value = old_values[t] 
            else:
                next_non_terminal = masks[t+1]
                next_value = old_values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + old_values
        
        # 归一化 Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 3. 展平数据 (Flatten) -> (Total_Samples, ...)
        # 此时 Batch 维度和 Time 维度混合，不再区分
        flat_states = states.view(-1, *states.shape[2:])
        flat_actions = actions.view(-1, *actions.shape[2:])
        flat_log_probs = old_log_probs.view(-1)
        flat_returns = returns.view(-1)
        flat_advantages = advantages.view(-1)
        flat_rewards = rewards.view(-1) # 用来筛选高分样本
        
        # === 4. 关键修改：成功样本增强 (Oversampling) ===
        # 在展平的数据中，找出 Reward 极高（说明成功了）的样本
        # 阈值 20.0 是安全的，因为成功奖励至少是 50.0
        success_indices = torch.nonzero(flat_rewards > 20.0).squeeze(-1)
        
        if len(success_indices) > 0:
            # 复制这些成功样本 N 次 (例如 3 次)
            repeat_times = 3
            
            aug_states = flat_states[success_indices].repeat(repeat_times, 1, 1)
            aug_actions = flat_actions[success_indices].repeat(repeat_times, 1, 1)
            aug_log_probs = flat_log_probs[success_indices].repeat(repeat_times)
            aug_returns = flat_returns[success_indices].repeat(repeat_times)
            aug_advantages = flat_advantages[success_indices].repeat(repeat_times)
            
            # 拼接到训练集中
            flat_states = torch.cat([flat_states, aug_states], dim=0)
            flat_actions = torch.cat([flat_actions, aug_actions], dim=0)
            flat_log_probs = torch.cat([flat_log_probs, aug_log_probs], dim=0)
            flat_returns = torch.cat([flat_returns, aug_returns], dim=0)
            flat_advantages = torch.cat([flat_advantages, aug_advantages], dim=0)
            
        # 5. PPO Update
        # 全量更新 (A100 显存够大，直接算)
        dist, values = self.model(flat_states)
        values = values.squeeze(-1)
        
        new_log_probs = dist.log_prob(flat_actions).sum(dim=(-1, -2))
        
        ratio = torch.exp(new_log_probs - flat_log_probs)
        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * flat_advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (flat_returns - values).pow(2).mean()
        entropy = dist.entropy().mean()
        
        loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), value_loss.item(), entropy.item()