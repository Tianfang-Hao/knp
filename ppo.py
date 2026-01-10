import torch
import torch.nn as nn

class PPOAgent:
    def __init__(self, model, optimizer, gamma=0.99, gae_lambda=0.95, 
                 clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 ppo_epoch=10, mini_batch_size=4096):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size

    def update(self, rollouts, last_value_prediction):
        """
        last_value_prediction: (Batch,) 最后一个状态的价值估计，用于 GAE Bootstrap
        """
        # 1. 整理数据
        states = torch.stack([r['state'] for r in rollouts]) 
        actions = torch.stack([r['action'] for r in rollouts])
        old_log_probs = torch.stack([r['log_prob'] for r in rollouts]).detach()
        rewards = torch.stack([r['reward'] for r in rollouts]) 
        old_values = torch.stack([r['value'] for r in rollouts]).squeeze(-1).detach()
        masks = torch.stack([r['mask'] for r in rollouts])

        # 2. GAE 计算 (修正版)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rollouts))):
            if t == len(rollouts) - 1:
                next_non_terminal = 1.0
                next_value = last_value_prediction # 使用传入的真实 Next Value
            else:
                next_non_terminal = masks[t+1]
                next_value = old_values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 3. Flatten
        flat_states = states.view(-1, *states.shape[2:])
        flat_actions = actions.view(-1, *actions.shape[2:])
        flat_log_probs = old_log_probs.view(-1)
        flat_returns = returns.view(-1)
        flat_advantages = advantages.view(-1)
        
        # 4. PPO Update Loop
        total_samples = flat_states.size(0)
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_counts = 0
        
        for epoch in range(self.ppo_epoch):
            indices = torch.randperm(total_samples, device=flat_states.device)
            
            for start in range(0, total_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                
                mb_states = flat_states[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]
                
                dist, values = self.model(mb_states)
                values = values.squeeze(-1)
                
                new_log_probs = dist.log_prob(mb_actions).sum(dim=(-1, -2))
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy = dist.entropy().mean()
                
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_counts += 1
        
        return (total_actor_loss/update_counts, 
                total_value_loss/update_counts, 
                total_entropy/update_counts)