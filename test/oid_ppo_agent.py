"""
OID-PPO Training Agent  -  FIXED VERSION
Key fixes:
  1. Much lower learning rates (3e-5 / 3e-4) to stop loss explosion
  2. Cosine-annealing LR scheduler to decay over training
  3. Tighter gradient clipping (0.2 instead of 0.5)
  4. Mini-batch PPO updates (batch_size=16) instead of per-step updates
  5. Entropy bonus decay so exploration narrows over time
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class PPOAgent:
    def __init__(self,
                 network: nn.Module,
                 env,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 epsilon: float = 0.2,
                 lr_actor: float = 3e-5,       # FIX 1: was 1e-4, too high
                 lr_critic: float = 3e-4,       # FIX 1: was 1e-3, too high
                 c_v: float = 0.5,
                 c_e: float = 0.01,
                 update_epochs: int = 4,
                 batch_size: int = 16,          # FIX 4: mini-batch size
                 n_episodes: int = 1000,        # needed for scheduler
                 device: str = 'cpu'):

        self.network = network.to(device)
        self.env = env
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c_v = c_v
        self.c_e = c_e
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.n_episodes = n_episodes

        # Separate param groups for actor / critic learning rates
        actor_params = (
            list(network.furniture_encoder.parameters()) +
            list(network.occupancy_encoder.parameters()) +
            list(network.shared_fc.parameters()) +
            list(network.actor_mean.parameters()) +
            list(network.actor_log_std.parameters())
        )
        self.optimizer = optim.Adam(
            [
                {'params': actor_params,             'lr': lr_actor},
                {'params': network.critic.parameters(), 'lr': lr_critic}
            ],
            betas=(0.9, 0.999), eps=1e-8
        )

        # FIX 2: cosine annealing decays LR smoothly to near-zero
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_episodes, eta_min=1e-7
        )

        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    # ── trajectory collection ──────────────────────────────────────────────────
    def collect_trajectory(self) -> Dict:
        trajectory = {
            'states': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': []
        }

        state = self.env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            state_tensor = self._state_to_tensor(state)

            with torch.no_grad():
                action, log_prob, value = self.network.get_action(
                    state_tensor, deterministic=False
                )

            action_np = action.cpu().numpy()[0]

            # Sigmoid rescaling so the centre is always inside the room
            # (fixes the "all placements land at (0,0)" bug)
            furniture = self.env.furniture_list[self.env.current_step]
            rot = int(np.round(action_np[2])) % 4
            if rot % 2 == 0:
                half_l, half_w = furniture.length / 2, furniture.width / 2
            else:
                half_l, half_w = furniture.width / 2, furniture.length / 2

            sig_x = 1.0 / (1.0 + np.exp(-action_np[0]))
            sig_y = 1.0 / (1.0 + np.exp(-action_np[1]))

            action_np[0] = half_l + sig_x * (self.env.N - 2 * half_l)
            action_np[1] = half_w + sig_y * (self.env.M - 2 * half_w)
            action_np[2] = np.clip(action_np[2], 0, 3)

            next_state, reward, done, info = self.env.step(action_np)

            trajectory['states'].append(state)
            trajectory['actions'].append(action_np)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value.cpu().item())
            trajectory['log_probs'].append(log_prob.cpu().item())
            trajectory['dones'].append(done)

            episode_reward += reward
            step_count += 1
            state = next_state

        trajectory['actions']   = np.array(trajectory['actions'])
        trajectory['rewards']   = np.array(trajectory['rewards'])
        trajectory['values']    = np.array(trajectory['values'])
        trajectory['log_probs'] = np.array(trajectory['log_probs'])
        trajectory['dones']     = np.array(trajectory['dones'])

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(step_count)
        return trajectory

    # ── GAE ───────────────────────────────────────────────────────────────────
    def compute_gae(self, trajectory: Dict) -> Tuple[np.ndarray, np.ndarray]:
        rewards = trajectory['rewards']
        values  = trajectory['values']

        advantages = np.zeros_like(rewards)
        last_gae   = 0.0

        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta         = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * last_gae

        returns = advantages + values
        return advantages, returns

    # ── PPO update with mini-batches ───────────────────────────────────────────
    def update_policy(self, trajectory: Dict):
        advantages, returns = self.compute_gae(trajectory)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states       = trajectory['states']
        actions      = torch.FloatTensor(trajectory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).to(self.device)
        adv_tensor   = torch.FloatTensor(advantages).to(self.device)
        ret_tensor   = torch.FloatTensor(returns).to(self.device)

        n_steps = len(states)
        indices = np.arange(n_steps)

        epoch_policy_loss = 0.0
        epoch_value_loss  = 0.0
        epoch_entropy     = 0.0
        n_updates = 0

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)

            # FIX 4: process in mini-batches instead of one-by-one
            for start in range(0, n_steps, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]

                # Build batched state tensors
                batch_cur  = torch.FloatTensor(
                    np.stack([states[i]['current_furniture'] for i in batch_idx])
                ).to(self.device)
                batch_next = torch.FloatTensor(
                    np.stack([states[i]['next_furniture'] for i in batch_idx])
                ).to(self.device)
                batch_occ  = torch.FloatTensor(
                    np.stack([states[i]['occupancy_map'] for i in batch_idx])
                ).unsqueeze(1).to(self.device)

                batch_state = {
                    'current_furniture': batch_cur,
                    'next_furniture':    batch_next,
                    'occupancy_map':     batch_occ
                }

                batch_actions      = actions[batch_idx]
                batch_old_lp       = old_log_probs[batch_idx]
                batch_adv          = adv_tensor[batch_idx]
                batch_ret          = ret_tensor[batch_idx]

                log_prob, value, entropy = self.network.evaluate_actions(
                    batch_state, batch_actions
                )

                ratio = torch.exp(log_prob - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((value.squeeze() - batch_ret) ** 2).mean()

                # FIX 5: decay entropy coefficient so exploration narrows
                ep_frac = len(self.episode_rewards) / max(self.n_episodes, 1)
                c_e_eff = self.c_e * max(0.1, 1.0 - ep_frac)

                loss = policy_loss + self.c_v * value_loss - c_e_eff * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # FIX 3: tighter gradient clip
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.2)
                self.optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss  += value_loss.item()
                epoch_entropy     += entropy.mean().item()
                n_updates         += 1

        # FIX 2: step the LR scheduler once per episode
        self.scheduler.step()

        n_updates = max(n_updates, 1)
        self.policy_losses.append(epoch_policy_loss / n_updates)
        self.value_losses.append(epoch_value_loss  / n_updates)
        self.entropy_losses.append(epoch_entropy   / n_updates)

    # ── single episode ─────────────────────────────────────────────────────────
    def train_episode(self) -> Dict:
        trajectory = self.collect_trajectory()
        self.update_policy(trajectory)

        info = {
            'reward':       self.episode_rewards[-1],
            'length':       self.episode_lengths[-1],
            'policy_loss':  self.policy_losses[-1]  if self.policy_losses  else 0,
            'value_loss':   self.value_losses[-1]   if self.value_losses   else 0,
            'entropy':      self.entropy_losses[-1] if self.entropy_losses else 0,
            'valid':        trajectory['rewards'][-1] > -5
        }
        return info

    # ── helpers ────────────────────────────────────────────────────────────────
    def _state_to_tensor(self, state: Dict) -> Dict:
        return {
            'current_furniture': torch.FloatTensor(
                state['current_furniture']).unsqueeze(0).to(self.device),
            'next_furniture':    torch.FloatTensor(
                state['next_furniture']).unsqueeze(0).to(self.device),
            'occupancy_map':     torch.FloatTensor(
                state['occupancy_map']).unsqueeze(0).to(self.device)
        }

    def get_statistics(self, window: int = 100) -> Dict:
        if not self.episode_rewards:
            return {}
        recent_rewards      = self.episode_rewards[-window:]
        recent_policy_losses = self.policy_losses[-window:] if self.policy_losses else [0]
        recent_value_losses  = self.value_losses[-window:]  if self.value_losses  else [0]
        return {
            'mean_reward':       np.mean(recent_rewards),
            'std_reward':        np.std(recent_rewards),
            'max_reward':        np.max(recent_rewards),
            'mean_policy_loss':  np.mean(recent_policy_losses),
            'mean_value_loss':   np.mean(recent_value_losses),
            'total_episodes':    len(self.episode_rewards),
            'current_lr_actor':  self.scheduler.get_last_lr()[0]
        }

    def save(self, filepath: str):
        torch.save({
            'network_state_dict':   self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode_rewards':      self.episode_rewards,
            'episode_lengths':      self.episode_lengths,
            'policy_losses':        self.policy_losses,
            'value_losses':         self.value_losses,
        }, filepath)
        print(f"Saved agent to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.policy_losses   = checkpoint['policy_losses']
        self.value_losses    = checkpoint['value_losses']
        print(f"Loaded agent from {filepath}")
