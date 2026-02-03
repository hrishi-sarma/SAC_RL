import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from furniture_env import FurnitureRecommendationEnvEnhanced
from sac_agent import RuleGuidedSAC


class FurnitureRecommendationTrainerEnhanced:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15,
        save_dir: str = 'models_enhanced'
    ):
        self.env = FurnitureRecommendationEnvEnhanced(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=grid_size,
            collision_buffer=collision_buffer
        )
        
        print("Enhanced Environment Settings:")
        print(f"  Grid Size: {grid_size}m ({grid_size*100:.0f}cm)")
        print(f"  Collision Buffer: {collision_buffer}m ({collision_buffer*100:.0f}cm)")
        print(f"  State Dimension: {self.env.observation_space.shape[0]}")
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        self.agent = RuleGuidedSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_entropy_tuning=True
        )
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.valid_placement_rates = []
    
    def train(
        self,
        num_episodes: int = 1000,
        batch_size: int = 256,
        update_every: int = 1,
        eval_every: int = 50,
        save_every: int = 100,
        warmup_episodes: int = 10
    ):
        print("=" * 70)
        print("Starting Enhanced Rule-Guided SAC Training")
        print("=" * 70)
        print(f"State Dimension: {self.env.observation_space.shape[0]}")
        print(f"Action Dimension: {self.env.action_space.shape[0]}")
        print(f"Max Items per Episode: {self.env.max_items}")
        print(f"Number of Episodes: {num_episodes}")
        print(f"Grid-Aligned: YES")
        print(f"Collision Buffer: {self.env.collision_buffer}m")
        print("=" * 70)
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            valid_placements = 0
            done = False
            
            while not done:
                if episode < warmup_episodes:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state, evaluate=False)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if info['reward_components']['valid_placement'] and not info['already_placed']:
                    valid_placements += 1
                
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                if episode >= warmup_episodes and episode_length % update_every == 0:
                    self.agent.update(batch_size)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.valid_placement_rates.append(valid_placements / episode_length if episode_length > 0 else 0)
            
            if (episode + 1) % eval_every == 0:
                eval_reward, eval_items, eval_valid_rate = self.evaluate(num_episodes=5)
                self.success_rates.append(eval_items / self.env.max_items)
                
                print("=" * 70)
                print(f"Episode {episode + 1}/{num_episodes}")
                print("=" * 70)
                print(f"Train Reward (avg last {eval_every}): {np.mean(self.episode_rewards[-eval_every:]):.3f}")
                print(f"Eval Reward: {eval_reward:.3f}")
                print(f"Items Placed: {eval_items:.2f}/{self.env.max_items}")
                print(f"Valid Placement Rate: {eval_valid_rate:.1%}")
                print(f"Success Rate: {eval_items / self.env.max_items:.1%}")
                if self.agent.alpha_values:
                    print(f"Temperature: {self.agent.alpha_values[-1]:.4f}")
                print("=" * 70)
            
            if (episode + 1) % save_every == 0:
                self.save_checkpoint(episode + 1)
        
        print("=" * 70)
        print("Training Complete!")
        print("=" * 70)
        
        self.save_checkpoint('final')
        self.plot_training_curves()
        self.print_final_stats()
    
    def evaluate(self, num_episodes: int = 10) -> tuple:
        total_reward = 0
        total_items = 0
        total_valid_rate = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            valid_count = 0
            total_attempts = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if info['reward_components']['valid_placement']:
                    valid_count += 1
                total_attempts += 1
            
            total_reward += episode_reward
            total_items += info['placed_items']
            total_valid_rate += valid_count / total_attempts if total_attempts > 0 else 0
        
        return (
            total_reward / num_episodes,
            total_items / num_episodes,
            total_valid_rate / num_episodes
        )
    
    def save_checkpoint(self, episode):
        filepath = os.path.join(self.save_dir, f'sac_enhanced_ep{episode}.pt')
        self.agent.save(filepath)
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'valid_placement_rates': self.valid_placement_rates,
            'actor_losses': self.agent.actor_losses,
            'critic_losses': self.agent.critic_losses,
            'alpha_values': self.agent.alpha_values
        }
        
        metrics_path = os.path.join(self.save_dir, f'metrics_ep{episode}.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(self.episode_rewards, alpha=0.3)
        if len(self.episode_rewards) > 20:
            window = 20
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window - 1, len(self.episode_rewards)), smoothed)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True)
        
        if self.success_rates:
            axes[0, 1].plot(self.success_rates)
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].grid(True)
        
        if self.valid_placement_rates:
            axes[0, 2].plot(self.valid_placement_rates, alpha=0.4)
            axes[0, 2].set_title('Valid Placement Rate')
            axes[0, 2].grid(True)
        
        if self.agent.actor_losses:
            axes[1, 0].plot(self.agent.actor_losses)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].grid(True)
        
        if self.agent.critic_losses:
            axes[1, 1].plot(self.agent.critic_losses)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].grid(True)
        
        if self.agent.alpha_values:
            axes[1, 2].plot(self.agent.alpha_values)
            axes[1, 2].set_title('Alpha')
            axes[1, 2].grid(True)
        
        save_path = os.path.join(self.save_dir, 'training_curves_enhanced.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_final_stats(self):
        print("=" * 70)
        print("FINAL TRAINING STATISTICS")
        print("=" * 70)
        
        if self.episode_rewards:
            last = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            print(f"Average Reward (last episodes): {np.mean(last):.3f} ± {np.std(last):.3f}")
        
        if self.success_rates:
            print(f"Best Success Rate: {np.max(self.success_rates):.1%}")
        
        if self.valid_placement_rates:
            recent = self.valid_placement_rates[-100:]
            print(f"Valid Placement Rate (recent): {np.mean(recent):.1%}")
        
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Total Updates: {len(self.agent.actor_losses)}")
        print(f"Replay Buffer Size: {len(self.agent.replay_buffer)}")
        print("=" * 70)


def main():
    room_layout_path = 'room_layout.json'
    catalog_path = 'furniture_catalog.json'
    max_items = 4
    
    grid_size = 0.3
    collision_buffer = 0.20
    
    trainer = FurnitureRecommendationTrainerEnhanced(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        max_items=max_items,
        grid_size=grid_size,
        collision_buffer=collision_buffer,
        save_dir='models_enhanced'
    )
    
    trainer.train(
        num_episodes=1000,
        batch_size=256,
        warmup_episodes=10,
        update_every=1,
        eval_every=50,
        save_every=100
    )


if __name__ == '__main__':
    main()