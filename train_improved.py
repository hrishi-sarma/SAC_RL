"""
IMPROVED Training Script - No Semantic Constraints
This version disables semantic constraints and increases exploration
to ensure the model learns to place ALL furniture types.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from furniture_env_semantic_old2 import FurnitureRecommendationEnvSemantic
from sac_agent import RuleGuidedSAC


class ImprovedFurnitureTrainer:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        max_items: int = 4,
        save_dir: str = 'models_improved'
    ):
        # Create environment with VERY RELAXED constraints
        self.env = FurnitureRecommendationEnvSemantic(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=0.5,              # LARGE grid for more options
            collision_buffer=0.12,       # SMALL buffer to fit more items
            wall_proximity=0.60          # VERY relaxed wall proximity
        )
        
        print("=" * 70)
        print("IMPROVED TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Grid Size: 0.5m (50cm) - COARSE for more options")
        print(f"Collision Buffer: 0.12m (12cm) - SMALL to fit more")
        print(f"Wall Proximity: 0.60m (60cm) - VERY relaxed")
        print(f"State Dimension: {self.env.observation_space.shape[0]}")
        print("Semantic Constraints: Should be DISABLED or VERY relaxed")
        print("=" * 70)
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        # Create agent with MORE EXPLORATION
        self.agent = RuleGuidedSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.3,  # HIGHER temperature = more exploration
            auto_entropy_tuning=True
        )
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.items_placed_history = []
    
    def train(
        self,
        num_episodes: int = 1000,
        batch_size: int = 256,
        warmup_episodes: int = 50,  # LONGER warmup for more exploration
        eval_every: int = 50,
        save_every: int = 100
    ):
        print("=" * 70)
        print("Starting IMPROVED Training (No Strict Constraints)")
        print("=" * 70)
        print(f"Total Episodes: {num_episodes}")
        print(f"Warmup Episodes: {warmup_episodes} (random exploration)")
        print(f"Goal: Place {self.env.max_items} items successfully")
        print("=" * 70)
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            items_placed = 0
            
            while not done:
                # LONGER warmup period
                if episode < warmup_episodes:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state, evaluate=False)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Track successful placements
                if info['reward_components'].get('valid_placement', False) and not info.get('already_placed', False):
                    items_placed += 1
                
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update more frequently after warmup
                if episode >= warmup_episodes and len(self.agent.replay_buffer) >= batch_size:
                    self.agent.update(batch_size)
                
                state = next_state
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            self.items_placed_history.append(items_placed)
            
            # Evaluation
            if (episode + 1) % eval_every == 0:
                eval_reward, eval_items = self.evaluate(num_episodes=5)
                self.success_rates.append(eval_items / self.env.max_items)
                
                print("=" * 70)
                print(f"Episode {episode + 1}/{num_episodes}")
                print("=" * 70)
                print(f"Train Reward (avg last {eval_every}): {np.mean(self.episode_rewards[-eval_every:]):.3f}")
                print(f"Train Items (avg last {eval_every}): {np.mean(self.items_placed_history[-eval_every:]):.2f}/{self.env.max_items}")
                print(f"Eval Reward: {eval_reward:.3f}")
                print(f"Eval Items: {eval_items:.2f}/{self.env.max_items}")
                print(f"Success Rate: {eval_items / self.env.max_items:.1%}")
                if self.agent.alpha_values:
                    print(f"Exploration (alpha): {self.agent.alpha_values[-1]:.4f}")
                print(f"Replay Buffer: {len(self.agent.replay_buffer)}")
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
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
            total_items += info['placed_items']
        
        return (
            total_reward / num_episodes,
            total_items / num_episodes
        )
    
    def save_checkpoint(self, episode):
        filepath = os.path.join(self.save_dir, f'sac_improved_ep{episode}.pt')
        self.agent.save(filepath)
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'items_placed': self.items_placed_history,
            'success_rates': self.success_rates
        }
        
        metrics_path = os.path.join(self.save_dir, f'metrics_ep{episode}.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Episode rewards
        axes[0].plot(self.episode_rewards, alpha=0.3)
        if len(self.episode_rewards) > 20:
            window = 20
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window - 1, len(self.episode_rewards)), smoothed)
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)
        
        # Items placed
        axes[1].plot(self.items_placed_history, alpha=0.3)
        if len(self.items_placed_history) > 20:
            window = 20
            smoothed = np.convolve(self.items_placed_history, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window - 1, len(self.items_placed_history)), smoothed)
        axes[1].axhline(y=self.env.max_items, color='r', linestyle='--', label='Max Items')
        axes[1].set_title('Items Placed per Episode')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Items Placed')
        axes[1].legend()
        axes[1].grid(True)
        
        save_path = os.path.join(self.save_dir, 'training_curves_improved.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def print_final_stats(self):
        print("=" * 70)
        print("FINAL TRAINING STATISTICS")
        print("=" * 70)
        
        if self.episode_rewards:
            last = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            print(f"Average Reward (last 100): {np.mean(last):.3f} ± {np.std(last):.3f}")
        
        if self.items_placed_history:
            last = self.items_placed_history[-100:]
            print(f"Average Items Placed (last 100): {np.mean(last):.2f}/{self.env.max_items}")
            print(f"Best: {np.max(self.items_placed_history)}")
        
        if self.success_rates:
            print(f"Best Success Rate: {np.max(self.success_rates):.1%}")
            print(f"Final Success Rate: {self.success_rates[-1]:.1%}")
        
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print("=" * 70)


def main():
    print("\n" + "🎯" * 35)
    print("IMPROVED TRAINING - NO STRICT CONSTRAINTS")
    print("🎯" * 35 + "\n")
    
    trainer = ImprovedFurnitureTrainer(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog.json',
        max_items=4,
        save_dir='models_improved'
    )
    
    trainer.train(
        num_episodes=1000,
        batch_size=256,
        warmup_episodes=50,  # More exploration
        eval_every=50,
        save_every=100
    )
    
    print("\n" + "✅" * 35)
    print("TRAINING COMPLETE!")
    print("Test with: python test_improved.py")
    print("✅" * 35 + "\n")


if __name__ == '__main__':
    main()
