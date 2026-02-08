import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from furniture_env_logical import FurnitureRecommendationEnvLogical
from sac_agent import RuleGuidedSAC


class FurnitureRecommendationTrainerLogical:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15,
        save_dir: str = 'models_logical'
    ):
        self.env = FurnitureRecommendationEnvLogical(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=grid_size,
            collision_buffer=collision_buffer
        )
        
        print("=" * 70)
        print("LOGICAL PLACEMENT SYSTEM - Environment Initialized")
        print("=" * 70)
        print(f"  Grid Size: {grid_size}m ({grid_size*100:.0f}cm)")
        print(f"  Collision Buffer: {collision_buffer}m ({collision_buffer*100:.0f}cm)")
        print(f"  State Dimension: {self.env.observation_space.shape[0]}")
        print(f"  Placement Rules: ENABLED")
        print("=" * 70)
        
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
        self.logical_placement_scores = []
    
    def train(
        self,
        num_episodes: int = 1000,
        batch_size: int = 256,
        update_every: int = 1,
        eval_every: int = 50,
        save_every: int = 100,
        warmup_episodes: int = 10
    ):
        print("\n" + "=" * 70)
        print("Starting Logical Placement Training")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Batch Size: {batch_size}")
        print(f"Warmup Episodes: {warmup_episodes}")
        print(f"Logical Placement Weight: 3.0x (highest priority)")
        print("=" * 70 + "\n")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            valid_placements = 0
            logical_placement_sum = 0
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
                    logical_placement_sum += info['reward_components'].get('logical_placement', 0)
                
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                if episode >= warmup_episodes and episode_length % update_every == 0:
                    self.agent.update(batch_size)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.valid_placement_rates.append(valid_placements / episode_length if episode_length > 0 else 0)
            
            avg_logical = logical_placement_sum / valid_placements if valid_placements > 0 else 0
            self.logical_placement_scores.append(avg_logical)
            
            if (episode + 1) % eval_every == 0:
                eval_reward, eval_items, eval_valid_rate, eval_logical = self.evaluate(num_episodes=5)
                self.success_rates.append(eval_items / self.env.max_items)
                
                print("\n" + "=" * 70)
                print(f"Episode {episode + 1}/{num_episodes}")
                print("=" * 70)
                print(f"Train Reward (avg last {eval_every}): {np.mean(self.episode_rewards[-eval_every:]):.3f}")
                print(f"Eval Reward: {eval_reward:.3f}")
                print(f"Items Placed: {eval_items:.2f}/{self.env.max_items}")
                print(f"Valid Placement Rate: {eval_valid_rate:.1%}")
                print(f"Logical Placement Score: {eval_logical:.3f}/1.0")
                print(f"Success Rate: {eval_items / self.env.max_items:.1%}")
                if self.agent.alpha_values:
                    print(f"Temperature (α): {self.agent.alpha_values[-1]:.4f}")
                print("=" * 70 + "\n")
            
            if (episode + 1) % save_every == 0:
                self.save_checkpoint(episode + 1)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70 + "\n")
        
        self.save_checkpoint('final')
        self.plot_training_curves()
        self.print_final_stats()
    
    def evaluate(self, num_episodes: int = 10) -> tuple:
        total_reward = 0
        total_items = 0
        total_valid_rate = 0
        total_logical_score = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            valid_count = 0
            logical_sum = 0
            total_attempts = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if info['reward_components']['valid_placement']:
                    valid_count += 1
                    logical_sum += info['reward_components'].get('logical_placement', 0)
                total_attempts += 1
            
            total_reward += episode_reward
            total_items += info['placed_items']
            total_valid_rate += valid_count / total_attempts if total_attempts > 0 else 0
            total_logical_score += logical_sum / valid_count if valid_count > 0 else 0
        
        return (
            total_reward / num_episodes,
            total_items / num_episodes,
            total_valid_rate / num_episodes,
            total_logical_score / num_episodes
        )
    
    def save_checkpoint(self, episode):
        filepath = os.path.join(self.save_dir, f'sac_logical_ep{episode}.pt')
        self.agent.save(filepath)
        print(f"Model saved: {filepath}")
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'valid_placement_rates': self.valid_placement_rates,
            'logical_placement_scores': self.logical_placement_scores,
            'actor_losses': self.agent.actor_losses,
            'critic_losses': self.agent.critic_losses,
            'alpha_values': self.agent.alpha_values
        }
        
        metrics_path = os.path.join(self.save_dir, f'metrics_ep{episode}.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > 20:
            window = 20
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window - 1, len(self.episode_rewards)), smoothed, label='Smoothed')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Success rate
        if self.success_rates:
            axes[0, 1].plot(self.success_rates, marker='o')
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].set_title('Success Rate (Items Placed / Max Items)')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Logical placement score (NEW!)
        if self.logical_placement_scores:
            axes[0, 2].plot(self.logical_placement_scores, alpha=0.4, label='Raw')
            if len(self.logical_placement_scores) > 20:
                window = 20
                smoothed = np.convolve(self.logical_placement_scores, np.ones(window)/window, mode='valid')
                axes[0, 2].plot(range(window - 1, len(self.logical_placement_scores)), smoothed, label='Smoothed')
            axes[0, 2].set_title('Logical Placement Score')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Score (0-1)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Actor loss
        if self.agent.actor_losses:
            axes[1, 0].plot(self.agent.actor_losses, alpha=0.6)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Critic loss
        if self.agent.critic_losses:
            axes[1, 1].plot(self.agent.critic_losses, alpha=0.6)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Alpha (temperature)
        if self.agent.alpha_values:
            axes[1, 2].plot(self.agent.alpha_values)
            axes[1, 2].set_title('Temperature (α)')
            axes[1, 2].set_xlabel('Update Step')
            axes[1, 2].set_ylabel('α')
            axes[1, 2].grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, 'training_curves_logical.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved: {save_path}")
    
    def print_final_stats(self):
        print("=" * 70)
        print("FINAL TRAINING STATISTICS")
        print("=" * 70)
        
        if self.episode_rewards:
            last = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            print(f"Average Reward (last episodes): {np.mean(last):.3f} ± {np.std(last):.3f}")
        
        if self.success_rates:
            print(f"Best Success Rate: {np.max(self.success_rates):.1%}")
            print(f"Final Success Rate: {self.success_rates[-1]:.1%}")
        
        if self.valid_placement_rates:
            recent = self.valid_placement_rates[-100:]
            print(f"Valid Placement Rate (recent): {np.mean(recent):.1%}")
        
        if self.logical_placement_scores:
            recent = self.logical_placement_scores[-100:]
            print(f"Logical Placement Score (recent): {np.mean(recent):.3f}/1.0")
        
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Total Updates: {len(self.agent.actor_losses)}")
        print(f"Replay Buffer Size: {len(self.agent.replay_buffer)}")
        print("=" * 70)


def main():
    # Paths
    room_layout_path = 'room_layout.json'
    catalog_path = 'furniture_catalog_enhanced.json'
    
    # Check if files exist
    if not os.path.exists(room_layout_path):
        print(f"Error: {room_layout_path} not found!")
        print("Please ensure room_layout.json is in the current directory.")
        return
    
    if not os.path.exists(catalog_path):
        print(f"Error: {catalog_path} not found!")
        print("Please ensure furniture_catalog_enhanced.json is in the current directory.")
        return
    
    # Training parameters
    max_items = 4
    grid_size = 0.3
    collision_buffer = 0.20
    
    print("\n" + "🪑 " * 30)
    print("FURNITURE RECOMMENDATION SYSTEM - LOGICAL PLACEMENT")
    print("🪑 " * 30 + "\n")
    
    trainer = FurnitureRecommendationTrainerLogical(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        max_items=max_items,
        grid_size=grid_size,
        collision_buffer=collision_buffer,
        save_dir='models_logical'
    )
    
    trainer.train(
        num_episodes=1000,
        batch_size=256,
        warmup_episodes=10,
        update_every=1,
        eval_every=50,
        save_every=100
    )
    
    print("\n✅ Training complete! Check models_logical/ for saved models.")
    print("Run test_sac_logical.py to evaluate the trained model.\n")


if __name__ == '__main__':
    main()
