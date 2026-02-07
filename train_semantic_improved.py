"""
IMPROVED Training Script - WITH SEMANTIC CONSTRAINTS PROPERLY ENFORCED
This version enables semantic constraints to guide the model to learn
realistic furniture placement patterns.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from furniture_env_semantic_improved_v1_backup import FurnitureRecommendationEnvSemantic
from sac_agent import RuleGuidedSAC


class ImprovedSemanticTrainer:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        max_items: int = 4,
        save_dir: str = 'models_semantic_improved'
    ):
        # Create environment with SEMANTIC CONSTRAINTS ENABLED
        self.env = FurnitureRecommendationEnvSemantic(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=0.3,              # Fine grid for precision
            collision_buffer=0.15,       # Standard buffer
            wall_proximity=0.35,         # Wall proximity threshold
            enforce_semantic=True        # *** ENABLE SEMANTIC CONSTRAINTS ***
        )
        
        print("=" * 70)
        print("IMPROVED SEMANTIC TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Grid Size: 0.3m (30cm) - FINE for precision")
        print(f"Collision Buffer: 0.15m (15cm) - STANDARD")
        print(f"Wall Proximity: 0.35m (35cm) - For wall items")
        print(f"Semantic Constraints: ENABLED ✅")
        print(f"State Dimension: {self.env.observation_space.shape[0]}")
        print("=" * 70)
        print("\nSemantic Rules Enforced:")
        print("  ✅ Wall items (shelves, mirrors) → Must be near walls")
        print("  ✅ Decorative ladder → Must lean on walls")
        print("  ✅ Plant stands → Must be in corners OR near windows")
        print("  ✅ Floor lamps → Must be near seating (within 3.5m)")
        print("  ✅ Table lamps → Must be on table surfaces")
        print("  ✅ Storage → Should be near walls (perimeter)")
        print("  ✅ Ottomans → Should be near seating")
        print("=" * 70)
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        # Create agent with MODERATE EXPLORATION
        self.agent = RuleGuidedSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,  # Standard temperature
            auto_entropy_tuning=True
        )
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.items_placed_history = []
        self.semantic_scores = []
    
    def train(
        self,
        num_episodes: int = 1500,  # More episodes for learning constraints
        batch_size: int = 256,
        warmup_episodes: int = 100,  # Longer warmup to explore
        eval_every: int = 50,
        save_every: int = 100
    ):
        print("=" * 70)
        print("Starting IMPROVED SEMANTIC Training")
        print("=" * 70)
        print(f"Total Episodes: {num_episodes}")
        print(f"Warmup Episodes: {warmup_episodes} (random exploration)")
        print(f"Goal: Learn to place {self.env.max_items} items with semantic rules")
        print("=" * 70)
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            items_placed = 0
            semantic_score_sum = 0
            semantic_count = 0
            
            while not done:
                # Longer warmup for exploration
                if episode < warmup_episodes:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state, evaluate=False)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Track successful placements
                if info.get('valid_placement', False) and not info.get('already_placed', False):
                    items_placed += 1
                    
                    # Track semantic scores
                    if 'reward_components' in info:
                        sem_score = info['reward_components'].get('semantic_correctness', 0)
                        semantic_score_sum += sem_score
                        semantic_count += 1
                
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update after warmup
                if episode >= warmup_episodes and len(self.agent.replay_buffer) >= batch_size:
                    self.agent.update(batch_size)
                
                state = next_state
                episode_reward += reward
            
            self.episode_rewards.append(episode_reward)
            self.items_placed_history.append(items_placed)
            
            # Track average semantic score
            if semantic_count > 0:
                avg_sem = semantic_score_sum / semantic_count
                self.semantic_scores.append(avg_sem)
            else:
                self.semantic_scores.append(0.0)
            
            # Evaluation
            if (episode + 1) % eval_every == 0:
                eval_reward, eval_items, eval_semantic = self.evaluate(num_episodes=5)
                self.success_rates.append(eval_items / self.env.max_items)
                
                print("=" * 70)
                print(f"Episode {episode + 1}/{num_episodes}")
                print("=" * 70)
                print(f"Train Reward (avg last {eval_every}): {np.mean(self.episode_rewards[-eval_every:]):.3f}")
                print(f"Train Items (avg last {eval_every}): {np.mean(self.items_placed_history[-eval_every:]):.2f}/{self.env.max_items}")
                print(f"Train Semantic (avg last {eval_every}): {np.mean(self.semantic_scores[-eval_every:]):.3f}/3.0")
                print(f"Eval Reward: {eval_reward:.3f}")
                print(f"Eval Items: {eval_items:.2f}/{self.env.max_items}")
                print(f"Eval Semantic: {eval_semantic:.3f}/3.0")
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
        total_semantic = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            semantic_sum = 0
            semantic_count = 0
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if info.get('valid_placement') and 'reward_components' in info:
                    sem = info['reward_components'].get('semantic_correctness', 0)
                    semantic_sum += sem
                    semantic_count += 1
            
            total_reward += episode_reward
            total_items += info['placed_items']
            if semantic_count > 0:
                total_semantic += semantic_sum / semantic_count
        
        return (
            total_reward / num_episodes,
            total_items / num_episodes,
            total_semantic / num_episodes
        )
    
    def save_checkpoint(self, episode):
        filepath = os.path.join(self.save_dir, f'sac_semantic_improved_ep{episode}.pt')
        self.agent.save(filepath)
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'items_placed': self.items_placed_history,
            'success_rates': self.success_rates,
            'semantic_scores': self.semantic_scores
        }
        
        metrics_path = os.path.join(self.save_dir, f'metrics_ep{episode}.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        axes[0, 0].grid(True)
        
        # Items placed
        axes[0, 1].plot(self.items_placed_history, alpha=0.3, label='Raw')
        if len(self.items_placed_history) > 20:
            window = 20
            smoothed = np.convolve(self.items_placed_history, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window - 1, len(self.items_placed_history)), smoothed, label='Smoothed')
        axes[0, 1].axhline(y=self.env.max_items, color='r', linestyle='--', label='Max Items')
        axes[0, 1].set_title('Items Placed per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Items Placed')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Semantic scores
        axes[1, 0].plot(self.semantic_scores, alpha=0.3, label='Raw')
        if len(self.semantic_scores) > 20:
            window = 20
            smoothed = np.convolve(self.semantic_scores, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window - 1, len(self.semantic_scores)), smoothed, label='Smoothed')
        axes[1, 0].axhline(y=3.0, color='g', linestyle='--', label='Max Score')
        axes[1, 0].set_title('Semantic Correctness Score')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Semantic Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Success rate
        if self.success_rates:
            x_vals = range(len(self.success_rates))
            axes[1, 1].plot(x_vals, self.success_rates, 'o-', label='Success Rate')
            axes[1, 1].axhline(y=1.0, color='g', linestyle='--', label='100%')
            axes[1, 1].set_title('Success Rate (Evaluation)')
            axes[1, 1].set_xlabel('Evaluation Point')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        save_path = os.path.join(self.save_dir, 'training_curves_semantic_improved.png')
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
        
        if self.semantic_scores:
            last = self.semantic_scores[-100:]
            print(f"Average Semantic Score (last 100): {np.mean(last):.3f}/3.0")
            print(f"Best: {np.max(self.semantic_scores):.3f}/3.0")
        
        if self.success_rates:
            print(f"Best Success Rate: {np.max(self.success_rates):.1%}")
            print(f"Final Success Rate: {self.success_rates[-1]:.1%}")
        
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print("=" * 70)


def main():
    print("\n" + "🎯" * 35)
    print("IMPROVED SEMANTIC TRAINING - CONSTRAINTS ENABLED")
    print("🎯" * 35 + "\n")
    
    # Use relative paths that work on any OS
    trainer = ImprovedSemanticTrainer(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog.json',
        max_items=4,
        save_dir='models_semantic_improved'
    )
    
    trainer.train(
        num_episodes=1500,      # More episodes to learn constraints
        batch_size=256,
        warmup_episodes=100,    # More exploration
        eval_every=50,
        save_every=100
    )
    
    print("\n" + "✅" * 35)
    print("TRAINING COMPLETE!")
    print("Test with improved testing script")
    print("✅" * 35 + "\n")


if __name__ == '__main__':
    main()
