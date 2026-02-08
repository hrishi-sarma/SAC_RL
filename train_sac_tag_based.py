"""
Tag-Based Context-Aware Training Script

This version leverages the rich tag metadata in the catalog to train
a more intelligent agent that understands furniture context and relationships.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from furniture_env_tag_based import FurnitureRecommendationEnvTagBased
from sac_agent import RuleGuidedSAC


class FurnitureRecommendationTrainerTagBased:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15,
        save_dir: str = 'models_tag_based'
    ):
        self.env = FurnitureRecommendationEnvTagBased(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=grid_size,
            collision_buffer=collision_buffer
        )
        
        print("=" * 80)
        print("TAG-BASED CONTEXT-AWARE PLACEMENT SYSTEM - Environment Initialized")
        print("=" * 80)
        print(f"  Grid Size: {grid_size}m ({grid_size*100:.0f}cm)")
        print(f"  Collision Buffer: {collision_buffer}m ({collision_buffer*100:.0f}cm)")
        print(f"  State Dimension: {self.env.observation_space.shape[0]}")
        print(f"  Tag-Based Features: ENABLED")
        print(f"  Spatial Zones: {len(self.env.spatial_zones)} types")
        print(f"  Tag Relationships: {len(self.env.tag_relationships)} defined")
        print("=" * 80)
        
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
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.valid_placement_rates = []
        
        # Tag-based metrics
        self.tag_placement_scores = []
        self.spatial_constraint_scores = []
        self.functional_requirement_scores = []
        self.zone_preference_scores = []
        self.tag_compatibility_scores = []
    
    def train(
        self,
        num_episodes: int = 1000,
        batch_size: int = 256,
        update_every: int = 1,
        eval_every: int = 50,
        save_every: int = 100,
        warmup_episodes: int = 10
    ):
        print("\n" + "=" * 80)
        print("Starting Tag-Based Context-Aware Training")
        print("=" * 80)
        print(f"Episodes: {num_episodes}")
        print(f"Batch Size: {batch_size}")
        print(f"Warmup Episodes: {warmup_episodes}")
        print("\nReward Component Weights:")
        print("  • Tag-Based Placement: 5.0x (HIGHEST)")
        print("  • Spatial Constraints: 4.0x")
        print("  • Functional Requirements: 3.5x")
        print("  • Zone Preference: 3.0x")
        print("  • Tag Compatibility: 2.5x")
        print("=" * 80 + "\n")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            valid_placements = 0
            
            # Tag-based metric accumulators
            tag_placement_sum = 0
            spatial_sum = 0
            functional_sum = 0
            zone_sum = 0
            compatibility_sum = 0
            
            done = False
            
            while not done:
                if episode < warmup_episodes:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state, evaluate=False)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Track tag-based metrics
                if info['reward_components']['valid_placement'] and not info['already_placed']:
                    valid_placements += 1
                    tag_placement_sum += info['reward_components'].get('tag_based_placement', 0)
                    spatial_sum += info['reward_components'].get('spatial_constraints', 0)
                    functional_sum += info['reward_components'].get('functional_requirements', 0)
                    zone_sum += info['reward_components'].get('zone_preference', 0)
                    compatibility_sum += info['reward_components'].get('tag_compatibility', 0)
                
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                if episode >= warmup_episodes and episode_length % update_every == 0:
                    self.agent.update(batch_size)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.valid_placement_rates.append(valid_placements / episode_length if episode_length > 0 else 0)
            
            # Store tag-based metrics
            if valid_placements > 0:
                self.tag_placement_scores.append(tag_placement_sum / valid_placements)
                self.spatial_constraint_scores.append(spatial_sum / valid_placements)
                self.functional_requirement_scores.append(functional_sum / valid_placements)
                self.zone_preference_scores.append(zone_sum / valid_placements)
                self.tag_compatibility_scores.append(compatibility_sum / valid_placements)
            else:
                self.tag_placement_scores.append(0)
                self.spatial_constraint_scores.append(0)
                self.functional_requirement_scores.append(0)
                self.zone_preference_scores.append(0)
                self.tag_compatibility_scores.append(0)
            
            # Evaluation
            if (episode + 1) % eval_every == 0:
                eval_metrics = self.evaluate(num_episodes=5)
                self.success_rates.append(eval_metrics['success_rate'])
                
                print("\n" + "=" * 80)
                print(f"Episode {episode + 1}/{num_episodes}")
                print("=" * 80)
                print(f"Train Reward (avg last {eval_every}): {np.mean(self.episode_rewards[-eval_every:]):.3f}")
                print(f"Eval Reward: {eval_metrics['avg_reward']:.3f}")
                print(f"Items Placed: {eval_metrics['avg_items']:.2f}/{self.env.max_items}")
                print(f"Valid Placement Rate: {eval_metrics['valid_rate']:.1%}")
                print(f"Success Rate: {eval_metrics['success_rate']:.1%}")
                print("\nTag-Based Metrics (Recent):")
                print(f"  Tag Placement Score: {np.mean(self.tag_placement_scores[-eval_every:]):.3f}/1.0")
                print(f"  Spatial Constraints: {np.mean(self.spatial_constraint_scores[-eval_every:]):.3f}/1.0")
                print(f"  Functional Requirements: {np.mean(self.functional_requirement_scores[-eval_every:]):.3f}/1.0")
                print(f"  Zone Preference: {np.mean(self.zone_preference_scores[-eval_every:]):.3f}/1.0")
                print(f"  Tag Compatibility: {np.mean(self.tag_compatibility_scores[-eval_every:]):.3f}/1.0")
                if self.agent.alpha_values:
                    print(f"Temperature (α): {self.agent.alpha_values[-1]:.4f}")
                print("=" * 80 + "\n")
            
            # Save checkpoint
            if (episode + 1) % save_every == 0:
                self.save_checkpoint(episode + 1)
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80 + "\n")
        
        self.save_checkpoint('final')
        self.plot_training_curves()
        self.print_final_stats()
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """Evaluate current policy"""
        total_reward = 0
        total_items = 0
        total_valid_rate = 0
        
        tag_placement_sum = 0
        spatial_sum = 0
        functional_sum = 0
        zone_sum = 0
        compatibility_sum = 0
        valid_count = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_valid = 0
            total_attempts = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if info['reward_components']['valid_placement']:
                    episode_valid += 1
                    tag_placement_sum += info['reward_components'].get('tag_based_placement', 0)
                    spatial_sum += info['reward_components'].get('spatial_constraints', 0)
                    functional_sum += info['reward_components'].get('functional_requirements', 0)
                    zone_sum += info['reward_components'].get('zone_preference', 0)
                    compatibility_sum += info['reward_components'].get('tag_compatibility', 0)
                    valid_count += 1
                
                total_attempts += 1
            
            total_reward += episode_reward
            total_items += info['placed_items']
            total_valid_rate += episode_valid / total_attempts if total_attempts > 0 else 0
        
        return {
            'avg_reward': total_reward / num_episodes,
            'avg_items': total_items / num_episodes,
            'valid_rate': total_valid_rate / num_episodes,
            'success_rate': (total_items / num_episodes) / self.env.max_items,
            'tag_placement': tag_placement_sum / valid_count if valid_count > 0 else 0,
            'spatial': spatial_sum / valid_count if valid_count > 0 else 0,
            'functional': functional_sum / valid_count if valid_count > 0 else 0,
            'zone': zone_sum / valid_count if valid_count > 0 else 0,
            'compatibility': compatibility_sum / valid_count if valid_count > 0 else 0
        }
    
    def save_checkpoint(self, episode):
        """Save model and metrics"""
        filepath = os.path.join(self.save_dir, f'sac_tag_based_ep{episode}.pt')
        self.agent.save(filepath)
        print(f"Model saved: {filepath}")
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'valid_placement_rates': self.valid_placement_rates,
            'tag_placement_scores': self.tag_placement_scores,
            'spatial_constraint_scores': self.spatial_constraint_scores,
            'functional_requirement_scores': self.functional_requirement_scores,
            'zone_preference_scores': self.zone_preference_scores,
            'tag_compatibility_scores': self.tag_compatibility_scores,
            'actor_losses': self.agent.actor_losses,
            'critic_losses': self.agent.critic_losses,
            'alpha_values': self.agent.alpha_values
        }
        
        metrics_path = os.path.join(self.save_dir, f'metrics_ep{episode}.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
    
    def plot_training_curves(self):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        
        # 1. Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > 20:
            window = 20
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window - 1, len(self.episode_rewards)), smoothed, label='Smoothed')
        axes[0, 0].set_title('Episode Rewards', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Success rate
        if self.success_rates:
            axes[0, 1].plot(self.success_rates, marker='o', linewidth=2, markersize=4)
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% target')
            axes[0, 1].set_title('Success Rate', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Tag placement score
        if self.tag_placement_scores:
            axes[0, 2].plot(self.tag_placement_scores, alpha=0.4, label='Raw', color='purple')
            if len(self.tag_placement_scores) > 20:
                window = 20
                smoothed = np.convolve(self.tag_placement_scores, np.ones(window)/window, mode='valid')
                axes[0, 2].plot(range(window - 1, len(self.tag_placement_scores)), 
                               smoothed, label='Smoothed', color='darkviolet', linewidth=2)
            axes[0, 2].set_title('Tag-Based Placement Score', fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Score (0-1)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Spatial constraints score
        if self.spatial_constraint_scores:
            axes[1, 0].plot(self.spatial_constraint_scores, alpha=0.4, label='Raw', color='blue')
            if len(self.spatial_constraint_scores) > 20:
                window = 20
                smoothed = np.convolve(self.spatial_constraint_scores, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(range(window - 1, len(self.spatial_constraint_scores)), 
                               smoothed, label='Smoothed', color='darkblue', linewidth=2)
            axes[1, 0].set_title('Spatial Constraints Score', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Score (0-1)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Functional requirements score
        if self.functional_requirement_scores:
            axes[1, 1].plot(self.functional_requirement_scores, alpha=0.4, label='Raw', color='green')
            if len(self.functional_requirement_scores) > 20:
                window = 20
                smoothed = np.convolve(self.functional_requirement_scores, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(range(window - 1, len(self.functional_requirement_scores)), 
                               smoothed, label='Smoothed', color='darkgreen', linewidth=2)
            axes[1, 1].set_title('Functional Requirements Score', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Score (0-1)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Zone preference score
        if self.zone_preference_scores:
            axes[1, 2].plot(self.zone_preference_scores, alpha=0.4, label='Raw', color='orange')
            if len(self.zone_preference_scores) > 20:
                window = 20
                smoothed = np.convolve(self.zone_preference_scores, np.ones(window)/window, mode='valid')
                axes[1, 2].plot(range(window - 1, len(self.zone_preference_scores)), 
                               smoothed, label='Smoothed', color='darkorange', linewidth=2)
            axes[1, 2].set_title('Zone Preference Score', fontsize=12, fontweight='bold')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Score (0-1)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Tag compatibility score
        if self.tag_compatibility_scores:
            axes[2, 0].plot(self.tag_compatibility_scores, alpha=0.4, label='Raw', color='red')
            if len(self.tag_compatibility_scores) > 20:
                window = 20
                smoothed = np.convolve(self.tag_compatibility_scores, np.ones(window)/window, mode='valid')
                axes[2, 0].plot(range(window - 1, len(self.tag_compatibility_scores)), 
                               smoothed, label='Smoothed', color='darkred', linewidth=2)
            axes[2, 0].set_title('Tag Compatibility Score', fontsize=12, fontweight='bold')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Score (0-1)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Actor loss
        if self.agent.actor_losses:
            axes[2, 1].plot(self.agent.actor_losses, alpha=0.6, color='teal')
            axes[2, 1].set_title('Actor Loss', fontsize=12, fontweight='bold')
            axes[2, 1].set_xlabel('Update Step')
            axes[2, 1].set_ylabel('Loss')
            axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Critic loss
        if self.agent.critic_losses:
            axes[2, 2].plot(self.agent.critic_losses, alpha=0.6, color='brown')
            axes[2, 2].set_title('Critic Loss', fontsize=12, fontweight='bold')
            axes[2, 2].set_xlabel('Update Step')
            axes[2, 2].set_ylabel('Loss')
            axes[2, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Tag-Based Context-Aware Training Metrics', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        save_path = os.path.join(self.save_dir, 'training_curves_tag_based.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved: {save_path}")
    
    def print_final_stats(self):
        """Print final training statistics"""
        print("=" * 80)
        print("FINAL TRAINING STATISTICS")
        print("=" * 80)
        
        if self.episode_rewards:
            last = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            print(f"Average Reward (last episodes): {np.mean(last):.3f} ± {np.std(last):.3f}")
        
        if self.success_rates:
            print(f"Best Success Rate: {np.max(self.success_rates):.1%}")
            print(f"Final Success Rate: {self.success_rates[-1]:.1%}")
        
        if self.valid_placement_rates:
            recent = self.valid_placement_rates[-100:]
            print(f"Valid Placement Rate (recent): {np.mean(recent):.1%}")
        
        print("\nTag-Based Metrics (Recent 100 Episodes):")
        if self.tag_placement_scores:
            recent = self.tag_placement_scores[-100:]
            print(f"  Tag Placement Score: {np.mean(recent):.3f}/1.0")
        
        if self.spatial_constraint_scores:
            recent = self.spatial_constraint_scores[-100:]
            print(f"  Spatial Constraints: {np.mean(recent):.3f}/1.0")
        
        if self.functional_requirement_scores:
            recent = self.functional_requirement_scores[-100:]
            print(f"  Functional Requirements: {np.mean(recent):.3f}/1.0")
        
        if self.zone_preference_scores:
            recent = self.zone_preference_scores[-100:]
            print(f"  Zone Preference: {np.mean(recent):.3f}/1.0")
        
        if self.tag_compatibility_scores:
            recent = self.tag_compatibility_scores[-100:]
            print(f"  Tag Compatibility: {np.mean(recent):.3f}/1.0")
        
        print(f"\nTotal Episodes: {len(self.episode_rewards)}")
        print(f"Total Updates: {len(self.agent.actor_losses)}")
        print(f"Replay Buffer Size: {len(self.agent.replay_buffer)}")
        print("=" * 80)


def main():
    """Main training function"""
    room_layout_path = 'room_layout.json'
    catalog_path = 'furniture_catalog_enhanced.json'
    
    # Check if files exist
    if not os.path.exists(room_layout_path):
        print(f"Error: {room_layout_path} not found!")
        return
    
    if not os.path.exists(catalog_path):
        print(f"Error: {catalog_path} not found!")
        return
    
    # Training parameters
    max_items = 4
    grid_size = 0.3
    collision_buffer = 0.20
    
    print("\n" + "🪑 " * 35)
    print("TAG-BASED CONTEXT-AWARE FURNITURE PLACEMENT SYSTEM")
    print("🪑 " * 35 + "\n")
    
    trainer = FurnitureRecommendationTrainerTagBased(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        max_items=max_items,
        grid_size=grid_size,
        collision_buffer=collision_buffer,
        save_dir='models_tag_based'
    )
    
    trainer.train(
        num_episodes=1000,
        batch_size=256,
        warmup_episodes=10,
        update_every=1,
        eval_every=50,
        save_every=100
    )
    
    print("\n✅ Training complete! Check models_tag_based/ for saved models.")
    print("Next: Create test script to evaluate the trained model.\n")


if __name__ == '__main__':
    main()
