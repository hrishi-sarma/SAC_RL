"""
Train SAC Agent with GNN Integration

Main training loop that combines graph-based spatial reasoning with RL
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import json
import os
from datetime import datetime
from collections import deque

from furniture_env_gnn import FurnitureRecommendationEnvGNN
from sac_agent_gnn import SACAgentWithGNN
from graph_encoder import create_graph_encoder
from graph_replay_buffer import GraphReplayBuffer


class GNNSACTrainer:
    """Trainer for SAC agent with GNN integration"""
    
    def __init__(
        self,
        env: FurnitureRecommendationEnvGNN,
        agent: SACAgentWithGNN,
        replay_buffer: GraphReplayBuffer,
        save_dir: str = 'models_gnn',
        log_dir: str = 'logs_gnn'
    ):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
        self.loss_history = {
            'actor_loss': [],
            'critic_loss': [],
            'q1_value': [],
            'q2_value': []
        }
    
    def train(
        self,
        num_episodes: int = 1000,
        batch_size: int = 64,
        start_steps: int = 1000,
        update_after: int = 1000,
        update_every: int = 50,
        updates_per_step: int = 1,
        save_every: int = 100,
        eval_every: int = 50,
        eval_episodes: int = 10
    ):
        """
        Main training loop
        
        Args:
            num_episodes: Total episodes to train
            batch_size: Batch size for updates
            start_steps: Initial random exploration steps
            update_after: Start updates after this many steps
            update_every: Update frequency
            updates_per_step: Number of updates per step
            save_every: Save model frequency
            eval_every: Evaluation frequency
            eval_episodes: Number of episodes for evaluation
        """
        print("=" * 60)
        print("Starting GNN-Enhanced SAC Training")
        print("=" * 60)
        
        total_steps = 0
        best_eval_reward = -np.inf
        
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Get current graph data
                graph_data = self.env.get_graph_data()
                
                # Select action
                if total_steps < start_steps:
                    # Random exploration
                    action = self.env.action_space.sample()
                else:
                    # Policy action
                    action = self.agent.select_action(state, graph_data, deterministic=False)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Get next graph data
                next_graph_data = self.env.get_graph_data()
                
                # Store transition
                self.replay_buffer.push(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    graph_data=graph_data,
                    next_graph_data=next_graph_data
                )
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Update agent
                if (total_steps >= update_after and 
                    total_steps % update_every == 0 and
                    len(self.replay_buffer) >= batch_size):
                    
                    for _ in range(updates_per_step):
                        batch = self.replay_buffer.sample(batch_size)
                        # Always update actor (was skipping every other, causing slow policy learning)
                        losses = self.agent.update(batch, update_actor=True)
                        
                        # Log losses
                        for key, value in losses.items():
                            if key in self.loss_history:
                                self.loss_history[key].append(value)
            
            # Episode finished
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Steps: {total_steps} | "
                      f"Buffer: {len(self.replay_buffer)}")
            
            # Evaluation
            if episode % eval_every == 0:
                eval_reward, success_rate = self.evaluate(eval_episodes)
                self.success_rate_history.append(success_rate)
                
                print(f"\n{'='*60}")
                print(f"Evaluation after episode {episode}:")
                print(f"  Average Reward: {eval_reward:.2f}")
                print(f"  Success Rate: {success_rate*100:.1f}%")
                print(f"{'='*60}\n")
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save(os.path.join(self.save_dir, 'best_model.pt'))
                    print(f"New best model saved! Reward: {eval_reward:.2f}")
            
            # Save checkpoint
            if episode % save_every == 0:
                self.agent.save(os.path.join(self.save_dir, f'checkpoint_{episode}.pt'))
                self.save_metrics()
                self.plot_training_curves()
        
        print("\nTraining completed!")
        self.agent.save(os.path.join(self.save_dir, 'final_model.pt'))
        self.save_metrics()
        self.plot_training_curves()
    
    def evaluate(self, num_episodes: int = 10) -> tuple:
        """
        Evaluate agent performance
        
        Returns:
            average_reward: Mean episode reward
            success_rate: Fraction of successful episodes
        """
        self.agent.eval()
        
        eval_rewards = []
        successes = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                graph_data = self.env.get_graph_data()
                action = self.agent.select_action(state, graph_data, deterministic=True)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            
            # Count as success if placed all items
            if info.get('placed_items', 0) >= self.env.max_items:
                successes += 1
        
        self.agent.train()
        
        return np.mean(eval_rewards), successes / num_episodes
    
    def save_metrics(self):
        """Save training metrics to file"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rate_history': self.success_rate_history,
            'loss_history': {k: v for k, v in self.loss_history.items()}
        }
        
        filepath = os.path.join(self.log_dir, 'training_metrics.json')
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) > 10:
            smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(smoothed, label='Smoothed (10 eps)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Success rate
        if self.success_rate_history:
            axes[0, 1].plot(self.success_rate_history, marker='o')
            axes[0, 1].set_xlabel('Evaluation')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Success Rate Over Time')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1.1])
        
        # Actor loss
        if self.loss_history['actor_loss']:
            axes[1, 0].plot(self.loss_history['actor_loss'], alpha=0.5)
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Critic loss
        if self.loss_history['critic_loss']:
            axes[1, 1].plot(self.loss_history['critic_loss'], alpha=0.5)
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {filepath}")


def main():
    """Main training function"""
    
    # Paths
    room_layout_path = 'uploads/room_layout.json'
    catalog_path = 'uploads/furniture_catalog_enhanced.json'
    
    # Hyperparameters
    config = {
        'num_episodes': 10000,
        'batch_size': 128,
        'start_steps': 1000,      # ~125 episodes of random exploration
        'update_after': 1000,
        'update_every': 10,       # update more frequently (was 50)
        'updates_per_step': 2,    # 2 gradient steps per env step
        'save_every': 200,
        'eval_every': 100,
        'eval_episodes': 10,
        
        # Network params
        'hidden_dim': 256,
        'graph_embedding_dim': 32,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'lr_graph': 1e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.5,             # higher initial temperature = more exploration
        'auto_tune_alpha': True,
        
        # Buffer
        'buffer_capacity': 100000
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = FurnitureRecommendationEnvGNN(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        max_items=4
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create graph encoder
    print("Creating graph encoder...")
    graph_encoder = create_graph_encoder(
        node_feature_dim=22,
        edge_feature_dim=10,
        hidden_dim=64,
        output_dim=config['graph_embedding_dim'],
        device=device
    )
    
    # Create agent
    print("Creating SAC agent...")
    agent = SACAgentWithGNN(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_encoder=graph_encoder,
        hidden_dim=config['hidden_dim'],
        graph_embedding_dim=config['graph_embedding_dim'],
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        lr_graph=config['lr_graph'],
        gamma=config['gamma'],
        tau=config['tau'],
        alpha=config['alpha'],
        auto_tune_alpha=config['auto_tune_alpha'],
        device=device
    )
    
    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = GraphReplayBuffer(capacity=config['buffer_capacity'])
    
    # Create trainer
    trainer = GNNSACTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer
    )
    
    # Save config
    with open('models_gnn/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    print("\nStarting training...")
    trainer.train(
        num_episodes=config['num_episodes'],
        batch_size=config['batch_size'],
        start_steps=config['start_steps'],
        update_after=config['update_after'],
        update_every=config['update_every'],
        updates_per_step=config['updates_per_step'],
        save_every=config['save_every'],
        eval_every=config['eval_every'],
        eval_episodes=config['eval_episodes']
    )


if __name__ == "__main__":
    main()