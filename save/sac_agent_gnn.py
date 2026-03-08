"""
SAC Agent with Graph Neural Network Integration

Main RL agent that uses GNN embeddings for improved spatial reasoning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import os

from sac_networks_gnn import GNNEnhancedActor, GNNEnhancedCritic
from graph_encoder import SceneGraphEncoder
from graph_replay_buffer import GraphReplayBuffer


class SACAgentWithGNN:
    """
    Soft Actor-Critic agent enhanced with Graph Neural Networks
    
    Combines traditional RL with graph-based spatial reasoning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        graph_encoder: SceneGraphEncoder,
        hidden_dim: int = 256,
        graph_embedding_dim: int = 32,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_graph: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            state_dim: Dimension of state observations
            action_dim: Dimension of action space
            graph_encoder: Pre-initialized SceneGraphEncoder
            hidden_dim: Hidden layer size for actor/critic
            graph_embedding_dim: Output dimension of graph encoder
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            lr_graph: Learning rate for graph encoder
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Temperature parameter
            auto_tune_alpha: Whether to automatically tune temperature
            device: Device to run on
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        
        # Graph encoder
        self.graph_encoder = graph_encoder.to(device)
        
        # Actor network
        self.actor = GNNEnhancedActor(
            state_dim=state_dim,
            action_dim=action_dim,
            graph_embedding_dim=graph_embedding_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Critic networks (double Q-learning)
        self.critic = GNNEnhancedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            graph_embedding_dim=graph_embedding_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.critic_target = GNNEnhancedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            graph_embedding_dim=graph_embedding_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.graph_optimizer = optim.Adam(self.graph_encoder.parameters(), lr=lr_graph)
        
        # Temperature parameter
        if auto_tune_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        
        # Training mode
        self.train()
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.actor.train(mode)
        self.critic.train(mode)
        self.graph_encoder.train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self.train(False)
    
    def select_action(
        self,
        state: np.ndarray,
        graph_data: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action given state and graph data
        
        Args:
            state: State observation
            graph_data: Dictionary with 'node_features', 'edge_index', 'edge_attr'
            deterministic: If True, return mean action
            
        Returns:
            action: Numpy array
        """
        # Convert graph data to tensors
        node_features = torch.FloatTensor(graph_data['node_features']).to(self.device)
        edge_index = torch.LongTensor(graph_data['edge_index']).to(self.device)
        edge_attr = torch.FloatTensor(graph_data['edge_attr']).to(self.device)
        
        # Get graph embedding
        with torch.no_grad():
            _, graph_embedding = self.graph_encoder(node_features, edge_index, edge_attr)
            # Squeeze batch dim: [1, D] -> [D] so get_action unsqueezes correctly to [1, D]
            graph_embedding = graph_embedding.squeeze(0).cpu().numpy()
        
        # Get action from actor
        action = self.actor.get_action(state, graph_embedding, deterministic)
        
        return action
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        update_actor: bool = True
    ) -> Dict[str, float]:
        """
        Update agent parameters
        
        Args:
            batch: Batch of transitions from replay buffer
            update_actor: Whether to update actor (delayed updates)
            
        Returns:
            Dictionary of loss values
        """
        # Move data to device
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # Process graph embeddings
        graph_embeddings = self._batch_encode_graphs(batch['graph_data'])
        next_graph_embeddings = self._batch_encode_graphs(batch['next_graph_data'])
        
        # ========== Update Critic ==========
        with torch.no_grad():
            # Sample next actions from current policy
            next_action, next_log_prob, _ = self.actor.sample(next_state, next_graph_embeddings)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action, next_graph_embeddings)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * (target_q - self.alpha * next_log_prob)
        
        # Current Q-values
        current_q1, current_q2 = self.critic(state, action, graph_embeddings)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        self.graph_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.graph_optimizer.step()
        
        losses = {
            'critic_loss': critic_loss.item(),
            'q1_value': current_q1.mean().item(),
            'q2_value': current_q2.mean().item()
        }
        
        # ========== Update Actor ==========
        if update_actor:
            # Re-encode graphs for actor update (need fresh computation graph)
            graph_embeddings_actor = self._batch_encode_graphs(batch['graph_data'])
            
            # Sample actions from current policy
            new_action, log_prob, _ = self.actor.sample(state, graph_embeddings_actor)
            
            # Q-values for new actions
            q1_new, q2_new = self.critic(state, new_action, graph_embeddings_actor)
            q_new = torch.min(q1_new, q2_new)
            
            # Actor loss
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            losses['actor_loss'] = actor_loss.item()
            losses['log_prob'] = log_prob.mean().item()
            
            # ========== Update Temperature ==========
            if self.auto_tune_alpha:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
                losses['alpha'] = self.alpha
                losses['alpha_loss'] = alpha_loss.item()
        
        # ========== Soft Update Target Network ==========
        self._soft_update(self.critic, self.critic_target)
        
        return losses
    
    def _batch_encode_graphs(self, graph_data_list: Dict[str, list]) -> torch.Tensor:
        """
        Encode a batch of graphs
        
        Args:
            graph_data_list: Dictionary with lists of node_features, edge_index, edge_attr
            
        Returns:
            graph_embeddings: [batch_size, graph_embedding_dim]
        """
        batch_size = len(graph_data_list['node_features'])
        graph_embeddings = []
        
        for i in range(batch_size):
            node_features = torch.FloatTensor(graph_data_list['node_features'][i]).to(self.device)
            edge_index = torch.LongTensor(graph_data_list['edge_index'][i]).to(self.device)
            edge_attr = torch.FloatTensor(graph_data_list['edge_attr'][i]).to(self.device)
            
            # Encode single graph
            _, graph_embedding = self.graph_encoder(node_features, edge_index, edge_attr)
            graph_embeddings.append(graph_embedding)
        
        # Stack into batch
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        
        return graph_embeddings
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath: str):
        """Save agent state"""
        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'graph_encoder': self.graph_encoder.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'graph_optimizer': self.graph_optimizer.state_dict()
        }
        
        if self.auto_tune_alpha:
            state['log_alpha'] = self.log_alpha
            state['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(state, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        if not os.path.exists(filepath):
            print(f"File {filepath} not found!")
            return
        
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.graph_encoder.load_state_dict(state['graph_encoder'])
        
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.graph_optimizer.load_state_dict(state['graph_optimizer'])
        
        if self.auto_tune_alpha and 'log_alpha' in state:
            self.log_alpha = state['log_alpha']
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        
        print(f"Agent loaded from {filepath}")


if __name__ == "__main__":
    # Test agent
    print("Testing SACAgentWithGNN...")
    
    from graph_encoder import create_graph_encoder
    
    state_dim = 100
    action_dim = 5
    graph_embedding_dim = 32
    
    # Create graph encoder
    graph_encoder = create_graph_encoder(
        node_feature_dim=22,
        edge_feature_dim=10,
        output_dim=graph_embedding_dim
    )
    
    # Create agent
    agent = SACAgentWithGNN(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_encoder=graph_encoder,
        graph_embedding_dim=graph_embedding_dim
    )
    
    # Test action selection
    state = np.random.randn(state_dim)
    graph_data = {
        'node_features': np.random.randn(10, 22),
        'edge_index': np.random.randint(0, 10, (2, 15)),
        'edge_attr': np.random.randn(15, 10)
    }
    
    action = agent.select_action(state, graph_data, deterministic=False)
    print(f"Action: {action}")
    print(f"Action shape: {action.shape}")
    
    print("Test passed!")