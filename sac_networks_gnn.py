"""
SAC Actor-Critic Networks with Graph Embedding Integration

Modified to accept and utilize graph embeddings from GNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class GNNEnhancedActor(nn.Module):
    """
    Actor network that combines state observations with graph embeddings
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        graph_embedding_dim: int = 32,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super(GNNEnhancedActor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Combined input dimension
        combined_dim = state_dim + graph_embedding_dim
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Mean and log_std heads
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
        # Initialize output layers with smaller weights
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)
    
    def forward(
        self, 
        state: torch.Tensor, 
        graph_embedding: torch.Tensor
    ) -> tuple:
        """
        Forward pass
        
        Args:
            state: [batch_size, state_dim]
            graph_embedding: [batch_size, graph_embedding_dim]
            
        Returns:
            mean: [batch_size, action_dim]
            log_std: [batch_size, action_dim]
        """
        # Concatenate state and graph embedding
        combined = torch.cat([state, graph_embedding], dim=-1)
        
        # Feature extraction
        features = self.feature_net(combined)
        
        # Get mean and log_std
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(
        self, 
        state: torch.Tensor, 
        graph_embedding: torch.Tensor
    ) -> tuple:
        """
        Sample action from policy
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            mean: Mean of distribution
        """
        mean, log_std = self.forward(state, graph_embedding)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        
        return action, log_prob, mean
    
    def get_action(
        self, 
        state: np.ndarray, 
        graph_embedding: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Get action for deployment (numpy interface)
        
        Args:
            state: State observation
            graph_embedding: Graph embedding
            deterministic: If True, return mean action
            
        Returns:
            action: Numpy array
        """
        device = next(self.parameters()).device
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        graph_embedding = torch.FloatTensor(graph_embedding).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(state, graph_embedding)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.sample(state, graph_embedding)
        
        return action.cpu().numpy()[0]


class GNNEnhancedCritic(nn.Module):
    """
    Critic network (Q-function) that combines state, action, and graph embeddings
    
    Uses double Q-learning with two Q-networks
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        graph_embedding_dim: int = 32,
        hidden_dim: int = 256
    ):
        super(GNNEnhancedCritic, self).__init__()
        
        combined_dim = state_dim + action_dim + graph_embedding_dim
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for net in [self.q1_net, self.q2_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        graph_embedding: torch.Tensor
    ) -> tuple:
        """
        Forward pass through both Q-networks
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            graph_embedding: [batch_size, graph_embedding_dim]
            
        Returns:
            q1: Q-value from first network
            q2: Q-value from second network
        """
        # Concatenate inputs
        combined = torch.cat([state, action, graph_embedding], dim=-1)
        
        q1 = self.q1_net(combined)
        q2 = self.q2_net(combined)
        
        return q1, q2
    
    def q1(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        graph_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Get Q-value from first network only"""
        combined = torch.cat([state, action, graph_embedding], dim=-1)
        return self.q1_net(combined)


class GNNEnhancedValue(nn.Module):
    """
    Value network that estimates state value using graph embeddings
    
    Optional component for SAC (can be derived from Q-networks)
    """
    
    def __init__(
        self, 
        state_dim: int,
        graph_embedding_dim: int = 32,
        hidden_dim: int = 256
    ):
        super(GNNEnhancedValue, self).__init__()
        
        combined_dim = state_dim + graph_embedding_dim
        
        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.value_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor,
        graph_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: [batch_size, state_dim]
            graph_embedding: [batch_size, graph_embedding_dim]
            
        Returns:
            value: [batch_size, 1]
        """
        combined = torch.cat([state, graph_embedding], dim=-1)
        return self.value_net(combined)


class GraphAttentionFusion(nn.Module):
    """
    Attention-based fusion of state features and graph embeddings
    
    Can be used as an alternative to simple concatenation
    """
    
    def __init__(
        self,
        state_dim: int,
        graph_embedding_dim: int,
        output_dim: int
    ):
        super(GraphAttentionFusion, self).__init__()
        
        self.state_transform = nn.Linear(state_dim, output_dim)
        self.graph_transform = nn.Linear(graph_embedding_dim, output_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        graph_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse state and graph embeddings using attention
        
        Args:
            state: [batch_size, state_dim]
            graph_embedding: [batch_size, graph_embedding_dim]
            
        Returns:
            fused: [batch_size, output_dim]
        """
        # Transform inputs
        state_features = self.state_transform(state).unsqueeze(1)  # [B, 1, D]
        graph_features = self.graph_transform(graph_embedding).unsqueeze(1)  # [B, 1, D]
        
        # Concatenate for attention
        features = torch.cat([state_features, graph_features], dim=1)  # [B, 2, D]
        
        # Self-attention
        attended, _ = self.attention(features, features, features)
        
        # Pool and project
        fused = attended.mean(dim=1)  # [B, D]
        fused = self.output_proj(fused)
        
        return fused


def create_gnn_enhanced_networks(
    state_dim: int,
    action_dim: int,
    graph_embedding_dim: int = 32,
    hidden_dim: int = 256,
    device: str = 'cpu'
) -> tuple:
    """
    Factory function to create GNN-enhanced SAC networks
    
    Returns:
        actor: GNNEnhancedActor
        critic: GNNEnhancedCritic
        critic_target: GNNEnhancedCritic (target network)
    """
    actor = GNNEnhancedActor(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_embedding_dim=graph_embedding_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    critic = GNNEnhancedCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_embedding_dim=graph_embedding_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    critic_target = GNNEnhancedCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_embedding_dim=graph_embedding_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Initialize target network
    critic_target.load_state_dict(critic.state_dict())
    
    return actor, critic, critic_target


if __name__ == "__main__":
    # Test networks
    print("Testing GNN-Enhanced SAC Networks...")
    
    state_dim = 100
    action_dim = 5
    graph_embedding_dim = 32
    batch_size = 8
    
    # Create networks
    actor, critic, critic_target = create_gnn_enhanced_networks(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_embedding_dim=graph_embedding_dim
    )
    
    # Create dummy data
    state = torch.randn(batch_size, state_dim)
    graph_embedding = torch.randn(batch_size, graph_embedding_dim)
    
    # Test actor
    action, log_prob, mean = actor.sample(state, graph_embedding)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    
    # Test critic
    q1, q2 = critic(state, action, graph_embedding)
    print(f"Q1 shape: {q1.shape}")
    print(f"Q2 shape: {q2.shape}")
    
    print("All tests passed!")