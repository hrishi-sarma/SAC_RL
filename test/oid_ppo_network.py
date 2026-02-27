"""
OID-PPO Neural Network Architecture
Following Paper Figure 1 and Section: Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for OID-PPO
    Architecture from Paper Figure 1
    
    Components:
    1. Furniture Encoders: Two identical L-layer MLPs with GELU
    2. Occupancy Encoder: CNN with 3 convolutional layers
    3. Actor Head: Outputs μ and log(σ) for diagonal Gaussian policy
    4. Critic Head: Outputs value V(s)
    """
    
    def __init__(self, occupancy_map_shape: tuple, 
                 furniture_dim: int = 4,
                 hidden_dim: int = 128,
                 cnn_channels: list = [16, 32, 64]):
        """
        Initialize network.
        
        Args:
            occupancy_map_shape: (H, W) shape of occupancy map
            furniture_dim: Dimension of furniture descriptor (4: length, width, height, area)
            hidden_dim: Hidden dimension for MLPs
            cnn_channels: Channel progression for CNN
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.occupancy_shape = occupancy_map_shape
        self.furniture_dim = furniture_dim
        self.hidden_dim = hidden_dim
        
        # ===== Furniture Descriptor Encoders =====
        # Paper: "identical L-layer MLPs"
        # Using L=3 layers with GELU activation
        self.furniture_encoder = nn.Sequential(
            nn.Linear(furniture_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # ===== Occupancy Map Encoder (CNN) =====
        # Paper Figure 1: 3 convolutional layers with GELU
        self.occupancy_encoder = nn.Sequential(
            nn.Conv2d(1, cnn_channels[0], kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size()
        
        # ===== Shared Representation =====
        # Paper: "concatenated and passed through fully connected layers"
        combined_dim = hidden_dim + hidden_dim + self.cnn_output_size  # ψ_t + ψ_{t+1} + ψ_O
        
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU()
        )
        
        # ===== Actor Head (Diagonal Gaussian Policy) =====
        # Paper: "predict the mean μ and standard deviation σ"
        # Action: (x, y, rotation) ∈ ℝ³
        self.actor_mean = nn.Linear(256, 3)      # μ_t
        self.actor_log_std = nn.Linear(256, 3)   # log(σ_t)
        
        # Initialize actor head with small weights for stable initial policy
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
        nn.init.constant_(self.actor_log_std.bias, 0.0)
        
        # ===== Critic Head (Value Function) =====
        # Paper: "maps the shared embedding to a scalar value estimate"
        self.critic = nn.Linear(256, 1)
        
        # Initialize critic head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def _get_cnn_output_size(self):
        """Calculate CNN output dimension"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.occupancy_shape)
            cnn_output = self.occupancy_encoder(dummy_input)
            return cnn_output.shape[1]
    
    def forward(self, state_dict):
        """
        Forward pass through network.
        
        Args:
            state_dict: Dictionary with keys:
                - 'current_furniture': (batch, 4) current furniture descriptor e_t
                - 'next_furniture': (batch, 4) next furniture descriptor e_{t+1}
                - 'occupancy_map': (batch, 1, H, W) binary occupancy map O_t
        
        Returns:
            mu: (batch, 3) action mean
            log_std: (batch, 3) action log standard deviation
            value: (batch, 1) state value
        """
        # Encode current furniture ψ_t
        current_furniture = state_dict['current_furniture']
        psi_t = self.furniture_encoder(current_furniture)
        
        # Encode next furniture ψ_{t+1}
        next_furniture = state_dict['next_furniture']
        psi_t1 = self.furniture_encoder(next_furniture)
        
        # Encode occupancy map ψ_O
        occupancy_map = state_dict['occupancy_map']
        if len(occupancy_map.shape) == 3:
            occupancy_map = occupancy_map.unsqueeze(1)  # Add channel dimension
        psi_O = self.occupancy_encoder(occupancy_map)
        
        # Concatenate features h_t = concat[ψ_t, ψ_{t+1}, ψ_O]
        h_t = torch.cat([psi_t, psi_t1, psi_O], dim=-1)
        
        # Shared representation ϕ_t
        phi_t = self.shared_fc(h_t)
        
        # Actor outputs
        mu = self.actor_mean(phi_t)
        log_std = self.actor_log_std(phi_t)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
        
        # Critic output
        value = self.critic(phi_t)
        
        return mu, log_std, value
    
    def get_action(self, state_dict, deterministic=False):
        """
        Sample action from diagonal Gaussian policy.
        Paper: a_t = μ_t + σ_t ⊙ z, where z ~ N(0, I)
        
        Args:
            state_dict: State dictionary
            deterministic: If True, return mean (no sampling)
        
        Returns:
            action: (batch, 3) sampled action
            log_prob: (batch,) log probability of action
            value: (batch, 1) state value
        """
        mu, log_std, value = self.forward(state_dict)
        
        if deterministic:
            return mu, torch.zeros_like(mu[:,0]), value
        
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        
        return action, log_prob, value
    
    def evaluate_actions(self, state_dict, actions):
        """
        Evaluate actions for PPO update.
        
        Args:
            state_dict: State dictionary
            actions: (batch, 3) actions to evaluate
        
        Returns:
            log_probs: (batch,) log probabilities
            values: (batch, 1) state values
            entropy: (batch,) policy entropy
        """
        mu, log_std, value = self.forward(state_dict)
        
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, value, entropy


def test_network():
    """Test network architecture"""
    print("Testing ActorCriticNetwork...")
    
    # Create dummy inputs
    batch_size = 4
    occupancy_shape = (60, 50)  # 10cm resolution for 6m x 5m room
    
    network = ActorCriticNetwork(occupancy_shape)
    
    state_dict = {
        'current_furniture': torch.randn(batch_size, 4),
        'next_furniture': torch.randn(batch_size, 4),
        'occupancy_map': torch.rand(batch_size, 1, *occupancy_shape)
    }
    
    # Forward pass
    mu, log_std, value = network(state_dict)
    
    print(f"✓ Forward pass successful")
    print(f"  Action mean shape: {mu.shape}")
    print(f"  Action log_std shape: {log_std.shape}")
    print(f"  Value shape: {value.shape}")
    
    # Sample action
    action, log_prob, value = network.get_action(state_dict)
    
    print(f"✓ Action sampling successful")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    
    # Evaluate actions
    log_probs, values, entropy = network.evaluate_actions(state_dict, action)
    
    print(f"✓ Action evaluation successful")
    print(f"  Entropy shape: {entropy.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✅ Network architecture test passed!")


if __name__ == '__main__':
    test_network()
