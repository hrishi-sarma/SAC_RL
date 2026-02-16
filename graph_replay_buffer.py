"""
Replay Buffer with Graph Data Support

Stores transitions including graph representations
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque
import pickle


class GraphReplayBuffer:
    """
    Experience replay buffer that stores graph data alongside transitions
    
    Each transition contains:
    - state: Environment observation
    - action: Action taken
    - reward: Reward received
    - next_state: Next observation
    - done: Terminal flag
    - graph_data: Graph representation (node features, edge index, edge attributes)
    - next_graph_data: Next state graph representation
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        graph_data: Dict[str, np.ndarray],
        next_graph_data: Dict[str, np.ndarray]
    ):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
            graph_data: Dictionary with 'node_features', 'edge_index', 'edge_attr'
            next_graph_data: Graph data for next state
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'graph_data': graph_data,
            'next_graph_data': next_graph_data
        }
        
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary containing batched tensors
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Graph data lists
        node_features_list = []
        edge_index_list = []
        edge_attr_list = []
        
        next_node_features_list = []
        next_edge_index_list = []
        next_edge_attr_list = []
        
        for idx in indices:
            transition = self.buffer[idx]
            
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_states.append(transition['next_state'])
            dones.append(transition['done'])
            
            # Current graph data
            graph_data = transition['graph_data']
            node_features_list.append(graph_data['node_features'])
            edge_index_list.append(graph_data['edge_index'])
            edge_attr_list.append(graph_data['edge_attr'])
            
            # Next graph data
            next_graph_data = transition['next_graph_data']
            next_node_features_list.append(next_graph_data['node_features'])
            next_edge_index_list.append(next_graph_data['edge_index'])
            next_edge_attr_list.append(next_graph_data['edge_attr'])
        
        # Convert to tensors
        batch = {
            'state': torch.FloatTensor(np.array(states)),
            'action': torch.FloatTensor(np.array(actions)),
            'reward': torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            'next_state': torch.FloatTensor(np.array(next_states)),
            'done': torch.FloatTensor(np.array(dones)).unsqueeze(1),
            
            # Graph data
            'graph_data': {
                'node_features': node_features_list,
                'edge_index': edge_index_list,
                'edge_attr': edge_attr_list
            },
            'next_graph_data': {
                'node_features': next_node_features_list,
                'edge_index': next_edge_index_list,
                'edge_attr': next_edge_attr_list
            }
        }
        
        return batch
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)
    
    def save(self, filepath: str):
        """Save buffer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, filepath: str):
        """Load buffer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data, maxlen=self.capacity)


class PrioritizedGraphReplayBuffer(GraphReplayBuffer):
    """
    Prioritized Experience Replay with graph data support
    
    Samples transitions based on TD-error priorities
    """
    
    def __init__(
        self, 
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_frames: Number of frames over which to anneal beta to 1.0
        """
        super().__init__(capacity)
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Use deque for priorities as well
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        graph_data: Dict[str, np.ndarray],
        next_graph_data: Dict[str, np.ndarray]
    ):
        """Add transition with maximum priority"""
        super().push(state, action, reward, next_state, done, graph_data, next_graph_data)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritization
        
        Returns:
            batch: Dictionary of batched data
            indices: Sampled indices (for updating priorities)
            weights: Importance sampling weights
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get batch data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        node_features_list = []
        edge_index_list = []
        edge_attr_list = []
        
        next_node_features_list = []
        next_edge_index_list = []
        next_edge_attr_list = []
        
        for idx in indices:
            transition = self.buffer[idx]
            
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_states.append(transition['next_state'])
            dones.append(transition['done'])
            
            graph_data = transition['graph_data']
            node_features_list.append(graph_data['node_features'])
            edge_index_list.append(graph_data['edge_index'])
            edge_attr_list.append(graph_data['edge_attr'])
            
            next_graph_data = transition['next_graph_data']
            next_node_features_list.append(next_graph_data['node_features'])
            next_edge_index_list.append(next_graph_data['edge_index'])
            next_edge_attr_list.append(next_graph_data['edge_attr'])
        
        batch = {
            'state': torch.FloatTensor(np.array(states)),
            'action': torch.FloatTensor(np.array(actions)),
            'reward': torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            'next_state': torch.FloatTensor(np.array(next_states)),
            'done': torch.FloatTensor(np.array(dones)).unsqueeze(1),
            
            'graph_data': {
                'node_features': node_features_list,
                'edge_index': edge_index_list,
                'edge_attr': edge_attr_list
            },
            'next_graph_data': {
                'node_features': next_node_features_list,
                'edge_index': next_edge_index_list,
                'edge_attr': next_edge_attr_list
            }
        }
        
        self.frame += 1
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions
        
        Args:
            indices: Indices of transitions
            priorities: New priority values (typically TD-errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    # Test replay buffer
    print("Testing GraphReplayBuffer...")
    
    buffer = GraphReplayBuffer(capacity=1000)
    
    # Add some transitions
    for i in range(10):
        state = np.random.randn(100)
        action = np.random.randn(5)
        reward = np.random.randn()
        next_state = np.random.randn(100)
        done = False
        
        graph_data = {
            'node_features': np.random.randn(10, 22),
            'edge_index': np.random.randint(0, 10, (2, 15)),
            'edge_attr': np.random.randn(15, 10)
        }
        
        next_graph_data = {
            'node_features': np.random.randn(11, 22),
            'edge_index': np.random.randint(0, 11, (2, 18)),
            'edge_attr': np.random.randn(18, 10)
        }
        
        buffer.push(state, action, reward, next_state, done, graph_data, next_graph_data)
    
    # Sample batch
    batch = buffer.sample(batch_size=4)
    
    print(f"State batch shape: {batch['state'].shape}")
    print(f"Action batch shape: {batch['action'].shape}")
    print(f"Number of graphs in batch: {len(batch['graph_data']['node_features'])}")
    
    print("Test passed!")
