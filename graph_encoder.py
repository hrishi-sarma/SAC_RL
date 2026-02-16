"""
Graph Neural Network Encoder for Scene Understanding

Uses Graph Attention Networks (GAT) to encode spatial and functional relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import Optional, Tuple


class SceneGraphEncoder(nn.Module):
    """
    Graph Attention Network for encoding scene graphs
    
    Architecture:
    - 3 GAT layers with multi-head attention
    - Skip connections for better gradient flow
    - Global pooling for scene-level representation
    - Outputs both node embeddings and graph-level embedding
    """
    
    def __init__(
        self,
        node_feature_dim: int = 22,  # 5 (type) + 2 (pos) + 3 (dims) + 1 (candidate) + 11 (tags)
        edge_feature_dim: int = 10,  # 8 (type) + 1 (dist) + 1 (importance)
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(SceneGraphEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers
        self.conv1 = GATConv(
            node_feature_dim, 
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.conv3 = GATConv(
            hidden_dim,
            output_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        # Skip connection projections
        self.skip1 = nn.Linear(node_feature_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Global pooling
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool
        
        # Graph-level encoding
        self.graph_encoder = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch indices for graphs [num_nodes] (for batched processing)
            
        Returns:
            node_embeddings: [num_nodes, output_dim]
            graph_embedding: [batch_size, output_dim] or [1, output_dim]
        """
        # Create batch if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Process edge features if provided (not directly used by GAT but can influence)
        if edge_attr is not None:
            edge_weights = edge_attr[:, -1]  # Use importance as edge weight
        else:
            edge_weights = None
        
        # First GAT layer with skip connection
        identity = self.skip1(x)
        x1 = self.conv1(x, edge_index)
        x1 = F.elu(x1)
        x1 = self.bn1(x1)
        x1 = x1 + identity
        x1 = self.dropout(x1)
        
        # Second GAT layer with skip connection
        identity = self.skip2(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.elu(x2)
        x2 = self.bn2(x2)
        x2 = x2 + identity
        x2 = self.dropout(x2)
        
        # Third GAT layer (output layer)
        node_embeddings = self.conv3(x2, edge_index)
        node_embeddings = F.elu(node_embeddings)
        
        # Global pooling for graph-level representation
        graph_mean = self.pool_mean(node_embeddings, batch)
        graph_max = self.pool_max(node_embeddings, batch)
        
        # Combine pooled representations
        graph_combined = torch.cat([graph_mean, graph_max], dim=-1)
        graph_embedding = self.graph_encoder(graph_combined)
        
        return node_embeddings, graph_embedding
    
    def get_candidate_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        candidate_idx: int = -1
    ) -> torch.Tensor:
        """
        Get embedding specifically for the candidate furniture item
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            candidate_idx: Index of candidate node (default: last node)
            
        Returns:
            candidate_embedding: [output_dim]
        """
        node_embeddings, _ = self.forward(x, edge_index, edge_attr)
        return node_embeddings[candidate_idx]


class GraphBasedRewardCalculator(nn.Module):
    """
    Uses graph embeddings to predict placement quality scores
    
    This can be used alongside the rule-based rewards
    """
    
    def __init__(
        self,
        graph_embedding_dim: int = 32,
        hidden_dim: int = 64
    ):
        super(GraphBasedRewardCalculator, self).__init__()
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(graph_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict placement quality score
        
        Args:
            graph_embedding: [batch_size, embedding_dim]
            
        Returns:
            quality_score: [batch_size, 1] in range [0, 1]
        """
        return self.reward_predictor(graph_embedding)


class MultiTaskGraphEncoder(nn.Module):
    """
    Multi-task GNN that predicts multiple aspects of placement quality
    """
    
    def __init__(
        self,
        node_feature_dim: int = 22,
        edge_feature_dim: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 32
    ):
        super(MultiTaskGraphEncoder, self).__init__()
        
        # Shared encoder
        self.encoder = SceneGraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Task-specific heads
        self.collision_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.clearance_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.zone_alignment_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.tag_compatibility_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.overall_quality_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs
        
        Returns:
            Dictionary with predictions for each task
        """
        # Get embeddings
        node_embeddings, graph_embedding = self.encoder(x, edge_index, edge_attr, batch)
        
        # Task predictions
        predictions = {
            'collision_free': self.collision_head(graph_embedding),
            'clearance_satisfied': self.clearance_head(graph_embedding),
            'zone_aligned': self.zone_alignment_head(graph_embedding),
            'tag_compatible': self.tag_compatibility_head(graph_embedding),
            'overall_quality': self.overall_quality_head(graph_embedding),
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding
        }
        
        return predictions


def create_graph_encoder(
    node_feature_dim: int = 22,
    edge_feature_dim: int = 10,
    hidden_dim: int = 64,
    output_dim: int = 32,
    device: str = 'cpu'
) -> SceneGraphEncoder:
    """
    Factory function to create and initialize a graph encoder
    
    Args:
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        device: Device to place model on
        
    Returns:
        Initialized SceneGraphEncoder
    """
    encoder = SceneGraphEncoder(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    encoder = encoder.to(device)
    
    # Initialize weights
    for m in encoder.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return encoder


if __name__ == "__main__":
    # Test the encoder
    print("Testing SceneGraphEncoder...")
    
    # Create dummy data
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn(num_nodes, 22)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 10)
    
    # Create encoder
    encoder = create_graph_encoder()
    
    # Forward pass
    node_emb, graph_emb = encoder(x, edge_index, edge_attr)
    
    print(f"Node embeddings shape: {node_emb.shape}")
    print(f"Graph embedding shape: {graph_emb.shape}")
    print("Test passed!")
