"""
Scene Graph Builder
Constructs spatial scene graphs from room layout and furniture state
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set
from shapely.geometry import Point, Polygon
import json


class SceneGraphBuilder:
    """
    Builds graph representation of the room scene
    
    Nodes:
    - Furniture items (existing + candidates)
    - Zones (functional, priority)
    - Constraints (doors, windows, walls)
    
    Edges:
    - Spatial relationships (near, far, aligned)
    - Functional relationships (complements, requires)
    - Constraint relationships (violates, respects)
    """
    
    # Node types
    NODE_FURNITURE = 0
    NODE_ZONE = 1
    NODE_DOOR = 2
    NODE_WINDOW = 3
    NODE_WALL = 4
    
    # Edge types
    EDGE_NEAR = 0
    EDGE_FAR = 1
    EDGE_COMPLEMENTS = 2
    EDGE_REQUIRES = 3
    EDGE_VIOLATES = 4
    EDGE_AVOIDS = 5
    EDGE_ALIGNED_WITH = 6
    EDGE_IN_ZONE = 7
    
    def __init__(self, room_layout: Dict, placement_rules):
        """
        Args:
            room_layout: Room layout dictionary
            placement_rules: PlacementRulesGraph instance
        """
        self.room_layout = room_layout
        self.placement_rules = placement_rules
        
        # Room dimensions
        self.room_length = room_layout['room_info']['dimensions']['length']
        self.room_width = room_layout['room_info']['dimensions']['width']
        
        # Extract zones, doors, windows
        self.priority_zones = room_layout['available_space'].get('priority_zones', [])
        self.doors = room_layout.get('doors', [])
        self.windows = room_layout.get('windows', [])
        self.walls = room_layout.get('walls', [])
        
        # Catalog for tag lookup
        self.catalog = None
        
        # Distance thresholds
        self.near_threshold = 1.5
        self.far_threshold = 3.0
    
    def set_catalog(self, catalog: Dict):
        """Set furniture catalog for tag lookup"""
        self.catalog = catalog
    
    def build_graph(
        self, 
        existing_furniture: List[Dict], 
        candidate_item: Optional[Dict] = None,
        candidate_position: Optional[Tuple[float, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build scene graph from current state
        
        Args:
            existing_furniture: List of existing furniture items
            candidate_item: Candidate furniture item to place
            candidate_position: (x, y) position for candidate
            
        Returns:
            node_features: [num_nodes, node_feature_dim] tensor
            edge_index: [2, num_edges] tensor
            edge_attr: [num_edges, edge_feature_dim] tensor
        """
        nodes = []
        edges = []
        edge_features = []
        
        node_id = 0
        node_map = {}  # Maps (type, id) -> node_id
        
        # 1. Add existing furniture as nodes
        for furn in existing_furniture:
            node_features = self._create_furniture_node_features(furn)
            nodes.append(node_features)
            node_map[('furniture', furn['id'])] = node_id
            node_id += 1
        
        # 2. Add candidate furniture if provided
        candidate_node_id = None
        if candidate_item is not None and candidate_position is not None:
            candidate_dict = {
                'id': 'candidate',
                'type': candidate_item.get('type', 'unknown'),
                'position': {'x': candidate_position[0], 'y': candidate_position[1], 'z': 0.0},
                'dimensions': candidate_item.get('dimensions', {}),
                'tags': candidate_item.get('tags', {})
            }
            node_features = self._create_furniture_node_features(candidate_dict, is_candidate=True)
            nodes.append(node_features)
            candidate_node_id = node_id
            node_map[('furniture', 'candidate')] = node_id
            node_id += 1
        
        # 3. Add zone nodes
        for zone in self.priority_zones:
            node_features = self._create_zone_node_features(zone)
            nodes.append(node_features)
            node_map[('zone', zone['zone_name'])] = node_id
            node_id += 1
        
        # 4. Add door nodes
        for door in self.doors:
            node_features = self._create_door_node_features(door)
            nodes.append(node_features)
            node_map[('door', door['id'])] = node_id
            node_id += 1
        
        # 5. Add window nodes
        for window in self.windows:
            node_features = self._create_window_node_features(window)
            nodes.append(node_features)
            node_map[('window', window['id'])] = node_id
            node_id += 1
        
        # 6. Build edges
        all_furniture = existing_furniture.copy()
        if candidate_item is not None and candidate_position is not None:
            all_furniture.append({
                'id': 'candidate',
                'type': candidate_item.get('type', 'unknown'),
                'position': {'x': candidate_position[0], 'y': candidate_position[1], 'z': 0.0},
                'dimensions': candidate_item.get('dimensions', {}),
                'tags': candidate_item.get('tags', {})
            })
        
        # Furniture-to-Furniture edges
        for i, furn1 in enumerate(all_furniture):
            for j, furn2 in enumerate(all_furniture):
                if i >= j:
                    continue
                
                node1_id = node_map[('furniture', furn1['id'])]
                node2_id = node_map[('furniture', furn2['id'])]
                
                # Calculate distance
                dist = self._calculate_distance(furn1['position'], furn2['position'])
                
                # Add spatial edges
                if dist < self.near_threshold:
                    edges.append([node1_id, node2_id])
                    edge_features.append(self._create_edge_features(self.EDGE_NEAR, dist))
                    edges.append([node2_id, node1_id])  # bidirectional
                    edge_features.append(self._create_edge_features(self.EDGE_NEAR, dist))
                elif dist < self.far_threshold:
                    edges.append([node1_id, node2_id])
                    edge_features.append(self._create_edge_features(self.EDGE_FAR, dist))
                
                # Add functional edges (complements)
                if self._check_tag_compatibility(furn1, furn2):
                    edges.append([node1_id, node2_id])
                    edge_features.append(self._create_edge_features(self.EDGE_COMPLEMENTS, dist))
        
        # Furniture-to-Zone edges
        for furn in all_furniture:
            furn_node_id = node_map[('furniture', furn['id'])]
            furn_pos = (furn['position']['x'], furn['position']['y'])
            
            for zone in self.priority_zones:
                zone_node_id = node_map[('zone', zone['zone_name'])]
                zone_center = (zone['center']['x'], zone['center']['y'])
                dist = np.linalg.norm(np.array(furn_pos) - np.array(zone_center))
                
                # Check if furniture is in zone
                if dist <= zone['radius']:
                    edges.append([furn_node_id, zone_node_id])
                    edge_features.append(self._create_edge_features(self.EDGE_IN_ZONE, dist))
        
        # Furniture-to-Door edges (violation check)
        for furn in all_furniture:
            furn_node_id = node_map[('furniture', furn['id'])]
            furn_pos = (furn['position']['x'], furn['position']['y'])
            
            for door in self.doors:
                door_node_id = node_map[('door', door['id'])]
                door_pos = (door['position']['x'], door['position']['y'])
                dist = np.linalg.norm(np.array(furn_pos) - np.array(door_pos))
                
                clearance = door.get('clearance_radius', 1.2)
                
                if dist < clearance:
                    # Violation
                    edges.append([furn_node_id, door_node_id])
                    edge_features.append(self._create_edge_features(self.EDGE_VIOLATES, dist))
                else:
                    # Respects clearance
                    edges.append([furn_node_id, door_node_id])
                    edge_features.append(self._create_edge_features(self.EDGE_AVOIDS, dist))
        
        # Furniture-to-Window edges
        for furn in all_furniture:
            furn_node_id = node_map[('furniture', furn['id'])]
            furn_pos = (furn['position']['x'], furn['position']['y'])
            
            for window in self.windows:
                window_node_id = node_map[('window', window['id'])]
                window_pos = (window['position']['x'], window['position']['y'])
                dist = np.linalg.norm(np.array(furn_pos) - np.array(window_pos))
                
                # Check if blocking window
                if dist < 0.8:
                    edges.append([furn_node_id, window_node_id])
                    edge_features.append(self._create_edge_features(self.EDGE_VIOLATES, dist))
        
        # Convert to tensors
        node_features_tensor = torch.FloatTensor(np.array(nodes))
        
        if len(edges) > 0:
            edge_index_tensor = torch.LongTensor(np.array(edges).T)
            edge_attr_tensor = torch.FloatTensor(np.array(edge_features))
        else:
            # No edges - create empty tensors
            edge_index_tensor = torch.LongTensor(2, 0)
            edge_attr_tensor = torch.FloatTensor(0, 5)  # edge feature dim
        
        return node_features_tensor, edge_index_tensor, edge_attr_tensor
    
    def _create_furniture_node_features(self, furniture: Dict, is_candidate: bool = False) -> np.ndarray:
        """
        Create node features for furniture
        
        Features:
        - Node type (one-hot)
        - Position (x, y, normalized)
        - Dimensions (normalized)
        - Is candidate flag
        - Tag embeddings (aggregated)
        """
        features = []
        
        # Node type one-hot [furniture, zone, door, window, wall]
        node_type_onehot = [1, 0, 0, 0, 0]
        features.extend(node_type_onehot)
        
        # Position (normalized)
        pos_x = furniture['position']['x'] / self.room_length
        pos_y = furniture['position']['y'] / self.room_width
        features.extend([pos_x, pos_y])
        
        # Dimensions (normalized)
        dims = furniture.get('dimensions', {})
        length = dims.get('length', 1.0) / self.room_length
        width = dims.get('width', 1.0) / self.room_width
        height = dims.get('height', 1.0) / 3.0  # normalize by typical max height
        features.extend([length, width, height])
        
        # Is candidate
        features.append(1.0 if is_candidate else 0.0)
        
        # Tag features (binary indicators for common tags)
        common_tags = [
            'primary_seating', 'seating_companion', 'seating_accessory',
            'task_lighting', 'ambient_lighting', 'storage', 'wall_unit',
            'corner_decor', 'focal_point', 'surface', 'display_surface'
        ]
        
        item_tags = furniture.get('tags', {})
        primary_tags = item_tags.get('primary', [])
        secondary_tags = item_tags.get('secondary', [])
        all_tags = set(primary_tags + secondary_tags)
        
        tag_features = [1.0 if tag in all_tags else 0.0 for tag in common_tags]
        features.extend(tag_features)
        
        return np.array(features, dtype=np.float32)
    
    def _create_zone_node_features(self, zone: Dict) -> np.ndarray:
        """Create node features for zone"""
        features = []
        
        # Node type one-hot
        node_type_onehot = [0, 1, 0, 0, 0]
        features.extend(node_type_onehot)
        
        # Position (normalized)
        pos_x = zone['center']['x'] / self.room_length
        pos_y = zone['center']['y'] / self.room_width
        features.extend([pos_x, pos_y])
        
        # Dimensions (radius as "size")
        radius = zone['radius'] / max(self.room_length, self.room_width)
        features.extend([radius, radius, 0.0])  # symmetric, no height
        
        # Is candidate
        features.append(0.0)
        
        # Priority encoding
        priority_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        priority_val = priority_map.get(zone.get('priority', 'medium'), 0.5)
        
        # Tag features (mostly zeros for zones, but add priority as feature)
        tag_features = [priority_val] + [0.0] * 10
        features.extend(tag_features)
        
        return np.array(features, dtype=np.float32)
    
    def _create_door_node_features(self, door: Dict) -> np.ndarray:
        """Create node features for door"""
        features = []
        
        # Node type one-hot
        node_type_onehot = [0, 0, 1, 0, 0]
        features.extend(node_type_onehot)
        
        # Position (normalized)
        pos_x = door['position']['x'] / self.room_length
        pos_y = door['position']['y'] / self.room_width
        features.extend([pos_x, pos_y])
        
        # Dimensions
        width = door['dimensions']['width'] / self.room_length
        height = door['dimensions']['height'] / 3.0
        clearance = door.get('clearance_radius', 1.2) / self.room_length
        features.extend([width, height, clearance])
        
        # Is candidate
        features.append(0.0)
        
        # Tag features (high importance for clearance)
        tag_features = [1.0] + [0.0] * 10  # First feature = critical constraint
        features.extend(tag_features)
        
        return np.array(features, dtype=np.float32)
    
    def _create_window_node_features(self, window: Dict) -> np.ndarray:
        """Create node features for window"""
        features = []
        
        # Node type one-hot
        node_type_onehot = [0, 0, 0, 1, 0]
        features.extend(node_type_onehot)
        
        # Position (normalized)
        pos_x = window['position']['x'] / self.room_length
        pos_y = window['position']['y'] / self.room_width
        features.extend([pos_x, pos_y])
        
        # Dimensions
        width = window['dimensions']['width'] / self.room_length
        height = window['dimensions']['height'] / 3.0
        features.extend([width, height, 0.0])
        
        # Is candidate
        features.append(0.0)
        
        # Tag features
        tag_features = [0.7] + [0.0] * 10  # Medium importance
        features.extend(tag_features)
        
        return np.array(features, dtype=np.float32)
    
    def _create_edge_features(self, edge_type: int, distance: float) -> np.ndarray:
        """
        Create edge features
        
        Features:
        - Edge type (one-hot)
        - Distance (normalized)
        - Importance weight
        """
        features = []
        
        # Edge type one-hot [near, far, complements, requires, violates, avoids, aligned, in_zone]
        edge_type_onehot = [0] * 8
        edge_type_onehot[edge_type] = 1
        features.extend(edge_type_onehot)
        
        # Distance (normalized)
        max_room_diag = np.sqrt(self.room_length**2 + self.room_width**2)
        norm_dist = min(distance / max_room_diag, 1.0)
        features.append(norm_dist)
        
        # Importance weight based on edge type
        importance_map = {
            self.EDGE_VIOLATES: 1.0,
            self.EDGE_AVOIDS: 0.7,
            self.EDGE_NEAR: 0.8,
            self.EDGE_COMPLEMENTS: 0.9,
            self.EDGE_IN_ZONE: 0.85,
            self.EDGE_REQUIRES: 0.9,
            self.EDGE_FAR: 0.3,
            self.EDGE_ALIGNED_WITH: 0.6
        }
        importance = importance_map.get(edge_type, 0.5)
        features.append(importance)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate Euclidean distance between two positions"""
        p1 = np.array([pos1['x'], pos1['y']])
        p2 = np.array([pos2['x'], pos2['y']])
        return np.linalg.norm(p1 - p2)
    
    def _check_tag_compatibility(self, furn1: Dict, furn2: Dict) -> bool:
        """Check if two furniture items have compatible tags"""
        tags1 = furn1.get('tags', {})
        tags2 = furn2.get('tags', {})
        
        primary1 = set(tags1.get('primary', []))
        primary2 = set(tags2.get('primary', []))
        
        # Check for known complementary relationships
        complementary_pairs = [
            ('primary_seating', 'seating_companion'),
            ('primary_seating', 'seating_accessory'),
            ('seating_companion', 'task_lighting'),
            ('storage', 'display_surface'),
        ]
        
        for tag1 in primary1:
            for tag2 in primary2:
                if (tag1, tag2) in complementary_pairs or (tag2, tag1) in complementary_pairs:
                    return True
        
        return False
