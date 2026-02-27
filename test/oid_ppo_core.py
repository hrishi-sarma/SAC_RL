"""
OID-PPO: Optimal Interior Design using Proximal Policy Optimization
EXACT implementation following the paper: arXiv:2508.00364v1

Authors: Yoon et al., AAAI 2026
Implementation: Complete PyTorch version with all components from paper
"""

import numpy as np
import json
import heapq
from typing import List, Dict, Tuple, Optional

class FurnitureItem:
    """
    Represents a furniture item with its geometric properties.
    Used in MDP state representation (Section 2).
    """
    def __init__(self, id: str, name: str, dimensions: dict, type_name: str):
        self.id = id
        self.name = name
        self.length = dimensions['length']  # Along x-axis
        self.width = dimensions['width']    # Along y-axis  
        self.height = dimensions['height']
        self.type = type_name
        self.area = self.length * self.width  # Footprint area
        
    def get_footprint(self, x: float, y: float, rotation: int) -> np.ndarray:
        """
        Get 2D footprint polygon T(x,k;f) = z_k Π(f) + x
        (Paper Section 2: Problem Definition)
        
        Args:
            x, y: Position coordinates
            rotation: Rotation index k ∈ {0,1,2,3} for 0°,90°,180°,270°
            
        Returns:
            corners: (4, 2) array of corner coordinates
        """
        # Apply rotation: swap dimensions for 90° and 270°
        if rotation % 2 == 0:
            l, w = self.length, self.width
        else:
            l, w = self.width, self.length
            
        # Canonical footprint centered at origin
        corners = np.array([
            [-l/2, -w/2],
            [l/2, -w/2],
            [l/2, w/2],
            [-l/2, w/2]
        ])
        
        # Translate to position
        corners += np.array([x, y])
        return corners
    
    def get_front_direction(self, rotation: int) -> np.ndarray:
        """
        Get front-facing unit vector n_f
        Front is defined as positive y-direction at rotation=0
        
        Args:
            rotation: Rotation index k ∈ {0,1,2,3}
            
        Returns:
            n_f: Unit vector (2,) pointing to furniture front
        """
        angle = rotation * np.pi / 2
        return np.array([np.sin(angle), np.cos(angle)])
    
    def get_descriptor(self) -> np.ndarray:
        """
        Get geometric descriptor e_t for neural network input
        (Paper Section 2: State definition)
        
        Returns:
            descriptor: (4,) array [length, width, height, area]
        """
        return np.array([self.length, self.width, self.height, self.area])


class InteriorDesignEnv:
    """
    Interior Design Environment as Markov Decision Process (MDP)
    Following Paper Section 2: Problem Definition
    
    MDP formulation: M = ⟨S, A, P, R, γ⟩
    - S: State space (e_t, e_{t+1}, O_t)
    - A: Action space (x, y, k) ∈ ℝ² × {0,1,2,3}
    - P: Deterministic transition function
    - R: Reward function R_idg ∈ [-1, 1]
    - γ: Discount factor (0.99)
    """
    
    def __init__(self, 
                 room_dimensions: dict, 
                 door_positions: List[np.ndarray], 
                 furniture_list: List[FurnitureItem],
                 map_resolution: float = 0.10):
        """
        Initialize environment.
        
        Args:
            room_dimensions: {'length': N, 'width': M} in meters
            door_positions: List of door center positions
            furniture_list: List of FurnitureItem objects (MUST be sorted by area descending)
            map_resolution: Grid resolution in meters (default 0.10m = 10cm)
        """
        # Room dimensions (N × M from paper)
        self.N = room_dimensions['length']
        self.M = room_dimensions['width']
        self.room_bounds = np.array([[0, 0], [self.N, 0], [self.N, self.M], [0, self.M]])
        
        # Doors (set D from paper)
        self.doors = door_positions
        
        # Furniture list F (sorted by area - Proposition 1)
        self.furniture_list = furniture_list
        self.n_furniture = len(furniture_list)
        
        # Verify furniture is sorted by area (descending)
        areas = [f.area for f in furniture_list]
        assert all(areas[i] >= areas[i+1] for i in range(len(areas)-1)), \
            "Furniture must be sorted by area (descending) as per paper"
        
        # Diagonal length d_△ for normalization
        self.d_triangle = np.sqrt(self.N**2 + self.M**2)
        
        # Room center o
        self.room_center = np.array([self.N/2, self.M/2])
        
        # Reference spatial variance κ²_E (uniform distribution)
        # Paper Section 3: Visual Balance
        self.kappa_E_squared = (self.N**2 + self.M**2) / 12
        
        # Walls with normals (pointing inward)
        self.walls = [
            {'start': np.array([0, 0]), 'end': np.array([self.N, 0]), 'normal': np.array([0, 1])},      # South
            {'start': np.array([self.N, 0]), 'end': np.array([self.N, self.M]), 'normal': np.array([-1, 0])},  # East
            {'start': np.array([self.N, self.M]), 'end': np.array([0, self.M]), 'normal': np.array([0, -1])},  # North
            {'start': np.array([0, self.M]), 'end': np.array([0, 0]), 'normal': np.array([1, 0])}       # West
        ]
        
        # Define functional pairs P (parent, child, α_pc)
        # α = -1 for face-to-face, +1 for parallel
        # Paper Section 3: Pairwise Relationship
        self.functional_pairs = self._define_functional_pairs()
        
        # Occupancy map resolution
        self.map_resolution = map_resolution
        
        # State tracking
        self.current_step = 0
        self.placed_furniture = []
        self.occupancy_map = None
        
        # Episode statistics
        self.episode_invalid_count = 0
        
        self.reset()
    
    def _define_functional_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Define functional furniture pairs P with alignment preference α_pc
        Paper Section 3: Pairwise Relationship
        
        Returns:
            pairs: List of (parent_type, child_type, alpha) tuples
                   alpha = -1 for face-to-face alignment
                   alpha = +1 for parallel alignment
        """
        pairs = []
        
        # Coffee table pairs with sofa (face-to-face)
        if any(f.type == 'coffee_table' for f in self.furniture_list):
            pairs.append(('sofa', 'coffee_table', -1))
        
        # Table lamp pairs with coffee table (parallel/on top)
        if any(f.type == 'table_lamp' for f in self.furniture_list):
            pairs.append(('coffee_table', 'table_lamp', 1))
        
        # Magazine rack with sofa or coffee table
        if any(f.type == 'magazine_rack' for f in self.furniture_list):
            pairs.append(('sofa', 'magazine_rack', 1))
            pairs.append(('coffee_table', 'magazine_rack', 1))
        
        # Storage basket placement
        if any(f.type == 'storage_basket' for f in self.furniture_list):
            pairs.append(('sofa', 'storage_basket', 1))
        
        return pairs
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state s_0
        
        Returns:
            state: Initial state dict with keys:
                   - 'current_furniture': e_t (4,)
                   - 'next_furniture': e_{t+1} (4,)
                   - 'occupancy_map': O_t (H, W)
        """
        self.current_step = 0
        self.placed_furniture = []
        self.episode_invalid_count = 0
        
        # Initialize binary occupancy map O_0
        map_h = int(np.ceil(self.M / self.map_resolution))
        map_w = int(np.ceil(self.N / self.map_resolution))
        self.occupancy_map = np.zeros((map_h, map_w), dtype=np.float32)
        
        return self._get_state()
    
    def _get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current state s_t = (e_t, e_{t+1}, O_t)
        Paper Section 2: State definition
        
        Returns:
            state: Dictionary with current furniture descriptor,
                   next furniture descriptor, and occupancy map
        """
        # Current furniture descriptor e_t
        if self.current_step < self.n_furniture:
            e_t = self.furniture_list[self.current_step].get_descriptor()
        else:
            e_t = np.zeros(4, dtype=np.float32)
        
        # Next furniture descriptor e_{t+1}
        if self.current_step + 1 < self.n_furniture:
            e_t1 = self.furniture_list[self.current_step + 1].get_descriptor()
        else:
            e_t1 = np.zeros(4, dtype=np.float32)
        
        # Occupancy map O_t
        O_t = self.occupancy_map.copy()
        
        return {
            'current_furniture': e_t,
            'next_furniture': e_t1,
            'occupancy_map': O_t
        }
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step of the MDP: s_{t+1}, r_t, done ← P(s_t, a_t)
        Paper Section 2: Transition function P
        
        Args:
            action: (3,) array [x, y, rotation] where
                    x ∈ [0, N], y ∈ [0, M], rotation ∈ [0, 3]
        
        Returns:
            next_state: Next state s_{t+1}
            reward: Reward r_t (0 for intermediate, R_idg for terminal, φ for invalid)
            done: Whether episode terminated
            info: Additional information dict
        """
        if self.current_step >= self.n_furniture:
            raise ValueError("Episode already terminated (Proposition 1: H ≤ |F|)")
        
        x, y, rotation = action
        rotation = int(np.round(rotation)) % 4  # Discretize rotation
        
        furniture = self.furniture_list[self.current_step]
        
        # Check validity (Definition 1 from paper)
        if not self._is_valid_placement(furniture, x, y, rotation):
            # Invalid action → terminal state with penalty φ
            # Paper Section 2: "episode terminates immediately"
            reward = -10.0  # φ = -10 from paper
            done = True
            self.episode_invalid_count += 1
            
            info = {
                'valid': False,
                'reason': 'Invalid placement (collision or out of bounds)',
                'step': self.current_step,
                'furniture': furniture.name
            }
            
            # Transition to terminal state
            terminal_state = {
                'current_furniture': np.zeros(4, dtype=np.float32),
                'next_furniture': np.zeros(4, dtype=np.float32),
                'occupancy_map': self.occupancy_map
            }
            
            return terminal_state, reward, done, info
        
        # Valid placement - update environment
        self._update_occupancy_map(furniture, x, y, rotation)
        
        # Store placement information
        self.placed_furniture.append({
            'furniture': furniture,
            'position': np.array([x, y]),
            'rotation': rotation,
            'footprint': furniture.get_footprint(x, y, rotation),
            'front_direction': furniture.get_front_direction(rotation)
        })
        
        self.current_step += 1
        
        # Check if episode complete (Proposition 1: H ≤ |F|)
        if self.current_step >= self.n_furniture:
            # All furniture placed - calculate R_idg
            reward = self._calculate_reward()
            done = True
            info = {
                'valid': True,
                'complete': True,
                'num_placed': len(self.placed_furniture)
            }
        else:
            # Intermediate step - no reward yet (sparse reward)
            reward = 0.0
            done = False
            info = {
                'valid': True,
                'complete': False,
                'step': self.current_step
            }
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _is_valid_placement(self, furniture: FurnitureItem, 
                           x: float, y: float, rotation: int) -> bool:
        """
        Check if placement satisfies Definition 1 (Valid Placement) from paper
        
        Valid if and only if:
        1. T(x,k;f) ⊂ E (footprint inside room)
        2. T(x,k;f) ∩ O_t = ∅ (no overlap with existing furniture)
        
        Args:
            furniture: Furniture item to place
            x, y: Position coordinates
            rotation: Rotation index
            
        Returns:
            valid: True if placement is valid
        """
        footprint = furniture.get_footprint(x, y, rotation)
        
        # Check 1: Must be inside room boundary E
        if not self._is_inside_room(footprint):
            return False
        
        # Check 2: Must not overlap with occupancy O_t
        if self._has_overlap(footprint):
            return False
        
        return True
    
    def _is_inside_room(self, footprint: np.ndarray) -> bool:
        """Check if footprint is completely inside room E = [0,N] × [0,M]"""
        min_x, min_y = footprint.min(axis=0)
        max_x, max_y = footprint.max(axis=0)
        
        return (min_x >= 0 and max_x <= self.N and 
                min_y >= 0 and max_y <= self.M)
    
    def _has_overlap(self, footprint: np.ndarray) -> bool:
        """Check if footprint overlaps with occupancy map O_t"""
        min_x, min_y = footprint.min(axis=0)
        max_x, max_y = footprint.max(axis=0)
        
        # Convert to grid indices
        grid_min_x = max(0, int(min_x / self.map_resolution))
        grid_min_y = max(0, int(min_y / self.map_resolution))
        grid_max_x = min(self.occupancy_map.shape[1], int(np.ceil(max_x / self.map_resolution)))
        grid_max_y = min(self.occupancy_map.shape[0], int(np.ceil(max_y / self.map_resolution)))
        
        # Check for overlap
        if grid_max_x > grid_min_x and grid_max_y > grid_min_y:
            region = self.occupancy_map[grid_min_y:grid_max_y, grid_min_x:grid_max_x]
            if np.any(region > 0.5):
                return True
        
        return False
    
    def _update_occupancy_map(self, furniture: FurnitureItem,
                              x: float, y: float, rotation: int):
        """
        Update occupancy map: O_{t+1} = O_t ∪ T(x,k;f)
        """
        footprint = furniture.get_footprint(x, y, rotation)
        min_x, min_y = footprint.min(axis=0)
        max_x, max_y = footprint.max(axis=0)
        
        grid_min_x = max(0, int(min_x / self.map_resolution))
        grid_min_y = max(0, int(min_y / self.map_resolution))
        grid_max_x = min(self.occupancy_map.shape[1], int(np.ceil(max_x / self.map_resolution)))
        grid_max_y = min(self.occupancy_map.shape[0], int(np.ceil(max_y / self.map_resolution)))
        
        # Mark region as occupied
        if grid_max_x > grid_min_x and grid_max_y > grid_min_y:
            self.occupancy_map[grid_min_y:grid_max_y, grid_min_x:grid_max_x] = 1.0
    
    def _calculate_reward(self) -> float:
        """
        Calculate composite reward R_idg following Lemma 1 and Section 3
        
        R_idg = (1/6)(R_pair + R_a + R_v + R_path + R_b + R_al) ∈ [-1, 1]
        
        Returns:
            R_idg: Composite reward (arithmetic mean of 6 components)
        """
        # Calculate all six reward components
        R_pair = self._reward_pairwise()
        R_a = self._reward_accessibility()
        R_v = self._reward_visibility()
        R_path = self._reward_pathway()
        R_b = self._reward_balance()
        R_al = self._reward_alignment()
        
        # Store for analysis
        self.last_rewards = {
            'R_pair': float(R_pair),
            'R_a': float(R_a),
            'R_v': float(R_v),
            'R_path': float(R_path),
            'R_b': float(R_b),
            'R_al': float(R_al)
        }
        
        # Composite reward (Lemma 1: arithmetic mean)
        R_idg = (R_pair + R_a + R_v + R_path + R_b + R_al) / 6.0
        
        return float(R_idg)
    
    # Reward functions continue in next part...
