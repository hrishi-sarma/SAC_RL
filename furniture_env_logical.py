import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import copy


class FurnitureRecommendationEnvLogical(gym.Env):
    """
    Enhanced RL Environment with Logical Placement Rules
    
    Key Features:
    - Rule-based placement preferences (wall-mounted, corner, near-seating, etc.)
    - Intelligent distance-based rewards
    - Furniture-specific placement logic
    - Grid alignment for organized layouts
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        room_layout_path: str, 
        catalog_path: str, 
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15
    ):
        super().__init__()
        
        # Load data
        with open(room_layout_path, 'r') as f:
            self.room_layout = json.load(f)
        with open(catalog_path, 'r') as f:
            self.catalog = json.load(f)
        
        self.max_items = max_items
        self.current_step = 0
        self.max_steps = max_items * 3
        
        # Grid and buffer settings
        self.grid_size = grid_size
        self.collision_buffer = collision_buffer
        
        # Extract room parameters
        self.room_length = self.room_layout['room_info']['dimensions']['length']
        self.room_width = self.room_layout['room_info']['dimensions']['width']
        self.min_clearance = self.room_layout['constraints']['min_clearance']
        self.min_pathway = self.room_layout['constraints']['min_pathway_width']
        
        # Initialize state
        self.existing_furniture = copy.deepcopy(self.room_layout['existing_furniture'])
        self.placed_items = []
        self.budget_used = 0
        self.budget_max = self.room_layout['constraints']['budget_remaining']
        
        # Catalog info
        self.furniture_catalog = self.catalog['furniture_items']
        self.num_catalog_items = len(self.furniture_catalog)
        
        # Track placed catalog items
        self.placed_catalog_ids = set()
        
        # Build grid reference points
        self.grid_points_x = np.arange(0, self.room_length + self.grid_size, self.grid_size)
        self.grid_points_y = np.arange(0, self.room_width + self.grid_size, self.grid_size)
        
        # Identify corner zones
        self.corner_zones = self._identify_corner_zones()
        
        # Define observation space
        self.state_dim = self._calculate_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.8]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.2]),
            dtype=np.float32
        )
    
    def _identify_corner_zones(self, corner_radius: float = 1.0) -> List[Dict]:
        """Identify corner zones in the room"""
        return [
            {'center': (corner_radius, corner_radius), 'radius': corner_radius},
            {'center': (self.room_length - corner_radius, corner_radius), 'radius': corner_radius},
            {'center': (corner_radius, self.room_width - corner_radius), 'radius': corner_radius},
            {'center': (self.room_length - corner_radius, self.room_width - corner_radius), 'radius': corner_radius}
        ]
    
    def _is_in_corner(self, x: float, y: float) -> bool:
        """Check if position is in a corner zone"""
        for corner in self.corner_zones:
            cx, cy = corner['center']
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist <= corner['radius']:
                return True
        return False
    
    def _calculate_state_dim(self) -> int:
        """Calculate state vector dimension"""
        # Room features: 6
        # Existing furniture encoding: 10 * 8 = 80
        # Budget info: 2
        # Free space: 1
        # Step counter: 1
        # Catalog mask: 20
        # Spatial occupancy: 10
        return 6 + 80 + 2 + 1 + 1 + 20 + 10
    
    def _get_state(self) -> np.ndarray:
        """Generate state representation"""
        state = []
        
        # Room features
        state.extend([
            self.room_length / 10.0,
            self.room_width / 10.0,
            self.room_length * self.room_width / 100.0,
            self._encode_style(self.room_layout['room_info']['style_preference']),
            len(self.existing_furniture) / 10.0,
            self.current_step / self.max_steps
        ])
        
        # Existing furniture encoding
        max_furniture = 8
        furniture_features = []
        all_furniture = self.existing_furniture + self.placed_items
        
        for i in range(max_furniture):
            if i < len(all_furniture):
                item = all_furniture[i]
                furniture_features.extend([
                    item['position']['x'] / self.room_length,
                    item['position']['y'] / self.room_width,
                    item['dimensions']['length'] / 5.0,
                    item['dimensions']['width'] / 5.0,
                    item['dimensions']['height'] / 3.0,
                    item['rotation'] / np.pi,
                    self._encode_category(item['category']),
                    item.get('price', 0) / 1000.0,
                    self._encode_style(item.get('style', 'modern')),
                    1.0
                ])
            else:
                furniture_features.extend([0.0] * 10)
        
        state.extend(furniture_features)
        
        # Budget info
        state.extend([
            self.budget_used / self.budget_max,
            (self.budget_max - self.budget_used) / self.budget_max
        ])
        
        # Free space
        state.append(self._calculate_free_space())
        
        # Step counter
        state.append(self.current_step / self.max_steps)
        
        # Action mask
        action_mask = []
        for item in self.furniture_catalog:
            available = 1.0 if item['id'] not in self.placed_catalog_ids else 0.0
            action_mask.append(available)
        
        state.extend(action_mask)
        
        # Spatial occupancy zones
        occupancy = self._calculate_zone_occupancy()
        state.extend(occupancy)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_zone_occupancy(self) -> List[float]:
        """Calculate occupancy in different zones"""
        zones = []
        zone_width = self.room_length / 5
        zone_height = self.room_width / 2
        
        for i in range(2):
            for j in range(5):
                zone_x_min = j * zone_width
                zone_x_max = (j + 1) * zone_width
                zone_y_min = i * zone_height
                zone_y_max = (i + 1) * zone_height
                
                occupied = 0.0
                for item in self.existing_furniture + self.placed_items:
                    x = item['position']['x']
                    y = item['position']['y']
                    if zone_x_min <= x <= zone_x_max and zone_y_min <= y <= zone_y_max:
                        occupied = 1.0
                        break
                
                zones.append(occupied)
        
        return zones
    
    def _encode_style(self, style: str) -> float:
        """Encode style as float"""
        style_map = {
            'modern': 0.9,
            'modern_minimalist': 0.8,
            'contemporary': 0.7,
            'traditional': 0.5,
            'rustic': 0.3,
            'industrial': 0.4
        }
        return style_map.get(style, 0.5)
    
    def _encode_category(self, category: str) -> float:
        """Encode category as float"""
        category_map = {
            'seating': 0.9,
            'tables': 0.7,
            'storage': 0.5,
            'lighting': 0.3,
            'decor': 0.1
        }
        return category_map.get(category, 0.5)
    
    def _calculate_free_space(self) -> float:
        """Calculate percentage of free floor space"""
        total_area = self.room_length * self.room_width
        furniture_area = 0
        
        for item in self.existing_furniture + self.placed_items:
            furniture_area += item['dimensions']['length'] * item['dimensions']['width']
        
        return max(0.0, 1.0 - furniture_area / total_area)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.placed_items = []
        self.budget_used = 0
        self.placed_catalog_ids = set()
        
        state = self._get_state()
        info = {'placed_items': 0, 'budget_used': 0}
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        self.current_step += 1
        
        # Decode action
        catalog_idx_raw = int(action[0] * self.num_catalog_items)
        catalog_idx = np.clip(catalog_idx_raw, 0, self.num_catalog_items - 1)
        
        x_norm = np.clip(action[1], 0.0, 1.0)
        y_norm = np.clip(action[2], 0.0, 1.0)
        rot_norm = np.clip(action[3], 0.0, 1.0)
        scale = np.clip(action[4], 0.8, 1.2)
        
        # Snap to grid
        x = self._snap_to_grid(x_norm * self.room_length, self.grid_points_x)
        y = self._snap_to_grid(y_norm * self.room_width, self.grid_points_y)
        
        # Discrete rotation (0°, 90°, 180°, 270°)
        rotation_steps = [0, np.pi/2, np.pi, 3*np.pi/2]
        rotation_idx = int(rot_norm * 4)
        rotation_idx = np.clip(rotation_idx, 0, 3)
        rotation = rotation_steps[rotation_idx]
        
        # Get catalog item
        catalog_item = self.furniture_catalog[catalog_idx]
        
        # Check if already placed
        already_placed = catalog_item['id'] in self.placed_catalog_ids
        
        # Create furniture dict
        furniture = {
            'id': catalog_item['id'],
            'type': catalog_item['type'],
            'category': catalog_item['category'],
            'position': {'x': x, 'y': y, 'z': 0.0},
            'dimensions': {
                'length': catalog_item['dimensions']['length'] * scale,
                'width': catalog_item['dimensions']['width'] * scale,
                'height': catalog_item['dimensions']['height']
            },
            'rotation': rotation,
            'color': catalog_item['color'],
            'material': catalog_item['material'],
            'style': catalog_item['style'],
            'price': catalog_item['price'],
            'price_tier': catalog_item['price_tier']
        }
        
        # Validate placement
        valid, collision_type = self._validate_placement(furniture)
        
        # Calculate reward
        reward, reward_components = self._calculate_reward(
            furniture, catalog_item, valid, collision_type, already_placed
        )
        
        # Update state if valid
        if valid and not already_placed:
            self.placed_items.append(furniture)
            self.placed_catalog_ids.add(catalog_item['id'])
            self.budget_used += catalog_item['price']
        
        # Check termination
        terminated = len(self.placed_items) >= self.max_items
        truncated = self.current_step >= self.max_steps
        
        next_state = self._get_state()
        
        info = {
            'placed_items': len(self.placed_items),
            'budget_used': self.budget_used,
            'selected_furniture': catalog_item['type'],
            'valid_placement': valid,
            'reward_components': reward_components,
            'already_placed': already_placed
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _snap_to_grid(self, value: float, grid_points: np.ndarray) -> float:
        """Snap value to nearest grid point"""
        idx = np.argmin(np.abs(grid_points - value))
        return grid_points[idx]
    
    def _validate_placement(self, furniture: Dict) -> Tuple[bool, str]:
        """Validate furniture placement with enhanced collision detection"""
        x, y = furniture['position']['x'], furniture['position']['y']
        length, width = furniture['dimensions']['length'], furniture['dimensions']['width']
        
        # Check room bounds
        half_l, half_w = length / 2, width / 2
        if (x - half_l < 0 or x + half_l > self.room_length or
            y - half_w < 0 or y + half_w > self.room_width):
            return False, 'out_of_bounds'
        
        # Check door clearance
        for door in self.room_layout['doors']:
            door_x = door['position']['x']
            door_y = door['position']['y']
            clearance = door['clearance_radius']
            
            dist = np.sqrt((x - door_x)**2 + (y - door_y)**2)
            if dist < clearance:
                return False, 'blocks_door'
        
        # Check collision with existing furniture
        furniture_poly = self._create_furniture_polygon(furniture)
        
        for existing in self.existing_furniture + self.placed_items:
            existing_poly = self._create_furniture_polygon(existing)
            
            if furniture_poly.intersects(existing_poly):
                return False, 'collision'
            
            # Enhanced buffer check
            distance = furniture_poly.distance(existing_poly)
            if distance < self.collision_buffer:
                return False, 'collision'
        
        return True, 'valid'
    
    def _create_furniture_polygon(self, furniture: Dict) -> Polygon:
        """Create polygon for furniture footprint"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        length = furniture['dimensions']['length']
        width = furniture['dimensions']['width']
        rotation = furniture.get('rotation', 0)
        
        half_l, half_w = length / 2, width / 2
        corners = [
            (-half_l, -half_w),
            (half_l, -half_w),
            (half_l, half_w),
            (-half_l, half_w)
        ]
        
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_r - cy * sin_r + x
            ry = cx * sin_r + cy * cos_r + y
            rotated.append((rx, ry))
        
        return Polygon(rotated)
    
    def _calculate_reward(
        self, 
        furniture: Dict, 
        catalog_item: Dict, 
        valid: bool, 
        collision_type: str,
        already_placed: bool
    ) -> Tuple[float, Dict]:
        """
        Calculate reward with LOGICAL PLACEMENT RULES
        """
        components = {
            'valid_placement': valid,
            'collision': collision_type != 'valid',
            'out_of_bounds': collision_type == 'out_of_bounds',
            'blocks_door': collision_type == 'blocks_door',
            'already_placed': already_placed
        }
        
        if already_placed:
            total_reward = -5.0
            components['total'] = total_reward
            return total_reward, components
        
        if not valid:
            penalty_map = {
                'out_of_bounds': -3.0,
                'blocks_door': -3.0,
                'collision': -2.0
            }
            total_reward = penalty_map.get(collision_type, -2.0)
            components['total'] = total_reward
            return total_reward, components
        
        # Get placement rules from catalog
        placement_rules = catalog_item.get('placement_rules', {})
        placement_type = placement_rules.get('placement_type', 'free_standing')
        
        # CORE LOGICAL PLACEMENT REWARDS
        placement_reward = self._reward_logical_placement(furniture, catalog_item, placement_rules)
        components['logical_placement'] = placement_reward
        
        # Existing reward components (weighted lower now)
        components['functional_pairing'] = self._reward_functional_pairing(furniture, catalog_item)
        components['accessibility'] = self._reward_accessibility(furniture)
        components['clearance'] = self._reward_clearance(furniture)
        components['visual_balance'] = self._reward_visual_balance(furniture)
        components['grid_alignment'] = 1.0  # Always aligned due to grid
        components['parallel_placement'] = self._reward_parallel_placement(furniture)
        components['color_harmony'] = self._reward_color_harmony(furniture, catalog_item)
        components['diversity'] = self._reward_diversity(furniture, catalog_item)
        components['budget_efficiency'] = self._reward_budget_efficiency(catalog_item)
        components['completeness'] = self._reward_completeness()
        components['size_appropriateness'] = self._reward_size_appropriateness(furniture, catalog_item)
        
        # ADJUSTED WEIGHTS - Logical placement gets highest weight
        weights = {
            'logical_placement': 3.0,      # NEW: Highest priority
            'functional_pairing': 1.0,
            'accessibility': 1.5,
            'clearance': 1.2,
            'visual_balance': 1.0,
            'grid_alignment': 1.0,
            'parallel_placement': 1.0,
            'color_harmony': 0.8,
            'diversity': 1.5,
            'budget_efficiency': 0.5,
            'completeness': 0.8,
            'size_appropriateness': 0.8
        }
        
        total_reward = sum(components[k] * weights[k] for k in weights.keys())
        components['total'] = total_reward
        
        return total_reward, components
    
    def _reward_logical_placement(
        self, 
        furniture: Dict, 
        catalog_item: Dict, 
        placement_rules: Dict
    ) -> float:
        """
        NEW: Reward based on logical placement rules
        This is the core of the enhanced system
        """
        placement_type = placement_rules.get('placement_type', 'free_standing')
        x, y = furniture['position']['x'], furniture['position']['y']
        
        # Calculate distance to nearest wall
        dist_to_walls = [
            x,  # Distance to west wall
            self.room_length - x,  # Distance to east wall
            y,  # Distance to south wall
            self.room_width - y  # Distance to north wall
        ]
        min_wall_dist = min(dist_to_walls)
        
        # WALL-MOUNTED items (shelves, mirrors)
        if placement_type == 'wall_mounted':
            ideal_dist = placement_rules.get('ideal_distance_from_wall', 0.0)
            max_dist = placement_rules.get('max_distance_from_wall', 0.1)
            
            if min_wall_dist <= max_dist:
                # Perfect - right against the wall
                error = abs(min_wall_dist - ideal_dist)
                return max(0.0, 1.0 - error / max_dist)
            else:
                # Too far from wall
                return 0.0
        
        # AGAINST WALL items (bookcases, console tables, benches)
        elif placement_type == 'against_wall':
            ideal_dist = placement_rules.get('ideal_distance_from_wall', 0.0)
            max_dist = placement_rules.get('max_distance_from_wall', 0.3)
            
            if min_wall_dist <= max_dist:
                error = abs(min_wall_dist - ideal_dist)
                return max(0.0, 1.0 - error / max_dist)
            else:
                return max(0.0, 0.5 - (min_wall_dist - max_dist) / 2.0)
        
        # CORNER items (plant stands, bar carts, storage baskets)
        elif placement_type == 'corner':
            if self._is_in_corner(x, y):
                # Bonus for being in corner
                return 1.0
            else:
                # Penalty for not being in corner
                # Find distance to nearest corner
                corner_dists = [
                    np.sqrt((x - c['center'][0])**2 + (y - c['center'][1])**2)
                    for c in self.corner_zones
                ]
                min_corner_dist = min(corner_dists)
                return max(0.0, 1.0 - min_corner_dist / 2.0)
        
        # NEAR SEATING items (coffee tables, side tables, lamps, ottomans)
        elif placement_type == 'near_seating':
            ideal_dist = placement_rules.get('ideal_distance_from_seating', 1.0)
            max_dist = placement_rules.get('max_distance_from_seating', 2.5)
            
            # Find nearest seating furniture
            seating_types = ['sofa', 'armchair', 'chair', 'bench']
            min_seating_dist = float('inf')
            
            for item in self.existing_furniture + self.placed_items:
                if item['type'] in seating_types:
                    dist = np.sqrt(
                        (x - item['position']['x'])**2 + 
                        (y - item['position']['y'])**2
                    )
                    min_seating_dist = min(min_seating_dist, dist)
            
            if min_seating_dist == float('inf'):
                # No seating found - neutral
                return 0.5
            
            if min_seating_dist <= max_dist:
                # Within acceptable range - reward proximity to ideal
                error = abs(min_seating_dist - ideal_dist)
                return max(0.0, 1.0 - error / ideal_dist)
            else:
                # Too far from seating
                return max(0.0, 0.3 - (min_seating_dist - max_dist) / 3.0)
        
        # ROOM CENTER items (rugs)
        elif placement_type == 'room_center':
            room_cx, room_cy = self.room_length / 2, self.room_width / 2
            dist_from_center = np.sqrt((x - room_cx)**2 + (y - room_cy)**2)
            max_acceptable = 1.5
            
            if dist_from_center <= max_acceptable:
                return 1.0 - (dist_from_center / max_acceptable) * 0.5
            else:
                return 0.0
        
        # FREE STANDING items (default)
        else:
            # Should NOT be too close to walls
            min_dist_from_wall = placement_rules.get('min_distance_from_wall', 0.5)
            
            if min_wall_dist >= min_dist_from_wall:
                return 1.0
            else:
                return min_wall_dist / min_dist_from_wall
    
    def _reward_parallel_placement(self, furniture: Dict) -> float:
        """Reward furniture placed parallel to existing items"""
        if not self.placed_items and not self.existing_furniture:
            return 1.0
        
        current_rotation = furniture.get('rotation', 0)
        current_rot_norm = current_rotation % (np.pi / 2)
        
        alignment_scores = []
        for existing in self.existing_furniture + self.placed_items:
            existing_rot = existing.get('rotation', 0)
            existing_rot_norm = existing_rot % (np.pi / 2)
            
            rot_diff = abs(current_rot_norm - existing_rot_norm)
            
            if rot_diff < np.pi / 16:
                alignment_scores.append(1.0)
            elif abs(rot_diff - np.pi/2) < np.pi / 16:
                alignment_scores.append(0.8)
            else:
                alignment_scores.append(0.3)
        
        if alignment_scores:
            return max(alignment_scores)
        return 0.5
    
    def _reward_functional_pairing(self, furniture: Dict, catalog_item: Dict) -> float:
        """Reward functional pairing with distance-sensitive scoring"""
        pairs_with = catalog_item.get('functional_properties', {}).get('pairs_with', [])
        
        if not pairs_with:
            return 0.5
        
        all_furniture = self.existing_furniture + self.placed_items
        best_score = 0.0
        
        for item in all_furniture:
            if item['type'] in pairs_with:
                dist = np.sqrt(
                    (furniture['position']['x'] - item['position']['x'])**2 +
                    (furniture['position']['y'] - item['position']['y'])**2
                )
                
                if item['type'] == 'sofa':
                    ideal_dist = 1.5
                elif item['type'] in ['armchair', 'console_table']:
                    ideal_dist = 0.8
                else:
                    ideal_dist = 2.0
                
                dist_error = abs(dist - ideal_dist)
                score = np.exp(-dist_error / ideal_dist)
                best_score = max(best_score, score)
        
        if best_score == 0.0 and pairs_with:
            return -0.5
        
        return best_score
    
    def _reward_accessibility(self, furniture: Dict) -> float:
        """Reward accessible placement"""
        min_dist = float('inf')
        
        all_furniture = self.existing_furniture + self.placed_items
        for item in all_furniture:
            dist = np.sqrt(
                (furniture['position']['x'] - item['position']['x'])**2 +
                (furniture['position']['y'] - item['position']['y'])**2
            )
            min_dist = min(min_dist, dist)
        
        if min_dist >= self.min_clearance + self.collision_buffer:
            return 1.0
        elif min_dist >= self.min_clearance:
            return 0.7
        else:
            return 0.0
    
    def _reward_clearance(self, furniture: Dict) -> float:
        """Reward pathway clearance"""
        x, y = furniture['position']['x'], furniture['position']['y']
        l, w = furniture['dimensions']['length'], furniture['dimensions']['width']
        
        clearances = [
            x - l/2,
            self.room_length - (x + l/2),
            y - w/2,
            self.room_width - (y + w/2)
        ]
        
        min_clearance = min(clearances)
        
        if min_clearance >= self.min_pathway:
            return 1.0
        elif min_clearance >= self.min_pathway * 0.6:
            return 0.6
        else:
            return 0.0
    
    def _reward_visual_balance(self, furniture: Dict) -> float:
        """Reward visual balance"""
        all_furniture = self.existing_furniture + self.placed_items + [furniture]
        
        cx = np.mean([f['position']['x'] for f in all_furniture])
        cy = np.mean([f['position']['y'] for f in all_furniture])
        
        room_cx = self.room_length / 2
        room_cy = self.room_width / 2
        dist_from_center = np.sqrt((cx - room_cx)**2 + (cy - room_cy)**2)
        
        max_dist = np.sqrt(room_cx**2 + room_cy**2)
        balance_score = 1.0 - (dist_from_center / max_dist)
        
        return max(0.0, balance_score)
    
    def _reward_color_harmony(self, furniture: Dict, catalog_item: Dict) -> float:
        """Reward color harmony with room"""
        furniture_color = catalog_item['color']
        room_colors = self.room_layout['room_info']['color_scheme']
        
        for room_color in room_colors:
            if room_color.lower() in furniture_color.lower():
                return 1.0
        
        return 0.5
    
    def _reward_diversity(self, furniture: Dict, catalog_item: Dict) -> float:
        """Reward diverse furniture types"""
        placed_types = set(item['type'] for item in self.placed_items)
        
        if catalog_item['type'] not in placed_types:
            return 1.0
        else:
            return 0.0
    
    def _reward_budget_efficiency(self, catalog_item: Dict) -> float:
        """Reward efficient budget usage"""
        price = catalog_item['price']
        remaining_budget = self.budget_max - self.budget_used
        
        if price <= remaining_budget:
            budget_ratio = self.budget_used / self.budget_max
            return min(1.0, budget_ratio + 0.2)
        else:
            return 0.0
    
    def _reward_completeness(self) -> float:
        """Reward placing all allowed items"""
        completion_ratio = len(self.placed_items) / self.max_items
        return completion_ratio
    
    def _reward_size_appropriateness(self, furniture: Dict, catalog_item: Dict) -> float:
        """Enhanced size appropriateness"""
        room_area = self.room_length * self.room_width
        furniture_area = furniture['dimensions']['length'] * furniture['dimensions']['width']
        
        ratio = furniture_area / room_area
        
        if 0.02 <= ratio <= 0.10:
            item_score = 1.0
        elif 0.01 <= ratio < 0.02:
            item_score = 0.6
        elif 0.10 < ratio <= 0.15:
            item_score = 0.7
        else:
            item_score = 0.0
        
        if self.placed_items:
            sizes = [p['dimensions']['length'] * p['dimensions']['width'] 
                    for p in self.placed_items]
            sizes.append(furniture_area)
            size_variance = np.std(sizes) / np.mean(sizes)
            
            if 0.3 <= size_variance <= 0.7:
                variance_bonus = 0.3
            else:
                variance_bonus = 0.0
        else:
            variance_bonus = 0.0
        
        return min(1.0, item_score + variance_bonus)
    
    def get_recommendation_summary(self) -> Dict:
        """Get summary of recommendations"""
        return {
            'placed_items': self.placed_items,
            'total_items': len(self.placed_items),
            'budget_used': self.budget_used,
            'budget_remaining': self.budget_max - self.budget_used,
            'free_space_percentage': self._calculate_free_space() * 100
        }


# Backwards compatibility
FurnitureRecommendationEnvEnhanced = FurnitureRecommendationEnvLogical
