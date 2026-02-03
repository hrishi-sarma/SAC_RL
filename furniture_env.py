import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import copy


class FurnitureRecommendationEnvEnhanced(gym.Env):
    """
    Enhanced RL Environment with:
    - Stricter collision detection
    - Grid alignment for organized placement
    - Better spatial reasoning
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        room_layout_path: str, 
        catalog_path: str, 
        max_items: int = 4,
        grid_size: float = 0.3,  # Grid spacing in meters
        collision_buffer: float = 0.15  # Minimum gap between furniture
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
        
        # NEW: Grid and buffer settings
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
        
        # NEW: Build grid reference points
        self.grid_points_x = np.arange(0, self.room_length + self.grid_size, self.grid_size)
        self.grid_points_y = np.arange(0, self.room_width + self.grid_size, self.grid_size)
        
        # Define observation space (expanded to include grid info)
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
    
    def _calculate_state_dim(self) -> int:
        """Calculate state vector dimension"""
        # Room features: 6
        # Existing furniture encoding: 10 * 8 = 80
        # Budget info: 2
        # Free space: 1
        # Step counter: 1
        # Catalog mask: 20
        # NEW: Grid occupancy summary: 10 (spatial distribution)
        return 6 + 80 + 2 + 1 + 1 + 20 + 10
    
    def _get_state(self) -> np.ndarray:
        """Generate enhanced state representation"""
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
        
        # NEW: Spatial occupancy zones (divide room into 10 zones)
        occupancy = self._calculate_zone_occupancy()
        state.extend(occupancy)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_zone_occupancy(self) -> List[float]:
        """Calculate occupancy in different zones (for spatial awareness)"""
        # Divide room into 2x5 grid
        zones = []
        zone_width = self.room_length / 5
        zone_height = self.room_width / 2
        
        for i in range(2):
            for j in range(5):
                zone_x_min = j * zone_width
                zone_x_max = (j + 1) * zone_width
                zone_y_min = i * zone_height
                zone_y_max = (i + 1) * zone_height
                
                # Check if any furniture in this zone
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
        
        self.existing_furniture = copy.deepcopy(self.room_layout['existing_furniture'])
        self.placed_items = []
        self.current_step = 0
        self.budget_used = 0
        self.placed_catalog_ids = set()
        
        state = self._get_state()
        return state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with enhanced placement"""
        # Decode action
        furniture_idx = int(action[0] * self.num_catalog_items) % self.num_catalog_items
        x_raw = action[1] * self.room_length
        y_raw = action[2] * self.room_width
        
        # NEW: Snap to grid for aligned placement
        x = self._snap_to_grid(x_raw, self.grid_points_x)
        y = self._snap_to_grid(y_raw, self.grid_points_y)
        
        # NEW: Discrete rotations (0°, 90°, 180°, 270° only)
        rotation_idx = int(action[3] * 4) % 4
        rotation = rotation_idx * (np.pi / 2)
        
        scale = action[4]
        
        # Get selected furniture
        selected_furniture = self.furniture_catalog[furniture_idx]
        
        # Check if already placed
        already_placed = selected_furniture['id'] in self.placed_catalog_ids
        
        # Create furniture instance
        furniture_instance = {
            'id': f"{selected_furniture['id']}_placed_{self.current_step}",
            'type': selected_furniture['type'],
            'category': selected_furniture['category'],
            'position': {'x': x, 'y': y, 'z': 0.0},
            'dimensions': {
                'length': selected_furniture['dimensions']['length'] * scale,
                'width': selected_furniture['dimensions']['width'] * scale,
                'height': selected_furniture['dimensions']['height']
            },
            'rotation': rotation,
            'color': selected_furniture['color'],
            'material': selected_furniture['material'],
            'style': selected_furniture['style'],
            'price': selected_furniture['price'],
            'price_tier': selected_furniture['price_tier'],
            'catalog_ref': selected_furniture['id']
        }
        
        # Calculate reward with enhanced checks
        reward, reward_components = self._calculate_reward(
            furniture_instance, selected_furniture, already_placed
        )
        
        is_valid = reward_components['valid_placement']
        
        # Update state if valid
        if is_valid and not already_placed:
            self.placed_items.append(furniture_instance)
            self.budget_used += selected_furniture['price']
            self.placed_catalog_ids.add(selected_furniture['id'])
        
        self.current_step += 1
        
        # Termination conditions
        terminated = (
            self.current_step >= self.max_steps or
            len(self.placed_items) >= self.max_items or
            self.budget_used >= self.budget_max
        )
        
        truncated = False
        
        state = self._get_state()
        info = {
            'reward_components': reward_components,
            'placed_items': len(self.placed_items),
            'budget_used': self.budget_used,
            'selected_furniture': selected_furniture['name'],
            'already_placed': already_placed,
            'grid_aligned': True  # Always grid-aligned now
        }
        
        return state, reward, terminated, truncated, info
    
    def _snap_to_grid(self, value: float, grid_points: np.ndarray) -> float:
        """Snap a value to nearest grid point"""
        idx = np.abs(grid_points - value).argmin()
        return grid_points[idx]
    
    def _calculate_reward(
        self, 
        furniture: Dict, 
        catalog_item: Dict,
        already_placed: bool = False
    ) -> Tuple[float, Dict]:
        """Calculate enhanced reward with stricter collision detection"""
        rewards = {}
        
        # Penalty for already placed
        if already_placed:
            return -5.0, {
                'valid_placement': False,
                'already_placed': True,
                'collision': False,
                'out_of_bounds': False,
                'blocks_door': False,
                'total': -5.0
            }
        
        # NEW: Enhanced collision detection with buffer
        collision = self._check_collision_with_buffer(furniture)
        out_of_bounds = self._check_out_of_bounds(furniture)
        blocks_door = self._check_blocks_door(furniture)
        
        valid_placement = not (collision or out_of_bounds or blocks_door)
        
        if not valid_placement:
            return -3.0, {
                'valid_placement': False,
                'collision': collision,
                'out_of_bounds': out_of_bounds,
                'blocks_door': blocks_door,
                'total': -3.0
            }
        
        # Functional rewards
        rewards['functional_pairing'] = self._reward_functional_pairing(furniture, catalog_item)
        rewards['accessibility'] = self._reward_accessibility(furniture)
        rewards['clearance'] = self._reward_clearance(furniture)
        
        # Visual rewards
        rewards['visual_balance'] = self._reward_visual_balance(furniture)
        rewards['alignment'] = self._reward_alignment(furniture)
        rewards['color_harmony'] = self._reward_color_harmony(furniture, catalog_item)
        
        # NEW: Grid alignment bonus
        rewards['grid_alignment'] = 1.0  # Always 1.0 since we force grid alignment
        
        # NEW: Parallel placement reward
        rewards['parallel_placement'] = self._reward_parallel_placement(furniture)
        
        # Recommendation rewards
        rewards['diversity'] = self._reward_diversity(furniture, catalog_item)
        rewards['budget_efficiency'] = self._reward_budget_efficiency(catalog_item)
        rewards['completeness'] = self._reward_completeness()
        rewards['size_appropriateness'] = self._reward_size_appropriateness(furniture, catalog_item)
        
        # Enhanced weights
        weights = {
            'functional_pairing': 3.0,
            'accessibility': 2.5,
            'clearance': 3.5,  # Higher for buffer enforcement
            'visual_balance': 2.0,
            'alignment': 2.0,
            'color_harmony': 2.0,
            'grid_alignment': 2.0,  # NEW
            'parallel_placement': 2.5,  # NEW
            'diversity': 2.5,
            'budget_efficiency': 2.5,
            'completeness': 2.5,
            'size_appropriateness': 2.0
        }
        
        total_reward = sum(rewards[k] * weights[k] for k in rewards.keys())
        total_weight = sum(weights.values())
        normalized_reward = total_reward / total_weight
        
        reward_components = {
            'valid_placement': True,
            **rewards,
            'total': normalized_reward
        }
        
        return normalized_reward, reward_components
    
    def _check_collision_with_buffer(self, furniture: Dict) -> bool:
        """
        ENHANCED: Check collision with minimum buffer distance
        This ensures furniture doesn't touch or overlap
        """
        try:
            # Create buffered polygon (expand by buffer amount)
            new_poly = self._furniture_to_polygon(furniture)
            new_poly_buffered = new_poly.buffer(self.collision_buffer)
            
            for existing in self.existing_furniture + self.placed_items:
                existing_poly = self._furniture_to_polygon(existing)
                
                # Check if buffered polygon intersects
                # This creates a minimum gap of collision_buffer meters
                if new_poly_buffered.intersects(existing_poly):
                    return True
            
            return False
        except Exception as e:
            # If polygon creation fails, consider it a collision
            return True
    
    def _check_out_of_bounds(self, furniture: Dict) -> bool:
        """Check if furniture is outside room bounds with margin"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        length = furniture['dimensions']['length']
        width = furniture['dimensions']['width']
        rotation = furniture.get('rotation', 0)
        
        # Get actual polygon
        poly = self._furniture_to_polygon(furniture)
        room_box = box(0, 0, self.room_length, self.room_width)
        
        # Check if polygon is fully contained in room
        return not room_box.contains(poly)
    
    def _check_blocks_door(self, furniture: Dict) -> bool:
        """Check if furniture blocks door clearance"""
        furniture_poly = self._furniture_to_polygon(furniture)
        
        for door in self.room_layout.get('doors', []):
            door_x = door['position']['x']
            door_y = door['position']['y']
            clearance = door['clearance_radius']
            
            door_center = Point(door_x + door['dimensions']['width']/2, door_y)
            if furniture_poly.distance(door_center) < clearance:
                return True
        
        return False
    
    def _furniture_to_polygon(self, furniture: Dict) -> Polygon:
        """Convert furniture to Shapely polygon with proper rotation"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        length = furniture['dimensions']['length']
        width = furniture['dimensions']['width']
        rotation = furniture.get('rotation', 0)
        
        # Create rectangle corners
        corners = [
            (-length/2, -width/2),
            (length/2, -width/2),
            (length/2, width/2),
            (-length/2, width/2)
        ]
        
        # Rotate
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_r - cy * sin_r + x
            ry = cx * sin_r + cy * cos_r + y
            rotated.append((rx, ry))
        
        return Polygon(rotated)
    
    def _reward_parallel_placement(self, furniture: Dict) -> float:
        """
        NEW: Reward furniture placed parallel to existing items
        Encourages organized, aligned layouts
        """
        if not self.placed_items and not self.existing_furniture:
            return 1.0  # First item is always good
        
        current_rotation = furniture.get('rotation', 0)
        
        # Normalize rotation to [0, pi/2] for comparison
        current_rot_norm = current_rotation % (np.pi / 2)
        
        # Check alignment with existing furniture
        alignment_scores = []
        for existing in self.existing_furniture + self.placed_items:
            existing_rot = existing.get('rotation', 0)
            existing_rot_norm = existing_rot % (np.pi / 2)
            
            # Calculate rotation difference
            rot_diff = abs(current_rot_norm - existing_rot_norm)
            
            # Reward parallel or perpendicular alignment
            if rot_diff < np.pi / 16:  # Nearly parallel (within ~11°)
                alignment_scores.append(1.0)
            elif abs(rot_diff - np.pi/2) < np.pi / 16:  # Nearly perpendicular
                alignment_scores.append(0.8)
            else:
                alignment_scores.append(0.3)
        
        if alignment_scores:
            return max(alignment_scores)  # Best alignment
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
                
                # Ideal pairing distances
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
        
        # Enhanced clearance rewards
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
    
    def _reward_alignment(self, furniture: Dict) -> float:
        """Reward alignment with walls (already enforced by discrete rotations)"""
        # Since we enforce discrete 90° rotations, always aligned
        return 1.0
    
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
    
    def set_difficulty(self, level='medium'):
        """Adjust reward weights based on curriculum"""
        # (Same as original)
        pass
    
    def get_recommendation_summary(self) -> Dict:
        """Get summary of recommendations"""
        return {
            'placed_items': self.placed_items,
            'total_items': len(self.placed_items),
            'budget_used': self.budget_used,
            'budget_remaining': self.budget_max - self.budget_used,
            'free_space_percentage': self._calculate_free_space() * 100
        }