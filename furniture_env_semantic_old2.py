import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import copy


class FurnitureRecommendationEnvSemantic(gym.Env):
    """
    Enhanced RL Environment with SEMANTIC PLACEMENT CONSTRAINTS:
    - Wall-mounted items must be near walls
    - Plants near windows or corners
    - Lamps near seating areas
    - Storage in appropriate zones
    - Mirrors on walls, avoiding windows/doors
    - Type-specific placement logic
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        room_layout_path: str, 
        catalog_path: str, 
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15,
        wall_proximity: float = 0.35  # Distance from wall for wall items
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
        self.wall_proximity = wall_proximity  # NEW: Wall distance constraint
        
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
        
        # NEW: Identify wall segments (avoiding doors/windows)
        self._build_wall_segments()
        
        # NEW: Identify corner positions
        self._build_corner_positions()
        
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
    
    def _build_wall_segments(self):
        """Identify valid wall segments (avoiding doors and windows)"""
        self.wall_segments = []
        
        # North wall (y = room_width)
        self.wall_segments.append({
            'orientation': 'horizontal',
            'y': self.room_width,
            'x_start': 0,
            'x_end': self.room_length,
            'name': 'north'
        })
        
        # South wall (y = 0)
        self.wall_segments.append({
            'orientation': 'horizontal',
            'y': 0,
            'x_start': 0,
            'x_end': self.room_length,
            'name': 'south'
        })
        
        # East wall (x = room_length)
        self.wall_segments.append({
            'orientation': 'vertical',
            'x': self.room_length,
            'y_start': 0,
            'y_end': self.room_width,
            'name': 'east'
        })
        
        # West wall (x = 0)
        self.wall_segments.append({
            'orientation': 'vertical',
            'x': 0,
            'y_start': 0,
            'y_end': self.room_width,
            'name': 'west'
        })
    
    def _build_corner_positions(self):
        """Identify room corners"""
        self.corners = [
            {'x': 0, 'y': 0, 'name': 'southwest'},
            {'x': self.room_length, 'y': 0, 'name': 'southeast'},
            {'x': 0, 'y': self.room_width, 'name': 'northwest'},
            {'x': self.room_length, 'y': self.room_width, 'name': 'northeast'}
        ]
    
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
        
        free_ratio = 1.0 - (furniture_area / total_area)
        return max(0.0, min(1.0, free_ratio))
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.placed_items = []
        self.placed_catalog_ids = set()
        self.budget_used = 0
        self.current_step = 0
        
        state = self._get_state()
        info = {}
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in environment"""
        self.current_step += 1
        
        # Decode action
        catalog_idx_norm, x_norm, y_norm, rotation_norm, scale = action
        
        # Select catalog item
        catalog_idx = int(catalog_idx_norm * (self.num_catalog_items - 1))
        catalog_idx = np.clip(catalog_idx, 0, self.num_catalog_items - 1)
        catalog_item = self.furniture_catalog[catalog_idx]
        
        # Check if already placed
        if catalog_item['id'] in self.placed_catalog_ids:
            reward = -10.0
            state = self._get_state()
            terminated = len(self.placed_items) >= self.max_items
            truncated = self.current_step >= self.max_steps
            info = {
                'reward_components': {'already_placed': True},
                'selected_furniture': catalog_item['name'],
                'placed_items': len(self.placed_items),
                'budget_used': self.budget_used,
                'already_placed': True
            }
            return state, reward, terminated, truncated, info
        
        # Snap to grid
        x = self._snap_to_grid(x_norm * self.room_length, self.grid_points_x)
        y = self._snap_to_grid(y_norm * self.room_width, self.grid_points_y)
        
        # Discrete rotations (0, π/2, π, 3π/2)
        rotation_discrete = int(rotation_norm * 4) % 4
        rotation = rotation_discrete * (np.pi / 2)
        
        # Scale dimensions
        base_length = catalog_item['dimensions']['length']
        base_width = catalog_item['dimensions']['width']
        base_height = catalog_item['dimensions']['height']
        
        length = base_length * scale
        width = base_width * scale
        height = base_height
        
        # Create furniture object
        furniture = {
            'id': catalog_item['id'],
            'type': catalog_item['type'],
            'name': catalog_item['name'],
            'category': catalog_item['category'],
            'position': {'x': x, 'y': y, 'z': 0.0},
            'dimensions': {'length': length, 'width': width, 'height': height},
            'rotation': rotation,
            'color': catalog_item['color'],
            'material': catalog_item['material'],
            'style': catalog_item['style'],
            'price': catalog_item['price'],
            'price_tier': catalog_item['price_tier']
        }
        
        # Check validity
        valid_placement = self._check_valid_placement(furniture, catalog_item)
        
        # Calculate reward
        reward_components = self._calculate_reward_components(furniture, catalog_item, valid_placement)
        reward = sum(reward_components.values())
        
        # Update state if valid
        if valid_placement and not reward_components.get('collision', False):
            self.placed_items.append(furniture)
            self.placed_catalog_ids.add(catalog_item['id'])
            self.budget_used += catalog_item['price']
        
        # Check termination
        terminated = len(self.placed_items) >= self.max_items
        truncated = self.current_step >= self.max_steps
        
        state = self._get_state()
        info = {
            'reward_components': reward_components,
            'selected_furniture': catalog_item['name'],
            'placed_items': len(self.placed_items),
            'budget_used': self.budget_used,
            'already_placed': False
        }
        
        return state, reward, terminated, truncated, info
    
    def _snap_to_grid(self, value: float, grid_points: np.ndarray) -> float:
        """Snap value to nearest grid point"""
        idx = np.argmin(np.abs(grid_points - value))
        return grid_points[idx]
    
    def _check_valid_placement(self, furniture: Dict, catalog_item: Dict) -> bool:
        """Check if placement is valid with semantic constraints"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        l = furniture['dimensions']['length']
        w = furniture['dimensions']['width']
        
        # Check room bounds
        if x - l/2 < 0 or x + l/2 > self.room_length:
            return False
        if y - w/2 < 0 or y + w/2 > self.room_width:
            return False
        
        # Check door clearance
        for door in self.room_layout['doors']:
            door_x = door['position']['x']
            door_y = door['position']['y']
            clearance = door['clearance_radius']
            
            dist = np.sqrt((x - door_x)**2 + (y - door_y)**2)
            if dist < clearance:
                return False
        
        # Check collision with existing furniture (with buffer)
        furniture_poly = self._create_furniture_polygon(furniture)
        
        for existing in self.existing_furniture + self.placed_items:
            existing_poly = self._create_furniture_polygon(existing)
            
            # Buffer polygons for collision check
            buffered_existing = existing_poly.buffer(self.collision_buffer)
            
            if furniture_poly.intersects(buffered_existing):
                return False
        
        # NEW: SEMANTIC PLACEMENT CONSTRAINTS
        semantic_valid = self._check_semantic_constraints(furniture, catalog_item)
        if not semantic_valid:
            return False
        
        return True
    
    def _check_semantic_constraints(self, furniture: Dict, catalog_item: Dict) -> bool:
        """
        Check type-specific semantic placement constraints
        Returns False if placement violates logical rules
        
        *** CONSTRAINTS DISABLED FOR TESTING ***
        All semantic rules are turned off to allow maximum placement flexibility.
        """
        # ALL SEMANTIC CONSTRAINTS DISABLED
        # This allows furniture to be placed anywhere as long as:
        # - It fits in the room
        # - It doesn't collide with existing furniture
        # - It respects door clearances
        
        return True  # Always pass semantic checks
    
    def _is_near_wall(self, x: float, y: float, threshold: float) -> bool:
        """Check if position is near any wall"""
        dist_to_walls = [
            x,  # west wall
            self.room_length - x,  # east wall
            y,  # south wall
            self.room_width - y  # north wall
        ]
        return min(dist_to_walls) < threshold
    
    def _is_near_corner(self, x: float, y: float, threshold: float) -> bool:
        """Check if position is near any corner"""
        for corner in self.corners:
            dist = np.sqrt((x - corner['x'])**2 + (y - corner['y'])**2)
            if dist < threshold:
                return True
        return False
    
    def _is_near_window(self, x: float, y: float, threshold: float) -> bool:
        """Check if position is near any window"""
        for window in self.room_layout['windows']:
            win_x = window['position']['x']
            win_y = window['position']['y']
            dist = np.sqrt((x - win_x)**2 + (y - win_y)**2)
            if dist < threshold:
                return True
        return False
    
    def _is_near_door(self, x: float, y: float, threshold: float) -> bool:
        """Check if position is near any door"""
        for door in self.room_layout['doors']:
            door_x = door['position']['x']
            door_y = door['position']['y']
            dist = np.sqrt((x - door_x)**2 + (y - door_y)**2)
            if dist < threshold:
                return True
        return False
    
    def _is_near_seating(self, x: float, y: float, threshold: float) -> bool:
        """Check if position is near seating furniture"""
        all_furniture = self.existing_furniture + self.placed_items
        
        for item in all_furniture:
            if item['category'] == 'seating':
                item_x = item['position']['x']
                item_y = item['position']['y']
                dist = np.sqrt((x - item_x)**2 + (y - item_y)**2)
                if dist < threshold:
                    return True
        return False
    
    def _is_on_table(self, x: float, y: float) -> bool:
        """Check if position is on a table surface"""
        all_furniture = self.existing_furniture + self.placed_items
        
        for item in all_furniture:
            if 'table' in item['type']:
                item_x = item['position']['x']
                item_y = item['position']['y']
                # Must be very close to table position
                dist = np.sqrt((x - item_x)**2 + (y - item_y)**2)
                if dist < 0.5:  # Within 50cm
                    return True
        return False
    
    def _create_furniture_polygon(self, furniture: Dict) -> Polygon:
        """Create polygon for furniture footprint"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        l = furniture['dimensions']['length']
        w = furniture['dimensions']['width']
        rotation = furniture.get('rotation', 0)
        
        # Create rectangle corners
        corners = np.array([
            [-l/2, -w/2],
            [l/2, -w/2],
            [l/2, w/2],
            [-l/2, w/2]
        ])
        
        # Rotate
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotation_matrix = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])
        
        rotated = corners @ rotation_matrix.T
        rotated[:, 0] += x
        rotated[:, 1] += y
        
        return Polygon(rotated)
    
    def _calculate_reward_components(self, furniture: Dict, catalog_item: Dict, valid_placement: bool) -> Dict[str, float]:
        """Calculate individual reward components"""
        components = {}
        
        if not valid_placement:
            components['valid_placement'] = False
            components['collision'] = True
            components['total'] = -5.0
            return components
        
        components['valid_placement'] = True
        components['collision'] = False
        
        # Existing reward components
        components['functional_pairing'] = self._reward_functional_pairing(furniture, catalog_item)
        components['accessibility'] = self._reward_accessibility(furniture)
        components['clearance'] = self._reward_clearance(furniture)
        components['visual_balance'] = self._reward_visual_balance(furniture)
        components['alignment'] = self._reward_alignment(furniture)
        components['color_harmony'] = self._reward_color_harmony(furniture, catalog_item)
        components['diversity'] = self._reward_diversity(furniture, catalog_item)
        components['budget_efficiency'] = self._reward_budget_efficiency(catalog_item)
        components['completeness'] = self._reward_completeness()
        components['size_appropriateness'] = self._reward_size_appropriateness(furniture, catalog_item)
        components['grid_alignment'] = self._reward_grid_alignment(furniture)
        components['parallel_placement'] = self._reward_parallel_placement(furniture)
        
        # NEW: SEMANTIC PLACEMENT REWARDS
        components['semantic_correctness'] = self._reward_semantic_placement(furniture, catalog_item)
        components['wall_proximity'] = self._reward_wall_proximity(furniture, catalog_item)
        components['corner_bonus'] = self._reward_corner_placement(furniture, catalog_item)
        components['window_bonus'] = self._reward_window_placement(furniture, catalog_item)
        
        return components
    
    def _reward_semantic_placement(self, furniture: Dict, catalog_item: Dict) -> float:
        """Reward correct semantic placement"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        item_type = catalog_item['type']
        score = 0.0
        
        # Wall items near walls
        if catalog_item.get('functional_properties', {}).get('wall_mounted', False):
            if self._is_near_wall(x, y, self.wall_proximity):
                score += 1.5
        
        # Plants in corners or near windows
        if item_type == 'plant_stand':
            if self._is_near_corner(x, y, 1.0):
                score += 1.0
            if self._is_near_window(x, y, 1.5):
                score += 1.0
        
        # Lamps near seating
        if 'lamp' in item_type:
            if self._is_near_seating(x, y, 2.0):
                score += 1.0
        
        # Storage near walls
        if item_type in ['bookshelf', 'console_table', 'media_cabinet']:
            if self._is_near_wall(x, y, 0.8):
                score += 0.8
        
        return min(2.0, score)
    
    def _reward_wall_proximity(self, furniture: Dict, catalog_item: Dict) -> float:
        """Reward appropriate wall distance"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        item_type = catalog_item['type']
        
        # Wall-mount items should be very close
        if catalog_item.get('functional_properties', {}).get('wall_mounted', False):
            if self._is_near_wall(x, y, 0.3):
                return 1.0
            else:
                return 0.0
        
        # Storage against walls
        if item_type in ['bookshelf', 'console_table']:
            if self._is_near_wall(x, y, 0.5):
                return 0.8
        
        return 0.5
    
    def _reward_corner_placement(self, furniture: Dict, catalog_item: Dict) -> float:
        """Reward corner placement for appropriate items"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        item_type = catalog_item['type']
        
        if item_type in ['plant_stand', 'floor_lamp', 'decorative_ladder']:
            if self._is_near_corner(x, y, 0.8):
                return 1.0
        
        return 0.0
    
    def _reward_window_placement(self, furniture: Dict, catalog_item: Dict) -> float:
        """Reward window proximity for appropriate items"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        item_type = catalog_item['type']
        
        if item_type in ['plant_stand']:
            if self._is_near_window(x, y, 1.2):
                return 1.0
        
        return 0.0
    
    def _reward_grid_alignment(self, furniture: Dict) -> float:
        """Reward grid-aligned placement"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        
        # Check if on grid
        on_grid_x = any(abs(x - gp) < 0.01 for gp in self.grid_points_x)
        on_grid_y = any(abs(y - gp) < 0.01 for gp in self.grid_points_y)
        
        if on_grid_x and on_grid_y:
            return 1.0
        return 0.5
    
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
    
    def _reward_alignment(self, furniture: Dict) -> float:
        """Reward alignment with walls"""
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
    
    def get_recommendation_summary(self) -> Dict:
        """Get summary of recommendations"""
        return {
            'placed_items': self.placed_items,
            'total_items': len(self.placed_items),
            'budget_used': self.budget_used,
            'budget_remaining': self.budget_max - self.budget_used,
            'free_space_percentage': self._calculate_free_space() * 100
        }


# Backwards compatibility alias
FurnitureRecommendationEnv = FurnitureRecommendationEnvSemantic
