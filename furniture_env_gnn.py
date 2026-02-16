"""
GNN-Enhanced Furniture Placement Environment

This version integrates Graph Neural Networks for spatial reasoning alongside
tag-based placement rules.

Features:
- Graph-based scene representation
- Tag-based placement rules
- Spatial constraints and functional requirements
- Zone preferences and relationship mapping
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from typing import Dict, List, Tuple, Optional, Set
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import copy

# GNN imports
from placement_rules_graph import PlacementRulesGraph
from scene_graph_builder import SceneGraphBuilder


class FurnitureRecommendationEnvGNN(gym.Env):
    """
    GNN-Enhanced RL Environment
    
    Features:
    - Graph-based scene representation
    - Tag-based placement scoring
    - Context-aware spatial constraints
    - Functional requirement validation
    - Dynamic zone preference calculation
    - Tag compatibility rewards
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
        
        # Initialize placement rules graph
        self.placement_rules = PlacementRulesGraph()
        
        # Initialize scene graph builder (will be set after loading room)
        self.scene_graph_builder = None
        
        # Load data
        with open(room_layout_path, 'r') as f:
            self.room_layout = json.load(f)
        with open(catalog_path, 'r') as f:
            self.catalog = json.load(f)
        
        self.max_items = max_items
        self.current_step = 0
        self.max_steps = max_items * 6   # 6 attempts per item slot
        
        # Grid and buffer settings
        self.grid_size = grid_size
        self.collision_buffer = 0.35  # increased to prevent overlaps
        
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
        
        # Initialize scene graph builder with room layout
        self.scene_graph_builder = SceneGraphBuilder(self.room_layout, self.placement_rules)
        self.scene_graph_builder.set_catalog(self.catalog)
        
        # NEW: Build tag-based spatial maps
        self._build_tag_based_zones()
        self._build_relationship_map()
        
        # Define observation space
        self.state_dim = self._calculate_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0,  1.0,  1.0,  1.0,  1.0]),
            dtype=np.float32
        )
    
    def _build_tag_based_zones(self):
        """Build spatial zones based on furniture tags already in the room"""
        self.spatial_zones = {
            'seating_area': [],
            'conversation_zone': [],
            'peripheral': [],
            'corner': [],
            'entertainment': [],
            'storage': [],
            'high_traffic': [],
            'low_traffic': []
        }
        
        # Identify zones based on existing furniture
        for furniture in self.existing_furniture:
            x, y = furniture['position']['x'], furniture['position']['y']
            ftype = furniture['type']
            zone = furniture.get('functional_zone', '')
            
            # Seating creates conversation zones
            if furniture['category'] == 'seating':
                # Create a zone around seating
                radius = 2.5
                self.spatial_zones['seating_area'].append({
                    'center': (x, y),
                    'radius': radius,
                    'source': ftype,
                    'priority': 'high'
                })
                self.spatial_zones['conversation_zone'].append({
                    'center': (x, y),
                    'radius': 3.0,
                    'source': ftype
                })
            
            # Storage/entertainment creates specific zones
            if furniture['category'] == 'storage':
                self.spatial_zones['peripheral'].append({
                    'center': (x, y),
                    'radius': 1.5,
                    'source': ftype
                })
        
        # Add corner zones
        corner_radius = 1.0
        corners = [
            (corner_radius, corner_radius),
            (self.room_length - corner_radius, corner_radius),
            (corner_radius, self.room_width - corner_radius),
            (self.room_length - corner_radius, self.room_width - corner_radius)
        ]
        
        for corner_pos in corners:
            self.spatial_zones['corner'].append({
                'center': corner_pos,
                'radius': corner_radius,
                'source': 'room_corner'
            })
        
        # Identify high traffic areas (near doors)
        for door in self.room_layout.get('doors', []):
            dx, dy = door['position']['x'], door['position']['y']
            clearance = door.get('clearance_radius', 1.2)
            self.spatial_zones['high_traffic'].append({
                'center': (dx, dy),
                'radius': clearance * 1.5,
                'source': 'door'
            })
    
    def _build_relationship_map(self):
        """Build map of furniture relationships based on tags"""
        self.tag_relationships = {
            # What tags work well together
            'seating_companion': ['primary_seating', 'task_lighting', 'seating_accessory'],
            'primary_seating': ['seating_companion', 'seating_accessory', 'task_lighting'],
            'seating_accessory': ['primary_seating', 'seating_companion', 'task_lighting'],
            'task_lighting': ['primary_seating', 'seating_accessory'],
            'wall_unit': ['peripheral', 'storage'],
            'corner_decor': ['corner_friendly', 'low_traffic'],
            'flexible_seating': ['seating_companion', 'primary_seating']
        }
        
        # What tags conflict
        self.tag_conflicts = {
            'corner_decor': ['seating_companion', 'focal_point'],
            'wall_unit': ['seating_companion', 'conversation_zone'],
            'high_traffic': ['corner_decor', 'wall_unit']
        }
    
    def _get_furniture_tags(self, catalog_item: Dict) -> Tuple[str, List[str]]:
        """Extract primary and secondary tags from catalog item"""
        advanced = catalog_item.get('advanced_placement', {})
        primary_tag = advanced.get('primary_tag', 'unknown')
        secondary_tags = advanced.get('secondary_tags', [])
        return primary_tag, secondary_tags
    
    def _get_zone_preference(self, catalog_item: Dict) -> Dict:
        """Extract zone preferences from catalog item"""
        advanced = catalog_item.get('advanced_placement', {})
        return advanced.get('zone_preference', {})
    
    def _get_spatial_constraints(self, catalog_item: Dict) -> Dict:
        """Extract spatial constraints from catalog item"""
        advanced = catalog_item.get('advanced_placement', {})
        return advanced.get('spatial_constraints', {})
    
    def _get_functional_requirements(self, catalog_item: Dict) -> Dict:
        """Extract functional requirements from catalog item"""
        advanced = catalog_item.get('advanced_placement', {})
        return advanced.get('functional_requirements', {})
    
    def _calculate_state_dim(self) -> int:
        """Calculate state dimension"""
        room_features = 6
        furniture_encoding = 80
        budget_info = 2
        free_space = 1
        step_counter = 1
        catalog_mask = self.num_catalog_items
        spatial_occupancy = 10
        tag_context = 20  # NEW: Tag-based context features
        
        return (room_features + furniture_encoding + budget_info + 
                free_space + step_counter + catalog_mask + spatial_occupancy + tag_context)
    
    def _get_state(self) -> np.ndarray:
        """Generate current state representation with tag context"""
        state = []
        
        # Room features
        state.extend([
            self.room_length / 10.0,
            self.room_width / 10.0,
            len(self.existing_furniture) / 10.0,
            len(self.placed_items) / self.max_items,
            self.budget_used / self.budget_max if self.budget_max > 0 else 0,
            self.current_step / self.max_steps
        ])
        
        # Furniture encoding
        all_furniture = self.existing_furniture + self.placed_items
        furniture_vectors = []
        for furniture in all_furniture[:20]:
            vec = [
                furniture['position']['x'] / self.room_length,
                furniture['position']['y'] / self.room_width,
                furniture['dimensions']['length'] / 3.0,
                furniture['dimensions']['width'] / 3.0
            ]
            furniture_vectors.extend(vec)
        
        while len(furniture_vectors) < 80:
            furniture_vectors.append(0.0)
        state.extend(furniture_vectors[:80])
        
        # Budget info
        state.extend([
            self.budget_used / self.budget_max if self.budget_max > 0 else 0,
            (self.budget_max - self.budget_used) / self.budget_max if self.budget_max > 0 else 0
        ])
        
        # Free space
        free_area = self._calculate_free_area()
        state.append(free_area / (self.room_length * self.room_width))
        
        # Step counter
        state.append(self.current_step / self.max_steps)
        
        # Catalog availability mask
        catalog_mask = []
        for item in self.furniture_catalog:
            if item['id'] in self.placed_catalog_ids:
                catalog_mask.append(0.0)
            elif item['price'] > (self.budget_max - self.budget_used):
                catalog_mask.append(0.0)
            else:
                catalog_mask.append(1.0)
        state.extend(catalog_mask)
        
        # Spatial occupancy grid
        grid_occupancy = self._calculate_grid_occupancy()
        state.extend(grid_occupancy)
        
        # NEW: Tag-based context features
        tag_context = self._calculate_tag_context()
        state.extend(tag_context)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_tag_context(self) -> List[float]:
        """Calculate tag-based context features for current room state"""
        context = []
        
        # Count items by primary tag in room
        tag_counts = {
            'primary_seating': 0,
            'seating_companion': 0,
            'seating_accessory': 0,
            'task_lighting': 0,
            'wall_unit': 0,
            'corner_decor': 0,
            'flexible_seating': 0,
            'ambient_lighting': 0
        }
        
        all_furniture = self.existing_furniture + self.placed_items
        for furniture in all_furniture:
            # Match furniture type to tags
            ftype = furniture['type']
            if ftype in ['sofa', 'armchair']:
                tag_counts['primary_seating'] += 1
            elif ftype in ['coffee_table', 'ottoman']:
                tag_counts['seating_companion'] += 1
            elif ftype in ['side_table']:
                tag_counts['seating_accessory'] += 1
            elif ftype in ['floor_lamp', 'table_lamp']:
                tag_counts['task_lighting'] += 1
            elif ftype in ['bookshelf', 'console_table']:
                tag_counts['wall_unit'] += 1
            elif ftype in ['plant_stand']:
                tag_counts['corner_decor'] += 1
        
        # Normalize counts
        for tag in tag_counts:
            context.append(min(tag_counts[tag] / 3.0, 1.0))
        
        # Zone saturation (are zones full?)
        seating_zone_saturation = min(len([z for z in self.spatial_zones['seating_area']]) / 2.0, 1.0)
        corner_zone_saturation = min(len([f for f in all_furniture if self._is_in_corner(
            f['position']['x'], f['position']['y'])]) / 4.0, 1.0)
        peripheral_saturation = min(len([f for f in all_furniture if self._is_near_wall(
            f['position']['x'], f['position']['y'])]) / 4.0, 1.0)
        
        context.extend([seating_zone_saturation, corner_zone_saturation, peripheral_saturation])
        
        # Available budget ratio
        budget_ratio = (self.budget_max - self.budget_used) / self.budget_max if self.budget_max > 0 else 0
        context.append(budget_ratio)
        
        # Pad to 20 features
        while len(context) < 20:
            context.append(0.0)
        
        return context[:20]
    
    def _is_in_corner(self, x: float, y: float, threshold: float = 1.0) -> bool:
        """Check if position is in a corner zone"""
        for corner in self.spatial_zones['corner']:
            cx, cy = corner['center']
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist <= threshold:
                return True
        return False
    
    def _is_near_wall(self, x: float, y: float, threshold: float = 0.5) -> bool:
        """Check if position is near a wall"""
        return (x < threshold or x > self.room_length - threshold or
                y < threshold or y > self.room_width - threshold)
    
    def _calculate_grid_occupancy(self) -> List[float]:
        """Calculate spatial occupancy grid"""
        grid_x = 5
        grid_y = 2
        grid = np.zeros((grid_y, grid_x))
        
        cell_width = self.room_length / grid_x
        cell_height = self.room_width / grid_y
        
        all_furniture = self.existing_furniture + self.placed_items
        for furniture in all_furniture:
            x = furniture['position']['x']
            y = furniture['position']['y']
            
            grid_i = min(int(y / cell_height), grid_y - 1)
            grid_j = min(int(x / cell_width), grid_x - 1)
            
            grid[grid_i, grid_j] = 1.0
        
        return grid.flatten().tolist()
    
    def _calculate_free_area(self) -> float:
        """Calculate free floor area"""
        total_area = self.room_length * self.room_width
        occupied_area = 0
        
        all_furniture = self.existing_furniture + self.placed_items
        for furniture in all_furniture:
            length = furniture['dimensions']['length']
            width = furniture['dimensions']['width']
            occupied_area += length * width
        
        return max(0, total_area - occupied_area)
    
    def get_graph_data(
        self, 
        candidate_item: Optional[Dict] = None,
        candidate_position: Optional[Tuple[float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get graph representation of current scene
        
        Args:
            candidate_item: Candidate furniture item (from catalog)
            candidate_position: (x, y) position for candidate
            
        Returns:
            Dictionary with 'node_features', 'edge_index', 'edge_attr'
        """
        all_furniture = self.existing_furniture + self.placed_items
        
        # Build graph
        node_features, edge_index, edge_attr = self.scene_graph_builder.build_graph(
            existing_furniture=all_furniture,
            candidate_item=candidate_item,
            candidate_position=candidate_position
        )
        
        # Convert tensors to numpy for storage
        return {
            'node_features': node_features.cpu().numpy(),
            'edge_index': edge_index.cpu().numpy(),
            'edge_attr': edge_attr.cpu().numpy()
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.existing_furniture = copy.deepcopy(self.room_layout['existing_furniture'])
        self.placed_items = []
        self.budget_used = 0
        self.current_step = 0
        self.placed_catalog_ids = set()
        
        # Rebuild zones
        self._build_tag_based_zones()
        
        state = self._get_state()
        info = {}
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        self.current_step += 1

        # --- Decode action (all dims come from tanh in [-1, 1]) ---
        # Map catalog selection from [-1,1] -> [0,1] -> valid index
        catalog_raw = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        catalog_idx = int(catalog_raw * self.num_catalog_items)
        catalog_idx = int(np.clip(catalog_idx, 0, self.num_catalog_items - 1))

        # Redirect already-placed selections to the nearest available item
        if self.furniture_catalog[catalog_idx]['id'] in self.placed_catalog_ids:
            budget_left = self.budget_max - self.budget_used
            available = [i for i, item in enumerate(self.furniture_catalog)
                         if item['id'] not in self.placed_catalog_ids
                         and item['price'] <= budget_left]
            if available:
                catalog_idx = min(available, key=lambda i: abs(i - catalog_idx))

        x_norm      = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))
        y_norm      = float(np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0))
        rotation_norm = float(np.clip((action[3] + 1.0) / 2.0, 0.0, 1.0))
        scale       = float(np.clip(action[4], 0.8, 1.2))

        # Snap to grid
        x = self._snap_to_grid(x_norm * self.room_length, self.grid_points_x)
        y = self._snap_to_grid(y_norm * self.room_width, self.grid_points_y)

        rotation_options = [0, 90, 180, 270]
        rot_idx  = min(int(rotation_norm * len(rotation_options)), len(rotation_options) - 1)
        rotation = rotation_options[rot_idx]

        # Get furniture from catalog
        catalog_item = self.furniture_catalog[catalog_idx]

        # Check if already placed
        already_placed = catalog_item['id'] in self.placed_catalog_ids

        # --- Wall-snapping: force wall_unit items flush against nearest wall ---
        p_tag = catalog_item.get('advanced_placement', {}).get('primary_tag', '')
        sp_c  = catalog_item.get('advanced_placement', {}).get('spatial_constraints', {})
        # Also snap by type if catalog doesn't have proper tags
        wall_types = {'wall_shelf', 'bookshelf', 'console_table', 'wall_mount', 'wall_cabinet'}
        if (sp_c.get('must_touch_wall', False) or 
            p_tag == 'wall_unit' or 
            catalog_item['type'] in wall_types):
            raw_l  = catalog_item['dimensions']['length'] * scale
            raw_w  = catalog_item['dimensions']['width']  * scale
            half_depth = raw_w / 2.0   # depth from wall to centre
            d_left   = x
            d_right  = self.room_length - x
            d_bottom = y
            d_top    = self.room_width  - y
            nearest  = min(d_left, d_right, d_bottom, d_top)
            if nearest == d_left:
                x, rotation = half_depth, 0
            elif nearest == d_right:
                x, rotation = self.room_length - half_depth, 180
            elif nearest == d_bottom:
                y, rotation = half_depth, 90
            else:
                y, rotation = self.room_width - half_depth, 270

        # Create furniture instance
        furniture = {
            'id': f"{catalog_item['id']}_placed_{self.current_step}",
            'catalog_id': catalog_item['id'],
            'type': catalog_item['type'],
            'category': catalog_item['category'],
            'position': {'x': x, 'y': y, 'z': 0.0},
            'dimensions': {
                'length': catalog_item['dimensions']['length'] * scale,
                'width':  catalog_item['dimensions']['width']  * scale,
                'height': catalog_item['dimensions']['height']
            },
            'rotation': rotation,
            'color':      catalog_item['color'],
            'material':   catalog_item['material'],
            'style':      catalog_item['style'],
            'price_tier': catalog_item['price_tier']
        }
        
        # Validate placement
        valid_placement, violation_reason = self._validate_placement(furniture)
        
        # Calculate reward with tag-based context
        reward, reward_components = self._calculate_reward(
            furniture, catalog_item, valid_placement, already_placed
        )
        
        # Update state if valid
        if valid_placement and not already_placed:
            if catalog_item['price'] <= (self.budget_max - self.budget_used):
                self.placed_items.append(furniture)
                self.budget_used += catalog_item['price']
                self.placed_catalog_ids.add(catalog_item['id'])
                self._build_tag_based_zones()
        
        # Check termination
        terminated = len(self.placed_items) >= self.max_items
        truncated  = self.current_step >= self.max_steps
        
        next_state = self._get_state()
        
        info = {
            'placed_items':     len(self.placed_items),
            'budget_used':      self.budget_used,
            'valid_placement':  valid_placement,
            'already_placed':   already_placed,
            'violation_reason': violation_reason,
            'selected_furniture': catalog_item['type'],
            'reward_components':  reward_components
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _snap_to_grid(self, value: float, grid_points: np.ndarray) -> float:
        """Snap value to nearest grid point"""
        idx = np.argmin(np.abs(grid_points - value))
        return grid_points[idx]
    
    def _validate_placement(self, furniture: Dict) -> Tuple[bool, str]:
        """Validate furniture placement (rotation-aware)"""
        x, y    = furniture['position']['x'], furniture['position']['y']
        rotation = furniture.get('rotation', 0)

        # Swap dims for 90/270° rotations
        if rotation in (90, 270):
            length = furniture['dimensions']['width']
            width  = furniture['dimensions']['length']
        else:
            length = furniture['dimensions']['length']
            width  = furniture['dimensions']['width']

        # Check bounds (small tolerance for wall-snapped items)
        tol = 0.02
        if (x - length/2 < -tol or x + length/2 > self.room_length + tol or
            y - width/2  < -tol or y + width/2  > self.room_width  + tol):
            return False, "out_of_bounds"
        
        # Check collision with existing furniture
        new_poly = self._get_furniture_polygon(furniture, buffer=self.collision_buffer)
        
        all_furniture = self.existing_furniture + self.placed_items
        for existing in all_furniture:
            existing_poly = self._get_furniture_polygon(existing, buffer=self.collision_buffer)
            if new_poly.intersects(existing_poly):
                # Debug logging (can be commented out after testing)
                # print(f"⚠️ COLLISION: {furniture['type']} @ "
                #       f"({furniture['position']['x']:.2f}, {furniture['position']['y']:.2f}) "
                #       f"intersects {existing['type']} @ "
                #       f"({existing['position']['x']:.2f}, {existing['position']['y']:.2f})")
                return False, "collision"
        
        # Check door clearance
        for door in self.room_layout.get('doors', []):
            door_x = door['position']['x']
            door_y = door['position']['y']
            clearance = door.get('clearance_radius', 1.2)
            
            dist = np.sqrt((x - door_x)**2 + (y - door_y)**2)
            if dist < clearance:
                return False, "blocks_door"
        
        return True, "valid"
    
    def _get_furniture_polygon(self, furniture: Dict, buffer: float = 0.0) -> Polygon:
        """Create polygon representation of furniture with buffer (rotation-aware)"""
        x        = furniture['position']['x']
        y        = furniture['position']['y']
        rotation = furniture.get('rotation', 0)
        if rotation in (90, 270):
            length = furniture['dimensions']['width']  + buffer
            width  = furniture['dimensions']['length'] + buffer
        else:
            length = furniture['dimensions']['length'] + buffer
            width  = furniture['dimensions']['width']  + buffer
        return box(x - length/2, y - width/2, x + length/2, y + width/2)
    
    def _calculate_reward(
        self, 
        furniture: Dict, 
        catalog_item: Dict,
        valid_placement: bool,
        already_placed: bool
    ) -> Tuple[float, Dict]:
        """Calculate comprehensive reward with tag-based scoring"""
        
        components = {
            'valid_placement': 1.0 if valid_placement else 0.0,
            'already_placed_penalty': 0.0,
            'tag_based_placement': 0.0,
            'spatial_constraints': 0.0,
            'functional_requirements': 0.0,
            'zone_preference': 0.0,
            'tag_compatibility': 0.0,
            'diversity': 0.0,
            'accessibility': 0.0,
            'spatial_distribution': 0.0,  # NEW
            'grid_alignment': 1.0,
            'budget_efficiency': 0.0
        }
        
        # Base penalty for invalid placement
        if not valid_placement:
            return -5.0, components
        
        # Heavy penalty for already placed items (strong signal to pick something else)
        if already_placed:
            components['already_placed_penalty'] = -2.0
            return -2.0, components
        
        # Get tag-based information
        primary_tag, secondary_tags = self._get_furniture_tags(catalog_item)
        zone_pref = self._get_zone_preference(catalog_item)
        spatial_const = self._get_spatial_constraints(catalog_item)
        func_req = self._get_functional_requirements(catalog_item)
        
        # 1. TAG-BASED PLACEMENT SCORE (Highest Priority)
        components['tag_based_placement'] = self._reward_tag_based_placement(
            furniture, primary_tag, secondary_tags
        )
        
        # 2. SPATIAL CONSTRAINTS SCORE
        components['spatial_constraints'] = self._reward_spatial_constraints(
            furniture, spatial_const
        )
        
        # 3. FUNCTIONAL REQUIREMENTS SCORE
        components['functional_requirements'] = self._reward_functional_requirements(
            furniture, func_req
        )
        
        # 4. ZONE PREFERENCE SCORE
        components['zone_preference'] = self._reward_zone_preference(
            furniture, zone_pref
        )
        
        # 5. TAG COMPATIBILITY SCORE
        components['tag_compatibility'] = self._reward_tag_compatibility(
            primary_tag, secondary_tags
        )
        
        # 6. DIVERSITY SCORE
        components['diversity'] = self._reward_diversity(catalog_item)
        
        # 7. ACCESSIBILITY SCORE
        components['accessibility'] = self._reward_accessibility(furniture)
        
        # 8. SPATIAL DISTRIBUTION (prevent clustering)
        components['spatial_distribution'] = self._reward_spatial_distribution()
        
        # 9. BUDGET EFFICIENCY
        if catalog_item['price'] <= (self.budget_max - self.budget_used):
            components['budget_efficiency'] = 1.0 - (catalog_item['price'] / self.budget_max)
        
        # WEIGHTED TOTAL
        weights = {
            'tag_based_placement':  5.0,
            'spatial_constraints':  4.0,
            'functional_requirements': 3.5,
            'zone_preference':      3.0,
            'tag_compatibility':    2.5,
            'diversity':            2.5,
            'accessibility':        2.0,   # raised importance
            'spatial_distribution': 2.0,   # NEW - prevent clustering
            'grid_alignment':       1.0,
            'budget_efficiency':    0.8
        }
        
        total_reward = sum(components[k] * weights.get(k, 0) for k in components.keys())
        
        # Completion bonus: big reward for placing each successive item
        items_so_far = len(self.placed_items)  # not yet updated, so +1 = after this placement
        total_reward += 5.0 * (items_so_far + 1)
        if items_so_far + 1 >= self.max_items:
            total_reward += 20.0  # extra bonus for completing the room
        
        # Step efficiency penalty
        total_reward -= 0.2 * (self.current_step / self.max_steps)
        
        return total_reward, components
    
    def _reward_tag_based_placement(
        self, 
        furniture: Dict, 
        primary_tag: str, 
        secondary_tags: List[str]
    ) -> float:
        """Reward based on tag-appropriate placement"""
        x, y = furniture['position']['x'], furniture['position']['y']
        score = 0.5  # Base score
        
        # Primary tag placement logic
        if primary_tag == 'seating_companion':
            # Should be near seating area
            for zone in self.spatial_zones['seating_area']:
                cx, cy = zone['center']
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                ideal_dist = 0.7
                max_dist = 2.0
                if dist <= max_dist:
                    score = max(score, 1.0 - (abs(dist - ideal_dist) / max_dist))
        
        elif primary_tag == 'primary_seating':
            # Should be in seating area but not too close to existing seating
            for zone in self.spatial_zones['seating_area']:
                cx, cy = zone['center']
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if 1.5 <= dist <= 3.0:
                    score = max(score, 0.9)
        
        elif primary_tag == 'seating_accessory':
            # Should be very close to seating
            for zone in self.spatial_zones['seating_area']:
                cx, cy = zone['center']
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist <= 1.0:
                    score = max(score, 1.0 - (dist / 1.0))
        
        elif primary_tag == 'task_lighting':
            # Should be near seating but slightly offset
            for zone in self.spatial_zones['seating_area']:
                cx, cy = zone['center']
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if 0.5 <= dist <= 1.5:
                    score = max(score, 1.0 - (abs(dist - 1.0) / 1.5))
        
        elif primary_tag == 'wall_unit':
            # Snapped items sit half-depth from wall (typically 0.2-0.4m)
            if self._is_near_wall(x, y, threshold=0.5):
                score = 1.0
            else:
                score = 0.2
        
        elif primary_tag == 'corner_decor':
            # Should be in corner
            if self._is_in_corner(x, y):
                score = 1.0
            elif self._is_in_corner(x, y, threshold=1.5):
                score = 0.7
            else:
                score = 0.2
        
        elif primary_tag == 'flexible_seating':
            # Can be near seating or in conversation zone
            for zone in self.spatial_zones['conversation_zone']:
                cx, cy = zone['center']
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist <= 2.5:
                    score = max(score, 0.8)
        
        # Bonus for secondary tags
        secondary_bonus = 0.0
        if 'corner_friendly' in secondary_tags and self._is_in_corner(x, y):
            secondary_bonus += 0.2
        if 'needs_clearance' in secondary_tags:
            # Check if there's adequate clearance
            if self._check_adequate_clearance(furniture, min_clearance=0.8):
                secondary_bonus += 0.1
        if 'focal_point' in secondary_tags or 'focal_point_support' in secondary_tags:
            # Should be visible from multiple angles
            if self._is_central_position(x, y):
                secondary_bonus += 0.15
        
        return min(1.0, score + secondary_bonus)
    
    def _reward_spatial_constraints(
        self, 
        furniture: Dict,
        spatial_const: Dict
    ) -> float:
        """Reward based on spatial constraints from catalog"""
        if not spatial_const:
            return 0.5
        
        x, y = furniture['position']['x'], furniture['position']['y']
        score = 0.5
        
        # Check wall proximity constraints
        dist_from_wall = self._get_min_distance_to_wall(x, y)
        
        if 'ideal_distance_from_wall' in spatial_const:
            ideal = spatial_const['ideal_distance_from_wall']
            min_dist = ideal.get('min', 0.0)
            ideal_dist = ideal.get('ideal', 0.5)
            max_dist = ideal.get('max', 999)
            
            if min_dist <= dist_from_wall <= max_dist:
                error = abs(dist_from_wall - ideal_dist)
                score = max(score, 1.0 - (error / max(ideal_dist, 0.5)))
        
        # Check if must touch wall (snapped items have centre at half_depth from wall)
        if spatial_const.get('must_touch_wall', False):
            half_depth = furniture['dimensions']['width'] / 2.0
            if dist_from_wall <= half_depth + 0.05:
                score = 1.0
            else:
                score = 0.2
        
        # Check corner requirements
        if spatial_const.get('strongly_prefers_corner', False):
            if self._is_in_corner(x, y):
                score = max(score, 1.0)
            else:
                score = min(score, 0.4)
        
        # Check seating proximity
        if 'ideal_distance_from_sofa' in spatial_const or 'ideal_distance_from_seating' in spatial_const:
            seating_dist_rule = spatial_const.get('ideal_distance_from_sofa') or \
                               spatial_const.get('ideal_distance_from_seating')
            
            min_seating = seating_dist_rule.get('min', 0.4)
            ideal_seating = seating_dist_rule.get('ideal', 0.7)
            max_seating = seating_dist_rule.get('max', 2.0)
            
            nearest_seating_dist = self._get_nearest_seating_distance(x, y)
            
            if min_seating <= nearest_seating_dist <= max_seating:
                error = abs(nearest_seating_dist - ideal_seating)
                seating_score = 1.0 - (error / max(ideal_seating, 0.5))
                score = max(score, seating_score)
        
        return min(1.0, score)
    
    def _reward_functional_requirements(
        self,
        furniture: Dict,
        func_req: Dict
    ) -> float:
        """Reward based on functional requirements"""
        if not func_req:
            return 0.5
        
        x, y = furniture['position']['x'], furniture['position']['y']
        score = 0.5
        
        # Check window proximity for items needing natural light
        if func_req.get('needs_natural_light', False):
            window_dist = self._get_nearest_window_distance(x, y)
            max_window_dist = func_req.get('max_window_distance', 3.0)
            
            if window_dist <= max_window_dist:
                score = max(score, 1.0 - (window_dist / max_window_dist))
            else:
                score = min(score, 0.3)
        
        # Check if fills empty corner
        if func_req.get('fills_empty_corner', False):
            if self._is_in_corner(x, y) and not self._is_corner_occupied(x, y):
                score = max(score, 1.0)
        
        # Check illumination requirements
        if func_req.get('illuminates_seating_area', False):
            seating_dist = self._get_nearest_seating_distance(x, y)
            coverage_radius = func_req.get('light_coverage_radius', 2.5)
            
            if seating_dist <= coverage_radius:
                score = max(score, 0.9)
        
        # Check if reachable from seat
        if func_req.get('reachable_from_seat', False):
            seating_dist = self._get_nearest_seating_distance(x, y)
            max_reach = func_req.get('max_reach_distance', 0.8)
            
            if seating_dist <= max_reach:
                score = max(score, 1.0)
            elif seating_dist <= max_reach * 1.5:
                score = max(score, 0.6)
        
        return min(1.0, score)
    
    def _reward_zone_preference(
        self,
        furniture: Dict,
        zone_pref: Dict
    ) -> float:
        """Reward based on zone preferences"""
        if not zone_pref:
            return 0.5
        
        x, y = furniture['position']['x'], furniture['position']['y']
        score = 0.5
        
        primary_zone = zone_pref.get('primary_zone', '')
        
        # Check if in preferred zone
        if primary_zone == 'seating_area':
            for zone in self.spatial_zones['seating_area']:
                cx, cy = zone['center']
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist <= zone['radius']:
                    score = max(score, 1.0)
                    break
        
        elif primary_zone == 'corner':
            if self._is_in_corner(x, y):
                score = 1.0
            else:
                score = 0.3
        
        elif primary_zone == 'peripheral':
            if self._is_near_wall(x, y, threshold=0.8):
                score = 1.0
            else:
                score = 0.5
        
        # Check corner compatibility
        if zone_pref.get('corner_compatible', False):
            if self._is_in_corner(x, y):
                score = max(score, 0.8)
        
        # Check center room preference
        center_score = zone_pref.get('center_room_score', 0.0)
        if center_score > 0 and self._is_central_position(x, y):
            score = max(score, center_score)
        
        return min(1.0, score)
    
    def _reward_tag_compatibility(
        self,
        primary_tag: str,
        secondary_tags: List[str]
    ) -> float:
        """Reward based on tag compatibility with already placed items"""
        all_furniture = self.existing_furniture + self.placed_items
        
        if not all_furniture:
            return 0.5
        
        compatibility_score = 0.0
        compatible_count = 0
        conflict_count = 0
        
        # Check each placed item
        for placed in all_furniture:
            placed_type = placed['type']
            
            # Map type to tags (simplified)
            placed_tag = self._map_type_to_primary_tag(placed_type)
            
            # Check relationships
            if primary_tag in self.tag_relationships:
                if placed_tag in self.tag_relationships[primary_tag]:
                    compatible_count += 1
            
            # Check conflicts
            if primary_tag in self.tag_conflicts:
                if placed_tag in self.tag_conflicts[primary_tag]:
                    conflict_count += 1
        
        # Calculate score
        if compatible_count > 0:
            compatibility_score += min(0.7, compatible_count * 0.25)
        
        if conflict_count > 0:
            compatibility_score -= conflict_count * 0.3
        
        return max(0.0, min(1.0, 0.5 + compatibility_score))
    
    def _map_type_to_primary_tag(self, ftype: str) -> str:
        """Map furniture type to primary tag"""
        mapping = {
            'sofa': 'primary_seating',
            'armchair': 'primary_seating',
            'coffee_table': 'seating_companion',
            'side_table': 'seating_accessory',
            'floor_lamp': 'task_lighting',
            'table_lamp': 'task_lighting',
            'bookshelf': 'wall_unit',
            'console_table': 'wall_unit',
            'plant_stand': 'corner_decor',
            'ottoman': 'flexible_seating'
        }
        return mapping.get(ftype, 'unknown')
    
    def _reward_diversity(self, catalog_item: Dict) -> float:
        """Reward category diversity — penalise repeats strongly"""
        categories = [f['category'] for f in self.placed_items]
        types      = [f['type']     for f in self.placed_items]
        
        # Never reward placing the same type twice
        if catalog_item['type'] in types:
            return 0.0
        # Bonus for a brand-new category
        if catalog_item['category'] not in categories:
            return 1.0
        # Same category but different type is acceptable
        if categories.count(catalog_item['category']) < 2:
            return 0.6
        return 0.2
    
    def _reward_accessibility(self, furniture: Dict) -> float:
        """Reward maintaining accessibility — exponential penalty for proximity"""
        x, y = furniture['position']['x'], furniture['position']['y']
        
        all_furniture = self.existing_furniture + self.placed_items
        min_clearance = float('inf')
        
        for other in all_furniture:
            if other == furniture:
                continue
            
            ox, oy = other['position']['x'], other['position']['y']
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            min_clearance = min(min_clearance, dist)
        
        # Exponential penalty curve
        if min_clearance >= 1.8:
            return 1.0
        elif min_clearance >= 1.2:
            return 0.8
        elif min_clearance >= 0.8:
            return 0.4
        elif min_clearance >= 0.5:
            return 0.1
        else:
            return 0.0  # severe penalty for very close items
    
    def _reward_spatial_distribution(self) -> float:
        """Reward for spreading items across the room (prevent clustering)"""
        if len(self.placed_items) < 2:
            return 1.0
        
        # Calculate average pairwise distance
        positions = [(f['position']['x'], f['position']['y']) for f in self.placed_items]
        total_dist = 0.0
        pairs = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                              (positions[i][1] - positions[j][1])**2)
                total_dist += dist
                pairs += 1
        
        avg_dist = total_dist / pairs if pairs > 0 else 0.0
        room_diag = np.sqrt(self.room_length**2 + self.room_width**2)
        
        # Ideal: items spread ~25% of diagonal apart on average
        ideal_dist = room_diag * 0.25
        if avg_dist >= ideal_dist:
            return 1.0
        else:
            return min(1.0, avg_dist / ideal_dist)
    
    # Helper methods
    
    def _get_min_distance_to_wall(self, x: float, y: float) -> float:
        """Get minimum distance to any wall"""
        return min(x, self.room_length - x, y, self.room_width - y)
    
    def _get_nearest_seating_distance(self, x: float, y: float) -> float:
        """Get distance to nearest seating"""
        min_dist = float('inf')
        
        all_furniture = self.existing_furniture + self.placed_items
        for furniture in all_furniture:
            if furniture['category'] == 'seating':
                fx, fy = furniture['position']['x'], furniture['position']['y']
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 999
    
    def _get_nearest_window_distance(self, x: float, y: float) -> float:
        """Get distance to nearest window"""
        min_dist = float('inf')
        
        for window in self.room_layout.get('windows', []):
            wx, wy = window['position']['x'], window['position']['y']
            dist = np.sqrt((x - wx)**2 + (y - wy)**2)
            min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 999
    
    def _is_corner_occupied(self, x: float, y: float, threshold: float = 0.5) -> bool:
        """Check if a corner zone is already occupied"""
        all_furniture = self.existing_furniture + self.placed_items
        
        for furniture in all_furniture:
            fx, fy = furniture['position']['x'], furniture['position']['y']
            if self._is_in_corner(fx, fy):
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                if dist < threshold:
                    return True
        
        return False
    
    def _is_central_position(self, x: float, y: float, threshold: float = 1.5) -> bool:
        """Check if position is central in the room"""
        center_x = self.room_length / 2
        center_y = self.room_width / 2
        
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        return dist_from_center <= threshold
    
    def _check_adequate_clearance(
        self, 
        furniture: Dict, 
        min_clearance: float = 0.8
    ) -> bool:
        """Check if furniture has adequate clearance on all sides"""
        x, y = furniture['position']['x'], furniture['position']['y']
        
        all_furniture = self.existing_furniture + self.placed_items
        for other in all_furniture:
            if other == furniture:
                continue
            
            ox, oy = other['position']['x'], other['position']['y']
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            
            if dist < min_clearance:
                return False
        
        return True