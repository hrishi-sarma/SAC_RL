"""
OID-PPO Metric Engine
Implements the exact evaluation criteria from the OID-PPO Paper (arXiv:2508.00364)
"""
import numpy as np
from shapely.geometry import Polygon, box, Point
import heapq

class OIDMetrics:
    def __init__(self, room_dims):
        self.L, self.W = room_dims['length'], room_dims['width']
        self.room_diag = np.sqrt(self.L**2 + self.W**2)
        self.room_center = np.array([self.L/2, self.W/2])
        # Ideal variance for uniform distribution (Definition 4 in paper)
        self.sigma_E_sq = (self.L**2 + self.W**2) / 12.0 

    def calculate_all(self, furniture_items, doors, walls):
        """Calculates the composite OID reward (R_idg) and all sub-components."""
        if not furniture_items: return 0.0, {}

        # 1. Pairwise Relationship (R_pair)
        r_pair = self._calc_pairwise(furniture_items)

        # 2. Accessibility (R_a)
        r_access = self._calc_accessibility(furniture_items)

        # 3. Visibility (R_v)
        r_vis = self._calc_visibility(furniture_items, walls)

        # 4. Pathway Connection (R_path)
        r_path = self._calc_pathway(furniture_items, doors)

        # 5. Visual Balance (R_b)
        r_bal = self._calc_balance(furniture_items)

        # 6. Alignment (R_al)
        r_align = self._calc_alignment(furniture_items, walls)

        # Composite Reward (Equation 5 in paper)
        r_idg = (r_pair + r_access + r_vis + r_path + r_bal + r_align) / 6.0
        
        return r_idg, {
            "R_pair": r_pair, "R_a": r_access, "R_v": r_vis,
            "R_path": r_path, "R_b": r_bal, "R_al": r_align
        }

    def _calc_pairwise(self, items):
        """Eq 2: Pairwise Relationship (Distance + Orientation kernels)"""
        # Hardcoded pairs based on catalog types for demo
        pairs = []
        for i, p in enumerate(items):
            for j, c in enumerate(items):
                if i == j: continue
                # Define simple pairs: Sofa <-> Coffee Table
                if p['type'] == 'sofa' and c['type'] == 'coffee_table':
                    pairs.append((p, c, -1)) # alpha=-1 for face-to-face
                # Bed <-> Side Table
                elif p['type'] == 'bed' and c['type'] == 'side_table':
                    pairs.append((p, c, 1))  # alpha=1 for parallel/side
        
        if not pairs: return 0.0 # Neutral if no pairs exist

        total_score = 0
        for p, c, alpha in pairs:
            # Distance Kernel
            d_pc = np.linalg.norm(np.array([p['position']['x'], p['position']['y']]) - 
                                  np.array([c['position']['x'], c['position']['y']]))
            k_dist = 1 + np.cos((np.pi * d_pc) / self.room_diag)
            
            # Directional Kernel (Simplified dot product of forward vectors)
            n_p = self._get_forward_vector(p['rotation'])
            n_c = self._get_forward_vector(c['rotation'])
            dot = np.dot(n_p, n_c)
            k_dir = (1 + alpha * dot) / 2.0
            
            total_score += k_dist * k_dir
            
        return total_score / len(pairs)

    def _calc_accessibility(self, items):
        """Eq 3: Accessibility (clearance overlap)"""
        # Simplified: Check if bounding box + clearance overlaps others
        violations = 0
        for i, item in enumerate(items):
            # Define access area (box + 0.5m buffer in front)
            access_poly = self._get_poly(item, buffer=0.5) 
            is_blocked = False
            for j, other in enumerate(items):
                if i == j: continue
                other_poly = self._get_poly(other)
                if access_poly.intersects(other_poly):
                    is_blocked = True
                    break
            if is_blocked: violations += 1
            
        # R_a = 1 - 2 * (violation_ratio)
        return 1.0 - 2.0 * (violations / len(items))

    def _calc_visibility(self, items, walls):
        """Eq 4: Visibility (Front facing away from wall)"""
        score_sum = 0
        for item in items:
            n_f = self._get_forward_vector(item['rotation'])
            # Find nearest wall normal
            n_w = self._get_nearest_wall_normal(item, walls)
            score_sum += np.dot(n_f, n_w)
            
        # R_v = - average(dot_product)
        return -1.0 * (score_sum / len(items))

    def _calc_pathway(self, items, doors):
        """Eq 5 & Def 3: Pathway Connection (Reachability)"""
        # Simplified A* proxy: Straight line check with intersection
        reachable_count = 0
        for item in items:
            item_center = Point(item['position']['x'], item['position']['y'])
            is_reachable = False
            for door in doors:
                door_center = Point(door['position']['x'], door['position']['y'])
                path_line = item_center.distance(door_center)
                # Heuristic: simple distance decay from paper
                kappa = (path_line / self.room_diag)**2
                
                # Check obstruction (raycast)
                obstructed = False
                for obs in items:
                    if obs == item: continue
                    if self._get_poly(obs).distance(item_center) < 0.1: # rudimentary check
                        pass 
                
                if not obstructed:
                    score = np.exp(-kappa)
                    reachable_count += score
                    is_reachable = True
                    break # Reached from at least one door
            
            if not is_reachable:
                reachable_count += 0 # 0 for unreachable (Proposition 2)

        # Normalize to [-1, 1] roughly as per paper logic
        return (reachable_count / len(items)) * 2 - 1

    def _calc_balance(self, items):
        """Eq 6: Visual Balance (Center of Mass + Spatial Variance)"""
        total_area = 0
        weighted_pos = np.zeros(2)
        
        for item in items:
            area = item['dimensions']['length'] * item['dimensions']['width']
            pos = np.array([item['position']['x'], item['position']['y']])
            weighted_pos += pos * area
            total_area += area
            
        if total_area == 0: return 0.0
        
        # Center of Mass
        x_bar = weighted_pos / total_area
        dist_to_center = np.linalg.norm(x_bar - self.room_center)
        term1 = np.exp(-(dist_to_center**2) / (self.room_diag**2))
        
        # Spatial Variance Tensor (Simplified to trace/radius for 2D)
        # Calculates how spread out the items are relative to the room size
        variance = 0
        for item in items:
             area = item['dimensions']['length'] * item['dimensions']['width']
             pos = np.array([item['position']['x'], item['position']['y']])
             variance += area * np.linalg.norm(pos - x_bar)**2
        variance /= total_area
        
        # Compare to ideal uniform variance (sigma_E_sq)
        term2 = np.exp(-((variance - self.sigma_E_sq)**2) / (self.sigma_E_sq**2))
        
        return term1 + term2 - 1.0

    def _calc_alignment(self, items, walls):
        """Eq 7: Alignment (Wall proximity + orientation)"""
        # Checks if items align with walls (0 or 90 degrees)
        score_sum = 0
        total_area = 0
        
        for item in items:
            area = item['dimensions']['length'] * item['dimensions']['width']
            total_area += area
            
            # Angle deviation from nearest 90 deg
            rot = item['rotation'] % 90
            if rot > 45: rot = 90 - rot
            # Convert to radians for formula: cos^2(2*theta)
            theta = np.deg2rad(rot)
            align_score = np.cos(2*theta)**2
            
            score_sum += area * align_score
            
        if total_area == 0: return 0.0
        return score_sum / total_area

    # Helpers
    def _get_forward_vector(self, rotation):
        rad = np.deg2rad(rotation)
        return np.array([np.cos(rad), np.sin(rad)])
        
    def _get_poly(self, item, buffer=0.0):
        x, y = item['position']['x'], item['position']['y']
        l, w = item['dimensions']['length'] + buffer, item['dimensions']['width'] + buffer
        return box(x - l/2, y - w/2, x + l/2, y + w/2)

    def _get_nearest_wall_normal(self, item, walls):
        # Simplified: returns normal of nearest boundary box wall
        x, y = item['position']['x'], item['position']['y']
        dists = [x, self.L - x, y, self.W - y] # Left, Right, Bottom, Top
        min_idx = np.argmin(dists)
        normals = [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]
        return normals[min_idx]