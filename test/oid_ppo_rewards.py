"""
OID-PPO Reward Functions  -  FIXED VERSION
Key fix: R_path A* searches to furniture VICINITY (0.6m radius),
         not the exact occupied cell (which is always blocked).
         This was the cause of R_path = -1.0 every episode.
"""

import numpy as np
import heapq
from typing import Tuple
from typing import List



def _reward_pairwise(self) -> float:
    if len(self.functional_pairs) == 0:
        return 0.0

    existing_furniture = [
        {'type': 'sofa',    'position': np.array([3.0, 4.5]), 'front_direction': np.array([0, -1])},
        {'type': 'tv_unit', 'position': np.array([3.0, 0.3]), 'front_direction': np.array([0,  1])}
    ]

    all_furniture = existing_furniture.copy()
    for pf in self.placed_furniture:
        all_furniture.append({
            'type':           pf['furniture'].type,
            'position':       pf['position'],
            'front_direction': pf['front_direction']
        })

    total_reward = 0.0
    pair_count   = 0

    for parent_type, child_type, alpha in self.functional_pairs:
        parents  = [f for f in all_furniture if f['type'] == parent_type]
        children = [f for f in all_furniture if f['type'] == child_type]

        for parent in parents:
            for child in children:
                d_pc   = np.linalg.norm(parent['position'] - child['position'])
                K_dist = (1 + np.cos(np.pi * d_pc / self.d_triangle)) / 2

                n_p = parent['front_direction']
                n_c = child['front_direction']
                K_dir = (1 + alpha * np.dot(n_p, n_c)) / 2

                total_reward += K_dist * K_dir
                pair_count   += 1

    R_pair = total_reward / pair_count if pair_count > 0 else 0.0
    return float(np.clip(R_pair, -1, 1))


def _reward_accessibility(self) -> float:
    total_violation_ratio = 0.0

    for pf in self.placed_furniture:
        furniture  = pf['furniture']
        position   = pf['position']
        rotation   = pf['rotation']
        front_dir  = pf['front_direction']
        perp_dir   = np.array([-front_dir[1], front_dir[0]])

        access_directions = [
            (front_dir,  0.6),
            (-front_dir, 0.4),
            (perp_dir,   0.4),
            (-perp_dir,  0.4)
        ]

        total_access_area = 0.0
        violated_area     = 0.0

        for direction, clearance_offset in access_directions:
            perp_width = furniture.width if rotation % 2 == 0 else furniture.length
            dir_access_area = clearance_offset * perp_width
            total_access_area += dir_access_area

            n_samples = 5
            dir_violations = 0

            for i in range(n_samples):
                offset_dist = clearance_offset * (i + 1) / n_samples
                check_point = position + direction * offset_dist

                for other_pf in self.placed_furniture:
                    if other_pf is pf:
                        continue
                    is_paired = any(
                        (furniture.type == c and other_pf['furniture'].type == p) or
                        (furniture.type == p and other_pf['furniture'].type == c)
                        for p, c, _ in self.functional_pairs
                    )
                    if not is_paired and self._point_in_polygon(check_point, other_pf['footprint']):
                        dir_violations += 1
                        break

            violated_area += (dir_violations / n_samples) * dir_access_area

        if total_access_area > 0:
            total_violation_ratio += violated_area / total_access_area

    n = len(self.placed_furniture)
    R_a = 1 - (2 / n) * total_violation_ratio if n > 0 else 1.0
    return float(np.clip(R_a, -1, 1))


def _reward_visibility(self) -> float:
    total_penalty = 0.0

    for pf in self.placed_furniture:
        nearest_wall, _ = self._find_nearest_wall(pf['position'])
        dot_product = np.dot(pf['front_direction'], nearest_wall['normal'])
        total_penalty += dot_product

    n = len(self.placed_furniture)
    R_v = -total_penalty / n if n > 0 else 0.0
    return float(np.clip(R_v, -1, 1))


def _reward_pathway(self) -> float:
    """
    FIX: A* now searches to the VICINITY of furniture (any free cell within
    0.6 m), not the exact occupied cell of the furniture centre.
    The old code tried to path-find into an occupied grid cell, which A*
    correctly rejects, giving R_path = -1 every time.
    """
    total_score = 0.0

    for pf in self.placed_furniture:
        position = pf['position']

        min_door_dist = min(
            np.linalg.norm(position - door) for door in self.doors
        )
        kappa = (min_door_dist / self.d_triangle) ** 2

        # FIX: check vicinity reachability
        is_reachable = self._is_reachable_from_door(position)

        if is_reachable:
            term = np.exp(-kappa)
        else:
            term = -(1 - np.exp(-kappa))

        total_score += term

    n = len(self.placed_furniture)
    if n == 0:
        return 0.0

    R_path = 1 - (2 / n) * sum(
        0 if self._is_reachable_from_door(pf['position'])
        else 1
        for pf in self.placed_furniture
    )
    # Recompute properly
    total = 0.0
    for pf in self.placed_furniture:
        pos = pf['position']
        min_door_dist = min(np.linalg.norm(pos - d) for d in self.doors)
        kappa = (min_door_dist / self.d_triangle) ** 2
        if self._is_reachable_from_door(pos):
            total += np.exp(-kappa)
        else:
            total += -(1.0)

    R_path = total / n
    return float(np.clip(R_path, -1, 1))


def _reward_balance(self) -> float:
    if not self.placed_furniture:
        return 0.0

    positions = np.array([pf['position'] for pf in self.placed_furniture])
    centroid  = positions.mean(axis=0)

    exp1 = np.exp(
        -np.linalg.norm(centroid - self.room_center) ** 2
        / (2 * self.d_triangle ** 2)
    )

    if len(positions) > 1:
        cov = np.cov(positions.T)
        sigma_F = np.trace(cov) if cov.ndim == 2 else float(cov)
    else:
        sigma_F = 0.0

    kappa_E_sq = self.kappa_E_squared
    exp2 = np.exp(
        -(sigma_F - kappa_E_sq) ** 2
        / (kappa_E_sq ** 2 + 1e-8)
    )

    R_b = exp1 + exp2 - 1.0
    return float(np.clip(R_b, -1, 1))


def _reward_alignment(self) -> float:
    total_weighted_score = 0.0
    total_weight         = 0.0

    for pf in self.placed_furniture:
        furniture = pf['furniture']
        rotation  = pf['rotation']
        position  = pf['position']

        if rotation % 2 == 0:
            orientation     = np.array([1.0, 0.0])
            long_axis_length = furniture.length
        else:
            orientation     = np.array([0.0, 1.0])
            long_axis_length = furniture.width

        nearest_wall, min_wall_dist = self._find_nearest_wall(position)
        wall_vec    = nearest_wall['end'] - nearest_wall['start']
        wall_tangent = wall_vec / (np.linalg.norm(wall_vec) + 1e-8)

        dot   = abs(np.dot(orientation, wall_tangent))
        theta = np.arccos(np.clip(dot, 0.0, 1.0))
        omega = min_wall_dist / (long_axis_length + 1e-8)

        angular_term   = np.cos(2 * theta) ** 2
        proximity_term = 1 - np.tanh(2 * omega) ** 2

        weight = furniture.area
        total_weighted_score += weight * angular_term * proximity_term
        total_weight         += weight

    R_al = total_weighted_score / total_weight if total_weight > 0 else 0.0
    return float(np.clip(R_al, -1, 1))


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_nearest_wall(self, position: np.ndarray) -> Tuple[dict, float]:
    min_dist     = float('inf')
    nearest_wall = None
    for wall in self.walls:
        dist = self._point_to_segment_distance(position, wall['start'], wall['end'])
        if dist < min_dist:
            min_dist     = dist
            nearest_wall = wall
    return nearest_wall, min_dist


def _point_to_segment_distance(self, point, seg_start, seg_end) -> float:
    seg_vec   = seg_end - seg_start
    point_vec = point   - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq == 0:
        return float(np.linalg.norm(point - seg_start))
    t = max(0.0, min(1.0, np.dot(point_vec, seg_vec) / seg_len_sq))
    return float(np.linalg.norm(point - (seg_start + t * seg_vec)))


def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y   = point
    n      = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _is_reachable_from_door(self, position: np.ndarray) -> bool:
    """
    FIX: search to any FREE cell within CLEARANCE_M metres of the furniture
    centre, not the furniture cell itself (which is occupied and A* rejects).
    """
    CLEARANCE_M   = 0.6          # look for path entry within 60 cm
    clearance_px  = max(1, int(CLEARANCE_M / self.map_resolution))

    tgt_gx = int(position[0] / self.map_resolution)
    tgt_gy = int(position[1] / self.map_resolution)

    H, W = self.occupancy_map.shape

    # Build list of free goal cells in a square neighbourhood
    goals = []
    for dy in range(-clearance_px, clearance_px + 1):
        for dx in range(-clearance_px, clearance_px + 1):
            gx, gy = tgt_gx + dx, tgt_gy + dy
            if 0 <= gx < W and 0 <= gy < H:
                if self.occupancy_map[gy, gx] < 0.5:
                    goals.append((gx, gy))

    if not goals:
        return False

    for door in self.doors:
        door_gx = int(np.clip(door[0] / self.map_resolution, 0, W - 1))
        door_gy = int(np.clip(door[1] / self.map_resolution, 0, H - 1))

        # Find nearest free start cell to door
        start = None
        for r in range(3):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    sx, sy = door_gx + dx, door_gy + dy
                    if 0 <= sx < W and 0 <= sy < H and self.occupancy_map[sy, sx] < 0.5:
                        start = (sx, sy)
                        break
                if start:
                    break
            if start:
                break

        if start is None:
            continue

        if self._astar_search_multi(start, goals):
            return True

    return False


def _astar_search_multi(self, start: Tuple[int, int],
                        goals: List[Tuple[int, int]]) -> bool:
    """A* from start to any cell in goals list."""
    if not goals:
        return False

    goal_set = set(goals)
    if start in goal_set:
        return True

    H, W = self.occupancy_map.shape

    # Heuristic: min Manhattan distance to any goal
    def heuristic(pos):
        return min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in goals)

    open_set = []
    heapq.heappush(open_set, (heuristic(start), start))
    g_score  = {start: 0}
    MAX_ITER = 5000
    iters    = 0

    while open_set and iters < MAX_ITER:
        iters += 1
        _, current = heapq.heappop(open_set)

        if current in goal_set:
            return True

        cx, cy = current
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:   # 8-connected
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if self.occupancy_map[ny, nx] > 0.5:
                continue

            move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
            tg = g_score[current] + move_cost

            if (nx, ny) not in g_score or tg < g_score[(nx, ny)]:
                g_score[(nx, ny)] = tg
                f = tg + heuristic((nx, ny))
                heapq.heappush(open_set, (f, (nx, ny)))

    return False


# keep old name as alias so oid_ppo_complete.py still works
def _astar_search(self, start, goal):
    return self._astar_search_multi(start, [goal])


# type hint needed by _astar_search_multi
