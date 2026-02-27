"""
evaluate_oid_ppo_criteria.py
────────────────────────────
Post-hoc evaluator that scores a trained SAC-GNN agent using the
six reward components from the OID-PPO paper (Yoon et al., 2025):

    Ridg = (1/6) * (Rpair + Ra + Rv + Rpath + Rb + Ral)  ∈ [-1, 1]

All six sub-rewards are independently implemented from the paper's
definitions and are then averaged to produce Ridg for direct
comparison with Table 1 of the paper.

Inference time per episode is also measured.

Usage
-----
    python evaluate_oid_ppo_criteria.py --model models_gnn/best_model.pt
                                        --episodes 10
                                        --fn 4

Outputs
-------
    outputs/oid_ppo_eval_results.json   — full per-episode breakdown
    outputs/oid_ppo_comparison.txt      — human-readable comparison table
"""

import argparse
import json
import os
import time
import copy
import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── project imports ────────────────────────────────────────────────────────────
from furniture_env_gnn import FurnitureRecommendationEnvGNN
from sac_agent_gnn import SACAgentWithGNN
from graph_encoder import create_graph_encoder


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _furniture_polygon(f: Dict) -> List[Tuple[float, float]]:
    """Return the 4 corners of a furniture footprint (axis-aligned after rotation)."""
    x, y = f["position"]["x"], f["position"]["y"]
    rot = f.get("rotation", 0)
    l = f["dimensions"]["length"]
    w = f["dimensions"]["width"]
    # Swap dims for 90/270 rotations (axis-aligned bounding box)
    if rot in (90, 270):
        l, w = w, l
    half_l, half_w = l / 2.0, w / 2.0
    return [
        (x - half_l, y - half_w),
        (x + half_l, y - half_w),
        (x + half_l, y + half_w),
        (x - half_l, y + half_w),
    ]


def _polygon_area(corners: List[Tuple[float, float]]) -> float:
    """Shoelace formula."""
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def _polygon_centroid(corners: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(corners)
    cx = sum(c[0] for c in corners) / n
    cy = sum(c[1] for c in corners) / n
    return cx, cy


def _rect_area(f: Dict) -> float:
    return f["dimensions"]["length"] * f["dimensions"]["width"]


def _front_direction(f: Dict) -> np.ndarray:
    """
    Canonical front direction vector for a furniture item.
    Convention:  rotation=0   → +Y  (faces "north")
                 rotation=90  → +X  (faces "east")
                 rotation=180 → -Y  (faces "south")
                 rotation=270 → -X  (faces "west")
    """
    rot = f.get("rotation", 0)
    mapping = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}
    dx, dy = mapping.get(rot, (0, 1))
    return np.array([dx, dy], dtype=float)


def _room_diagonal(room_l: float, room_w: float) -> float:
    return math.sqrt(room_l ** 2 + room_w ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# OID-PPO Reward Components
# ══════════════════════════════════════════════════════════════════════════════

class OIDPPOEvaluator:
    """
    Implements the six OID-PPO design-guideline rewards.

    Parameters
    ----------
    room_layout : dict
        Parsed room_layout.json.
    functional_pairs : list of (str, str)
        Furniture-type pairs that form functional units.
        e.g. [("sofa", "coffee_table"), ("sofa", "side_table")]
    clearance_per_type : dict
        Minimum clearance (m) required on each side of a furniture type.
        Defaults to 0.6 m for all types if not specified.
    """

    # ── default functional pairs (sofa-centric living room) ───────────────────
    DEFAULT_PAIRS = [
        ("sofa",        "coffee_table"),
        ("sofa",        "side_table"),
        ("sofa",        "table_lamp"),
        ("coffee_table","sofa"),
        ("tv_unit",     "sofa"),
    ]

    # alpha_pc per pair: +1 = parallel preferred, -1 = face-to-face preferred
    # Living room pairs (sofa/coffee_table etc.) face the same direction → alpha=+1
    DEFAULT_PAIR_ALPHA = {
        ("sofa",        "coffee_table"): +1.0,
        ("sofa",        "side_table"):   +1.0,
        ("sofa",        "table_lamp"):   +1.0,
        ("coffee_table","sofa"):         +1.0,
        ("tv_unit",     "sofa"):         -1.0,  # tv faces toward sofa
    }

    DEFAULT_CLEARANCE = 0.6   # metres

    def __init__(
        self,
        room_layout: Dict,
        functional_pairs: Optional[List[Tuple[str, str]]] = None,
        clearance_per_type: Optional[Dict[str, float]] = None,
    ):
        self.room_layout = room_layout
        dims = room_layout["room_info"]["dimensions"]
        self.room_l = dims["length"]
        self.room_w = dims["width"]
        self.d_diag = _room_diagonal(self.room_l, self.room_w)

        self.functional_pairs = set(
            functional_pairs if functional_pairs is not None else self.DEFAULT_PAIRS
        )
        self.pair_alpha = self.DEFAULT_PAIR_ALPHA
        self.clearance = clearance_per_type or {}

        self.doors = room_layout.get("doors", [])

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Rpair  – Pairwise relationship reward
    # ─────────────────────────────────────────────────────────────────────────

    def rpair(self, placed: List[Dict]) -> float:
        """
        Equation (paper):
            Kdist(p,c) = 1 + cos(π * d_pc / d_△)        ∈ [0, 2]
            Kdir(p,c)  = 1 + α_pc * <n_p, n_c>²
                         α=+1 → [1,2] (rewards parallel)
                         α=-1 → [0,1] (rewards perpendicular)
            Rpair = (1/|P|) * Σ Kdist * Kdir

        Product range:
            α=+1: [0,4]  → normalise: raw/2 - 1  → [-1,1]
            α=-1: [0,2]  → normalise: raw - 1    → [-1,1]
        We compute per-pair, then average.
        """
        if not placed:
            return 0.0

        scored_pairs = []
        for i, fi in enumerate(placed):
            for j, fj in enumerate(placed):
                if i == j:
                    continue
                pair_key = (fi["type"], fj["type"])
                if pair_key not in self.functional_pairs:
                    continue

                xi, yi = fi["position"]["x"], fi["position"]["y"]
                xj, yj = fj["position"]["x"], fj["position"]["y"]
                d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                kdist = 1.0 + math.cos(math.pi * d / self.d_diag)

                ni = _front_direction(fi)
                nj = _front_direction(fj)
                dot = float(np.dot(ni, nj))
                alpha = self.pair_alpha.get(pair_key, +1.0)
                kdir = 1.0 + alpha * dot ** 2

                # Normalise product to [-1,1] per pair before averaging
                product = kdist * kdir
                if alpha > 0:
                    # product ∈ [0,4] → [-1,1]
                    norm_score = product / 2.0 - 1.0
                else:
                    # product ∈ [0,2] → [-1,1]
                    norm_score = product - 1.0

                scored_pairs.append(norm_score)

        if not scored_pairs:
            return 0.0

        return float(np.clip(np.mean(scored_pairs), -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Ra  – Accessibility reward
    # ─────────────────────────────────────────────────────────────────────────

    def ra(self, placed: List[Dict]) -> float:
        """
        For each furniture item f, compute the required clearance area Uf
        (four rectangular strips of width Ω around its footprint), then
        measure what fraction of that area is blocked by non-paired items.

            Ra = 1 - (2 / |F|) * Σ |ν(f)| / |Uf|

        Uses axis-aligned approximation for speed.
        """
        if len(placed) < 2:
            return 1.0

        all_f = placed
        scores = []
        for i, fi in enumerate(all_f):
            xi, yi = fi["position"]["x"], fi["position"]["y"]
            rot = fi.get("rotation", 0)
            fl = fi["dimensions"]["length"] if rot not in (90, 270) else fi["dimensions"]["width"]
            fw = fi["dimensions"]["width"]  if rot not in (90, 270) else fi["dimensions"]["length"]

            omega = self.clearance.get(fi["type"], self.DEFAULT_CLEARANCE)

            # Four clearance strips (top, bottom, left, right)
            # We approximate as rectangles and compute overlap
            strips = [
                # (strip_xmin, strip_ymin, strip_xmax, strip_ymax)
                (xi - fl / 2,        yi + fw / 2,        xi + fl / 2,        yi + fw / 2 + omega),  # top
                (xi - fl / 2,        yi - fw / 2 - omega, xi + fl / 2,        yi - fw / 2),           # bottom
                (xi - fl / 2 - omega, yi - fw / 2,        xi - fl / 2,        yi + fw / 2),            # left
                (xi + fl / 2,        yi - fw / 2,        xi + fl / 2 + omega, yi + fw / 2),            # right
            ]

            u_total = sum((s[2] - s[0]) * (s[3] - s[1]) for s in strips)
            if u_total <= 0:
                scores.append(1.0)
                continue

            # Non-paired blockers
            non_paired = [
                fj for j, fj in enumerate(all_f)
                if j != i and (fi["type"], fj["type"]) not in self.functional_pairs
            ]

            violated = 0.0
            for strip in strips:
                sx1, sy1, sx2, sy2 = strip
                for fj in non_paired:
                    xj, yj = fj["position"]["x"], fj["position"]["y"]
                    rotj = fj.get("rotation", 0)
                    jl = fj["dimensions"]["length"] if rotj not in (90, 270) else fj["dimensions"]["width"]
                    jw = fj["dimensions"]["width"]  if rotj not in (90, 270) else fj["dimensions"]["length"]
                    bx1 = xj - jl / 2
                    by1 = yj - jw / 2
                    bx2 = xj + jl / 2
                    by2 = yj + jw / 2

                    # Intersection area
                    ox = max(0, min(sx2, bx2) - max(sx1, bx1))
                    oy = max(0, min(sy2, by2) - max(sy1, by1))
                    violated += ox * oy

            violation_ratio = min(1.0, violated / u_total)
            scores.append(1.0 - violation_ratio)

        mean_score = float(np.mean(scores))
        # Paper: Ra = 1 - (2/|F|)*Σ|ν|/|U|
        # Our per-item score is already 1 - ratio, mean gives [0,1]
        # Remap to [-1, 1]
        return 2.0 * mean_score - 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Rv  – Visibility reward
    # ─────────────────────────────────────────────────────────────────────────

    def rv(self, placed: List[Dict]) -> float:
        """
        Penalises furniture whose front faces a wall.

            Rv = -(1/|F|) * Σ <n_f, n_wall(f)>

        n_wall(f) = outward normal of the nearest wall.
        """
        if not placed:
            return 0.0

        walls = {
            "south": {"normal": np.array([0.0, -1.0]), "y": 0.0},
            "north": {"normal": np.array([0.0,  1.0]), "y": self.room_w},
            "west":  {"normal": np.array([-1.0, 0.0]), "x": 0.0},
            "east":  {"normal": np.array([ 1.0, 0.0]), "x": self.room_l},
        }

        total = 0.0
        for f in placed:
            x, y = f["position"]["x"], f["position"]["y"]

            dists = {
                "south": y,
                "north": self.room_w - y,
                "west":  x,
                "east":  self.room_l - x,
            }
            nearest_wall = min(dists, key=dists.get)
            nw = walls[nearest_wall]["normal"]

            nf = _front_direction(f)
            dot = float(np.dot(nf, nw))
            total += dot

        rv_val = -total / len(placed)
        return float(np.clip(rv_val, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Rpath  – Pathway connection reward
    # ─────────────────────────────────────────────────────────────────────────

    def rpath(self, placed: List[Dict], grid_res: float = 0.25) -> float:
        """
        Computes A* reachability from each door to each furniture centroid.

            Rpath = 1 - (2/|F|) * Σ [(1 - I_f) + exp(-κ_f) * I_f]

        where κ_f = (d_door / d_△)² and I_f = reachable flag.

        Uses a discretised occupancy grid.
        """
        if not placed or not self.doors:
            return 0.0

        # Build occupancy grid
        nx = int(math.ceil(self.room_l / grid_res)) + 1
        ny = int(math.ceil(self.room_w / grid_res)) + 1
        grid = np.zeros((ny, nx), dtype=bool)

        # Mark furniture footprints as obstacles
        for f in placed:
            x, y = f["position"]["x"], f["position"]["y"]
            rot = f.get("rotation", 0)
            fl = f["dimensions"]["length"] if rot not in (90, 270) else f["dimensions"]["width"]
            fw = f["dimensions"]["width"]  if rot not in (90, 270) else f["dimensions"]["length"]
            x0, x1 = x - fl / 2, x + fl / 2
            y0, y1 = y - fw / 2, y + fw / 2
            i0 = max(0, int(y0 / grid_res))
            i1 = min(ny - 1, int(y1 / grid_res))
            j0 = max(0, int(x0 / grid_res))
            j1 = min(nx - 1, int(x1 / grid_res))
            grid[i0:i1 + 1, j0:j1 + 1] = True

        # Door positions (clamped inside grid)
        door_cells = []
        for door in self.doors:
            dx, dy = door["position"]["x"], door["position"]["y"]
            di = int(np.clip(dy / grid_res, 0, ny - 1))
            dj = int(np.clip(dx / grid_res, 0, nx - 1))
            door_cells.append((di, dj))

        def nearest_free_cell(fi, fj):
            """Return the nearest unobstructed cell adjacent to (fi, fj)."""
            if not grid[fi, fj]:
                return (fi, fj)
            for radius in range(1, 6):
                for di2 in range(-radius, radius + 1):
                    for dj2 in range(-radius, radius + 1):
                        if abs(di2) != radius and abs(dj2) != radius:
                            continue
                        ni2, nj2 = fi + di2, fj + dj2
                        if 0 <= ni2 < ny and 0 <= nj2 < nx and not grid[ni2, nj2]:
                            return (ni2, nj2)
            return None

        def astar_distance(start, goal):
            """Returns path length in grid cells, or None if unreachable."""
            open_heap = []
            heapq.heappush(open_heap, (0, start))
            dist = {start: 0}
            gx, gy = goal

            while open_heap:
                _, (ci, cj) = heapq.heappop(open_heap)
                if (ci, cj) == goal:
                    return dist[goal]
                for di2, dj2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni2, nj2 = ci + di2, cj + dj2
                    if 0 <= ni2 < ny and 0 <= nj2 < nx and not grid[ni2, nj2]:
                        nd = dist[(ci, cj)] + 1
                        if (ni2, nj2) not in dist or nd < dist[(ni2, nj2)]:
                            dist[(ni2, nj2)] = nd
                            h = abs(ni2 - gx) + abs(nj2 - gy)
                            heapq.heappush(open_heap, (nd + h, (ni2, nj2)))
            return None

        total = 0.0
        for f in placed:
            x, y = f["position"]["x"], f["position"]["y"]
            fi = int(np.clip(y / grid_res, 0, ny - 1))
            fj = int(np.clip(x / grid_res, 0, nx - 1))

            # Target the nearest free cell adjacent to the furniture footprint
            # (the centroid cell itself is always an obstacle)
            goal_cell = nearest_free_cell(fi, fj)
            if goal_cell is None:
                total += 1.0  # completely surrounded → unreachable
                continue

            # Try all doors, pick shortest path
            best_path = None
            best_door_dist = float("inf")

            for dc in door_cells:
                # Also resolve door cell to nearest free cell (door may be on boundary)
                door_free = nearest_free_cell(dc[0], dc[1])
                if door_free is None:
                    continue
                path_len = astar_distance(door_free, goal_cell)
                if path_len is not None:
                    ddoor = math.sqrt(
                        (dc[0] * grid_res - y) ** 2 + (dc[1] * grid_res - x) ** 2
                    )
                    if best_path is None or path_len < best_path:
                        best_path = path_len
                        best_door_dist = ddoor

            if best_path is None:
                total += 1.0   # unreachable → maximum penalty term
            else:
                kappa = (best_door_dist / self.d_diag) ** 2
                total += math.exp(-kappa)

        rpath_val = 1.0 - (2.0 / len(placed)) * total
        return float(np.clip(rpath_val, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Rb  – Visual balance reward
    # ─────────────────────────────────────────────────────────────────────────

    def rb(self, placed: List[Dict]) -> float:
        """
        Two-term reward:
          1. Distance of area-weighted centroid from room centre.
          2. Frobenius distance of spatial variance tensor from κ²_E * I.

            Rb = exp(-‖x̄_F - o‖² / (2*d²_△))
               + exp(-‖Σ_F - κ²_E * I‖²_F / κ⁴_E)
               - 1
        """
        if not placed:
            return 0.0

        areas = np.array([_rect_area(f) for f in placed])
        total_area = areas.sum()
        if total_area == 0:
            return 0.0

        positions = np.array(
            [[f["position"]["x"], f["position"]["y"]] for f in placed]
        )
        weights = areas / total_area
        centroid = (positions * weights[:, None]).sum(axis=0)

        room_center = np.array([self.room_l / 2.0, self.room_w / 2.0])

        # Term 1 – centroid displacement
        t1 = math.exp(
            -np.dot(centroid - room_center, centroid - room_center)
            / (2 * self.d_diag ** 2)
        )

        # Spatial variance tensor
        diffs = positions - centroid                   # [N, 2]
        sigma = np.einsum("n,ni,nj->ij", weights, diffs, diffs)

        kappa2 = (self.room_l ** 2 + self.room_w ** 2) / 12.0
        identity = np.eye(2)
        frob_sq = np.sum((sigma - kappa2 * identity) ** 2)

        t2 = math.exp(-frob_sq / (kappa2 ** 2 + 1e-8))

        rb_val = t1 + t2 - 1.0
        return float(np.clip(rb_val, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Ral  – Alignment reward
    # ─────────────────────────────────────────────────────────────────────────

    def ral(self, placed: List[Dict]) -> float:
        """
        Rewards furniture whose long axis is parallel/perpendicular to
        the nearest wall, weighted by footprint area and proximity.

            Ral = Σ [Π(f) * cos²(2ϑ_f) * (1 - tanh²(ω_f))]
                  ─────────────────────────────────────────
                               Σ Π(f)

        ϑ_f = angular deviation from wall tangent.
        ω_f = normalised distance to wall (d_f / long_axis_length_f).
        """
        if not placed:
            return 0.0

        # Wall tangents (unit vectors along walls)
        wall_tangents = {
            "south": np.array([1.0, 0.0]),
            "north": np.array([1.0, 0.0]),
            "west":  np.array([0.0, 1.0]),
            "east":  np.array([0.0, 1.0]),
        }

        numerator = 0.0
        denom = 0.0

        for f in placed:
            x, y = f["position"]["x"], f["position"]["y"]
            rot = f.get("rotation", 0)
            fl = f["dimensions"]["length"] if rot not in (90, 270) else f["dimensions"]["width"]
            fw = f["dimensions"]["width"]  if rot not in (90, 270) else f["dimensions"]["length"]
            area = fl * fw
            long_axis = max(fl, fw)

            # Nearest wall
            dists = {
                "south": y - fw / 2,
                "north": self.room_w - y - fw / 2,
                "west":  x - fl / 2,
                "east":  self.room_l - x - fl / 2,
            }
            nearest_wall = min(dists, key=lambda k: abs(dists[k]))
            d_wall = abs(dists[nearest_wall])

            tau = wall_tangents[nearest_wall]

            # Long-axis orientation vector
            nf = _front_direction(f)
            u_f = nf  # unit orientation vector (same as front direction)

            cos_theta = abs(float(np.dot(u_f, tau)))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = math.acos(cos_theta)

            omega = d_wall / (long_axis + 1e-8)

            score = (math.cos(2 * theta) ** 2) * (1.0 - math.tanh(omega) ** 2)
            numerator += area * score
            denom += area

        if denom == 0:
            return 0.0

        ral_val = numerator / denom
        # ral_val ∈ [0, 1], remap to [-1, 1]
        return float(np.clip(2.0 * ral_val - 1.0, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Composite Ridg
    # ─────────────────────────────────────────────────────────────────────────

    def ridg(self, placed: List[Dict]) -> Dict[str, float]:
        """
        Compute all six sub-rewards and their mean Ridg.

        Returns a dict with keys: rpair, ra, rv, rpath, rb, ral, ridg.
        """
        components = {
            "rpair": self.rpair(placed),
            "ra":    self.ra(placed),
            "rv":    self.rv(placed),
            "rpath": self.rpath(placed),
            "rb":    self.rb(placed),
            "ral":   self.ral(placed),
        }
        components["ridg"] = float(np.mean(list(components.values())))
        return components


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_path: str,
    room_layout_path: str,
    catalog_path: str,
    num_episodes: int = 10,
    deterministic: bool = True,
    device: str = "cpu",
) -> Dict:
    """
    Load the trained SAC-GNN agent, run it for `num_episodes`, score each
    final layout with OIDPPOEvaluator, and return aggregated results.
    """
    print("\n" + "=" * 66)
    print(" SAC-GNN  vs  OID-PPO  —  Evaluation")
    print("=" * 66)

    # ── environment ────────────────────────────────────────────────────────────
    env = FurnitureRecommendationEnvGNN(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        max_items=4,
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ── agent ──────────────────────────────────────────────────────────────────
    graph_encoder = create_graph_encoder(
        node_feature_dim=22,
        edge_feature_dim=10,
        output_dim=32,
        device=device,
    )
    agent = SACAgentWithGNN(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_encoder=graph_encoder,
        device=device,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    agent.load(model_path)
    agent.eval()
    print(f"  Model loaded  : {model_path}")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Device        : {device}")

    # ── evaluator ──────────────────────────────────────────────────────────────
    with open(room_layout_path, "r") as fh:
        room_layout = json.load(fh)
    evaluator = OIDPPOEvaluator(room_layout)

    # ── per-episode records ────────────────────────────────────────────────────
    records = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward_env = 0.0
        step_inference_times = []

        while not done:
            graph_data = env.get_graph_data()

            t0 = time.perf_counter()
            action = agent.select_action(state, graph_data, deterministic=deterministic)
            t1 = time.perf_counter()
            step_inference_times.append(t1 - t0)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward_env += reward
            state = next_state

        # Score final layout with OID-PPO criteria
        # Include existing furniture so Rpath / Ra / Rv see the full scene
        all_placed = env.existing_furniture + env.placed_items
        oid_scores = evaluator.ridg(all_placed)

        inference_total = sum(step_inference_times)
        inference_per_step = float(np.mean(step_inference_times)) if step_inference_times else 0.0

        record = {
            "episode":              ep,
            "env_reward":           float(episode_reward_env),
            "items_placed":         len(env.placed_items),
            "budget_used":          float(env.budget_used),
            "inference_total_s":    round(inference_total, 4),
            "inference_per_step_s": round(inference_per_step, 4),
            **{f"oid_{k}": round(v, 4) for k, v in oid_scores.items()},
        }
        records.append(record)

        print(
            f"  Ep {ep:>3d} | Items={record['items_placed']} "
            f"| Ridg={oid_scores['ridg']:+.3f} "
            f"[pair={oid_scores['rpair']:+.2f} "
            f"acc={oid_scores['ra']:+.2f} "
            f"vis={oid_scores['rv']:+.2f} "
            f"path={oid_scores['rpath']:+.2f} "
            f"bal={oid_scores['rb']:+.2f} "
            f"aln={oid_scores['ral']:+.2f}] "
            f"| t={inference_total:.3f}s"
        )

    # ── aggregate ──────────────────────────────────────────────────────────────
    ridg_vals = [r["oid_ridg"] for r in records]
    inf_vals  = [r["inference_total_s"] for r in records]

    component_keys = ["oid_rpair", "oid_ra", "oid_rv", "oid_rpath", "oid_rb", "oid_ral"]
    component_means = {k: float(np.mean([r[k] for r in records])) for k in component_keys}

    aggregated = {
        "model":              model_path,
        "num_episodes":       num_episodes,
        "fn":                 4,
        "room_shape":         "square",
        "avg_ridg":           round(float(np.mean(ridg_vals)),  4),
        "std_ridg":           round(float(np.std(ridg_vals)),   4),
        "avg_items_placed":   round(float(np.mean([r["items_placed"]  for r in records])), 2),
        "success_rate":       round(float(np.mean([r["items_placed"] >= 4 for r in records])), 4),
        "avg_inference_s":    round(float(np.mean(inf_vals)), 4),
        "std_inference_s":    round(float(np.std(inf_vals)),  4),
        "component_means":    {k.replace("oid_", ""): round(v, 4) for k, v in component_means.items()},
        "episodes":           records,
    }

    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table formatter
# ══════════════════════════════════════════════════════════════════════════════

OID_PPO_TABLE1 = {
    # Fn=4, Square room — from Table 1 of the paper
    "MH":       {"reward": 0.281, "time_s": 70.28,  "p_loss": None,  "v_loss": None},
    "MOPSO":    {"reward": 0.338, "time_s": 96.82,  "p_loss": None,  "v_loss": None},
    "DDPG":     {"reward": 0.803, "time_s":  1.247, "p_loss": 0.065, "v_loss": 0.233},
    "TD3":      {"reward": 0.823, "time_s":  1.181, "p_loss": 0.061, "v_loss": 0.206},
    "SAC":      {"reward": 0.903, "time_s":  2.547, "p_loss": 0.045, "v_loss": 0.134},
    "OID-PPO":  {"reward": 0.971, "time_s":  3.181, "p_loss": 0.009, "v_loss": 0.026},
}


def format_comparison_table(results: Dict) -> str:
    """Render a side-by-side comparison table (Fn=4, Square)."""

    ridg = results["avg_ridg"]
    tinf = results["avg_inference_s"]

    lines = []
    lines.append("=" * 70)
    lines.append("  OID-PPO Paper  (Table 1)  vs  SAC-GNN  —  Fn=4, Square Room")
    lines.append("=" * 70)
    lines.append(f"  {'Method':<12}  {'Ridg':>7}  {'Delta vs SAC-GNN':>17}  {'Time (s)':>9}")
    lines.append("-" * 70)

    for method, vals in OID_PPO_TABLE1.items():
        delta = vals["reward"] - ridg
        lines.append(
            f"  {method:<12}  {vals['reward']:>7.3f}  {delta:>+17.3f}  {vals['time_s']:>9.3f}"
        )

    lines.append("-" * 70)
    lines.append(
        f"  {'SAC-GNN':<12}  {ridg:>7.3f}  {'(this model)':>17}  {tinf:>9.3f}"
    )
    lines.append("=" * 70)

    lines.append("\n  Component breakdown (SAC-GNN vs paper baselines):")
    lines.append("  " + "-" * 55)
    lines.append(f"  {'Component':<10}  {'SAC-GNN':>8}  {'Paper SAC':>10}  {'OID-PPO':>9}")
    lines.append("  " + "-" * 55)

    # Paper doesn't publish per-component breakdown, so we annotate qualitatively
    comp_labels = {
        "rpair": ("Rpair", "pair"),
        "ra":    ("Ra   ", "acc"),
        "rv":    ("Rv   ", "vis"),
        "rpath": ("Rpath", "path"),
        "rb":    ("Rb   ", "bal"),
        "ral":   ("Ral  ", "aln"),
    }
    cm = results["component_means"]
    for key, (label, _) in comp_labels.items():
        val = cm.get(key, float("nan"))
        lines.append(f"  {label:<10}  {val:>8.3f}  {'N/A':>10}  {'N/A':>9}")

    lines.append("  " + "-" * 55)
    lines.append(f"  {'Ridg':10}  {ridg:>8.3f}  {0.903:>10.3f}  {0.971:>9.3f}")
    lines.append("\n  Note: Ridg values use OID-PPO formula applied post-hoc to SAC-GNN")
    lines.append("  placements. Paper component breakdowns are not published in Table 1.")
    lines.append("=" * 70)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAC-GNN vs OID-PPO criteria")
    parser.add_argument("--model",    default="models_gnn/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--room",     default="uploads/room_layout.json",
                        help="Path to room_layout.json")
    parser.add_argument("--catalog",  default="uploads/furniture_catalog_enhanced.json",
                        help="Path to furniture catalog JSON")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_evaluation(
        model_path=args.model,
        room_layout_path=args.room,
        catalog_path=args.catalog,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        device=device,
    )

    os.makedirs("outputs", exist_ok=True)

    # ── save JSON ──────────────────────────────────────────────────────────────
    json_path = "outputs/oid_ppo_eval_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Full results  → {json_path}")

    # ── save comparison table ─────────────────────────────────────────────────
    table = format_comparison_table(results)
    print("\n" + table)

    txt_path = "outputs/oid_ppo_comparison.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print(f"\n  Comparison table → {txt_path}")
