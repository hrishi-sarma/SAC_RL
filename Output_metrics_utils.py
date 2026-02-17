import numpy as np
import torch


def compute_graph_violations(graph_data):
    """
    Count violation edges from scene graph.
    EDGE_VIOLATES = 4 (from SceneGraphBuilder)
    """
    edge_attr = graph_data["edge_attr"]
    if edge_attr is None or len(edge_attr) == 0:
        return 0, 0

    # Edge type is first 8 values (one-hot)
    edge_types = edge_attr[:, :8]
    violate_mask = edge_types[:, 4]  # index 4 = EDGE_VIOLATES

    total_edges = len(edge_types)
    violations = int(np.sum(violate_mask))

    return violations, total_edges


def compute_budget_utilization(env):
    if env.budget_max == 0:
        return 0.0
    return env.budget_used / env.budget_max


def aggregate_episode_metrics(metrics_list):
    """
    Aggregate metrics across episodes
    """
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    return aggregated
