import torch
import numpy as np
import json
import os

from furniture_env_gnn import FurnitureRecommendationEnvGNN
from sac_agent_gnn import SACAgentWithGNN
from graph_encoder import create_graph_encoder
from Output_metrics_utils import (
    compute_graph_violations,
    compute_budget_utilization,
    aggregate_episode_metrics
)


def evaluate_model(
    model_path,
    num_episodes=20,
    deterministic=True,
    save_path="outputs/detailed_metrics.json"
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = FurnitureRecommendationEnvGNN(
        room_layout_path="uploads/room_layout.json",
        catalog_path="uploads/furniture_catalog_enhanced.json",
        max_items=4
    )

    graph_encoder = create_graph_encoder(
        node_feature_dim=22,
        edge_feature_dim=10,
        output_dim=32,
        device=device
    )

    agent = SACAgentWithGNN(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        graph_encoder=graph_encoder,
        device=device
    )

    agent.load(model_path)
    agent.eval()

    all_episode_metrics = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False

        episode_reward = 0
        violations = 0
        total_edges = 0
        placements = 0

        while not done:
            graph_data = env.get_graph_data()
            action = agent.select_action(state, graph_data, deterministic=deterministic)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            # Count violations
            v, t = compute_graph_violations(graph_data)
            violations += v
            total_edges += t

            if info["valid_placement"] and not info["already_placed"]:
                placements += 1

            state = next_state

        violation_rate = 0 if total_edges == 0 else violations / total_edges
        budget_util = compute_budget_utilization(env)

        episode_metrics = {
            "episode_reward": episode_reward,
            "placements": placements,
            "violation_rate": violation_rate,
            "budget_utilization": budget_util,
            "success": 1 if placements >= env.max_items else 0
        }

        all_episode_metrics.append(episode_metrics)

        print(f"Episode {ep+1}: {episode_metrics}")

    aggregated = aggregate_episode_metrics(all_episode_metrics)

    final_results = {
    "Average Return": aggregated["episode_reward_mean"],
    "Return Std": aggregated["episode_reward_std"],
    "Success Rate (%)": aggregated["success_mean"] * 100,
    "Average Placements": aggregated["placements_mean"],
    "Constraint Violation Rate (%)": aggregated["violation_rate_mean"] * 100,
    "Collision-Free Rate (%)": (1 - aggregated["violation_rate_mean"]) * 100,
    "Budget Utilization (%)": aggregated["budget_utilization_mean"] * 100
}


    os.makedirs("outputs", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\nFinal Aggregated Metrics:")
    print(json.dumps(aggregated, indent=2))

    return final_results


if __name__ == "__main__":
    evaluate_model("models_gnn/best_model.pt")
