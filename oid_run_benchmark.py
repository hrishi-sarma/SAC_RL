"""
OID-PPO Benchmark Runner
Comparing GNN-SAC performance against OID-PPO Paper Baselines
"""
import torch
import numpy as np
import json
import pandas as pd
from tabulate import tabulate

from furniture_env_gnn import FurnitureRecommendationEnvGNN
from sac_agent_gnn import SACAgentWithGNN
from graph_encoder import create_graph_encoder
from oid_metrics import OIDMetrics

# --- CONFIGURATION ---
MODEL_PATH = 'models_gnn/best_model.pt' # Ensure this points to your trained model
ROOM_PATH = 'C:\\Users\\shobh\\Shobhit\\VS\\Proj\\SAC_RL_GNN\\SAC_RL\\uploads\\room_layout.json' #C:\Users\shobh\Shobhit\VS\Proj\SAC_RL_GNN\SAC_RL\uploads\room_layout.json
CATALOG_PATH = 'C:\\Users\\shobh\\Shobhit\\VS\\Proj\\SAC_RL_GNN\\SAC_RL\\uploads\\furniture_catalog_enhanced.json'
NUM_TEST_EPISODES = 5

# --- PAPER BASELINES (from Table 1 in PDF) ---
# We use the "Rectangle" row since your room is 6.0 x 5.0 (Rectangle)
BASELINES = {
    'OID-PPO (Paper)': 0.962,  # Best in paper for Fn=4
    'SAC (Paper Baseline)': 0.891, # Their implementation of SAC
    'PPO (Paper Baseline)': 0.941,
    'TD3 (Paper Baseline)': 0.792
}

def run_benchmark():
    # 1. Setup Environment
    env = FurnitureRecommendationEnvGNN(ROOM_PATH, CATALOG_PATH, max_items=4)
    oid_scorer = OIDMetrics({'length': env.room_length, 'width': env.room_width})
    
    # 2. Load Agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph_encoder = create_graph_encoder(node_feature_dim=22, edge_feature_dim=10, output_dim=32, device=device)
    agent = SACAgentWithGNN(env.observation_space.shape[0], env.action_space.shape[0], graph_encoder, device=device)
    
    try:
        agent.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load model ({e}). Running with random weights for demo.")

    # 3. Run Testing
    print(f"\nRunning {NUM_TEST_EPISODES} Episodes...")
    results = []
    
    for ep in range(NUM_TEST_EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            graph_data = env.get_graph_data()
            action = agent.select_action(state, graph_data, deterministic=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        # 4. Calculate OID-Specific Metrics on the FINAL Layout
        # Combine existing + placed items for evaluation
        all_items = env.existing_furniture + env.placed_items
        
        # Calculate scores using the OID Engine
        r_idg, components = oid_scorer.calculate_all(
            all_items, 
            env.room_layout.get('doors', []),
            env.room_layout.get('walls', [])
        )
        
        results.append({
            'Episode': ep + 1,
            'OID_Score': r_idg,
            **components
        })

    # 5. Aggregate Results
    df = pd.DataFrame(results)
    avg_score = df['OID_Score'].mean()
    
    # 6. Generate Comparison Table
    print("\n" + "="*50)
    print("BENCHMARK RESULTS: YOUR GNN-SAC vs OID-PPO PAPER")
    print("="*50)
    
    comparison_data = []
    # Add Paper Baselines
    for model, score in BASELINES.items():
        comparison_data.append([model, score, "Paper (Table 1)"])
        
    # Add Your Model
    comparison_data.append(["**Your GNN-SAC**", f"**{avg_score:.3f}**", "Evaluated via OID Metrics"])
    
    print(tabulate(comparison_data, headers=["Model", "Reward (R_idg)", "Source"], tablefmt="grid"))
    
    print("\nDetailed Component Breakdown (Your Model):")
    print(df.mean(numeric_only=True).to_frame(name="Average Score").to_markdown())

    # 7. Interpretation
    print("\n--- INTERPRETATION ---")
    if avg_score > 0.962:
        print("RESULT: SUPERIOR. Your model outperforms the OID-PPO state-of-the-art.")
        print("Why? GNNs likely capture spatial relationships (Pairwise/Alignment) better than CNNs.")
    elif avg_score > 0.891:
        print("RESULT: COMPETITIVE. Your model beats the SAC baseline from the paper.")
        print("This validates that your Graph encoding is working better than standard encodings.")
    else:
        print("RESULT: BELOW BASELINE. Check 'Detailed Component Breakdown'.")
        print("Low 'R_path' often means furniture is blocking doors.")
        print("Low 'R_vis' means furniture is facing walls.")

if __name__ == "__main__":
    run_benchmark()