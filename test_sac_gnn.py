"""
Test GNN-Enhanced SAC Agent

Load trained model and visualize furniture placement recommendations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import json
import os
from typing import Dict, List

from furniture_env_gnn import FurnitureRecommendationEnvGNN
from sac_agent_gnn import SACAgentWithGNN
from graph_encoder import create_graph_encoder


def visualize_placement(
    env: FurnitureRecommendationEnvGNN,
    save_path: str = 'placement_visualization_gnn.png'
):
    """
    Visualize the furniture placement
    
    Args:
        env: Environment with placed furniture
        save_path: Path to save visualization
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Room boundaries
    room_rect = Rectangle(
        (0, 0), 
        env.room_length, 
        env.room_width,
        fill=False, 
        edgecolor='black', 
        linewidth=3
    )
    ax.add_patch(room_rect)
    
    # Draw doors
    for door in env.room_layout.get('doors', []):
        door_x = door['position']['x']
        door_y = door['position']['y']
        clearance = door.get('clearance_radius', 1.2)
        
        # Door position
        door_marker = plt.Circle(
            (door_x, door_y), 
            0.1, 
            color='brown', 
            zorder=10
        )
        ax.add_patch(door_marker)
        
        # Clearance zone (dashed circle)
        clearance_circle = plt.Circle(
            (door_x, door_y),
            clearance,
            fill=False,
            edgecolor='brown',
            linestyle='--',
            linewidth=1.5,
            alpha=0.5,
            zorder=1
        )
        ax.add_patch(clearance_circle)
        
        ax.text(door_x, door_y - 0.3, 'Door', 
                ha='center', fontsize=8, color='brown')
    
    # Draw windows
    for window in env.room_layout.get('windows', []):
        window_x = window['position']['x']
        window_y = window['position']['y']
        window_width = window['dimensions']['width']
        
        # Window marker
        window_rect = Rectangle(
            (window_x - window_width/2, window_y - 0.1),
            window_width,
            0.2,
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=2,
            alpha=0.7,
            zorder=2
        )
        ax.add_patch(window_rect)
        ax.text(window_x, window_y + 0.4, 'Window',
                ha='center', fontsize=8, color='blue')
    
    # Draw priority zones
    zone_colors = {'high': 'orange', 'medium': 'yellow', 'low': 'lightgreen'}
    for zone in env.room_layout['available_space'].get('priority_zones', []):
        zone_circle = plt.Circle(
            (zone['center']['x'], zone['center']['y']),
            zone['radius'],
            fill=True,
            facecolor=zone_colors.get(zone.get('priority', 'medium'), 'gray'),
            alpha=0.15,
            edgecolor='none',
            zorder=0
        )
        ax.add_patch(zone_circle)
    
    # Draw existing furniture
    for furniture in env.existing_furniture:
        x = furniture['position']['x']
        y = furniture['position']['y']
        rot = furniture.get('rotation', 0)
        if rot in (90, 270):
            length, width = furniture['dimensions']['width'], furniture['dimensions']['length']
        else:
            length, width = furniture['dimensions']['length'], furniture['dimensions']['width']
        
        furn_rect = FancyBboxPatch(
            (x - length/2, y - width/2),
            length,
            width,
            boxstyle="round,pad=0.05",
            facecolor='lightgray',
            edgecolor='gray',
            linewidth=2,
            alpha=0.7,
            zorder=5
        )
        ax.add_patch(furn_rect)
        
        # Label
        ax.text(x, y, f"Existing:\n{furniture['type']}",
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw placed furniture
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(env.placed_items), 1)))
    
    for i, furniture in enumerate(env.placed_items):
        x = furniture['position']['x']
        y = furniture['position']['y']
        rot = furniture.get('rotation', 0)
        if rot in (90, 270):
            length, width = furniture['dimensions']['width'], furniture['dimensions']['length']
        else:
            length, width = furniture['dimensions']['length'], furniture['dimensions']['width']
        
        furn_rect = FancyBboxPatch(
            (x - length/2, y - width/2),
            length,
            width,
            boxstyle="round,pad=0.05",
            facecolor=colors[i],
            edgecolor='darkblue',
            linewidth=2.5,
            alpha=0.8,
            zorder=6
        )
        ax.add_patch(furn_rect)
        
        # Label
        label_text = f"NEW ({i+1}):\n{furniture['type']}"
        ax.text(x, y, label_text,
                ha='center', va='center', fontsize=9,
                weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', 
                         edgecolor='darkblue', linewidth=1.5, alpha=0.9))
    
    # Set axis properties
    ax.set_xlim(-0.5, env.room_length + 0.5)
    ax.set_ylim(-0.5, env.room_width + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (m)', fontsize=12)
    ax.set_ylabel('Width (m)', fontsize=12)
    ax.set_title(
        f'GNN-Enhanced Furniture Placement\n'
        f'Total Items: {len(env.placed_items)} | Budget Used: ${env.budget_used:.0f}',
        fontsize=14,
        weight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='lightgray', edgecolor='gray', label='Existing Furniture'),
        patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='New Placement'),
        patches.Circle((0, 0), 0.1, facecolor='brown', label='Door'),
        patches.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Window'),
        patches.Circle((0, 0), 0.1, fill=False, edgecolor='brown', 
                      linestyle='--', label='Door Clearance')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def test_model(
    model_path: str = 'models_gnn/best_model.pt',
    num_episodes: int = 5,
    visualize: bool = True,
    deterministic: bool = True
):
    """
    Test trained model
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of test episodes
        visualize: Whether to create visualizations
        deterministic: If True, use mean action; if False, sample from policy
    """
    # Paths
    room_layout_path = 'uploads/room_layout.json'
    catalog_path = 'uploads/furniture_catalog_enhanced.json'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = FurnitureRecommendationEnvGNN(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        max_items=4
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create graph encoder
    graph_encoder = create_graph_encoder(
        node_feature_dim=22,
        edge_feature_dim=10,
        output_dim=32,
        device=device
    )
    
    # Create agent
    agent = SACAgentWithGNN(
        state_dim=state_dim,
        action_dim=action_dim,
        graph_encoder=graph_encoder,
        device=device
    )
    
    # Load model
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Model not found at {model_path}")
        return
    
    agent.eval()
    
    # Run test episodes
    print(f"\nRunning {num_episodes} test episodes...")
    
    all_rewards = []
    all_placements = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        episode_info = {
            'placements': [],
            'rewards': [],
            'total_reward': 0
        }
        
        while not done:
            # Get graph data
            graph_data = env.get_graph_data()
            
            # Select action (use flag to control deterministic vs stochastic)
            action = agent.select_action(state, graph_data, deterministic=deterministic)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            # Log placement
            print(f"\nStep {step}:")
            print(f"  Selected: {info['selected_furniture']}")
            print(f"  Valid: {info['valid_placement']}")
            print(f"  Reward: {reward:.3f}")
            
            if info['valid_placement'] and not info['already_placed']:
                print(f"  ✓ Placed successfully")
                episode_info['placements'].append({
                    'step': step,
                    'furniture': info['selected_furniture'],
                    'reward': reward
                })
            
            episode_info['rewards'].append(reward)
            
            state = next_state
        
        episode_info['total_reward'] = episode_reward
        all_rewards.append(episode_reward)
        all_placements.append(len(env.placed_items))
        
        print(f"\nEpisode Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Items Placed: {len(env.placed_items)}/{env.max_items}")
        print(f"  Budget Used: ${env.budget_used:.0f}/${env.budget_max:.0f}")
        
        # Visualize last episode
        if visualize and episode == num_episodes - 1:
            print("\nCreating visualization...")
            visualize_placement(
                env, 
                save_path='outputs/placement_gnn_final.png'
            )
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("Test Results Summary")
    print(f"{'='*60}")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Placements: {np.mean(all_placements):.1f} ± {np.std(all_placements):.1f}")
    print(f"Success Rate: {(np.array(all_placements) >= env.max_items).mean()*100:.1f}%")
    
    # Save results
    results = {
        'num_episodes': num_episodes,
        'rewards': [float(r) for r in all_rewards],
        'placements': [int(p) for p in all_placements],
        'avg_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'avg_placements': float(np.mean(all_placements)),
        'success_rate': float((np.array(all_placements) >= env.max_items).mean())
    }
    
    results_path = 'outputs/test_results_gnn.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GNN-Enhanced SAC Agent')
    parser.add_argument('--model', type=str, default='models_gnn/best_model.pt',
                       help='Path to model file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (sample) instead of deterministic (mean)')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        num_episodes=args.episodes,
        visualize=not args.no_viz,
        deterministic=not args.stochastic
    )