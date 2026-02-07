"""
Simple Test Script for Improved Semantic Model
Run this after training completes
"""

import numpy as np
import matplotlib.pyplot as plt
from furniture_env_semantic_improved_v1_backup import FurnitureRecommendationEnvSemantic
from sac_agent import RuleGuidedSAC
import os
import glob

def find_best_model():
    """Find the best trained model"""
    # Look for final model first
    if os.path.exists('models_semantic_improved/sac_semantic_improved_epfinal.pt'):
        return 'models_semantic_improved/sac_semantic_improved_epfinal.pt'
    
    # Otherwise get latest checkpoint
    models = glob.glob('models_semantic_improved/sac_semantic_improved_ep*.pt')
    if models:
        # Remove 'final' from list
        models = [m for m in models if 'final' not in m]
        if models:
            # Get episode numbers and find max
            episodes = []
            for m in models:
                try:
                    ep = int(m.split('ep')[1].split('.')[0])
                    episodes.append((ep, m))
                except:
                    continue
            if episodes:
                episodes.sort(reverse=True)
                return episodes[0][1]
    
    return None

def test_model():
    """Test the trained model"""
    
    # Find model
    model_path = find_best_model()
    
    if not model_path:
        print("❌ No model found!")
        print("Please train first: python train_semantic_improved.py")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Create environment
    env = FurnitureRecommendationEnvSemantic(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog.json',
        max_items=4,
        grid_size=0.3,
        collision_buffer=0.15,
        wall_proximity=0.35,
        enforce_semantic=True
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = RuleGuidedSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high
    )
    
    # Load model
    agent.load(model_path)
    print("✅ Model loaded successfully")
    
    # Run test episode
    print("\n" + "=" * 70)
    print("RUNNING TEST EPISODE")
    print("=" * 70)
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    semantic_scores = []
    
    while not done:
        step += 1
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if info.get('valid_placement'):
            sem_score = info['reward_components'].get('semantic_correctness', 0)
            semantic_scores.append(sem_score)
            
            print(f"\nStep {step}:")
            print(f"  Furniture: {info['selected_furniture']}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Semantic Score: {sem_score:.3f}/3.0")
            print(f"  Budget Used: ${info['budget_used']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"Total Items Placed: {info['placed_items']}/4")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Average Semantic Score: {np.mean(semantic_scores):.3f}/3.0" if semantic_scores else "N/A")
    print(f"Budget Used: ${info['budget_used']}")
    print(f"Free Space: {info['free_space']*100:.1f}%")
    
    # Visualization
    visualize_layout(env)
    
    print("\n✅ Test complete! Check 'test_layout.png' for visualization")

def visualize_layout(env):
    """Create a simple visualization of the layout"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw room
    ax.add_patch(plt.Rectangle((0, 0), env.room_length, env.room_width, 
                               fill=False, edgecolor='black', linewidth=2))
    
    # Draw existing furniture (faded)
    for item in env.existing_furniture:
        x, y = item['position']['x'], item['position']['y']
        l, w = item['dimensions']['length'], item['dimensions']['width']
        rect = plt.Rectangle((x - l/2, y - w/2), l, w, 
                            fill=True, facecolor='lightgray', edgecolor='gray',
                            alpha=0.5, linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, item['type'][:4], ha='center', va='center', 
               fontsize=8, color='gray')
    
    # Draw placed furniture (colored)
    colors = {
        'seating': 'salmon',
        'tables': 'lightblue',
        'storage': 'lightgreen',
        'lighting': 'orange',
        'decor': 'plum'
    }
    
    for i, item in enumerate(env.placed_items):
        x, y = item['position']['x'], item['position']['y']
        l, w = item['dimensions']['length'], item['dimensions']['width']
        color = colors.get(item['category'], 'yellow')
        
        rect = plt.Rectangle((x - l/2, y - w/2), l, w,
                            fill=True, facecolor=color, edgecolor='black',
                            linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, f"{i+1}.{item['type'][:4]}", ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # Draw doors
    for door in env.room_layout['doors']:
        dx, dy = door['position']['x'], door['position']['y']
        ax.plot(dx, dy, 'rs', markersize=10, label='Door' if door == env.room_layout['doors'][0] else '')
    
    # Draw windows
    for window in env.room_layout['windows']:
        wx, wy = window['position']['x'], window['position']['y']
        ax.plot(wx, wy, 'b^', markersize=10, label='Window' if window == env.room_layout['windows'][0] else '')
    
    ax.set_xlim(-0.5, env.room_length + 0.5)
    ax.set_ylim(-0.5, env.room_width + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Length (m)', fontsize=12)
    ax.set_ylabel('Width (m)', fontsize=12)
    ax.set_title('Furniture Layout (Improved Semantic Model)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('test_layout.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("\n" + "🏠" * 35)
    print("TESTING IMPROVED SEMANTIC MODEL")
    print("🏠" * 35 + "\n")
    
    test_model()
