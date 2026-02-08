import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from furniture_env_logical import FurnitureRecommendationEnvLogical
from sac_agent import RuleGuidedSAC


class FurnitureRecommendationTesterLogical:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        model_path: str,
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15
    ):
        
        print("\n" + "=" * 70)
        print("LOGICAL PLACEMENT TESTING")
        print("=" * 70)
        
        self.env = FurnitureRecommendationEnvLogical(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=grid_size,
            collision_buffer=collision_buffer
        )
        
        print(f"Environment loaded with logical placement rules")
        print(f"  Grid Size: {grid_size}m")
        print(f"  Collision Buffer: {collision_buffer}m")
        print(f"  State Dimension: {self.env.observation_space.shape[0]}")
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        self.agent = RuleGuidedSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high
        )
        
        if os.path.exists(model_path):
            try:
                self.agent.load(model_path)
                print(f"✅ Model loaded: {model_path}")
            except RuntimeError as e:
                print(f"❌ Error loading model: {e}")
                sys.exit(1)
        else:
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        print("=" * 70 + "\n")
    
    def test_episode(self, visualize: bool = True, verbose: bool = True):
        state, _ = self.env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        recommendations = []
        reward_details = []
        
        if verbose:
            print("=" * 70)
            print("Starting Furniture Recommendation Episode")
            print("=" * 70 + "\n")
        
        while not done:
            action = self.agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            recommendations.append({
                'step': step + 1,
                'furniture': info['selected_furniture'],
                'reward': reward,
                'reward_components': info['reward_components'],
                'budget_used': info['budget_used']
            })
            reward_details.append(info['reward_components'])
            
            if verbose:
                print(f"Step {step + 1}:")
                print(f"  Selected: {info['selected_furniture']}")
                print(f"  Reward: {reward:.3f}")
                print(f"  Valid Placement: {info['reward_components']['valid_placement']}")
                
                if info['reward_components']['valid_placement']:
                    logical_score = info['reward_components'].get('logical_placement', 0)
                    print(f"  Logical Placement Score: {logical_score:.3f}/1.0")
                
                print(f"  Budget Used: ${info['budget_used']:.0f}")
                print()
            
            state = next_state
            episode_reward += reward
            step += 1
        
        summary = self.env.get_recommendation_summary()
        
        if verbose:
            print("=" * 70)
            print("Episode Summary")
            print("=" * 70)
            print(f"Total Items Placed: {summary['total_items']}/{self.env.max_items}")
            print(f"Total Budget Used: ${summary['budget_used']:.0f}")
            print(f"Budget Remaining: ${summary['budget_remaining']:.0f}")
            print(f"Free Space: {summary['free_space_percentage']:.1f}%")
            print(f"Total Episode Reward: {episode_reward:.3f}")
            
            # Calculate average logical placement score
            valid_rewards = [r for r in reward_details if r['valid_placement']]
            if valid_rewards:
                avg_logical = np.mean([r.get('logical_placement', 0) for r in valid_rewards])
                print(f"Average Logical Placement Score: {avg_logical:.3f}/1.0")
            
            print("=" * 70 + "\n")
        
        if visualize:
            self.visualize_layout(summary, recommendations)
            self.visualize_reward_breakdown(reward_details)
        
        return episode_reward, summary, recommendations
    
    def visualize_layout(self, summary: dict, recommendations: list):
        fig, ax = plt.subplots(1, 1, figsize=(14, 11))
        
        room_length = self.env.room_length
        room_width = self.env.room_width
        
        # Draw room boundary
        ax.add_patch(Rectangle((0, 0), room_length, room_width, 
                               fill=False, edgecolor='black', linewidth=2.5))
        
        # Draw grid
        grid_size = self.env.grid_size
        for x in np.arange(0, room_length + grid_size, grid_size):
            ax.axvline(x, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.4)
        for y in np.arange(0, room_width + grid_size, grid_size):
            ax.axhline(y, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.4)
        
        # Draw corner zones
        for corner in self.env.corner_zones:
            cx, cy = corner['center']
            radius = corner['radius']
            circle = plt.Circle((cx, cy), radius, fill=False, 
                              edgecolor='purple', linestyle='--', 
                              linewidth=1.5, alpha=0.3, label='Corner Zone')
            ax.add_artist(circle)
        
        # Draw doors
        for door in self.env.room_layout['doors']:
            door_x = door['position']['x']
            door_y = door['position']['y']
            door_width = door['dimensions']['width']
            ax.add_patch(Rectangle((door_x, door_y), door_width, 0.15, 
                                  fill=True, facecolor='brown', 
                                  edgecolor='black', linewidth=1.5))
            
            # Door clearance
            circle = plt.Circle((door_x + door_width / 2, door_y), 
                              door['clearance_radius'], 
                              fill=False, edgecolor='brown', 
                              linestyle='--', alpha=0.4)
            ax.add_artist(circle)
        
        # Draw windows
        for window in self.env.room_layout['windows']:
            win_x = window['position']['x']
            win_y = window['position']['y']
            win_width = window['dimensions']['width']
            ax.add_patch(Rectangle((win_x, win_y), win_width, 0.08, 
                                  fill=True, facecolor='lightblue', 
                                  edgecolor='blue', linewidth=1.5))
        
        # Color map
        color_map = {
            'seating': '#FF6B6B',
            'tables': '#4ECDC4',
            'storage': '#45B7D1',
            'lighting': '#FFA07A',
            'decor': '#98D8C8'
        }
        
        # Draw existing furniture (semi-transparent)
        for item in self.env.existing_furniture:
            self._draw_furniture(ax, item, color_map, alpha=0.4, label='Existing')
        
        # Draw placed items (full opacity)
        for item in summary['placed_items']:
            self._draw_furniture(ax, item, color_map, alpha=0.85, label='Recommended')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['seating'], alpha=0.85, label='Seating'),
            Patch(facecolor=color_map['tables'], alpha=0.85, label='Tables'),
            Patch(facecolor=color_map['storage'], alpha=0.85, label='Storage'),
            Patch(facecolor=color_map['lighting'], alpha=0.85, label='Lighting'),
            Patch(facecolor=color_map['decor'], alpha=0.85, label='Decor'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        ax.set_xlim(-0.5, room_length + 0.5)
        ax.set_ylim(-0.5, room_width + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Length (m)', fontsize=11)
        ax.set_ylabel('Width (m)', fontsize=11)
        ax.set_title('Furniture Recommendation Layout (Logical Placement)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig('recommendation_layout_logical.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Layout visualization saved: recommendation_layout_logical.png")
    
    def _draw_furniture(self, ax, item, color_map, alpha=0.8, label=''):
        x = item['position']['x']
        y = item['position']['y']
        length = item['dimensions']['length']
        width = item['dimensions']['width']
        rotation = item['rotation']
        category = item['category']
        
        color = color_map.get(category, 'gray')
        
        rect = FancyBboxPatch(
            (x - length / 2, y - width / 2),
            length,
            width,
            boxstyle="round,pad=0.05",
            linewidth=1.8,
            edgecolor='black',
            facecolor=color,
            alpha=alpha,
            transform=ax.transData
        )
        
        import matplotlib.transforms as transforms
        t = transforms.Affine2D().rotate_around(x, y, rotation) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y, item['type'][:5], ha='center', va='center', 
               fontsize=7, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    
    def visualize_reward_breakdown(self, reward_details: list):
        if not reward_details:
            return
        
        valid_rewards = [r for r in reward_details if r['valid_placement']]
        if not valid_rewards:
            print("⚠️  No valid placements to analyze")
            return
        
        # Extract component names (excluding flags)
        exclude_keys = ['valid_placement', 'collision', 'out_of_bounds', 'blocks_door', 'total', 'already_placed']
        component_names = [k for k in valid_rewards[0].keys() if k not in exclude_keys]
        
        # Calculate averages
        avg_rewards = {name: np.mean([r.get(name, 0) for r in valid_rewards]) 
                      for name in component_names}
        
        # Sort by value
        sorted_items = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        # Color bars based on value
        colors = ['#2ecc71' if v > 0.7 else '#f39c12' if v > 0.4 else '#e74c3c' 
                 for v in values]
        
        bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
        
        # Highlight logical_placement
        if 'logical_placement' in names:
            idx = names.index('logical_placement')
            bars[idx].set_edgecolor('purple')
            bars[idx].set_linewidth(3)
        
        ax.set_xlabel('Average Reward Value', fontsize=11, fontweight='bold')
        ax.set_title('Reward Component Breakdown (Valid Placements Only)', 
                    fontsize=13, fontweight='bold')
        ax.set_xlim([0, max(values) * 1.15])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reward_breakdown_logical.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Reward breakdown saved: reward_breakdown_logical.png")
    
    def batch_test(self, num_episodes: int = 20):
        print(f"\nRunning batch test with {num_episodes} episodes...")
        
        total_rewards = []
        items_placed = []
        budget_used = []
        collision_counts = []
        logical_scores = []
        
        for i in range(num_episodes):
            reward, summary, recommendations = self.test_episode(visualize=False, verbose=False)
            total_rewards.append(reward)
            items_placed.append(summary['total_items'])
            budget_used.append(summary['budget_used'])
            
            # Count collisions
            collisions = sum(1 for r in recommendations 
                           if not r['reward_components']['valid_placement'] 
                           and r['reward_components'].get('collision', False))
            collision_counts.append(collisions)
            
            # Average logical placement score
            valid_rewards = [r['reward_components'] for r in recommendations 
                           if r['reward_components']['valid_placement']]
            if valid_rewards:
                avg_logical = np.mean([r.get('logical_placement', 0) for r in valid_rewards])
                logical_scores.append(avg_logical)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Reward distribution
        axes[0, 0].hist(total_rewards, bins=15, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(total_rewards), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(total_rewards):.2f}')
        axes[0, 0].set_title('Reward Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Episode Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Items placed
        axes[0, 1].hist(items_placed, bins=range(0, self.env.max_items + 2), 
                       color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(items_placed), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(items_placed):.2f}')
        axes[0, 1].set_title('Items Placed Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Items Placed')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Budget usage
        axes[0, 2].hist(budget_used, bins=15, color='#f39c12', edgecolor='black', alpha=0.7)
        axes[0, 2].axvline(np.mean(budget_used), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: ${np.mean(budget_used):.0f}')
        axes[0, 2].set_title('Budget Usage Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Budget Used ($)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # Collisions
        axes[1, 0].hist(collision_counts, bins=range(0, max(collision_counts) + 2), 
                       color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Collisions per Episode (LOWER IS BETTER)', fontweight='bold')
        axes[1, 0].set_xlabel('Collision Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(alpha=0.3)
        
        # Logical placement scores
        if logical_scores:
            axes[1, 1].hist(logical_scores, bins=15, color='#9b59b6', edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(np.mean(logical_scores), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean: {np.mean(logical_scores):.3f}')
            axes[1, 1].set_title('Logical Placement Score Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Logical Placement Score (0-1)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
BATCH TEST RESULTS
{'='*30}
Episodes: {num_episodes}

Rewards:
  Mean: {np.mean(total_rewards):.2f}
  Std: {np.std(total_rewards):.2f}

Items Placed:
  Mean: {np.mean(items_placed):.2f}
  Success Rate: {np.mean(items_placed)/self.env.max_items:.1%}

Logical Placement:
  Mean Score: {np.mean(logical_scores):.3f}/1.0
  Std: {np.std(logical_scores):.3f}

Collisions:
  Mean: {np.mean(collision_counts):.2f}
  Zero Collisions: {sum(1 for c in collision_counts if c == 0)}/{num_episodes}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, 
                       verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('batch_test_results_logical.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Batch test results saved: batch_test_results_logical.png\n")
        
        # Print summary
        print("=" * 70)
        print("BATCH TEST SUMMARY")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average Items Placed: {np.mean(items_placed):.2f}/{self.env.max_items}")
        print(f"Success Rate: {np.mean(items_placed)/self.env.max_items:.1%}")
        print(f"Average Logical Score: {np.mean(logical_scores):.3f}/1.0")
        print(f"Average Collisions: {np.mean(collision_counts):.2f}")
        print(f"Episodes with Zero Collisions: {sum(1 for c in collision_counts if c == 0)}/{num_episodes}")
        print("=" * 70)


def find_model_path() -> str:
    """Find the most recent trained model"""
    
    # Check for final model
    candidate = os.path.join('models_logical', 'sac_logical_epfinal.pt')
    if os.path.exists(candidate):
        return candidate
    
    # Find latest numbered checkpoint
    pattern = os.path.join('models_logical', 'sac_logical_ep*.pt')
    matches = glob.glob(pattern)
    if matches:
        def _ep_num(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                return int(stem.split('ep')[-1])
            except ValueError:
                return -1
        matches.sort(key=_ep_num)
        return matches[-1]
    
    print("❌ No trained model found in models_logical/")
    print("Please train the model first using train_sac_logical.py")
    return ''


def main():
    room_layout_path = 'room_layout.json'
    catalog_path = 'furniture_catalog_enhanced.json'
    max_items = 4
    
    # Check if files exist
    if not os.path.exists(room_layout_path):
        print(f"❌ {room_layout_path} not found!")
        return
    
    if not os.path.exists(catalog_path):
        print(f"❌ {catalog_path} not found!")
        return
    
    # Find model
    model_path = find_model_path()
    if not model_path:
        return
    
    print("\n" + "🪑 " * 30)
    print("FURNITURE RECOMMENDATION SYSTEM - LOGICAL PLACEMENT TEST")
    print("🪑 " * 30)
    
    tester = FurnitureRecommendationTesterLogical(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        model_path=model_path,
        max_items=max_items,
        grid_size=0.3,
        collision_buffer=0.20
    )
    
    # Single episode test with visualization
    print("Running single episode test with visualization...")
    tester.test_episode(visualize=True, verbose=True)
    
    # Batch test
    tester.batch_test(num_episodes=20)
    
    print("\n✅ Testing complete!")
    print("Check the generated PNG files for visualizations.\n")


if __name__ == '__main__':
    main()
