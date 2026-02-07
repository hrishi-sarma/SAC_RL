import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import os
import sys

from furniture_env_semantic_old2 import FurnitureRecommendationEnvSemantic
from sac_agent import RuleGuidedSAC


class FurnitureRecommendationTesterSemantic:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        model_path: str,
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15,
        wall_proximity: float = 0.35
    ):
        
        print("Using SEMANTIC environment (grid + collision + semantic constraints)")
        self.env = FurnitureRecommendationEnvSemantic(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=grid_size,
            collision_buffer=collision_buffer,
            wall_proximity=wall_proximity
        )
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        print(f"Environment state dimension: {state_dim}")
        print(f"Environment action dimension: {action_dim}")
        
        self.agent = RuleGuidedSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high
        )
        
        if os.path.exists(model_path):
            try:
                self.agent.load(model_path)
                print(f"Successfully loaded model from: {model_path}")
            except RuntimeError as e:
                print(f"Error loading model: {e}")
                print("Dimension mismatch detected!")
                print(f"Environment has state_dim = {state_dim}")
                sys.exit(1)
        else:
            print(f"Warning: Model not found at {model_path}")
    
    def test_episode(self, visualize: bool = True, verbose: bool = True):
        state, _ = self.env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        recommendations = []
        reward_details = []
        
        if verbose:
            print("=" * 60)
            print("Starting Furniture Recommendation Episode (SEMANTIC)")
            print("=" * 60)
        
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
                print(f"\nStep {step + 1}:")
                print(f"Selected: {info['selected_furniture']}")
                print(f"Reward: {reward:.3f}")
                print(f"Valid Placement: {info['reward_components']['valid_placement']}")
                
                # Show semantic rewards
                if 'semantic_correctness' in info['reward_components']:
                    print(f"Semantic Score: {info['reward_components']['semantic_correctness']:.2f}")
                if 'wall_proximity' in info['reward_components']:
                    print(f"Wall Proximity: {info['reward_components']['wall_proximity']:.2f}")
                if 'corner_bonus' in info['reward_components']:
                    print(f"Corner Bonus: {info['reward_components']['corner_bonus']:.2f}")
                
                print(f"Budget Used: {info['budget_used']:.0f}")
            
            state = next_state
            episode_reward += reward
            step += 1
        
        summary = self.env.get_recommendation_summary()
        
        if verbose:
            print("=" * 60)
            print("Episode Summary")
            print("=" * 60)
            print(f"Total Items Placed: {summary['total_items']}")
            print(f"Total Budget Used: {summary['budget_used']:.0f}")
            print(f"Budget Remaining: {summary['budget_remaining']:.0f}")
            print(f"Free Space: {summary['free_space_percentage']:.1f}%")
            print(f"Total Episode Reward: {episode_reward:.3f}")
            
            # Calculate average semantic score
            semantic_scores = [r.get('semantic_correctness', 0) for r in reward_details if r.get('valid_placement', False)]
            if semantic_scores:
                print(f"Avg Semantic Score: {np.mean(semantic_scores):.2f}/2.0")
            
            print("=" * 60)
        
        if visualize:
            self.visualize_layout(summary, recommendations)
            self.visualize_reward_breakdown(reward_details)
        
        return episode_reward, summary, recommendations
    
    def visualize_layout(self, summary: dict, recommendations: list):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        room_length = self.env.room_length
        room_width = self.env.room_width
        
        # Room boundary
        ax.add_patch(Rectangle((0, 0), room_length, room_width, fill=False, edgecolor='black', linewidth=2))
        
        # Grid
        if hasattr(self.env, 'grid_size'):
            grid_size = self.env.grid_size
            for x in np.arange(0, room_length + grid_size, grid_size):
                ax.axvline(x, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)
            for y in np.arange(0, room_width + grid_size, grid_size):
                ax.axhline(y, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)
        
        # Doors
        for door in self.env.room_layout['doors']:
            door_x = door['position']['x']
            door_y = door['position']['y']
            door_width = door['dimensions']['width']
            ax.add_patch(Rectangle((door_x, door_y), door_width, 0.1, fill=True, facecolor='brown', edgecolor='black', linewidth=2))
            circle = Circle((door_x + door_width / 2, door_y), door['clearance_radius'], fill=False, edgecolor='brown', linestyle='--', alpha=0.5)
            ax.add_artist(circle)
            ax.text(door_x + door_width / 2, door_y - 0.3, 'DOOR', ha='center', fontsize=8, fontweight='bold')
        
        # Windows
        for window in self.env.room_layout['windows']:
            win_x = window['position']['x']
            win_y = window['position']['y']
            win_width = window['dimensions']['width']
            ax.add_patch(Rectangle((win_x, win_y), win_width, 0.05, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2))
            ax.text(win_x + win_width / 2, win_y + 0.15, 'WINDOW', ha='center', fontsize=7, style='italic')
        
        # Corners (visual markers)
        for corner in self.env.corners:
            ax.plot(corner['x'], corner['y'], 'ko', markersize=4, alpha=0.3)
        
        color_map = {
            'seating': '#FF6B6B',
            'tables': '#4ECDC4',
            'storage': '#45B7D1',
            'lighting': '#FFA07A',
            'decor': '#98D8C8'
        }
        
        # Existing furniture
        for item in self.env.existing_furniture:
            self._draw_furniture(ax, item, color_map, alpha=0.5, label_prefix='')
        
        # New furniture
        for i, item in enumerate(summary['placed_items']):
            self._draw_furniture(ax, item, color_map, alpha=0.8, label_prefix=f'{i+1}.')
        
        ax.set_xlim(-0.5, room_length + 0.5)
        ax.set_ylim(-0.5, room_width + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Length (m)', fontsize=12)
        ax.set_ylabel('Width (m)', fontsize=12)
        ax.set_title('Furniture Recommendation Layout (SEMANTIC CONSTRAINTS)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['seating'], label='Seating'),
            Patch(facecolor=color_map['tables'], label='Tables'),
            Patch(facecolor=color_map['storage'], label='Storage'),
            Patch(facecolor=color_map['lighting'], label='Lighting'),
            Patch(facecolor=color_map['decor'], label='Decor'),
            Patch(facecolor='lightgray', label='Existing', alpha=0.5)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('recommendation_layout_semantic.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Layout visualization saved to recommendation_layout_semantic.png")
    
    def _draw_furniture(self, ax, item, color_map, alpha=0.8, label_prefix=''):
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
            linewidth=1.5,
            edgecolor='black',
            facecolor=color,
            alpha=alpha,
            transform=ax.transData
        )
        
        import matplotlib.transforms as transforms
        t = transforms.Affine2D().rotate_around(x, y, rotation) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        
        label_text = f"{label_prefix}{item['type'][:4]}"
        ax.text(x, y, label_text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    def visualize_reward_breakdown(self, reward_details: list):
        if not reward_details:
            return
        
        valid_rewards = [r for r in reward_details if r.get('valid_placement', False)]
        if not valid_rewards:
            return
        
        component_names = [k for k in valid_rewards[0].keys() if k not in ['valid_placement', 'collision', 'out_of_bounds', 'blocks_door', 'total', 'already_placed']]
        avg_rewards = {name: np.mean([r.get(name, 0) for r in valid_rewards]) for name in component_names}
        
        # Sort by value
        sorted_items = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Color code semantic components
        colors = []
        for name in names:
            if 'semantic' in name or 'wall' in name or 'corner' in name or 'window' in name:
                colors.append('#2ecc71')  # Green for semantic
            else:
                colors.append('#3498db')  # Blue for others
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        bars = ax.barh(names, values, color=colors)
        ax.set_xlabel('Average Reward Value', fontsize=12)
        ax.set_title('Reward Component Breakdown (Semantic Constraints)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Semantic Rewards'),
            Patch(facecolor='#3498db', label='Standard Rewards')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('reward_breakdown_semantic.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Reward breakdown saved to reward_breakdown_semantic.png")
    
    def batch_test(self, num_episodes: int = 20):
        total_rewards = []
        items_placed = []
        budget_used = []
        collision_counts = []
        semantic_scores = []
        
        print(f"\nRunning batch test with {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            reward, summary, recommendations = self.test_episode(visualize=False, verbose=False)
            total_rewards.append(reward)
            items_placed.append(summary['total_items'])
            budget_used.append(summary['budget_used'])
            
            # Count collisions
            collisions = sum(1 for r in recommendations if not r['reward_components'].get('valid_placement', True) and r['reward_components'].get('collision', False))
            collision_counts.append(collisions)
            
            # Calculate semantic scores
            sem_scores = [r['reward_components'].get('semantic_correctness', 0) for r in recommendations if r['reward_components'].get('valid_placement', False)]
            if sem_scores:
                semantic_scores.append(np.mean(sem_scores))
            else:
                semantic_scores.append(0)
        
        # Print statistics
        print("\n" + "=" * 60)
        print("BATCH TEST RESULTS")
        print("=" * 60)
        print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average Items Placed: {np.mean(items_placed):.2f} / {self.env.max_items}")
        print(f"Success Rate: {np.mean(items_placed) / self.env.max_items:.1%}")
        print(f"Average Budget Used: ${np.mean(budget_used):.0f}")
        print(f"Average Collisions: {np.mean(collision_counts):.2f}")
        print(f"Average Semantic Score: {np.mean(semantic_scores):.2f}/2.0")
        print("=" * 60)
        
        # Visualize batch results
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Rewards
        axes[0, 0].hist(total_rewards, bins=15, color='#3498db', edgecolor='black')
        axes[0, 0].axvline(np.mean(total_rewards), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].set_xlabel('Episode Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Items placed
        axes[0, 1].hist(items_placed, bins=range(0, self.env.max_items + 2), color='#2ecc71', edgecolor='black')
        axes[0, 1].axvline(np.mean(items_placed), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 1].set_xlabel('Items Placed')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Items Placed Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Budget
        axes[0, 2].hist(budget_used, bins=15, color='#e74c3c', edgecolor='black')
        axes[0, 2].axvline(np.mean(budget_used), color='darkred', linestyle='--', linewidth=2, label='Mean')
        axes[0, 2].set_xlabel('Budget Used ($)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Budget Usage Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # Collisions
        axes[1, 0].hist(collision_counts, bins=range(0, max(collision_counts) + 2), color='#e67e22', edgecolor='black')
        axes[1, 0].set_xlabel('Collision Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Collisions per Episode (LOWER IS BETTER)')
        axes[1, 0].grid(alpha=0.3)
        
        # Semantic scores
        axes[1, 1].hist(semantic_scores, bins=15, color='#9b59b6', edgecolor='black')
        axes[1, 1].axvline(np.mean(semantic_scores), color='purple', linestyle='--', linewidth=2, label='Mean')
        axes[1, 1].set_xlabel('Semantic Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Semantic Correctness Score')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # Scatter: rewards vs items
        axes[1, 2].scatter(items_placed, total_rewards, alpha=0.6, s=80, color='#16a085')
        axes[1, 2].set_xlabel('Items Placed')
        axes[1, 2].set_ylabel('Episode Reward')
        axes[1, 2].set_title('Reward vs Items Placed')
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('batch_test_results_semantic.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Batch test results saved to batch_test_results_semantic.png")


def find_model_path() -> str:
    import glob
    
    # Try semantic models first
    candidate = os.path.join('models_semantic', 'sac_semantic_epfinal.pt')
    if os.path.exists(candidate):
        return candidate
    
    pattern = os.path.join('models_semantic', 'sac_semantic_ep*.pt')
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
    
    print("No semantic model found!")
    return ''


def main():
    room_layout_path = 'room_layout.json'
    catalog_path = 'furniture_catalog.json'
    max_items = 4
    
    model_path = find_model_path()
    if not model_path:
        print("Error: No trained model found. Please run train_sac_semantic.py first.")
        sys.exit(1)
    
    tester = FurnitureRecommendationTesterSemantic(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        model_path=model_path,
        max_items=max_items,
        grid_size=0.3,
        collision_buffer=0.15,
        wall_proximity=0.35
    )
    
    print("\n" + "🏠" * 30)
    print("SEMANTIC FURNITURE RECOMMENDATION TESTING")
    print("🏠" * 30 + "\n")
    
    # Run single test with visualization
    tester.test_episode(visualize=True, verbose=True)
    
    # Run batch test  C:\Users\shobh\Shobhit\VS\NLP\class\files\test_sac_semantic.py
    tester.batch_test(num_episodes=20)
    
    print("\n" + "=" * 60)
    print("✓ TESTING COMPLETE")
    print("=" * 60)
    print("Check the outputs/ folder for visualization files:")
    print("  - recommendation_layout_semantic.png")
    print("  - reward_breakdown_semantic.png")
    print("  - batch_test_results_semantic.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
