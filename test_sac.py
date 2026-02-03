import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import os
import sys

from furniture_env import FurnitureRecommendationEnvEnhanced
from sac_agent import RuleGuidedSAC


class FurnitureRecommendationTester:
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        model_path: str,
        max_items: int = 4,
        grid_size: float = 0.3,
        collision_buffer: float = 0.15
    ):
        
        print("Using ENHANCED environment (grid + collision buffer)")
        self.env = FurnitureRecommendationEnvEnhanced(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=grid_size,
            collision_buffer=collision_buffer
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
            print("Starting Furniture Recommendation Episode")
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
                print(f"Step {step + 1}:")
                print(f"Selected: {info['selected_furniture']}")
                print(f"Reward: {reward:.3f}")
                print(f"Valid Placement: {info['reward_components']['valid_placement']}")
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
            print("=" * 60)
        
        if visualize:
            self.visualize_layout(summary, recommendations)
            self.visualize_reward_breakdown(reward_details)
        
        return episode_reward, summary, recommendations
    
    def visualize_layout(self, summary: dict, recommendations: list):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        room_length = self.env.room_length
        room_width = self.env.room_width
        
        ax.add_patch(Rectangle((0, 0), room_length, room_width, fill=False, edgecolor='black', linewidth=2))
        
        if hasattr(self.env, 'grid_size'):
            grid_size = self.env.grid_size
            for x in np.arange(0, room_length + grid_size, grid_size):
                ax.axvline(x, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)
            for y in np.arange(0, room_width + grid_size, grid_size):
                ax.axhline(y, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)
        
        for door in self.env.room_layout['doors']:
            door_x = door['position']['x']
            door_y = door['position']['y']
            door_width = door['dimensions']['width']
            ax.add_patch(Rectangle((door_x, door_y), door_width, 0.1, fill=True, facecolor='brown', edgecolor='black'))
            circle = plt.Circle((door_x + door_width / 2, door_y), door['clearance_radius'], fill=False, edgecolor='brown', linestyle='--', alpha=0.5)
            ax.add_artist(circle)
        
        for window in self.env.room_layout['windows']:
            win_x = window['position']['x']
            win_y = window['position']['y']
            win_width = window['dimensions']['width']
            ax.add_patch(Rectangle((win_x, win_y), win_width, 0.05, fill=True, facecolor='lightblue', edgecolor='blue'))
        
        color_map = {
            'seating': '#FF6B6B',
            'tables': '#4ECDC4',
            'storage': '#45B7D1',
            'lighting': '#FFA07A',
            'decor': '#98D8C8'
        }
        
        for item in self.env.existing_furniture:
            self._draw_furniture(ax, item, color_map, alpha=0.5)
        
        for item in summary['placed_items']:
            self._draw_furniture(ax, item, color_map, alpha=0.8)
        
        ax.set_xlim(-0.5, room_length + 0.5)
        ax.set_ylim(-0.5, room_width + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.set_title('Furniture Recommendation Layout (Enhanced)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('recommendation_layout_enhanced2.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _draw_furniture(self, ax, item, color_map, alpha=0.8):
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
        
        ax.text(x, y, item['type'][:4], ha='center', va='center', fontsize=8, fontweight='bold')
    
    def visualize_reward_breakdown(self, reward_details: list):
        if not reward_details:
            return
        
        valid_rewards = [r for r in reward_details if r['valid_placement']]
        if not valid_rewards:
            return
        
        component_names = [k for k in valid_rewards[0].keys() if k not in ['valid_placement', 'collision', 'out_of_bounds', 'blocks_door', 'total', 'already_placed']]
        avg_rewards = {name: np.mean([r.get(name, 0) for r in valid_rewards]) for name in component_names}
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        names = list(avg_rewards.keys())
        values = list(avg_rewards.values())
        
        bars = ax.barh(names, values)
        ax.set_xlabel('Average Reward Value')
        ax.set_title('Reward Component Breakdown')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reward_breakdown_enhanced2.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_test(self, num_episodes: int = 20):
        total_rewards = []
        items_placed = []
        budget_used = []
        collision_counts = []
        
        for _ in range(num_episodes):
            reward, summary, recommendations = self.test_episode(visualize=False, verbose=False)
            total_rewards.append(reward)
            items_placed.append(summary['total_items'])
            budget_used.append(summary['budget_used'])
            collisions = sum(1 for r in recommendations if not r['reward_components']['valid_placement'] and r['reward_components'].get('collision', False))
            collision_counts.append(collisions)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].hist(total_rewards)
        axes[0, 1].hist(items_placed)
        axes[1, 0].hist(budget_used)
        axes[1, 1].hist(collision_counts)
        
        plt.tight_layout()
        plt.savefig('batch_test_results_enhanced2.png', dpi=150, bbox_inches='tight')
        plt.close()


def find_model_path() -> str:
    import glob
    
    candidate = os.path.join('models_enhanced', 'sac_enhanced_epfinal.pt')
    if os.path.exists(candidate):
        return candidate
    
    pattern = os.path.join('models_enhanced', 'sac_enhanced_ep*.pt')
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
    
    candidate = os.path.join('models', 'sac_furniture_epfinal.pt')
    if os.path.exists(candidate):
        return candidate
    
    return ''


def main():
    room_layout_path = 'room_layout.json'
    catalog_path = 'furniture_catalog.json'
    max_items = 4
    
    model_path = find_model_path()
    if not model_path:
        sys.exit(1)
    
    tester = FurnitureRecommendationTester(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        model_path=model_path,
        max_items=max_items,
        grid_size=0.3,
        collision_buffer=0.15
    )
    
    tester.test_episode(visualize=True, verbose=True)
    tester.batch_test(num_episodes=20)


if __name__ == '__main__':
    main()