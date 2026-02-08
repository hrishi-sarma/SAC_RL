"""
Test Script - Tag-Based Context-Aware Placement System

This script evaluates the trained tag-based model and generates:
1. Visual layout with tag-based annotations
2. Detailed reward component breakdown
3. Batch testing statistics
4. Tag-based performance analysis
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arc, Wedge
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from furniture_env_tag_based import FurnitureRecommendationEnvTagBased
from sac_agent import RuleGuidedSAC


class TagBasedTester:
    """Test and visualize tag-based placement system"""
    
    def __init__(
        self,
        room_layout_path: str,
        catalog_path: str,
        model_path: str,
        max_items: int = 4
    ):
        self.env = FurnitureRecommendationEnvTagBased(
            room_layout_path=room_layout_path,
            catalog_path=catalog_path,
            max_items=max_items,
            grid_size=0.3,
            collision_buffer=0.15
        )
        
        # Create agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        self.agent = RuleGuidedSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dim=256
        )
        
        # Load trained model
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"✅ Loaded model from: {model_path}")
        else:
            print(f"⚠️  Model not found: {model_path}")
            print("   Running with untrained agent (for testing setup)")
        
        self.room_layout_path = room_layout_path
        self.catalog_path = catalog_path
    
    def run_single_recommendation(self):
        """Run single episode and collect detailed metrics"""
        state, _ = self.env.reset()
        
        episode_data = {
            'furniture_placed': [],
            'positions': [],
            'rewards': [],
            'reward_components': [],
            'tags': [],
            'total_reward': 0
        }
        
        done = False
        step = 0
        
        while not done:
            action = self.agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_data['total_reward'] += reward
            
            # Store placement info
            if info['reward_components']['valid_placement'] and not info['already_placed']:
                furniture_type = info['selected_furniture']
                
                # Find catalog item to get tags
                catalog_item = next((item for item in self.env.furniture_catalog 
                                   if item['type'] == furniture_type), None)
                
                if catalog_item:
                    primary_tag, secondary_tags = self.env._get_furniture_tags(catalog_item)
                    
                    episode_data['furniture_placed'].append(furniture_type)
                    episode_data['rewards'].append(reward)
                    episode_data['reward_components'].append(info['reward_components'])
                    episode_data['tags'].append({
                        'primary': primary_tag,
                        'secondary': secondary_tags
                    })
                    
                    # Get position of last placed item
                    if self.env.placed_items:
                        last_item = self.env.placed_items[-1]
                        episode_data['positions'].append({
                            'x': last_item['position']['x'],
                            'y': last_item['position']['y'],
                            'type': last_item['type']
                        })
            
            state = next_state
            step += 1
        
        return episode_data
    
    def visualize_layout_with_tags(self, episode_data, save_path='recommendation_layout_tag_based.png'):
        """Create detailed visualization with tag annotations"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw room
        room_rect = patches.Rectangle(
            (0, 0), 
            self.env.room_length, 
            self.env.room_width,
            linewidth=3, 
            edgecolor='black', 
            facecolor='#f5f5f0',
            alpha=0.3
        )
        ax.add_patch(room_rect)
        
        # Draw grid
        for x in self.env.grid_points_x:
            ax.axvline(x, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')
        for y in self.env.grid_points_y:
            ax.axhline(y, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')
        
        # Draw spatial zones
        self._draw_spatial_zones(ax)
        
        # Draw doors with clearance
        for door in self.env.room_layout.get('doors', []):
            dx = door['position']['x']
            dy = door['position']['y']
            dw = door['dimensions']['width']
            clearance = door.get('clearance_radius', 1.2)
            
            # Door
            door_rect = patches.Rectangle(
                (dx - dw/2, dy - 0.1),
                dw, 0.2,
                linewidth=2,
                edgecolor='brown',
                facecolor='saddlebrown',
                alpha=0.8
            )
            ax.add_patch(door_rect)
            
            # Clearance arc
            arc = Arc(
                (dx, dy), 
                clearance * 2, 
                clearance * 2,
                angle=0, 
                theta1=0, 
                theta2=180,
                linewidth=2, 
                color='brown', 
                linestyle='--',
                alpha=0.5
            )
            ax.add_patch(arc)
        
        # Draw windows
        for window in self.env.room_layout.get('windows', []):
            wx = window['position']['x']
            wy = window['position']['y']
            ww = window['dimensions']['width']
            
            window_rect = patches.Rectangle(
                (wx - ww/2, wy - 0.1),
                ww, 0.2,
                linewidth=2,
                edgecolor='skyblue',
                facecolor='lightblue',
                alpha=0.7
            )
            ax.add_patch(window_rect)
            ax.text(wx, wy + 0.3, 'Window', ha='center', fontsize=8, color='blue')
        
        # Tag-based color scheme
        tag_colors = {
            'seating_companion': '#FF6B6B',
            'primary_seating': '#4ECDC4',
            'seating_accessory': '#45B7D1',
            'task_lighting': '#FFA07A',
            'wall_unit': '#96CEB4',
            'corner_decor': '#DDA15E',
            'flexible_seating': '#BC6C25',
            'ambient_lighting': '#FFDAB9'
        }
        
        # Draw existing furniture
        for furniture in self.env.existing_furniture:
            self._draw_furniture_piece(
                ax, furniture, 
                color='lightgray', 
                alpha=0.4, 
                label_prefix='Existing:\n',
                show_tag=False
            )
        
        # Draw recommended furniture with tag info
        for i, furniture in enumerate(self.env.placed_items):
            if i < len(episode_data['tags']):
                tag_info = episode_data['tags'][i]
                primary_tag = tag_info['primary']
                color = tag_colors.get(primary_tag, '#95a5a6')
                
                self._draw_furniture_piece(
                    ax, furniture,
                    color=color,
                    alpha=0.85,
                    label_prefix=f'NEW ({i+1}):\n',
                    show_tag=True,
                    tag_info=tag_info
                )
        
        # Add legend for tags
        self._add_tag_legend(ax, tag_colors)
        
        # Add title and labels
        ax.set_xlim(-0.5, self.env.room_length + 0.5)
        ax.set_ylim(-0.5, self.env.room_width + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Length (m)', fontsize=12)
        ax.set_ylabel('Width (m)', fontsize=12)
        
        title = f'Tag-Based Furniture Recommendation\n'
        title += f'Total Items: {len(self.env.placed_items)} | '
        title += f'Total Reward: {episode_data["total_reward"]:.2f}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Layout visualization saved: {save_path}")
    
    def _draw_spatial_zones(self, ax):
        """Draw spatial zones on the plot"""
        # Seating areas - light red
        for zone in self.env.spatial_zones.get('seating_area', []):
            cx, cy = zone['center']
            radius = zone['radius']
            circle = Circle(
                (cx, cy), radius,
                fill=True, facecolor='red', alpha=0.05,
                edgecolor='red', linewidth=1.5, linestyle='--'
            )
            ax.add_patch(circle)
            ax.text(cx, cy, 'Seating\nZone', ha='center', va='center',
                   fontsize=8, color='red', alpha=0.5, fontweight='bold')
        
        # Corner zones - orange
        for zone in self.env.spatial_zones.get('corner', []):
            cx, cy = zone['center']
            radius = zone['radius']
            circle = Circle(
                (cx, cy), radius,
                fill=True, facecolor='orange', alpha=0.08,
                edgecolor='orange', linewidth=1.5, linestyle=':'
            )
            ax.add_patch(circle)
    
    def _draw_furniture_piece(self, ax, furniture, color, alpha, label_prefix, show_tag, tag_info=None):
        """Draw a furniture piece with optional tag annotation"""
        x = furniture['position']['x']
        y = furniture['position']['y']
        length = furniture['dimensions']['length']
        width = furniture['dimensions']['width']
        
        # Draw furniture box
        furniture_box = FancyBboxPatch(
            (x - length/2, y - width/2),
            length, width,
            boxstyle="round,pad=0.05",
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha
        )
        ax.add_patch(furniture_box)
        
        # Label
        label = f"{label_prefix}{furniture['type']}"
        ax.text(x, y, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='none'))
        
        # Tag annotation
        if show_tag and tag_info:
            primary = tag_info['primary']
            secondary = tag_info['secondary']
            
            tag_text = f"[{primary}]"
            if secondary:
                tag_text += f"\n{', '.join(secondary[:2])}"
                if len(secondary) > 2:
                    tag_text += f"\n+{len(secondary)-2} more"
            
            ax.text(x, y - width/2 - 0.3, tag_text,
                   ha='center', va='top', fontsize=7,
                   style='italic', color=color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            alpha=0.9, edgecolor=color, linewidth=1))
    
    def _add_tag_legend(self, ax, tag_colors):
        """Add legend explaining tag colors"""
        legend_elements = []
        
        for tag, color in sorted(tag_colors.items()):
            # Format tag name
            display_name = tag.replace('_', ' ').title()
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='black', 
                            label=display_name, alpha=0.85)
            )
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.02, 1), fontsize=9,
                 title='Primary Tags', title_fontsize=10, framealpha=0.9)
    
    def visualize_reward_breakdown(self, episode_data, save_path='reward_breakdown_tag_based.png'):
        """Visualize detailed reward component breakdown"""
        if not episode_data['reward_components']:
            print("⚠️  No valid placements to analyze")
            return
        
        # Average components across all placements
        avg_components = {}
        component_names = list(episode_data['reward_components'][0].keys())
        
        for comp in component_names:
            if comp not in ['valid_placement', 'already_placed_penalty']:
                values = [rc[comp] for rc in episode_data['reward_components']]
                avg_components[comp] = np.mean(values)
        
        # Define weights
        weights = {
            'tag_based_placement': 5.0,
            'spatial_constraints': 4.0,
            'functional_requirements': 3.5,
            'zone_preference': 3.0,
            'tag_compatibility': 2.5,
            'diversity': 1.5,
            'accessibility': 1.5,
            'grid_alignment': 1.0,
            'budget_efficiency': 0.8
        }
        
        # Sort by weight
        sorted_components = sorted(
            [(k, v, weights.get(k, 0)) for k, v in avg_components.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Component scores
        components = [item[0] for item in sorted_components]
        scores = [item[1] for item in sorted_components]
        component_weights = [item[2] for item in sorted_components]
        
        # Color coding
        colors = []
        for comp in components:
            if comp == 'tag_based_placement':
                colors.append('#9b59b6')  # Purple
            elif comp == 'spatial_constraints':
                colors.append('#3498db')  # Blue
            elif comp == 'functional_requirements':
                colors.append('#2ecc71')  # Green
            elif comp == 'zone_preference':
                colors.append('#f39c12')  # Orange
            elif comp == 'tag_compatibility':
                colors.append('#e74c3c')  # Red
            else:
                colors.append('#95a5a6')  # Gray
        
        bars = ax1.barh(range(len(components)), scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Highlight tag-based components
        for i, comp in enumerate(components):
            if comp in ['tag_based_placement', 'spatial_constraints', 
                       'functional_requirements', 'zone_preference', 'tag_compatibility']:
                bars[i].set_linewidth(3)
                bars[i].set_edgecolor(colors[i])
        
        ax1.set_yticks(range(len(components)))
        ax1.set_yticklabels([c.replace('_', ' ').title() for c in components], fontsize=10)
        ax1.set_xlabel('Score (0-1)', fontsize=12, fontweight='bold')
        ax1.set_title('Reward Component Scores', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1.1)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (score, weight) in enumerate(zip(scores, component_weights)):
            ax1.text(score + 0.02, i, f'{score:.3f} (×{weight:.1f})', 
                    va='center', fontsize=9, fontweight='bold')
        
        # Right plot: Weighted contributions
        weighted_values = [s * w for s, w in zip(scores, component_weights)]
        
        bars2 = ax2.barh(range(len(components)), weighted_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for i, comp in enumerate(components):
            if comp in ['tag_based_placement', 'spatial_constraints', 
                       'functional_requirements', 'zone_preference', 'tag_compatibility']:
                bars2[i].set_linewidth(3)
                bars2[i].set_edgecolor(colors[i])
        
        ax2.set_yticks(range(len(components)))
        ax2.set_yticklabels([c.replace('_', ' ').title() for c in components], fontsize=10)
        ax2.set_xlabel('Weighted Contribution', fontsize=12, fontweight='bold')
        ax2.set_title('Weighted Reward Contributions', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, val in enumerate(weighted_values):
            ax2.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Reward breakdown saved: {save_path}")
    
    def batch_test(self, num_episodes=20, save_path='batch_test_results_tag_based.png'):
        """Run multiple episodes and analyze performance"""
        print(f"\n{'='*80}")
        print(f"Running Batch Test: {num_episodes} Episodes")
        print(f"{'='*80}\n")
        
        results = {
            'total_rewards': [],
            'items_placed': [],
            'budget_used': [],
            'collisions': [],
            'tag_placement_scores': [],
            'spatial_scores': [],
            'functional_scores': [],
            'zone_scores': [],
            'compatibility_scores': []
        }
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            collision_count = 0
            
            tag_scores = []
            spatial_scores = []
            functional_scores = []
            zone_scores = []
            compatibility_scores = []
            
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Track collisions
                if not info['valid_placement'] and not info['already_placed']:
                    if info.get('violation_reason') == 'collision':
                        collision_count += 1
                
                # Track tag-based scores
                if info['reward_components']['valid_placement']:
                    tag_scores.append(info['reward_components'].get('tag_based_placement', 0))
                    spatial_scores.append(info['reward_components'].get('spatial_constraints', 0))
                    functional_scores.append(info['reward_components'].get('functional_requirements', 0))
                    zone_scores.append(info['reward_components'].get('zone_preference', 0))
                    compatibility_scores.append(info['reward_components'].get('tag_compatibility', 0))
            
            results['total_rewards'].append(episode_reward)
            results['items_placed'].append(info['placed_items'])
            results['budget_used'].append(info['budget_used'])
            results['collisions'].append(collision_count)
            results['tag_placement_scores'].append(np.mean(tag_scores) if tag_scores else 0)
            results['spatial_scores'].append(np.mean(spatial_scores) if spatial_scores else 0)
            results['functional_scores'].append(np.mean(functional_scores) if functional_scores else 0)
            results['zone_scores'].append(np.mean(zone_scores) if zone_scores else 0)
            results['compatibility_scores'].append(np.mean(compatibility_scores) if compatibility_scores else 0)
            
            if (ep + 1) % 5 == 0:
                print(f"  Episode {ep+1}/{num_episodes} complete")
        
        # Create visualization
        self._visualize_batch_results(results, num_episodes, save_path)
        
        # Print summary
        print(f"\n{'='*80}")
        print("Batch Test Results Summary")
        print(f"{'='*80}")
        print(f"Average Reward: {np.mean(results['total_rewards']):.2f} ± {np.std(results['total_rewards']):.2f}")
        print(f"Average Items Placed: {np.mean(results['items_placed']):.2f}/{self.env.max_items}")
        print(f"Success Rate: {sum([1 for x in results['items_placed'] if x == self.env.max_items]) / num_episodes * 100:.1f}%")
        print(f"Average Budget Used: ${np.mean(results['budget_used']):.0f}")
        print(f"Zero Collision Episodes: {sum([1 for x in results['collisions'] if x == 0])}/{num_episodes}")
        print(f"\nTag-Based Metrics:")
        print(f"  Tag Placement Score: {np.mean(results['tag_placement_scores']):.3f} ± {np.std(results['tag_placement_scores']):.3f}")
        print(f"  Spatial Constraints: {np.mean(results['spatial_scores']):.3f} ± {np.std(results['spatial_scores']):.3f}")
        print(f"  Functional Requirements: {np.mean(results['functional_scores']):.3f} ± {np.std(results['functional_scores']):.3f}")
        print(f"  Zone Preference: {np.mean(results['zone_scores']):.3f} ± {np.std(results['zone_scores']):.3f}")
        print(f"  Tag Compatibility: {np.mean(results['compatibility_scores']):.3f} ± {np.std(results['compatibility_scores']):.3f}")
        print(f"{'='*80}\n")
        
        return results
    
    def _visualize_batch_results(self, results, num_episodes, save_path):
        """Visualize batch test results"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Reward distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(results['total_rewards'], bins=15, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(results['total_rewards']), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('Total Reward', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax1.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Items placed distribution
        ax2 = fig.add_subplot(gs[0, 1])
        unique, counts = np.unique(results['items_placed'], return_counts=True)
        ax2.bar(unique, counts, color='#2ecc71', alpha=0.7, edgecolor='black', width=0.6)
        ax2.set_xlabel('Items Placed', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Items Placed Distribution', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(0, self.env.max_items + 1))
        ax2.grid(alpha=0.3, axis='y')
        
        # 3. Budget usage
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(results['budget_used'], bins=15, color='#f39c12', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(results['budget_used']), color='red', linestyle='--', linewidth=2, label='Mean')
        ax3.set_xlabel('Budget Used ($)', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax3.set_title('Budget Usage Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Collisions
        ax4 = fig.add_subplot(gs[1, 0])
        unique, counts = np.unique(results['collisions'], return_counts=True)
        ax4.bar(unique, counts, color='#e74c3c', alpha=0.7, edgecolor='black', width=0.6)
        ax4.set_xlabel('Collision Count', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax4.set_title('Collision Distribution', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')
        
        # 5. Tag placement scores
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(results['tag_placement_scores'], marker='o', color='#9b59b6', linewidth=2, markersize=4, alpha=0.7)
        ax5.axhline(np.mean(results['tag_placement_scores']), color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Episode', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax5.set_title('Tag Placement Scores', fontsize=12, fontweight='bold')
        ax5.set_ylim([0, 1.1])
        ax5.grid(alpha=0.3)
        
        # 6. Tag-based metrics comparison
        ax6 = fig.add_subplot(gs[1, 2])
        tag_metrics = {
            'Tag\nPlacement': np.mean(results['tag_placement_scores']),
            'Spatial\nConstraints': np.mean(results['spatial_scores']),
            'Functional\nReqs': np.mean(results['functional_scores']),
            'Zone\nPreference': np.mean(results['zone_scores']),
            'Tag\nCompatibility': np.mean(results['compatibility_scores'])
        }
        colors_metrics = ['#9b59b6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax6.bar(range(len(tag_metrics)), tag_metrics.values(), 
                      color=colors_metrics, alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_xticks(range(len(tag_metrics)))
        ax6.set_xticklabels(tag_metrics.keys(), fontsize=9, fontweight='bold')
        ax6.set_ylabel('Average Score', fontsize=10, fontweight='bold')
        ax6.set_title('Tag-Based Metrics Comparison', fontsize=12, fontweight='bold')
        ax6.set_ylim([0, 1.1])
        ax6.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Target: 0.8')
        ax6.legend()
        ax6.grid(alpha=0.3, axis='y')
        
        # 7-9. Summary statistics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        summary_text = f"""
BATCH TEST SUMMARY ({num_episodes} episodes)
{'='*100}

OVERALL PERFORMANCE:
  • Average Total Reward: {np.mean(results['total_rewards']):.2f} ± {np.std(results['total_rewards']):.2f}
  • Success Rate: {sum([1 for x in results['items_placed'] if x == self.env.max_items]) / num_episodes * 100:.1f}% ({sum([1 for x in results['items_placed'] if x == self.env.max_items])}/{num_episodes} episodes)
  • Average Items Placed: {np.mean(results['items_placed']):.2f} / {self.env.max_items}
  • Zero Collision Episodes: {sum([1 for x in results['collisions'] if x == 0])}/{num_episodes}
  • Average Budget Used: ${np.mean(results['budget_used']):.0f} ± ${np.std(results['budget_used']):.0f}

TAG-BASED PERFORMANCE:
  • Tag Placement Score: {np.mean(results['tag_placement_scores']):.3f} ± {np.std(results['tag_placement_scores']):.3f}
  • Spatial Constraints Score: {np.mean(results['spatial_scores']):.3f} ± {np.std(results['spatial_scores']):.3f}
  • Functional Requirements Score: {np.mean(results['functional_scores']):.3f} ± {np.std(results['functional_scores']):.3f}
  • Zone Preference Score: {np.mean(results['zone_scores']):.3f} ± {np.std(results['zone_scores']):.3f}
  • Tag Compatibility Score: {np.mean(results['compatibility_scores']):.3f} ± {np.std(results['compatibility_scores']):.3f}

QUALITY INDICATORS:
  • Min Reward: {np.min(results['total_rewards']):.2f} | Max Reward: {np.max(results['total_rewards']):.2f}
  • Reward Std Dev: {np.std(results['total_rewards']):.2f} (Lower is more consistent)
  • Tag Score Consistency: {np.std(results['tag_placement_scores']):.3f} (Lower is better)
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Tag-Based Context-Aware Placement - Batch Test Results', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Batch test results saved: {save_path}")


def main():
    """Main testing function"""
    print("\n" + "🪑 " * 35)
    print("TAG-BASED CONTEXT-AWARE PLACEMENT SYSTEM - TESTING")
    print("🪑 " * 35 + "\n")
    
    # Paths
    room_layout_path = 'room_layout.json'
    catalog_path = 'furniture_catalog_enhanced.json'
    model_path = 'models_tag_based/sac_tag_based_epfinal.pt'
    
    # Check if files exist
    if not os.path.exists(room_layout_path):
        print(f"❌ Error: {room_layout_path} not found!")
        return
    
    if not os.path.exists(catalog_path):
        print(f"❌ Error: {catalog_path} not found!")
        return
    
    if not os.path.exists(model_path):
        print(f"⚠️  Warning: Trained model not found at {model_path}")
        print("   Please train the model first: python train_sac_tag_based.py")
        print("   Continuing with untrained agent for demonstration...\n")
    
    # Create tester
    tester = TagBasedTester(
        room_layout_path=room_layout_path,
        catalog_path=catalog_path,
        model_path=model_path,
        max_items=4
    )
    
    # Run single recommendation
    print("=" * 80)
    print("Running Single Recommendation Test")
    print("=" * 80)
    episode_data = tester.run_single_recommendation()
    
    print(f"\n✅ Placed {len(episode_data['furniture_placed'])} items:")
    for i, (ftype, tag_info) in enumerate(zip(episode_data['furniture_placed'], episode_data['tags'])):
        print(f"   {i+1}. {ftype} [{tag_info['primary']}]")
    print(f"\nTotal Reward: {episode_data['total_reward']:.2f}\n")
    
    # Generate visualizations
    tester.visualize_layout_with_tags(episode_data)
    tester.visualize_reward_breakdown(episode_data)
    
    # Run batch test
    tester.batch_test(num_episodes=20)
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  1. recommendation_layout_tag_based.png - Visual layout with tag annotations")
    print("  2. reward_breakdown_tag_based.png - Detailed reward component analysis")
    print("  3. batch_test_results_tag_based.png - Comprehensive batch statistics")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()